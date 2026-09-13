"""Plan-and-Execute 编排器：Harness 双模式之二。

流程：规划（LLM 产出步骤 DAG）→ 拓扑分层 → 按层并行执行（无依赖步骤并发）
→ 每步校验（失败 → 反思归因 → 带反馈重试一次）→ 综合器产出最终回答 → 终验。
每层执行后落 Checkpoint，断点续跑时已完成步骤直接跳过。
"""

from __future__ import annotations

import asyncio
import time

from ..events import EventBus, Event, LLM_CALL, PLAN_CREATED, REFLECT, STEP_FINISHED, STEP_STARTED
from ..llm.base import LLMClient, LLMMessage
from ..tools.base import ToolContext
from ..utils import extract_json, now_iso, truncate
from .reflection import FailureAttributor
from .state import STATUS_DONE, STATUS_FAILED, AgentState
from .verifier import Verifier


class PlanExecuteAgent:
    def __init__(self, llm: LLMClient, react_agent, verifier: Verifier,
                 attributor: FailureAttributor, config=None, event_bus: EventBus | None = None) -> None:
        self.llm = llm
        self.react = react_agent
        self.verifier = verifier
        self.attributor = attributor
        self.config = config
        self.event_bus = event_bus

    # ------------------------------------------------------------ 规划
    async def _make_plan(self, state: AgentState, ctx: ToolContext) -> list[dict]:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.react.registry.list_tools())
        feedback = f"\n{state.failure_feedback}" if state.failure_feedback else ""
        resp = await self.llm.chat([
            LLMMessage(role="system",
                       content=("你是任务规划器。[MODE: PLAN] 将目标拆解为可执行步骤（每步应尽量绑定一个工具）。"
                                "只输出 JSON：{\"steps\": [{\"id\": \"s1\", \"description\": \"...\", \"depends_on\": []}]}")),
            LLMMessage(role="user",
                       content=f"目标：{state.goal}\n可用工具：\n{tool_lines}{feedback}"),
        ], json_mode=True, role="planner")
        state.tokens_used += resp.usage.total_tokens
        if self.event_bus is not None:
            await self.event_bus.emit(Event(type=LLM_CALL, trace_id=ctx.trace_id,
                                            session_id=ctx.session_id, payload={
                                                "role": "planner", "model": resp.model,
                                                "content": truncate(resp.content, 2000),
                                                "prompt_tokens": resp.usage.prompt_tokens,
                                                "completion_tokens": resp.usage.completion_tokens,
                                                "latency_ms": 0}))
        data = extract_json(resp.content) or {}
        steps = data.get("steps") if isinstance(data, dict) else None
        return self._validate_plan(steps or [])

    @staticmethod
    def _validate_plan(steps: list[dict]) -> list[dict]:
        """规划后校验：id 唯一、依赖存在、无环 → 异常时退化为单步直答。"""
        clean: list[dict] = []
        ids: set[str] = set()
        for s in steps:
            sid = str(s.get("id") or f"s{len(clean) + 1}")
            desc = str(s.get("description") or "").strip()
            if not desc or sid in ids:
                continue
            deps = [d for d in (s.get("depends_on") or []) if d in ids]
            clean.append({"id": sid, "description": desc, "depends_on": deps})
            ids.add(sid)
        if not clean:
            return [{"id": "s1", "description": "直接理解并回答用户目标", "depends_on": []}]
        # 环检测：尝试拓扑排序，失败则去掉全部依赖退化为顺序执行
        resolved: set[str] = set()
        for _ in range(len(clean)):
            for s in clean:
                if s["id"] not in resolved and all(d in resolved for d in s["depends_on"]):
                    resolved.add(s["id"])
        if len(resolved) != len(clean):
            for s in clean:
                s["depends_on"] = []
        return clean

    @staticmethod
    def _topo_levels(steps: list[dict]) -> list[list[dict]]:
        by_id = {s["id"]: s for s in steps}
        done: set[str] = set()
        levels: list[list[dict]] = []
        remaining = list(steps)
        while remaining:
            level = [s for s in remaining if all(d in done for d in s["depends_on"])]
            if not level:  # 兜底：剩余步骤顺序执行
                level = [remaining[0]]
            levels.append(level)
            done.update(s["id"] for s in level)
            remaining = [s for s in remaining if s["id"] not in done]
        return levels

    # ------------------------------------------------------------ 单步执行
    async def _run_step(self, state: AgentState, step: dict, ctx: ToolContext,
                        memory=None, feedback: str = "") -> dict:
        prior = "\n".join(f"- [{sid}] {r.get('result', '')[:120]}"
                          for sid, r in state.step_results.items() if r.get("status") == "done")
        step_goal = f"{step['description']}（总目标：{truncate(state.goal, 120)}）"
        sub_state = AgentState(session_id=f"{state.session_id}:{step['id']}",
                               goal=step_goal + (f"\n{feedback}" if feedback else ""),
                               mode=state.mode)
        if prior:
            sub_state.scratchpad = [{"thought": "已完成的前序步骤结果：", "observation": prior}]
        await self._emit(ctx, STEP_STARTED, {"step_id": step["id"], "description": step["description"]})
        await self.react.run(sub_state, ctx, memory=memory,
                             max_steps=self.config.harness.max_step_react_steps if self.config else 4)
        ok = sub_state.status == STATUS_DONE and bool(sub_state.final_answer)
        state.tokens_used += sub_state.tokens_used
        state.tool_calls.extend(sub_state.tool_calls)
        return {"step": step, "ok": ok, "result": sub_state.final_answer or sub_state.error,
                "sub_state": sub_state}

    async def _emit(self, ctx: ToolContext, event_type: str, payload: dict) -> None:
        if self.event_bus is not None:
            await self.event_bus.emit(Event(type=event_type, trace_id=ctx.trace_id,
                                            session_id=ctx.session_id, payload=payload))

    # ------------------------------------------------------------ 主流程
    async def run(self, state: AgentState, ctx: ToolContext, memory=None, checkpoint=None) -> AgentState:
        started = time.perf_counter()
        max_rounds = self.config.harness.max_repair_rounds if self.config else 2

        for round_no in range(max_rounds + 1):
            state.rounds = round_no
            if not state.plan or round_no > 0:
                state.plan = await self._make_plan(state, ctx)
                await self._emit(ctx, PLAN_CREATED, {"steps": state.plan, "round": round_no})

            levels = self._topo_levels(state.plan)
            sem = asyncio.Semaphore(self.config.harness.parallel_workers if self.config else 3)

            for level in levels:
                async def _guarded(step: dict):
                    async with sem:
                        if state.step_results.get(step["id"], {}).get("status") == "done":
                            return None  # 断点续跑：已完成步骤跳过
                        outcome = await self._run_step(state, step, ctx, memory)
                        result: dict = {"status": "done" if outcome["ok"] else "failed",
                                        "result": outcome["result"]}
                        # 每步校验 + 单步反思重试
                        if outcome["ok"]:
                            verdict = await self.verifier.verify(state.goal, outcome["result"])
                            if not verdict.passed:
                                analysis = await self.attributor.attribute(state.goal, verdict.reason)
                                await self._emit(ctx, REFLECT, {"step_id": step["id"], **analysis})
                                retry = await self._run_step(
                                    state, step, ctx, memory,
                                    feedback=FailureAttributor.feedback_text(analysis))
                                if retry["ok"]:
                                    result = {"status": "done", "result": retry["result"]}
                        if result["status"] != "done":
                            analysis = await self.attributor.attribute(
                                state.goal, result.get("result", ""))
                            await self._emit(ctx, REFLECT, {"step_id": step["id"], **analysis})
                            state.failure_feedback = FailureAttributor.feedback_text(analysis)
                        return step["id"], result

                outcomes = await asyncio.gather(*[_guarded(s) for s in level])
                for outcome in outcomes:
                    if outcome is None:
                        continue
                    sid, result = outcome
                    state.step_results[sid] = result
                    await self._emit(ctx, STEP_FINISHED, {"step_id": sid, **result})
                    if checkpoint:
                        checkpoint.save(state, now_iso())

            all_ok = all(r.get("status") == "done" for r in state.step_results.values()) \
                and len(state.step_results) >= len(state.plan)
            if all_ok:
                final = await self._synthesize(state, ctx)
                verdict = await self.verifier.verify(state.goal, final, is_final=True)
                if verdict.passed:
                    state.final_answer = final
                    state.status = STATUS_DONE
                    if checkpoint:
                        checkpoint.save(state, now_iso())
                    return state
                state.failure_feedback = f"最终回答未通过校验：{verdict.reason}"
            else:
                failed = [sid for sid, r in state.step_results.items() if r.get("status") != "done"]
                state.failure_feedback = (state.failure_feedback or "") + \
                    f"（失败步骤：{', '.join(failed)}）"
            # 进入下一轮前重置步骤结果，携带反思反馈重新规划
            if round_no < max_rounds:
                state.step_results = {sid: r for sid, r in state.step_results.items()
                                      if r.get("status") == "done"}
                state.plan = []

        state.status = STATUS_FAILED
        state.error = state.error or "重规划轮次耗尽仍未完成任务"
        if checkpoint:
            checkpoint.save(state, now_iso())
        return state

    async def _synthesize(self, state: AgentState, ctx: ToolContext) -> str:
        results = "\n".join(f"[{sid}] {r.get('result', '')}" for sid, r in state.step_results.items())
        if len(state.step_results) <= 1:
            return next(iter(state.step_results.values()), {}).get("result", "")
        resp = await self.llm.chat([
            LLMMessage(role="system",
                       content=("你是结果综合器。[MODE: SYNTH] 将各步骤结果综合为一段连贯的最终回答，"
                                "直接给用户，不要复述步骤编号。")),
            LLMMessage(role="user", content=f"目标：{state.goal}\n各步骤结果：\n{results}"),
        ], role="synthesizer")
        state.tokens_used += resp.usage.total_tokens
        return resp.content.strip()
