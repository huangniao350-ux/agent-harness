"""Multi-Agent 协同引擎（对应简历：规划/执行/校验多智能体协同框架）。

架构：
- PlannerAgent：把目标拆解为任务并按能力画像分配给执行者（research/ops 两类画像）；
- ExecutorAgent：角色化执行者（差异化 system prompt），内部复用 ReAct Agent 执行任务，
  多个执行者并行处理无依赖任务（asyncio 并发 + 信号量限流）；
- VerifierAgent：校验执行结果，不通过时生成纠错指令回传给原执行者重做（协同纠错闭环）；
- Blackboard：共享黑板承载任务、结果与消息，带版本号实现一致性保障；
- 冲突处理：同一事实键出现不一致结果时记录冲突并交给校验者裁决。

与 Plan-and-Execute 的关系：Plan-Execute 是单 Agent 内的任务拆解；本模块在其之上
引入"角色分工 + 消息传递 + 结果反馈纠错"的协同层。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..events import AGENT_MESSAGE, LLM_CALL, PLAN_CREATED, EventBus, Event, REFLECT, STEP_FINISHED, STEP_STARTED
from ..llm.base import LLMClient, LLMMessage
from ..tools.base import ToolContext
from ..utils import extract_json, truncate
from ..harness.reflection import FailureAttributor
from ..harness.state import STATUS_DONE, STATUS_FAILED, AgentState


@dataclass
class BlackboardEntry:
    key: str
    value: Any
    version: int = 1
    author: str = ""
    ts: float = field(default_factory=time.time)


class Blackboard:
    """共享黑板：写入即版本自增；同键不同值视为冲突并记录。"""

    def __init__(self) -> None:
        self._data: dict[str, BlackboardEntry] = {}
        self.conflicts: list[dict] = []

    def write(self, key: str, value: Any, author: str) -> None:
        entry = self._data.get(key)
        if entry is None:
            self._data[key] = BlackboardEntry(key=key, value=value, author=author)
            return
        if entry.value != value:
            self.conflicts.append({"key": key, "existing": str(entry.value)[:80],
                                   "incoming": str(value)[:80],
                                   "resolution": "保留先写入结果（校验者裁决）"})
            return  # 一致性保障：先写入者胜，冲突记录供校验者裁决
        entry.version += 1

    def read(self, key: str) -> Any:
        entry = self._data.get(key)
        return entry.value if entry else None

    def snapshot(self) -> dict[str, Any]:
        return {k: e.value for k, e in self._data.items()}


ROLE_PROMPTS = {
    "planner": "你是规划智能体（Planner）。负责任务拆解与分工，只输出 JSON。",
    "research": ("你是信息检索执行者（Research Executor），擅长查询、检索与数据核实。"
                 "完成分配给你的子任务，给出明确、可验证的结论。"),
    "ops": ("你是事务办理执行者（Ops Executor），擅长在业务系统中执行操作（提交、办理、写入）。"
            "完成分配给你的子任务，报告操作结果与单据号。"),
    "verifier": "你是校验智能体（Verifier）。独立核验执行结果，发现问题必须指出并给出纠错指令。",
}


class MultiAgentOrchestrator:
    def __init__(self, llm: LLMClient, react_agent, verifier, attributor: FailureAttributor,
                 config=None, event_bus: EventBus | None = None) -> None:
        self.llm = llm
        self.react = react_agent
        self.verifier = verifier
        self.attributor = attributor
        self.config = config
        self.event_bus = event_bus
        self.executors: dict[str, str] = {"research": "research", "ops": "ops"}

    async def _emit(self, ctx: ToolContext, event_type: str, payload: dict) -> None:
        if self.event_bus is not None:
            await self.event_bus.emit(Event(type=event_type, trace_id=ctx.trace_id,
                                            session_id=ctx.session_id, payload=payload))

    async def _chat(self, messages: list[LLMMessage], state: AgentState, ctx: ToolContext,
                    role: str = "planner", json_mode: bool = False) -> str:
        resp = await self.llm.chat(messages, json_mode=json_mode, role=role)
        state.tokens_used += resp.usage.total_tokens
        if self.event_bus is not None:
            await self.event_bus.emit(Event(type=LLM_CALL, trace_id=ctx.trace_id,
                                            session_id=ctx.session_id, payload={
                                                "role": role, "model": resp.model,
                                                "content": truncate(resp.content, 2000),
                                                "prompt_tokens": resp.usage.prompt_tokens,
                                                "completion_tokens": resp.usage.completion_tokens,
                                                "latency_ms": 0}))
        return resp.content

    # ------------------------------------------------------------ 规划与分工
    async def _plan_and_assign(self, state: AgentState, ctx: ToolContext) -> list[dict]:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.react.registry.list_tools())
        content = await self._chat([
            LLMMessage(role="system", content=ROLE_PROMPTS["planner"] +
                       " 将目标拆解为任务，并分配给执行者（executor 只能是 research 或 ops）。"
                       ' 只输出 JSON：{"tasks": [{"id": "t1", "description": "...", '
                       '"executor": "research", "depends_on": []}]}'),
            LLMMessage(role="user", content=f"目标：{state.goal}\n可用工具：\n{tool_lines}"),
        ], state, ctx, role="planner", json_mode=True)
        data = extract_json(content) or {}
        tasks = data.get("tasks") if isinstance(data, dict) else None
        clean: list[dict] = []
        ids: set[str] = set()
        for t in (tasks or []):
            tid = str(t.get("id") or f"t{len(clean) + 1}")
            desc = str(t.get("description") or "").strip()
            if not desc or tid in ids:
                continue
            clean.append({"id": tid, "description": desc,
                          "executor": t.get("executor") if t.get("executor") in self.executors else "research",
                          "depends_on": [d for d in (t.get("depends_on") or []) if d in ids]})
            ids.add(tid)
        if not clean:
            clean = [{"id": "t1", "description": f"完成目标：{state.goal}",
                      "executor": "research", "depends_on": []}]
        return clean

    # ------------------------------------------------------------ 执行者
    async def _executor_run(self, state: AgentState, task: dict, ctx: ToolContext,
                            memory=None, correction: str = "") -> dict:
        role = task["executor"]
        await self._emit(ctx, STEP_STARTED, {"agent": role, "step_id": task["id"],
                                             "description": task["description"]})
        await self._emit(ctx, AGENT_MESSAGE, {"from": "planner", "to": role,
                                              "content": truncate(task["description"], 100)})
        sub_state = AgentState(
            session_id=f"{state.session_id}:{task['id']}",
            goal=f"{task['description']}（总目标：{truncate(state.goal, 120)}）"
                 + (f"\n{correction}" if correction else ""),
            mode=state.mode)
        await self.react.run(sub_state, ctx, memory=memory,
                             max_steps=self.config.harness.max_step_react_steps if self.config else 4,
                             system_extra=ROLE_PROMPTS[role])
        state.tokens_used += sub_state.tokens_used
        state.tool_calls.extend(sub_state.tool_calls)
        await self._emit(ctx, AGENT_MESSAGE, {"from": role, "to": "verifier",
                                              "content": truncate(sub_state.final_answer, 100)})
        return {"task": task, "ok": sub_state.status == STATUS_DONE,
                "result": sub_state.final_answer or sub_state.error}

    # ------------------------------------------------------------ 校验者
    async def _verifier_check(self, state: AgentState, task: dict, result_text: str,
                              ctx: ToolContext) -> tuple[bool, str]:
        verdict = await self.verifier.verify(state.goal, result_text)
        await self._emit(ctx, AGENT_MESSAGE, {"from": "verifier", "to": task["executor"],
                                              "content": truncate(verdict.reason, 100)})
        return verdict.passed, verdict.reason

    # ------------------------------------------------------------ 主流程
    async def run(self, state: AgentState, ctx: ToolContext, memory=None, checkpoint=None) -> AgentState:
        max_correction_rounds = 1
        blackboard = Blackboard()
        tasks = await self._plan_and_assign(state, ctx)
        state.plan = tasks
        await self._emit(ctx, PLAN_CREATED, {"tasks": tasks})
        results: dict[str, dict] = {}

        by_id = {t["id"]: t for t in tasks}
        done: set[str] = set()
        pending = list(tasks)
        while pending:
            ready = [t for t in pending if all(d in done for d in t["depends_on"])]
            if not ready:
                ready = [pending[0]]
            sem = asyncio.Semaphore(self.config.harness.parallel_workers if self.config else 3)

            async def _worker(task: dict) -> tuple[str, dict]:
                async with sem:
                    outcome = await self._executor_run(state, task, ctx, memory)
                    passed, reason = await self._verifier_check(state, task, outcome["result"], ctx)
                    # 协同纠错闭环：校验不通过 → 带纠错指令重做一次
                    if not passed and max_correction_rounds > 0:
                        analysis = await self.attributor.attribute(state.goal, reason)
                        await self._emit(ctx, REFLECT, {"step_id": task["id"], **analysis})
                        correction = FailureAttributor.feedback_text(analysis)
                        await self._emit(ctx, AGENT_MESSAGE, {"from": "verifier", "to": task["executor"],
                                                              "content": truncate(correction, 120)})
                        outcome = await self._executor_run(state, task, ctx, memory, correction)
                        passed, _ = await self._verifier_check(state, task, outcome["result"], ctx)
                    blackboard.write(f"task:{task['id']}", outcome["result"], task["executor"])
                    return task["id"], {"status": "done" if (outcome["ok"] and passed) else "failed",
                                        "result": outcome["result"]}

            outcomes = await asyncio.gather(*[_worker(t) for t in ready])
            for tid, res in outcomes:
                results[tid] = res
                done.add(tid)
                pending = [t for t in pending if t["id"] != tid]
                await self._emit(ctx, STEP_FINISHED, {"step_id": tid, **res})
            if checkpoint:
                from ..utils import now_iso
                state.step_results = {tid: {"status": r["status"], "result": r["result"]}
                                      for tid, r in results.items()}
                checkpoint.save(state, now_iso())

        if blackboard.conflicts:
            await self._emit(ctx, REFLECT, {"stage": "consistency",
                                            "cause": "多执行者结果冲突",
                                            "suggestion": "以校验者裁决为准",
                                            "conflicts": blackboard.conflicts})

        state.step_results = {tid: {"status": r["status"], "result": r["result"]}
                              for tid, r in results.items()}
        good = [r["result"] for r in results.values() if r["status"] == "done"]
        if good and len(good) == len(results):
            state.final_answer = await self._aggregate(state, ctx, good)
            state.status = STATUS_DONE
        else:
            state.status = STATUS_FAILED
            state.error = "部分任务未能通过校验者核验"
        if checkpoint:
            from ..utils import now_iso
            checkpoint.save(state, now_iso())
        return state

    async def _aggregate(self, state: AgentState, ctx: ToolContext, results: list[str]) -> str:
        if len(results) == 1:
            return results[0]
        content = "\n".join(f"- {r[:150]}" for r in results)
        return await self._chat([
            LLMMessage(role="system", content=ROLE_PROMPTS["planner"] +
                       " 综合所有执行者的结果，输出一段面向用户的最终回答。"),
            LLMMessage(role="user", content=f"目标：{state.goal}\n执行者结果：\n{content}"),
        ], state, ctx, role="synthesizer")
