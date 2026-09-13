"""AgentRuntime：组合根，装配 LLM/工具/记忆/护栏/编排器/可观测 全部组件。

一次 run() 的完整生命周期：
1. 输入护栏（注入拦截 / 敏感信息脱敏）→ 拦截则直接返回，不进入模型；
2. Trace 开始（事件总线广播）；
3. 按 mode 分发到 ReAct / Plan-and-Execute / Multi-Agent 编排器；
4. 输出护栏（泄露检测 / 脱敏）；
5. 会话记忆沉淀：写入消息、超限压缩、LLM 抽取长期事实（去重/冲突处理）；
6. Trace 结束，返回 AgentResult。

断点续跑：session 命中未完成 Checkpoint 时，从快照恢复 AgentState 继续执行。
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from .config import RuntimeConfig
from .events import (FINAL_ANSWER, TASK_FINISHED, TASK_STARTED, EventBus, Event)
from .guardrails.guardrails import Guardrails
from .harness.checkpoint import CheckpointStore
from .harness.plan_execute import PlanExecuteAgent
from .harness.react import ReActAgent
from .harness.reflection import FailureAttributor
from .harness.state import (MODE_MULTI_AGENT, MODE_PLAN_EXECUTE, MODE_REACT, STATUS_DONE,
                            AgentResult, AgentState)
from .harness.verifier import Verifier
from .llm.base import LLMClient, LLMMessage
from .llm.factory import build_llm
from .memory.memory import LongTermStore, MemoryManager
from .multiagent.orchestrator import MultiAgentOrchestrator
from .observability.trace import TraceDB, TraceSubscriber
from .tools.base import ToolContext
from .tools.builtin import build_builtin_tools
from .tools.registry import ToolRegistry
from .utils import now_iso, truncate

VALID_MODES = (MODE_REACT, MODE_PLAN_EXECUTE, MODE_MULTI_AGENT)


class AgentRuntime:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig.from_env()
        self.config.ensure_dirs()
        self.event_bus = EventBus()
        self.llm: LLMClient = build_llm(self.config.llm)
        self.registry = ToolRegistry(self.config)
        for tool in build_builtin_tools():
            self.registry.register(tool)
        self.longterm = LongTermStore(self.config.db_path, self.config, self.event_bus)
        self._sessions: dict[str, MemoryManager] = {}
        self.guardrails = Guardrails(
            enabled=self.config.guardrails_enabled,
            audit_path=self.config.data_dir / "audit.jsonl",
            event_bus=self.event_bus)
        self.checkpoints = CheckpointStore(self.config.db_path)
        self.traces = TraceDB(self.config.db_path)
        if self.config.trace_enabled:
            self.event_bus.subscribe(TraceSubscriber(self.traces))
        # 编排器装配
        self.react = ReActAgent(self.llm, self.registry, self.config, self.event_bus)
        self.verifier = Verifier(self.llm)
        self.attributor = FailureAttributor(self.llm)
        self.plan_agent = PlanExecuteAgent(self.llm, self.react, self.verifier,
                                           self.attributor, self.config, self.event_bus)
        self.orchestrator = MultiAgentOrchestrator(self.llm, self.react, self.verifier,
                                                   self.attributor, self.config, self.event_bus)

    # ------------------------------------------------------------ 启动/关闭
    async def startup(self) -> None:
        """异步初始化：接入 MCP Server（进程内工具之外的远程工具来源）。"""
        if self.config.enable_mcp:
            cmd = [sys.executable, "-m", "agent_harness.tools.mcp"]
            cwd = str(Path(__file__).resolve().parent.parent)
            try:
                count = await self.registry.register_mcp(cmd, source_name="demo-unit-tools", cwd=cwd)
                if count:
                    print(f"[runtime] 已接入 MCP Server: {count} 个远程工具")
            except Exception as exc:
                print(f"[runtime] MCP 接入失败（不影响本地工具）: {exc}")

    async def close(self) -> None:
        await self.registry.close()

    # ------------------------------------------------------------ 会话
    def memory_for(self, session_id: str) -> MemoryManager:
        if session_id not in self._sessions:
            self._sessions[session_id] = MemoryManager(session_id, self.longterm,
                                                       self.config, self.event_bus)
        return self._sessions[session_id]

    # ------------------------------------------------------------ 主入口
    async def run(self, goal: str, mode: str = MODE_REACT, session_id: str | None = None,
                  resume: bool = False, policy=None) -> AgentResult:
        started = time.perf_counter()
        session_id = session_id or f"s-{uuid.uuid4().hex[:8]}"
        trace_id = uuid.uuid4().hex[:12]
        mode = mode if mode in VALID_MODES else MODE_REACT
        memory = self.memory_for(session_id)
        ctx = ToolContext(session_id=session_id, workspace_dir=self.config.workspace_dir,
                          memory=memory, event_bus=self.event_bus, trace_id=trace_id,
                          config=self.config)

        # 1. 输入护栏
        verdict = self.guardrails.check_input(goal)
        if verdict.action == "block":
            rules = "、".join(v["rule"] for v in verdict.violations)
            await self._emit(ctx, TASK_STARTED, {"goal": goal, "mode": mode, "blocked": True})
            answer = (f"请求已被安全护栏拦截（命中：{rules}）。"
                      "本平台禁止提示词注入类请求，请调整后重试。")
            await self._emit(ctx, FINAL_ANSWER, {"answer": answer, "blocked": True})
            await self._finish(ctx, started, state=None, answer=answer, status="blocked")
            return AgentResult(session_id=session_id, trace_id=trace_id, goal=goal, mode=mode,
                               answer=answer, success=False, steps=0, tool_calls=[],
                               tokens_used=0, latency_ms=self._latency(started),
                               error="guardrail_blocked")
        if verdict.action == "mask":
            goal = verdict.text  # 脱敏后的目标进入执行链路

        await self._emit(ctx, TASK_STARTED, {"goal": goal, "mode": mode})
        memory.add_message("user", goal)

        # 2. 断点续跑
        state: AgentState | None = None
        if resume:
            state = self.checkpoints.load(session_id)
            if state is not None and state.status == STATUS_DONE:
                state = None  # 已完成，无需续跑
        if state is None:
            state = AgentState(session_id=session_id, goal=goal, mode=mode)

        # 3. 分发执行
        if mode == MODE_PLAN_EXECUTE:
            final_state = await self.plan_agent.run(state, ctx, memory=memory,
                                                    checkpoint=self.checkpoints)
        elif mode == MODE_MULTI_AGENT:
            final_state = await self.orchestrator.run(state, ctx, memory=memory,
                                                      checkpoint=self.checkpoints)
        else:
            final_state = await self.react.run(state, ctx, memory=memory,
                                               checkpoint=self.checkpoints)

        # 4. 输出护栏
        answer = final_state.final_answer
        out_verdict = self.guardrails.check_output(answer or "")
        if out_verdict.action == "block":
            answer = "（输出已被安全护栏拦截：检测到疑似敏感信息泄露，详情见审计日志）"
        elif out_verdict.action == "mask":
            answer = out_verdict.text

        # 5. 记忆沉淀
        memory.add_message("assistant", answer or "")
        await memory.compress_if_needed(self.llm)
        await self._extract_facts(goal, memory)

        # 6. 收尾
        success = final_state.status == STATUS_DONE and bool(answer)
        if success:
            self.checkpoints.delete(session_id)
        await self._emit(ctx, FINAL_ANSWER, {"answer": answer, "status": final_state.status})
        await self._finish(ctx, started, state=final_state, answer=answer,
                           status=final_state.status)
        return AgentResult(session_id=session_id, trace_id=trace_id, goal=goal, mode=mode,
                           answer=answer or "", success=success, steps=final_state.steps_used,
                           tool_calls=final_state.tool_calls,
                           tokens_used=final_state.tokens_used,
                           latency_ms=self._latency(started), plan=final_state.plan,
                           error=final_state.error)

    # ------------------------------------------------------------ 辅助
    async def _extract_facts(self, goal: str, memory: MemoryManager) -> None:
        """任务结束后从用户输入中抽取长期事实（subject/key/value），写入时自动去重与冲突处理。"""
        if not goal:
            return
        resp = await self.llm.chat([
            LLMMessage(role="system",
                       content=("你是信息抽取器。[MODE: EXTRACT] 从用户输入中抽取稳定的个人信息事实"
                                "（如 name/dept/employee_id/preference）。只输出 JSON："
                                "{\"facts\": [{\"subject\": \"user\", \"key\": \"...\", \"value\": \"...\"}]}，"
                                "无可抽取事实时输出空数组。")),
            LLMMessage(role="user", content=goal),
        ], json_mode=True, role="default")
        from .utils import extract_json
        data = extract_json(resp.content) or {}
        for fact in (data.get("facts") or [])[:8]:
            try:
                self.longterm.add(str(fact.get("subject", "user")),
                                  str(fact.get("key", "")), str(fact.get("value", "")))
            except (AttributeError, TypeError):
                continue

    async def _emit(self, ctx: ToolContext, event_type: str, payload: dict) -> None:
        await self.event_bus.emit(Event(type=event_type, trace_id=ctx.trace_id,
                                        session_id=ctx.session_id, payload=payload))

    async def _finish(self, ctx: ToolContext, started: float, state, answer: str,
                      status: str) -> None:
        await self.event_bus.emit(Event(type=TASK_FINISHED, trace_id=ctx.trace_id,
                                        session_id=ctx.session_id, payload={
                                            "status": status,
                                            "answer": truncate(answer, 2000),
                                            "tokens": state.tokens_used if state else 0,
                                            "latency_ms": self._latency(started)}))

    @staticmethod
    def _latency(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 1)
