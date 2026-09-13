"""ReAct Agent Loop：Harness 双模式之一（对应简历：基于状态机实现 ReAct 执行链路）。

循环：LLM 思考 → 解析（Thought/Action/Action Input 或 Final Answer）→ 工具调用
（弹性执行层兜底）→ Observation 追加 → 下一轮。每步后落 Checkpoint。
所有状态收敛在 AgentState，循环本身无隐藏状态 —— 可中断、可续跑、可回放。
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..events import LLM_CALL, EventBus, Event
from ..llm.base import LLMClient, LLMMessage, LLMResponse
from ..tools.base import ToolContext
from ..tools.registry import ToolRegistry
from ..utils import extract_json, now_iso, truncate
from .state import STATUS_DONE, STATUS_INCOMPLETE, AgentState


def parse_react(text: str) -> dict[str, Any]:
    """解析 ReAct 输出，容忍格式抖动。返回 {thought, action, args, final}。"""
    out: dict[str, Any] = {"thought": "", "action": "", "args": {}, "final": ""}
    final_m = re.search(r"Final Answer\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    action_m = re.search(r"Action\s*[:：]\s*([A-Za-z_][\w.\-]*)", text, re.IGNORECASE)
    thought_m = re.search(r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final Answer)|$)",
                          text, re.DOTALL | re.IGNORECASE)
    if thought_m:
        out["thought"] = thought_m.group(1).strip()
    if final_m and not action_m:
        out["final"] = final_m.group(1).strip()
        return out
    if action_m:
        out["action"] = action_m.group(1).strip()
        input_m = re.search(r"Action Input\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if input_m:
            out["args"] = extract_json(input_m.group(1)) or {}
            if not isinstance(out["args"], dict):
                out["args"] = {}
        return out
    if not out["final"]:
        out["final"] = text.strip()  # 模型没按格式输出时，将原文作为回答兜底
    return out


def render_scratchpad(state: AgentState, limit: int = 8) -> str:
    if not state.scratchpad:
        return "（暂无）"
    lines = []
    for item in state.scratchpad[-limit:]:
        lines.append(f"Thought: {item.get('thought', '')}")
        if item.get("action"):
            lines.append(f"Action: {item['action']}")
            lines.append(f"Action Input: {item.get('args', {})}")
            lines.append(f"Observation: {truncate(item.get('observation', ''), 400)}")
        elif item.get("observation"):
            lines.append(f"Observation: {truncate(item['observation'], 400)}")
    return "\n".join(lines)


REACT_SYSTEM = """[MODE: REACT]
你是企业智能助理，需要通过逐步调用工具来完成用户目标。
{system_extra}
可用工具：
{tools}

严格按以下两种格式之一输出（不要输出多余内容）：
方式一（需要调用工具时）：
Thought: 你的思考
Action: 工具名（必须是可用工具列表中的名字）
Action Input: JSON格式的参数

方式二（可以直接给出最终回答时）：
Thought: 你的思考
Final Answer: 最终回答"""


class ReActAgent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, config=None,
                 event_bus: EventBus | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.config = config
        self.event_bus = event_bus

    async def _chat(self, messages: list[LLMMessage], state: AgentState,
                    ctx: ToolContext, role: str = "actor", json_mode: bool = False) -> LLMResponse:
        started = time.perf_counter()
        resp = await self.llm.chat(messages, json_mode=json_mode, role=role)
        latency = (time.perf_counter() - started) * 1000
        state.tokens_used += resp.usage.total_tokens
        if self.event_bus is not None:
            await self.event_bus.emit(Event(type=LLM_CALL, trace_id=ctx.trace_id,
                                            session_id=ctx.session_id, payload={
                                                "role": role, "model": resp.model,
                                                "content": truncate(resp.content, 2000),
                                                "prompt_tokens": resp.usage.prompt_tokens,
                                                "completion_tokens": resp.usage.completion_tokens,
                                                "latency_ms": round(latency, 1)}))
        return resp

    async def run(self, state: AgentState, ctx: ToolContext, memory=None,
                  max_steps: int | None = None, system_extra: str = "",
                  checkpoint=None) -> AgentState:
        max_steps = max_steps or (self.config.harness.max_react_steps if self.config else 8)
        tool_lines = "\n".join(t.schema_line() for t in self.registry.list_tools())
        system = REACT_SYSTEM.format(tools=tool_lines, system_extra=system_extra).strip()

        for _ in range(max_steps - state.steps_used):
            memory_block = memory.build_context(state.goal) if memory else ""
            user_content = "\n\n".join(filter(None, [
                memory_block,
                f"目标：{state.goal}",
                f"执行记录（Thought/Action/Observation）：\n{render_scratchpad(state)}",
            ]))
            messages = [LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user_content)]
            resp = await self._chat(messages, state, ctx)
            parsed = parse_react(resp.content)
            state.steps_used += 1

            if parsed["final"]:
                state.final_answer = parsed["final"]
                state.status = STATUS_DONE
                if checkpoint:
                    checkpoint.save(state, now_iso())
                return state

            if not parsed["action"]:
                # 既无 Action 也无 Final：把解析错误回灌给模型自我修正
                state.scratchpad.append({"thought": parsed["thought"],
                                         "observation": "ERROR: 输出格式不符合 ReAct 规范，请按指定格式重新输出。"})
                continue

            tool_result = await self.registry.call(parsed["action"], parsed["args"], ctx)
            state.tool_calls.append({"tool": parsed["action"], "args": parsed["args"],
                                     "ok": tool_result.ok})
            if tool_result.meta.get("degraded_from"):
                # 降级链路：备用工具的调用也要计入调用轨迹（评估可观测）
                state.tool_calls.append({"tool": tool_result.meta.get("tool", ""),
                                         "args": parsed["args"],
                                         "ok": tool_result.ok,
                                         "degraded_from": tool_result.meta["degraded_from"]})
            observation = tool_result.observe()
            state.scratchpad.append({"thought": parsed["thought"], "action": parsed["action"],
                                     "args": parsed["args"], "observation": observation})
            if checkpoint:
                checkpoint.save(state, now_iso())

        state.status = STATUS_INCOMPLETE
        state.error = state.error or f"达到最大步数上限（{max_steps}）仍未完成"
        return state
