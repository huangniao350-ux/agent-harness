"""事件总线：Harness 执行过程中的所有关键节点都以事件形式广播。

订阅方：
- CLI / API：实时流式展示（SSE）
- TraceSubscriber：落库形成可回放的执行轨迹
- 审计：Guardrails 事件写入审计日志

这种"执行与观测解耦"的设计，让可观测体系对核心链路零侵入。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .utils import now_iso

# 事件类型常量
TASK_STARTED = "task_started"
PLAN_CREATED = "plan_created"
STEP_STARTED = "step_started"
STEP_FINISHED = "step_finished"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
GUARDRAIL = "guardrail"
MEMORY_EVENT = "memory_event"
REFLECT = "reflect"
AGENT_MESSAGE = "agent_message"      # multi-agent 场景中角色间的消息
FINAL_ANSWER = "final_answer"
TASK_FINISHED = "task_finished"
ERROR = "error"


@dataclass
class Event:
    type: str
    trace_id: str = ""
    session_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "trace_id": self.trace_id, "session_id": self.session_id,
                "payload": self.payload, "ts": self.ts}


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """异步事件总线：emit 时顺序等待所有订阅者处理完成，保证 Trace 写入先于任务结束。"""

    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: Handler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        for handler in list(self._handlers):
            try:
                await handler(event)
            except Exception as exc:  # 观测失败不能影响主链路
                print(f"[event-bus] handler error: {exc!r}")

    async def emit_many(self, events: list[Event]) -> None:
        for e in events:
            await self.emit(e)
