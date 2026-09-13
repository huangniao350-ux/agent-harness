"""高危操作人工确认门（对应简历：高危操作设人工确认门）。

risk_level="high" 的工具（写操作类：请假提交/发邮件/写文件）在执行前必须经过确认：
- auto 模式（默认）：自动批准，但每次决策都写入审计并广播事件 —— 演示模式不中断流程；
- manual 模式：挂起等待人工决定，三条决定路径任选其一：
  1) prompt 回调（CLI 交互式 y/N）；
  2) decide() API（控制台/SSE 客户端调 POST /v1/agent/confirm）；
  3) 超时自动拒绝（防止任务永久挂起）。
所有批准/拒绝决策均落 JSONL 审计（stage=confirm），与 Guardrails 审计同一文件可回溯。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..events import Event, EventBus
from ..utils import now_iso
from .base import BaseTool, ToolContext

# 事件类型
CONFIRM_REQUEST = "confirmation_request"
CONFIRM_DECIDED = "confirmation_decided"


@dataclass
class ConfirmDecision:
    request_id: str
    tool: str
    args: dict
    approved: bool
    via: str = ""          # auto | user | prompt | timeout
    reason: str = ""


class ConfirmationGate:
    def __init__(self, mode: str = "auto", event_bus: EventBus | None = None,
                 audit_path: Path | None = None, timeout_s: float = 120.0,
                 prompt=None) -> None:
        if mode not in ("auto", "manual"):
            raise ValueError(f"未知确认门模式: {mode}")
        self.mode = mode
        self.event_bus = event_bus
        self.timeout_s = timeout_s
        self.prompt = prompt                     # async (tool, args, ctx) -> bool
        self._pending: dict[str, asyncio.Future] = {}
        from ..guardrails.guardrails import AuditLogger
        self.audit = AuditLogger(audit_path or Path(".agent_data/audit.jsonl"))

    # ------------------------------------------------------------ 入口
    async def confirm(self, tool: BaseTool, args: dict, ctx: ToolContext,
                      mode: str | None = None) -> ConfirmDecision:
        """高危工具执行前调用；返回批准/拒绝决定。低风险工具不应走到这里。

        mode：本次调用的生效模式（run 级覆盖），缺省用构造时的全局模式。
        """
        effective = mode if mode in ("auto", "manual") else self.mode
        if effective == "auto":
            return await self._finish(tool, args, ctx, approved=True, via="auto",
                                      reason="演示模式自动批准（决策已留痕）")

        # manual：优先交互式 prompt，否则挂起等待 decide()/超时
        if self.prompt is not None:
            request_id = uuid.uuid4().hex[:12]
            await self._emit(ctx, CONFIRM_REQUEST, self._payload(
                tool, args, ctx, request_id, note="等待操作者确认（交互式）"))
            try:
                approved = bool(await self.prompt(tool, args, ctx))
            except Exception:
                approved = False
            return await self._finish(tool, args, ctx, approved=approved,
                                      via="prompt",
                                      reason="" if approved else "操作者在交互提示中拒绝")

        request_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        await self._emit(ctx, CONFIRM_REQUEST, self._payload(
            tool, args, ctx, request_id, note="等待人工决定（调用 /v1/agent/confirm）"))
        try:
            approved = await asyncio.wait_for(future, timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return await self._finish(tool, args, ctx, approved=False, via="timeout",
                                      reason=f"确认等待超时（>{self.timeout_s:.0f}s），自动拒绝",
                                      request_id=request_id)
        finally:
            self._pending.pop(request_id, None)
        return await self._finish(tool, args, ctx, approved=bool(approved), via="user",
                                  reason="" if approved else "操作者拒绝了本次操作",
                                  request_id=request_id)

    # ------------------------------------------------------------ 外部决定
    def decide(self, request_id: str, approved: bool) -> bool:
        """API/控制台对某个待确认请求作出决定。返回该请求是否存在。"""
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(bool(approved))
        return True

    def pending_requests(self) -> list[dict]:
        return [{"request_id": rid} for rid in self._pending]

    # ------------------------------------------------------------ 内部
    async def _emit(self, ctx: ToolContext, event_type: str, payload: dict) -> None:
        if self.event_bus is not None:
            await self.event_bus.emit(Event(type=event_type, trace_id=ctx.trace_id,
                                            session_id=ctx.session_id, payload=payload))

    async def _finish(self, tool: BaseTool, args: dict, ctx: ToolContext, *, approved: bool,
                      via: str, reason: str, request_id: str = "") -> ConfirmDecision:
        decision = ConfirmDecision(request_id=request_id, tool=tool.name, args=args,
                                   approved=approved, via=via, reason=reason)
        self.audit.log("confirm", "approve" if approved else "deny",
                       [{"type": "confirmation", "rule": f"高危工具:{tool.name}",
                         "via": via, "reason": reason}], str(args))
        if self.event_bus is not None:
            await self.event_bus.emit(Event(
                type=CONFIRM_DECIDED, trace_id=ctx.trace_id, session_id=ctx.session_id,
                payload={"request_id": request_id, "tool": tool.name,
                         "approved": approved, "via": via, "reason": reason}))
        return decision

    @staticmethod
    def _payload(tool: BaseTool, args: dict, ctx: ToolContext, request_id: str,
                 note: str) -> dict:
        return {"request_id": request_id, "tool": tool.name, "args": args,
                "session_id": ctx.session_id, "note": note}
