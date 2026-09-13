"""统一 Tool Registry（对应简历：基于 MCP 协议的统一 Tool Registry 与 Skill 抽象层）。

- 动态注册：进程内工具直接注册；远程 MCP Server 通过 stdio JSON-RPC 接入；
- 发现：list_tools() 汇总本地与 MCP 工具，供 Prompt 渲染与 /tools 接口使用；
- 调用：统一走弹性执行层（校验/超时/重试/去重/降级），与工具实现完全解耦。
"""

from __future__ import annotations

from typing import Any

from ..events import EventBus
from .base import BaseTool, ToolContext, ToolResult
from .execution import ToolCallCache, execute_tool


class UnknownToolError(KeyError):
    pass


class ToolRegistry:
    def __init__(self, config=None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._mcp_sessions: list[Any] = []      # 保持 MCP 子进程会话引用
        self._cache = ToolCallCache()
        self.config = config
        self.confirmation_gate = None           # ConfirmationGate，由 Runtime 装配

    # ------------------------------------------------ 注册
    def register(self, tool: BaseTool, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    async def register_mcp(self, command: list[str], source_name: str = "",
                           cwd: str | None = None) -> int:
        """连接一个 stdio MCP Server，把其 tools 全部注册进来。返回接入的工具数。"""
        from .mcp import MCPClientSession, MCPToolAdapter
        session = MCPClientSession(command, cwd=cwd)
        await session.start()
        remote_tools = await session.list_tools()
        label = source_name or f"mcp://{command[-1]}"
        for spec in remote_tools:
            self.register(MCPToolAdapter(session, spec, source=label), replace=True)
        self._mcp_sessions.append(session)
        return len(remote_tools)

    async def close(self) -> None:
        for session in self._mcp_sessions:
            try:
                await session.close()
            except Exception:
                pass
        self._mcp_sessions.clear()

    # ------------------------------------------------ 发现
    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[BaseTool]:
        return sorted(self._tools.values(), key=lambda t: (t.source != "in-process", t.name))

    def spec(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "params": t.params,
                 "source": t.source} for t in self.list_tools()]

    # ------------------------------------------------ 调用
    async def call(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            tool = self.get(name)
        except UnknownToolError:
            hints = ", ".join(t.name for t in self.list_tools()) or "（无）"
            return ToolResult(ok=False, error=f"未知工具: {name}（可用工具: {hints}）",
                              meta={"stage": "registry"})
        # 高危操作确认门：写操作类工具执行前必须获得批准（决策留痕审计）
        if getattr(tool, "risk_level", "low") == "high" and self.confirmation_gate is not None:
            gate_mode = getattr(ctx, "confirm_mode", None) or self.confirmation_gate.mode
            if gate_mode != "off":
                decision = await self.confirmation_gate.confirm(tool, args, ctx, mode=gate_mode)
                if not decision.approved:
                    return ToolResult(
                        ok=False,
                        error=f"人工确认门拒绝：{decision.reason or '操作未被批准'}",
                        meta={"stage": "confirmation", "tool": name, "confirmation": "rejected"})
                confirmed_via = decision.via
            else:
                confirmed_via = None
        else:
            confirmed_via = None
        result = await execute_tool(
            tool, args, ctx, cache=self._cache,
            default_timeout=getattr(self.config.harness, "tool_call_timeout_s", 10.0) if self.config else 10.0,
            default_retries=getattr(self.config.harness, "tool_retries", 2) if self.config else 2,
        )
        if confirmed_via:
            # 批准通过的高危调用在轨迹中标记确认来源（auto/prompt/user）
            result.meta = {**result.meta, "confirmation": confirmed_via}
        return result
