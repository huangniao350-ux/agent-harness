"""手写最小 MCP（Model Context Protocol）实现：JSON-RPC 2.0 over stdio。

MCP 是 Anthropic 提出的模型-工具连接开放协议。本文件不依赖官方 SDK，直接实现协议核心：
- 服务端：initialize / notifications/initialized / tools/list / tools/call；
- 客户端：启动子进程，完成握手，发现工具并封装为注册中心可用的 Tool。

帧格式：stdio 上按行分隔的 JSON-RPC 消息（与 MCP 规范的 stdio transport 一致）。
演示服务端提供 unit_convert / timezone_convert 两个工具。
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Any

from .base import BaseTool, ToolResult

PROTOCOL_VERSION = "2024-11-05"


def _reply(msg_id: Any, result: dict) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}, ensure_ascii=False)


def _error(msg_id: Any, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}},
                      ensure_ascii=False)


# ---------------------------------------------------------------- 服务端工具

_UNIT_TABLE: dict[str, dict[str, float]] = {
    # 类别内基准换算
    "千米": 1000.0, "公里": 1000.0, "米": 1.0, "厘米": 0.01, "英寸": 0.0254, "英尺": 0.3048, "英里": 1609.344,
}
_WEIGHT_TABLE = {"千克": 1.0, "公斤": 1.0, "克": 0.001, "磅": 0.45359237}
_TZ_OFFSET = {"UTC": 0, "UTC+0": 0, "UTC+8": 8, "UTC+9": 9, "UTC+1": 1, "UTC-5": -5, "UTC-8": -8}


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "unit_convert",
            "description": "单位换算：长度（千米/米/英寸/英尺/英里等）与重量（千克/克/磅）、温度（摄氏/华氏）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "数值"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
        {
            "name": "timezone_convert",
            "description": "时区换算（演示环境使用固定偏移，不处理夏令时）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "如 15:00 或 15点30"},
                    "from_tz": {"type": "string", "description": "如 UTC+8"},
                    "to_tz": {"type": "string", "description": "如 UTC+0"},
                },
                "required": ["time", "from_tz", "to_tz"],
            },
        },
    ]


def _call_tool(name: str, arguments: dict) -> str:
    if name == "unit_convert":
        value = float(arguments["value"])
        src, dst = str(arguments["from_unit"]), str(arguments["to_unit"])
        if src in ("摄氏度", "°C") and dst in ("华氏度", "°F"):
            out = value * 9 / 5 + 32
        elif src in ("华氏度", "°F") and dst in ("摄氏度", "°C"):
            out = (value - 32) * 5 / 9
        else:
            table = _UNIT_TABLE if src in _UNIT_TABLE else _WEIGHT_TABLE
            table_dst = _UNIT_TABLE if dst in _UNIT_TABLE else _WEIGHT_TABLE
            if src not in table or dst not in table_dst:
                raise ValueError(f"不支持的单位: {src} → {dst}")
            out = value * table[src] / table_dst[dst]
        pretty = f"{out:.4f}".rstrip("0").rstrip(".")
        return f"{value} {src} = {pretty} {dst}"

    if name == "timezone_convert":
        time_s = str(arguments["time"]).replace("点", ":").replace("：", ":")
        src, dst = str(arguments["from_tz"]), str(arguments["to_tz"])
        if src not in _TZ_OFFSET or dst not in _TZ_OFFSET:
            raise ValueError(f"不支持的时区: {src}/{dst}（可用: {sorted(_TZ_OFFSET)}）")
        parts = time_s.split(":")
        hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        base = datetime(2026, 1, 1, hour % 24, minute)
        shifted = base + timedelta(hours=_TZ_OFFSET[dst] - _TZ_OFFSET[src])
        return f"{time_s}（{src}）= {shifted.strftime('%H:%M')}（{dst}）"
    raise ValueError(f"未知工具: {name}")


# ---------------------------------------------------------------- 服务端主体

class DemoMCPServer:
    """示例 MCP Server：从 stdin 读 JSON-RPC，向 stdout 写响应。"""

    def __init__(self, server_name: str = "agent-harness-demo-mcp") -> None:
        self.server_name = server_name

    def handle(self, msg: dict) -> str | None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        if method == "initialize":
            return _reply(msg_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.server_name, "version": "0.1.0"},
            })
        if method == "notifications/initialized":
            return None  # 通知无需响应
        if method == "tools/list":
            return _reply(msg_id, {"tools": _tool_specs()})
        if method == "tools/call":
            try:
                text = _call_tool(params["name"], params.get("arguments") or {})
                return _reply(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
            except (KeyError, ValueError, TypeError) as exc:
                return _reply(msg_id, {"content": [{"type": "text", "text": f"工具执行失败: {exc}"}],
                                       "isError": True})
        if msg_id is not None:
            return _error(msg_id, -32601, f"method not found: {method}")
        return None

    def serve(self) -> None:
        """阻塞式 stdio 主循环（服务端进程除读取 stdin 外无其他工作，直接同步处理）。"""
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = self.handle(msg)
            if out is not None:
                sys.stdout.write(out + "\n")
                sys.stdout.flush()


if __name__ == "__main__":  # python -m agent_harness.tools.mcp
    DemoMCPServer().serve()


# ---------------------------------------------------------------- 客户端

class MCPClientSession:
    """连接一个 stdio MCP Server 的客户端会话。"""

    def __init__(self, command: list[str], cwd: str | None = None) -> None:
        self.command = command
        self.cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._server_info: dict = {}

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=1024 * 1024, cwd=self.cwd)
        result = await self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "agent-harness", "version": "0.1.0"}})
        self._server_info = result.get("serverInfo", {})
        await self.notify("notifications/initialized", {})

    async def _readline(self) -> str:
        assert self._proc and self._proc.stdout
        line = await self._proc.stdout.readline()
        if not line:
            raise ConnectionError("MCP server stdout 已关闭")
        return line.decode("utf-8").strip()

    async def request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        msg_id = self._next_id
        assert self._proc and self._proc.stdin
        payload = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
                             ensure_ascii=False)
        self._proc.stdin.write(payload.encode("utf-8") + b"\n")
        await self._proc.stdin.drain()
        # 读取直到拿到对应 id 的响应（跳过服务端通知）
        while True:
            resp = json.loads(await self._readline())
            if resp.get("id") == msg_id:
                if "error" in resp:
                    raise RuntimeError(f"MCP 错误: {resp['error']}")
                return resp.get("result", {})

    async def notify(self, method: str, params: dict) -> None:
        assert self._proc and self._proc.stdin
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params},
                             ensure_ascii=False)
        self._proc.stdin.write(payload.encode("utf-8") + b"\n")
        await self._proc.stdin.drain()

    async def list_tools(self) -> list[dict]:
        result = await self.request("tools/list", {})
        return result.get("tools", [])

    async def call(self, name: str, arguments: dict) -> str:
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content") or []
        text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        if result.get("isError"):
            raise RuntimeError(text or "MCP 工具执行失败")
        return text

    @property
    def server_info(self) -> dict:
        return self._server_info

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()


class MCPToolAdapter(BaseTool):
    """把远程 MCP 工具适配为注册中心的 BaseTool（来源标记 mcp://xxx）。"""

    def __init__(self, session: MCPClientSession, spec: dict, source: str) -> None:
        self._session = session
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.params = spec.get("inputSchema") or {"type": "object", "properties": {}}
        self.source = f"mcp://{source}" if not source.startswith("mcp://") else source
        self.timeout_s = 8.0
        self.retries = 2
        self.fallback = None

    async def run(self, args: dict, ctx) -> ToolResult:
        try:
            text = await self._session.call(self.name, args)
            return ToolResult(ok=True, output=text, meta={"server": self._session.server_info.get("name", "")})
        except (RuntimeError, ConnectionError) as exc:
            return ToolResult(ok=False, error=f"MCP 工具 {self.name} 执行失败: {exc}")
