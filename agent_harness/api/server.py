"""FastAPI + SSE 服务（对应简历：FastAPI、SSE 流式响应、微服务化）。

- POST /v1/agent/run：SSE 流式返回执行事件（规划/工具调用/护栏/最终回答），
  与 CLI 的 ConsoleReporter 共用同一事件总线 —— 前后端一致的实时观测体验；
- GET  /v1/tools：工具清单（含 MCP 远程工具）；
- GET  /v1/traces/{trace_id}：Trace 查询；
- GET  /health：健康检查。

启动：python -m agent_harness server --port 8000
体验：curl -N -X POST localhost:8000/v1/agent/run -H "Content-Type: application/json" \
       -d '{"goal": "查一下张三的年假余额，如果够的话帮他提交3天的年假申请"}'

注意：本文件不能加 `from __future__ import annotations` —— 延迟求值会让 FastAPI
无法解析定义在 create_app 内部的 Pydantic 模型（被降级为 query 参数导致 422）。
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from ..config import RuntimeConfig
from ..events import Event
from ..harness.state import MODE_PLAN_EXECUTE, MODE_MULTI_AGENT, MODE_REACT
from ..runtime import AgentRuntime

_WEB_DIR = Path(__file__).parent / "static"

# 全局单例运行时（演示用单进程；生产按 worker 隔离）
_runtime: AgentRuntime | None = None


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime()
    return _runtime


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = get_runtime()
        await runtime.startup()
        yield
        await runtime.close()

    app = FastAPI(title="AgentHarness API", version="0.1.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index():
        """Harness 开发者控制台（面向平台研发/排查）。"""
        from fastapi.responses import HTMLResponse
        return HTMLResponse((_WEB_DIR / "console.html").read_text(encoding="utf-8"),
                            headers={"Cache-Control": "no-cache"})

    class RunRequest(BaseModel):
        goal: str
        mode: str = MODE_REACT
        session_id: str | None = None
        confirm: str | None = None   # 高危操作确认门模式覆盖：auto | manual | off

    class ConfirmRequest(BaseModel):
        request_id: str
        approve: bool

    @app.get("/health")
    async def health():
        return {"status": "ok", "llm_provider": get_runtime().config.llm.provider}

    @app.post("/v1/agent/confirm")
    async def confirm(req: ConfirmRequest):
        """人工确认门决定接口：对 confirmation_request 事件中的 request_id 批准/拒绝。"""
        ok = get_runtime().confirmation_gate.decide(req.request_id, req.approve)
        if not ok:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="确认请求不存在或已处理")
        return {"request_id": req.request_id, "approved": req.approve}

    @app.get("/v1/tools")
    async def tools():
        return {"tools": get_runtime().registry.spec()}

    @app.get("/v1/traces/{trace_id}")
    async def trace(trace_id: str):
        data = get_runtime().traces.get(trace_id)
        if data is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="trace not found")
        return data

    @app.post("/v1/agent/run")
    async def run_agent(req: RunRequest):
        runtime = get_runtime()
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        session_id = req.session_id or f"s-{uuid.uuid4().hex[:8]}"

        async def forward(event: Event) -> None:
            if event.session_id == session_id:
                await queue.put(event)

        runtime.event_bus.subscribe(forward)

        async def event_stream():
            task = asyncio.create_task(runtime.run(req.goal, mode=req.mode,
                                                   session_id=session_id,
                                                   confirm=req.confirm))
            try:
                yield _sse("session", {"session_id": session_id})
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield _sse(event.type, event.payload)
                    if event.type == "task_finished":
                        break
                task_result = await task
                yield _sse("result", task_result.to_dict())
            finally:
                runtime.event_bus.unsubscribe(forward)
                if not task.done():
                    task.cancel()

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
