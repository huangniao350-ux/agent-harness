"""弹性执行层（对应简历：超时控制、失败重试、防重复调用与降级机制）。

所有工具调用统一经过 execute_tool()：
1. Schema 校验失败 → 直接返回结构化错误 Observation（让模型可自我修正）；
2. 防重复调用：同会话内 (tool, args) 完全相同且成功过的调用直接复用缓存；
   同一失败调用重复出现时，在 Observation 中注入策略提示，抑制"原地打转"；
3. 超时控制：asyncio.wait_for；
4. 失败重试：仅对 RetryableError / 超时做指数退避重试（业务性错误重试无意义）；
5. 降级：重试耗尽后若工具声明了 fallback，则自动切换备用工具。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from ..events import TOOL_CALL, Event
from .base import BaseTool, RetryableError, ToolContext, ToolResult


class ToolCallCache:
    """会话级调用缓存与失败计数（防重复调用 / 打转检测）。"""

    def __init__(self) -> None:
        self._ok_cache: dict[str, ToolResult] = {}
        self._fail_count: dict[str, int] = {}

    @staticmethod
    def key(tool_name: str, args: dict, session_id: str = "") -> str:
        canonical = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        raw = f"{session_id}::{tool_name}::{canonical}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get_ok(self, key: str) -> ToolResult | None:
        return self._ok_cache.get(key)

    def put_ok(self, key: str, result: ToolResult) -> None:
        self._ok_cache[key] = result

    def bump_fail(self, key: str) -> int:
        self._fail_count[key] = self._fail_count.get(key, 0) + 1
        return self._fail_count[key]


async def execute_tool(tool: BaseTool, args: dict, ctx: ToolContext,
                       cache: ToolCallCache | None = None,
                       default_timeout: float = 10.0, default_retries: int = 2) -> ToolResult:
    started = time.perf_counter()

    # 1. Schema 校验（前置护栏：错误参数不落到底层工具）
    errors = tool.validate_args(args)
    if errors:
        return ToolResult(ok=False, error="参数校验失败: " + "; ".join(errors),
                          meta={"stage": "schema", "tool": tool.name})

    key = ToolCallCache.key(tool.name, args, ctx.session_id)
    # 2. 防重复调用：命中成功缓存直接复用
    if cache is not None:
        cached = cache.get_ok(key)
        if cached is not None:
            cached.meta = {**cached.meta, "dedup": True, "tool": tool.name}
            await _emit(ctx, tool, cached, time.perf_counter() - started, args)
            return cached

    timeout_s = tool.timeout_s or default_timeout
    retries = max(1, tool.retries or default_retries)
    last_error: str | None = None
    attempts = 0
    for attempt in range(1, retries + 1):
        attempts = attempt
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=timeout_s)
        except asyncio.TimeoutError:
            last_error = f"工具执行超时（>{timeout_s}s）"
        except RetryableError as exc:
            last_error = f"可重试错误: {exc}"
        except Exception as exc:  # 业务性错误：重试无意义，直接失败
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                                meta={"tool": tool.name, "attempts": 1})
            await _emit(ctx, tool, result, time.perf_counter() - started, args)
            return _with_hints(result, cache, key)
        else:
            result.meta = {**result.meta, "tool": tool.name, "attempts": attempts}
            if result.ok and cache is not None:
                cache.put_ok(key, result)
            if not result.ok and cache is not None and _is_retryable_failure(result):
                # 业务失败（如下游报错文本）也计入重试
                fail_n = cache.bump_fail(key)
                result.meta["fail_count"] = fail_n
            await _emit(ctx, tool, result, time.perf_counter() - started, args)
            return _with_hints(result, cache, key) if not result.ok else result
        # 指数退避后重试
        if attempt < retries:
            await asyncio.sleep(0.2 * (2 ** (attempt - 1)))

    # 3. 重试耗尽 → 降级到备用工具
    if tool.fallback is not None:
        fb_result = await execute_tool(tool.fallback, args, ctx, cache,
                                       default_timeout, default_retries)
        fb_result.meta = {**fb_result.meta, "degraded_from": tool.name, "tool": tool.fallback.name}
        await _emit(ctx, tool, fb_result, time.perf_counter() - started, args)
        return fb_result

    result = ToolResult(ok=False, error=last_error or "未知错误",
                        meta={"tool": tool.name, "attempts": attempts})
    await _emit(ctx, tool, result, time.perf_counter() - started, args)
    return _with_hints(result, cache, key)


def _is_retryable_failure(result: ToolResult) -> bool:
    return bool(result.error) and any(
        w in (result.error or "") for w in ("超时", "Timeout", "timeout", "不稳定", "5xx", "Connection")
    )


def _with_hints(result: ToolResult, cache: ToolCallCache | None, key: str) -> ToolResult:
    """失败 Observation 中注入策略提示，帮助模型跳出重复尝试。"""
    if cache is None:
        return result
    fail_n = cache.bump_fail(key)
    result.meta["fail_count"] = fail_n
    if fail_n >= 2:
        result.error = (f"{result.error} | 提示：同一调用已失败 {fail_n} 次，"
                        f"建议更换工具、调整参数或降级处理，避免原地重试")
    return result


async def _emit(ctx: ToolContext, tool: BaseTool, result: ToolResult,
                latency: float, args: dict) -> None:
    if ctx.event_bus is None:
        return
    payload: dict[str, Any] = {
        "tool": tool.name, "source": tool.source, "args": args,
        "ok": result.ok, "error": result.error, "latency_ms": round(latency * 1000, 1),
        "meta": {k: v for k, v in result.meta.items() if k != "args"},
    }
    await ctx.event_bus.emit(Event(type=TOOL_CALL, trace_id=ctx.trace_id,
                                   session_id=ctx.session_id, payload=payload))
