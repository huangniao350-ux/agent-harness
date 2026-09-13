"""LLM 响应缓存（对应简历：成本管控中的缓存机制）。

对完全相同的 (role, json_mode, messages) 请求直接复用上次的模型输出：
- 命中后本次调用零 Token 成本（usage 不再累计），TTFT 趋近于 0；
- Key 为规范化消息序列的 SHA256，进程内缓存，生产可平移至 Redis 等分布式缓存；
- 语义边界：只做精确匹配（不含语义缓存），确定性场景（Mock/temperature≈0）收益最大。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace

from .base import LLMClient, LLMMessage, LLMResponse, normalize_messages


class LLMResponseCache:
    """包装任意 LLMClient：精确匹配的响应缓存层。"""

    def __init__(self, inner: LLMClient, max_entries: int = 2048) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self._store: dict[str, LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "cached")

    @staticmethod
    def _key(messages: list[LLMMessage], json_mode: bool, role: str) -> str:
        canonical = json.dumps(
            {"role": role, "json": json_mode,
             "msgs": [m.to_dict() for m in normalize_messages(messages)]},
            ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def chat(self, messages, *, json_mode: bool = False, role: str = "default",
                   **kwargs) -> LLMResponse:
        key = self._key(messages, json_mode, role)
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            # 命中：零 Token 成本，TTFT 即查表耗时
            return replace(cached, cached=True, ttft_ms=0.0)
        self.misses += 1
        started = time.perf_counter()
        resp = await self.inner.chat(messages, json_mode=json_mode, role=role, **kwargs)
        if resp.ttft_ms <= 0:  # inner 未上报时以整次调用耗时兜底
            resp.ttft_ms = round((time.perf_counter() - started) * 1000, 2)
        if len(self._store) >= self.max_entries:
            self._store.clear()  # 简单防膨胀；生产换 LRU
        self._store[key] = resp
        return resp

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}
