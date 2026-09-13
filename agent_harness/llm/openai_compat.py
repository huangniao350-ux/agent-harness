"""OpenAI 兼容客户端：仅用标准库 urllib 实现 /chat/completions 调用（含 SSE 流式解析）。

支持 DeepSeek / 智谱 / Qwen / vLLM 等任意 OpenAI 兼容端点，生产建议替换为官方 SDK
或 httpx 异步客户端，这里保持零依赖并演示协议细节。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

from .base import LLMClient, LLMError, LLMMessage, LLMResponse, normalize_messages, usage_from_text


class OpenAICompatClient(LLMClient):
    def __init__(self, api_base: str, api_key: str, model: str, cheap_model: str = "",
                 temperature: float = 0.2, timeout_s: float = 60.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.cheap_model = cheap_model or model
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _model_for_role(self, role: str) -> str:
        return self.model if role in ("planner", "verifier", "synthesizer") else self.cheap_model

    def _request(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def chat(self, messages, *, json_mode: bool = False, role: str = "default", **_) -> LLMResponse:
        msgs = normalize_messages(messages)
        payload = {
            "model": self._model_for_role(role),
            "messages": [m.to_dict() for m in msgs],
            "temperature": self.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._request, payload)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"LLM HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"LLM 请求失败: {exc}") from exc
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"LLM 响应格式异常: {data!r}") from exc
        usage_raw = data.get("usage") or {}
        usage = usage_from_text(self.model, msgs, content)
        usage.prompt_tokens = usage_raw.get("prompt_tokens", usage.prompt_tokens)
        usage.completion_tokens = usage_raw.get("completion_tokens", usage.completion_tokens)
        return LLMResponse(content=content, model=data.get("model", self.model),
                           usage=usage, finish_reason=choice.get("finish_reason", "stop"))

    async def stream(self, messages, *, role: str = "default", **_):
        """SSE 流式增量输出（供 API 服务透传给前端）。"""
        msgs = normalize_messages(messages)
        payload = {
            "model": self._model_for_role(role),
            "messages": [m.to_dict() for m in msgs],
            "temperature": self.temperature,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )

        def _iter_sse():
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if line.startswith("data:"):
                        chunk = line[5:].strip()
                        if chunk and chunk != "[DONE]":
                            try:
                                item = json.loads(chunk)
                                delta = item["choices"][0].get("delta", {}).get("content")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

        loop = asyncio.get_running_loop()
        agen = _iter_sse()
        while True:
            try:
                chunk = await loop.run_in_executor(None, next, agen, _SENTINEL)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
                raise LLMError(f"LLM 流式请求失败: {exc}") from exc
            if chunk is _SENTINEL:
                break
            yield chunk


_SENTINEL = object()
