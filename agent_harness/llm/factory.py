"""LLM 工厂：根据配置构建 MockLLM 或 OpenAI 兼容客户端（模型路由在 config 中定义）。"""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMClient, LLMError, LLMMessage, LLMResponse
from .cache import LLMResponseCache
from .mock import MockLLM


def build_llm(config: LLMConfig) -> LLMClient:
    if config.provider == "mock":
        inner: LLMClient = MockLLM()
    elif config.provider == "openai":
        from .openai_compat import OpenAICompatClient
        inner = OpenAICompatClient(
            api_base=config.api_base, api_key=config.api_key, model=config.model,
            cheap_model=config.cheap_model, temperature=config.temperature,
            timeout_s=config.timeout_s,
        )
    else:
        raise ValueError(f"未知 LLM provider: {config.provider}")
    if config.enable_response_cache:
        return LLMResponseCache(inner)
    return inner


__all__ = ["build_llm", "MockLLM", "LLMClient", "LLMMessage", "LLMResponse", "LLMError",
           "LLMResponseCache"]
