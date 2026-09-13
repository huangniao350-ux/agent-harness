"""LLM 工厂：根据配置构建 MockLLM 或 OpenAI 兼容客户端（模型路由在 config 中定义）。"""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMClient, LLMError, LLMMessage, LLMResponse
from .mock import MockLLM


def build_llm(config: LLMConfig) -> LLMClient:
    if config.provider == "mock":
        return MockLLM()
    if config.provider == "openai":
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(
            api_base=config.api_base, api_key=config.api_key, model=config.model,
            cheap_model=config.cheap_model, temperature=config.temperature,
            timeout_s=config.timeout_s,
        )
    raise ValueError(f"未知 LLM provider: {config.provider}")


__all__ = ["build_llm", "MockLLM", "LLMClient", "LLMMessage", "LLMResponse", "LLMError"]
