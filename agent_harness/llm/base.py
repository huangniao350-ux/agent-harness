"""LLM 抽象层：统一消息/响应模型，上层不感知具体 provider。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..utils import estimate_tokens


@dataclass
class LLMMessage:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[LLMMessage | dict],
        *,
        json_mode: bool = False,
        role: str = "default",
    ) -> LLMResponse: ...


def normalize_messages(messages: list[LLMMessage | dict]) -> list[LLMMessage]:
    out: list[LLMMessage] = []
    for m in messages:
        if isinstance(m, LLMMessage):
            out.append(m)
        else:
            out.append(LLMMessage(role=m["role"], content=m["content"]))
    return out


def usage_from_text(model: str, messages: list[LLMMessage], completion: str) -> TokenUsage:
    prompt = sum(estimate_tokens(m.content) for m in messages)
    return TokenUsage(prompt_tokens=prompt, completion_tokens=estimate_tokens(completion))


def build_tool_lines(tools: list[Any]) -> str:
    """把工具列表渲染进 Prompt（名称 + 参数说明 + 描述）。"""
    lines = []
    for t in tools:
        params = ", ".join(f"{k}:{v.get('type', 'any')}" for k, v in (t.params.get("properties") or {}).items())
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines) if lines else "- （无可用工具）"
