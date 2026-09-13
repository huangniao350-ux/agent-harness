"""Agent 状态模型：Harness 全部状态显式可序列化，这是 Checkpoint 续跑的前提。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MODE_REACT = "react"
MODE_PLAN_EXECUTE = "plan_execute"
MODE_MULTI_AGENT = "multi_agent"

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_INCOMPLETE = "incomplete"


@dataclass
class AgentState:
    session_id: str
    goal: str
    mode: str = MODE_REACT
    status: str = STATUS_RUNNING
    scratchpad: list[dict[str, str]] = field(default_factory=list)   # ReAct 轨迹
    plan: list[dict[str, Any]] = field(default_factory=list)         # Plan-and-Execute 步骤
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)  # sid -> {status, result}
    rounds: int = 0                                                  # 反思/重规划轮数
    final_answer: str = ""
    error: str = ""
    failure_feedback: str = ""                                       # 反思结论，注入下一轮
    tool_calls: list[dict[str, Any]] = field(default_factory=list)   # {tool, args, ok}
    tokens_used: int = 0
    steps_used: int = 0
    llm_calls: int = 0                                               # LLM 调用次数（含子步骤）
    ttft_total_ms: float = 0.0                                       # TTFT 累计（求平均用）
    cache_hits: int = 0                                              # LLM 响应缓存命中次数

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def success(self) -> bool:
        return self.status == STATUS_DONE and bool(self.final_answer)


@dataclass
class AgentResult:
    session_id: str
    trace_id: str
    goal: str
    mode: str
    answer: str
    success: bool
    steps: int
    tool_calls: list[dict[str, Any]]
    tokens_used: int
    latency_ms: float
    plan: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    ttft_ms: float = 0.0        # 平均首字延迟（LLM 调用级）
    cache_hits: int = 0         # LLM 响应缓存命中次数

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
