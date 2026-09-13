"""运行时配置：支持环境变量覆盖，便于在 Mock 与真实 LLM 之间一键切换。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """provider: mock | openai（任意 OpenAI 兼容端点：DeepSeek / 智谱 / vLLM 等）"""

    provider: str = "mock"
    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    cheap_model: str = ""          # 模型路由：简单步骤用便宜模型，为空则复用 model
    temperature: float = 0.2
    timeout_s: float = 60.0

    def model_for_role(self, role: str) -> str:
        """模型路由策略：规划/校验用强模型，普通执行步骤用便宜模型（成本管控）。"""
        if role in ("planner", "verifier", "synthesizer"):
            return self.model
        return self.cheap_model or self.model


@dataclass
class HarnessConfig:
    max_react_steps: int = 8          # ReAct 单轮最大步数
    max_step_react_steps: int = 4     # Plan-and-Execute 中每个子步骤的最大 ReAct 步数
    max_repair_rounds: int = 2        # 反思后重新规划的最大轮数
    tool_call_timeout_s: float = 10.0
    tool_retries: int = 2             # 可重试错误的最大尝试次数
    checkpoint_enabled: bool = True
    parallel_workers: int = 3         # 并行步骤的信号量上限


@dataclass
class MemoryConfig:
    short_term_token_budget: int = 900      # 短期记忆超出后触发压缩
    context_token_budget: int = 1500        # 单次注入上下文总预算
    longterm_top_k: int = 3
    dedup_similarity_threshold: float = 0.86


@dataclass
class RuntimeConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data_dir: Path = field(default_factory=lambda: Path(".agent_data"))
    workspace_dir: Path = field(default_factory=lambda: Path(".agent_workspace"))
    guardrails_enabled: bool = True
    enable_mcp: bool = True
    trace_enabled: bool = True

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.db"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        cfg = cls()
        llm = cfg.llm
        llm.provider = os.environ.get("AGENT_LLM_PROVIDER", llm.provider).lower()
        llm.api_base = os.environ.get("AGENT_LLM_API_BASE", llm.api_base)
        llm.api_key = os.environ.get("AGENT_LLM_API_KEY", llm.api_key)
        llm.model = os.environ.get("AGENT_LLM_MODEL", llm.model)
        llm.cheap_model = os.environ.get("AGENT_LLM_CHEAP_MODEL", llm.cheap_model)
        if llm.provider != "mock" and not llm.api_key:
            # 未提供 Key 时自动回落 Mock，保证演示永不中断
            llm.provider = "mock"
        cfg.guardrails_enabled = os.environ.get("AGENT_GUARDRAILS", "1") not in ("0", "false")
        cfg.enable_mcp = os.environ.get("AGENT_ENABLE_MCP", "1") not in ("0", "false")
        cfg.trace_enabled = os.environ.get("AGENT_TRACE", "1") not in ("0", "false")
        cfg.harness.max_react_steps = int(os.environ.get("AGENT_MAX_STEPS", cfg.harness.max_react_steps))
        cfg.data_dir = Path(os.environ.get("AGENT_DATA_DIR", ".agent_data"))
        cfg.workspace_dir = Path(os.environ.get("AGENT_WORKSPACE_DIR", ".agent_workspace"))
        return cfg
