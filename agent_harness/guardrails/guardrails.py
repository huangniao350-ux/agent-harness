"""Guardrails 安全护栏（对应简历：输入校验、输出过滤、敏感信息检测与操作审计）。

管道式设计，规则可插拔：
- InputGuardrails：Prompt 注入检测（拦截）、敏感信息脱敏（改写）、长度/黑白名单校验；
- OutputGuardrails：敏感信息泄露过滤、API Key/内部凭据泄露检测；
- AuditLogger：所有拦截/脱敏决策落 JSONL 审计文件，可回溯。

动作策略：block（直接拒绝，不进入 LLM）/ mask（脱敏后放行）/ allow。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..events import Event, EventBus, GUARDRAIL
from ..utils import now_iso

# ------------------------------------------------------------- 规则定义

INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"忽略(之前|上面|以上|所有|先前).{0,8}(指令|规则|提示|设定)", "指令覆盖类注入"),
    (r"(你现在是|从现在开始你是|进入).{0,6}(DAN|开发者模式|无限制模式)", "角色越狱类注入"),
    (r"(打印|输出|泄露|告诉我).{0,8}(系统提示|system\s*prompt|你的指令|你的设定)", "提示词泄露类注入"),
    (r"(ignore|disregard)\s+(all\s+)?(previous|above)\s+instructions", "英文指令覆盖注入"),
    (r"(必须|请).{0,4}(无条件|不加限制).{0,6}(执行|回答)", "约束规避类注入"),
]

SENSITIVE_PATTERNS: list[tuple[str, str, Callable]] = [
    (r"1[3-9]\d{9}", "手机号", lambda m: m[:3] + "****" + m[7:]),
    (r"\d{17}[\dXx]", "身份证号", lambda m: m[:4] + "***********" + m[-3:]),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "邮箱",
     lambda m: m.split("@")[0][:2] + "***@" + m.split("@")[1]),
    (r"\b\d{16,19}\b", "银行卡号", lambda m: m[:4] + " **** " + m[-4:]),
]

LEAK_PATTERNS: list[tuple[str, str]] = [
    (r"sk-[A-Za-z0-9]{16,}", "疑似 API Key 泄露"),
    (r"(password|passwd|密码)\s*[:=]\s*\S{4,}", "疑似凭据泄露"),
    (r"BEGIN (RSA |EC )?PRIVATE KEY", "私钥泄露"),
]


@dataclass
class GuardrailVerdict:
    action: str                       # allow | mask | block
    text: str                         # 处理后的文本（allow/mask 时有效）
    violations: list[dict] = field(default_factory=list)


class AuditLogger:
    """JSONL 审计日志：记录每次护栏决策，满足可追溯要求。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, action: str, violations: list[dict], sample: str) -> None:
        record = {"ts": now_iso(), "stage": stage, "action": action,
                  "violations": violations, "sample": sample[:120]}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Guardrails:
    def __init__(self, enabled: bool = True, audit_path: Path | None = None,
                 event_bus: EventBus | None = None) -> None:
        self.enabled = enabled
        self.audit = AuditLogger(audit_path or Path(".agent_data/audit.jsonl"))
        self.event_bus = event_bus

    # ------------------------------------------------------------ 输入护栏
    async def check_input(self, text: str) -> GuardrailVerdict:
        if not self.enabled:
            return GuardrailVerdict(action="allow", text=text)
        violations: list[dict] = []
        # 1. Prompt 注入检测 → block
        for pattern, name in INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({"type": "prompt_injection", "rule": name})
        if violations:
            await self._record("input", "block", violations, text)
            return GuardrailVerdict(action="block", text=text, violations=violations)
        # 2. 敏感信息 → 脱敏放行
        masked, mask_hits = _mask_sensitive(text)
        if mask_hits:
            violations.extend(mask_hits)
            await self._record("input", "mask", violations, text)
            return GuardrailVerdict(action="mask", text=masked, violations=violations)
        return GuardrailVerdict(action="allow", text=text)

    # ------------------------------------------------------------ 输出护栏
    async def check_output(self, text: str) -> GuardrailVerdict:
        if not self.enabled:
            return GuardrailVerdict(action="allow", text=text)
        violations: list[dict] = []
        for pattern, name in LEAK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({"type": "leak", "rule": name})
        if violations:
            await self._record("output", "block", violations, text)
            return GuardrailVerdict(action="block", text=text, violations=violations)
        masked, mask_hits = _mask_sensitive(text)
        if mask_hits:
            violations.extend(mask_hits)
            await self._record("output", "mask", violations, text)
            return GuardrailVerdict(action="mask", text=masked, violations=violations)
        return GuardrailVerdict(action="allow", text=text)

    # ------------------------------------------------------------ 内部
    async def _record(self, stage: str, action: str, violations: list[dict], sample: str) -> None:
        self.audit.log(stage, action, violations, sample)
        if self.event_bus is not None:
            # 同步 await 广播：保证事件到达顺序与决策顺序一致（不允许乱序延迟到后续任务）
            await self.event_bus.emit(Event(
                type=GUARDRAIL, payload={"stage": stage, "action": action, "violations": violations}))


def _mask_sensitive(text: str) -> tuple[str, list[dict]]:
    hits: list[dict] = []
    out = text
    for pattern, name, masker in SENSITIVE_PATTERNS:
        def _sub(m, _name=name, _masker=masker):
            hits.append({"type": "sensitive", "rule": _name})
            return _masker(m.group(0))
        out = re.sub(pattern, _sub, out)
    return out, hits
