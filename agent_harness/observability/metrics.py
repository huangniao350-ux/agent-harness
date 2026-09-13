"""评估指标汇总（对应简历：任务成功率、工具调用准确率、延迟、Token 等核心指标）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskOutcome:
    task_id: str
    category: str
    goal: str
    success: bool
    expected_tools: list[str]
    used_tools: list[str]
    steps: int
    latency_ms: float
    tokens: int
    answer: str
    error: str = ""

    @property
    def tool_precision_ok(self) -> bool:
        """期望工具全部被使用，且未使用期望之外的工具（严格匹配）。"""
        if not self.expected_tools:
            return True
        return set(self.expected_tools).issubset(set(self.used_tools))


@dataclass
class EvalReport:
    outcomes: list[TaskOutcome] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.outcomes) or 1
        success = sum(1 for o in self.outcomes if o.success)
        tool_ok = sum(1 for o in self.outcomes if o.tool_precision_ok)
        return {
            "total": len(self.outcomes),
            "success": success,
            "success_rate": round(success / n, 4),
            "tool_accuracy": round(tool_ok / n, 4),
            "avg_steps": round(sum(o.steps for o in self.outcomes) / n, 2),
            "avg_latency_ms": round(sum(o.latency_ms for o in self.outcomes) / n, 1),
            "avg_tokens": round(sum(o.tokens for o in self.outcomes) / n, 1),
        }

    def by_category(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for cat in {o.category for o in self.outcomes}:
            sub = EvalReport([o for o in self.outcomes if o.category == cat])
            out[cat] = sub.summary()
        return out

    def render(self) -> str:
        s = self.summary()
        lines = [
            "=" * 72,
            "Agent 评估报告",
            "=" * 72,
            f"任务总数: {s['total']}  成功: {s['success']}  成功率: {s['success_rate']:.1%}",
            f"工具调用准确率: {s['tool_accuracy']:.1%}  平均步数: {s['avg_steps']}",
            f"平均延迟: {s['avg_latency_ms']:.0f}ms  平均Token: {s['avg_tokens']:.0f}",
            "-" * 72,
            f"{'ID':<14}{'类别':<14}{'结果':<6}{'步数':<5}{'工具'}",
            "-" * 72,
        ]
        for o in self.outcomes:
            mark = "PASS" if o.success else "FAIL"
            tools = ",".join(o.used_tools) or "-"
            lines.append(f"{o.task_id:<14}{o.category:<14}{mark:<6}{o.steps:<5}{tools[:40]}")
            if not o.success and o.error:
                lines.append(f"{'':<14}└─ {o.error[:80]}")
        lines.append("-" * 72)
        for cat, cs in self.by_category().items():
            lines.append(f"[{cat}] {cs['success']}/{cs['total']} 成功率 {cs['success_rate']:.1%}")
        return "\n".join(lines)
