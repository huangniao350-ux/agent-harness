"""结果校验器（对应简历 Multi-Agent 协同中的"校验"角色 / 执行结果反馈闭环）。

校验采用"规则前置 + LLM 兜底"两层：
1. 规则层：空结果、错误标记、明显未完成信号 —— 零成本、确定性；
2. LLM 层：判断结果是否真正达成目标语义（[MODE: VERIFY]）。
"""

from __future__ import annotations

import re

from ..llm.base import LLMClient, LLMMessage
from ..utils import extract_json, truncate

ERROR_MARKERS = re.compile(r"ERROR|Traceback|异常终止|CRITICAL", re.IGNORECASE)


class Verdict:
    def __init__(self, passed: bool, reason: str) -> None:
        self.passed = passed
        self.reason = reason

    def __repr__(self) -> str:
        return f"Verdict(passed={self.passed}, reason={self.reason!r})"


class Verifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def verify(self, goal: str, result_text: str, *, is_final: bool = False) -> Verdict:
        # ---- 规则层
        if not result_text or not result_text.strip():
            return Verdict(False, "结果为空")
        if ERROR_MARKERS.search(result_text):
            return Verdict(False, "结果包含错误标记（ERROR/Traceback）")
        # ---- LLM 层
        resp = await self.llm.chat([
            LLMMessage(role="system",
                       content=("你是结果校验器。[MODE: VERIFY] 判断执行结果是否达成目标。"
                                "只输出 JSON：{\"pass\": true/false, \"reason\": \"简要理由\"}")),
            LLMMessage(role="user",
                       content=f"目标：{truncate(goal, 300)}\n执行结果：{truncate(result_text, 900)}"
                               + ("\n（这是面向用户的最终回答）" if is_final else "")),
        ], json_mode=True, role="verifier")
        data = extract_json(resp.content) or {}
        passed = bool(data.get("pass", True))
        return Verdict(passed, str(data.get("reason", ""))[:200])
