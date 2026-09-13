"""失败归因与自我反思（对应简历：执行结果反馈与自我反思纠错闭环）。

失败时自动定位问题环节：planning（规划）/ tool_call（工具调用）/ context（上下文与检索）/
memory（记忆）/ model（模型能力），产出结构化归因结论与改进建议，作为反馈注入重新规划。
规则层先做确定性分类（错误指纹），LLM 层负责兜底与更细粒度的判断。
"""

from __future__ import annotations

import re

from ..llm.base import LLMClient, LLMMessage
from ..utils import extract_json, truncate

STAGES = ("planning", "tool_call", "context", "memory", "model")

_FINGERPRINTS: list[tuple[str, str, str, str]] = [
    # (正则, stage, cause, suggestion)
    (r"超时|Timeout|timed?\s*out|5\d\d|503|Connection|网络", "tool_call",
     "工具调用持续失败（超时/上游不稳定）", "改用备用工具或降级方案，必要时调整参数重试"),
    (r"未知工具|Unknown tool|没有这个工具|not in available", "planning",
     "规划引用了不存在的工具", "仅从可用工具列表中选择工具重新规划"),
    (r"参数校验失败|缺少必填|Schema", "tool_call",
     "工具参数构造错误", "对照工具参数说明补全参数后重试"),
    (r"未找到|没有找到|不存在|无结果|为空", "context",
     "检索/查询未覆盖所需信息", "更换检索关键词、扩大范围或先确认实体信息"),
    (r"达到最大步数|格式不符合", "model",
     "模型未能收敛（步数耗尽/格式漂移）", "拆小任务并明确每步工具绑定，减少单步自由度"),
]


class FailureAttributor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def attribute(self, goal: str, error_context: str) -> dict:
        # ---- 规则层：错误指纹匹配
        for pattern, stage, cause, suggestion in _FINGERPRINTS:
            if re.search(pattern, error_context, re.IGNORECASE):
                return {"stage": stage, "cause": cause, "suggestion": suggestion,
                        "source": "rule"}
        # ---- LLM 层兜底
        resp = await self.llm.chat([
            LLMMessage(role="system",
                       content=("你是 Agent 失败归因分析器。[MODE: REFLECT] 从规划/工具调用/上下文/记忆/"
                                "模型能力中定位问题环节。只输出 JSON："
                                "{\"stage\": \"...\", \"cause\": \"...\", \"suggestion\": \"...\"}")),
            LLMMessage(role="user", content=f"目标：{truncate(goal, 200)}\n失败上下文：{truncate(error_context, 900)}"),
        ], json_mode=True, role="verifier")
        data = extract_json(resp.content) or {}
        stage = data.get("stage") if data.get("stage") in STAGES else "planning"
        return {"stage": stage, "cause": str(data.get("cause", "未知原因")),
                "suggestion": str(data.get("suggestion", "重新规划任务")), "source": "llm"}

    @staticmethod
    def feedback_text(analysis: dict) -> str:
        return (f"【上轮失败归因】环节：{analysis['stage']}；原因：{analysis['cause']}；"
                f"改进建议：{analysis['suggestion']}。请在此基础上调整执行策略。")
