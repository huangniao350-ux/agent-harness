"""Replay 回放（对应简历：Trace/Replay 支持线上问题回放与定位）。

原理：Trace 中已记录每次 LLM 调用的输出内容，Replay 用 ReplayLLM 按调用序号
原样回放这些输出，在隔离数据目录中重跑同一目标，再对比两次执行的
工具调用序列与最终答案 —— 用于回归验证与问题复现（确定性重放）。
"""

from __future__ import annotations

from ..llm.base import LLMMessage, LLMResponse, normalize_messages, usage_from_text


class ReplayLLM:
    """按调用顺序回放 Trace 中记录的 LLM 输出。"""

    name = "replay"

    def __init__(self, recorded_outputs: list[dict]) -> None:
        # 每项: {"role": ..., "content": ...}，按调用顺序排列
        self.recorded = list(recorded_outputs)
        self.cursor = 0

    async def chat(self, messages, *, json_mode: bool = False, role: str = "default", **_) -> LLMResponse:
        if self.cursor >= len(self.recorded):
            # 超出记录（如版本差异导致调用次数变多）：退化为 MockLLM 处理
            from ..llm.mock import MockLLM
            return await MockLLM().chat(messages, json_mode=json_mode, role=role)
        item = self.recorded[self.cursor]
        self.cursor += 1
        msgs = normalize_messages(messages)
        return LLMResponse(content=item.get("content", ""),
                           model=f"replay:{item.get('role', role)}",
                           usage=usage_from_text("replay", msgs, item.get("content", "")))

    @classmethod
    def from_trace(cls, trace: dict) -> "ReplayLLM":
        outputs = [s["detail"] for s in trace.get("spans", [])
                   if s["type"] == "llm" and s.get("detail")]
        return cls(outputs)


def compare_traces(original: dict, replayed: dict) -> dict:
    """对比两次执行的工具调用序列与最终答案，输出差异报告。"""

    def tool_seq(trace: dict) -> list[tuple]:
        seq = []
        for s in trace.get("spans", []):
            if s["type"] == "tool":
                seq.append((s["name"], json_key(s["detail"].get("args") if isinstance(s.get("detail"), dict) else None)))
        return seq

    def json_key(args):
        import json
        try:
            return json.dumps(args, ensure_ascii=False, sort_keys=True) if args is not None else ""
        except TypeError:
            return str(args)

    orig_seq, rep_seq = tool_seq(original), tool_seq(replayed)
    tools_match = orig_seq == rep_seq
    answer_match = (original.get("answer") or "").strip() == (replayed.get("answer") or "").strip()
    return {
        "original_trace_id": original.get("trace_id"),
        "replayed_status": replayed.get("status"),
        "tool_calls_original": len(orig_seq),
        "tool_calls_replayed": len(rep_seq),
        "tool_sequence_match": tools_match,
        "answer_match": answer_match,
        "deterministic": tools_match and answer_match,
    }
