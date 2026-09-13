"""ReAct 解析器 / 混合检索 / Schema 校验 的纯函数单测（无 IO，毫秒级）。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_harness.harness.react import parse_react
from agent_harness.memory.retrieval import BM25Index, HashEmbedding, rrf_fuse
from agent_harness.tools.base import validate_schema


def test_parse_react_tool_call():
    text = "Thought: 需要查询员工\nAction: employee_lookup\nAction Input: {\"name\": \"张三\"}"
    parsed = parse_react(text)
    assert parsed["action"] == "employee_lookup"
    assert parsed["args"] == {"name": "张三"}
    assert parsed["final"] == ""
    assert "张三" in parsed["thought"] or parsed["thought"]


def test_parse_react_final_answer():
    text = "Thought: 已完成\nFinal Answer: 任务完成，请假单号 LR-1"
    parsed = parse_react(text)
    assert parsed["final"].startswith("任务完成")
    assert parsed["action"] == ""


def test_parse_react_multiline_json_and_fallback():
    text = 'Thought: x\\nAction: file_write\\nAction Input: {"path": "a.txt", "content": "第一行\\n第二行"}'
    parsed = parse_react(text)
    assert parsed["args"]["path"] == "a.txt"
    # 完全不合规输出 → 原文兜底为 final
    parsed2 = parse_react("我就直接回答了")
    assert parsed2["final"] == "我就直接回答了"


def test_bm25_and_rrf_fusion():
    bm25 = BM25Index()
    bm25.add("d1", "年假顺延规则 当年未休完可顺延次年")
    bm25.add("d2", "差旅报销流程 15个工作日内提交")
    hits = bm25.search("年假顺延", top_k=2)
    assert hits and hits[0].doc_id == "d1"
    # RRF：双路都命中的文档应排到只命中单路的前面
    vec = HashEmbedding()
    list1 = hits
    list2 = [type(hits[0])(doc_id="d2", text="x", score=0.9)]
    fused = rrf_fuse([list1, list2], top_k=2)
    assert fused[0].doc_id == "d1"  # d1 双路命中得分更高


def test_hash_embedding_stable_and_similarity():
    e = HashEmbedding()
    v1 = e.embed("年假顺延规则")
    v2 = e.embed("年假顺延规则")
    v3 = e.embed("采购审批流程")
    assert v1 == v2                       # 跨调用稳定（stable hash）
    assert e.similarity(v1, v2) > 0.99
    assert e.similarity(v1, v3) < e.similarity(v1, v2)


def test_validate_schema():
    schema = {"type": "object", "properties": {
        "name": {"type": "string"}, "days": {"type": "integer"},
        "leave_type": {"type": "string", "enum": ["年假", "事假"]}},
        "required": ["name"], "additionalProperties": False}
    assert validate_schema(schema, {"name": "张三", "days": 3, "leave_type": "年假"}) == []
    errs = validate_schema(schema, {"days": "3", "leave_type": "病假", "extra": 1})
    joined = "; ".join(errs)
    assert "name" in joined            # 缺必填
    assert "days" in joined            # 类型错误
    assert "leave_type" in joined      # 枚举越界
    assert "extra" in joined           # 多余参数


def test_async_wrapper():
    asyncio.run(asyncio.sleep(0))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("all tests passed")
