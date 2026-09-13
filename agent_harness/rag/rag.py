"""企业知识库 RAG：分块 → 双路索引 → 混合检索。

对应简历 RAG 链路：Chunking 策略 → Embedding/索引 → Hybrid Search（BM25 + 向量）→ RRF 融合 → TOP-K。
分块策略：按二级标题切分 + 320 字符窗口 packing（重叠 40 字符），块粒度与条款对齐，
避免"整篇文档一个 chunk 导致的召回稀释"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ..memory.retrieval import BM25Index, Hit, VectorIndex, rrf_fuse

CHUNK_SIZE = 320
CHUNK_OVERLAP = 40


@dataclass
class Chunk:
    doc_id: str
    doc: str          # 文档名
    heading: str      # 所属章节
    text: str


def chunk_markdown(text: str, doc: str) -> list[Chunk]:
    """按二级标题分块，超长块做滑动窗口 packing。"""
    doc_title = ""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        doc_title = m.group(1).strip()
    sections = re.split(r"\n(?=##\s)", text)
    chunks: list[Chunk] = []
    for section in sections:
        head_m = re.match(r"##\s+(.+)\n", section)
        heading = head_m.group(1).strip() if head_m else "总则"
        body = section.strip()
        if len(body) <= CHUNK_SIZE:
            if body:
                chunks.append(Chunk(doc_id="", doc=doc_title or doc, heading=heading, text=body))
            continue
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(body), step):
            piece = body[i : i + CHUNK_SIZE]
            if len(piece) > 50 or i == 0:
                chunks.append(Chunk(doc_id="", doc=doc_title or doc, heading=heading, text=piece))
    return chunks


class KnowledgeBase:
    """混合检索知识库（线程安全、惰性构建、进程内单例）。"""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir
        self._bm25 = BM25Index()
        self._vector = VectorIndex()
        self._chunks: dict[str, Chunk] = {}
        self._lock = Lock()
        self._built = False

    def build(self) -> None:
        with self._lock:
            if self._built:
                return
            files = sorted(self.corpus_dir.glob("*.md"))
            for path in files:
                text = path.read_text(encoding="utf-8")
                for idx, chunk in enumerate(chunk_markdown(text, path.stem)):
                    chunk.doc_id = f"{path.stem}#{idx}"
                    self._chunks[chunk.doc_id] = chunk
                    self._bm25.add(chunk.doc_id, chunk.text,
                                   {"doc": chunk.doc, "heading": chunk.heading})
                    self._vector.add(chunk.doc_id, chunk.text,
                                     {"doc": chunk.doc, "heading": chunk.heading})
            self._built = True

    def search(self, query: str, top_k: int = 3, min_bm25: float = 1.0) -> list[Hit]:
        """混合检索 + 相关性门槛。

        min_bm25：BM25 关键词证据阈值 —— 向量检索存在 hash 碰撞带来的弱相关噪声，
        完全无词汇证据的召回一律丢弃，让上层明确"知识库未覆盖"而非编造（幻觉治理）。
        """
        self.build()
        bm25_hits = self._bm25.search(query, top_k=top_k * 2)
        vec_hits = self._vector.search(query, top_k=top_k * 2)
        fused = rrf_fuse([bm25_hits, vec_hits], top_k=top_k * 2)
        bm25_scores = {h.doc_id: h.score for h in bm25_hits}
        kept = [h for h in fused if bm25_scores.get(h.doc_id, 0.0) >= min_bm25]
        out: list[Hit] = []
        for hit in kept[:top_k]:
            chunk = self._chunks.get(hit.doc_id)
            if chunk:
                hit.text = chunk.text
                hit.meta = {"doc": chunk.doc, "heading": chunk.heading}
                out.append(hit)
        return out

    @property
    def size(self) -> int:
        self.build()
        return len(self._chunks)


_KB: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """进程内单例：语料随包内置（rag/corpus/*.md），首次检索时构建索引。"""
    global _KB
    if _KB is None:
        corpus_dir = Path(__file__).parent / "corpus"
        _KB = KnowledgeBase(corpus_dir)
    return _KB
