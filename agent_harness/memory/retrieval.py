"""混合检索原语：BM25 关键词检索 + 轻量 Hash 向量检索 + RRF 融合。

对应简历中的 Hybrid Search（BM25 + Embedding）+ RRF 融合：
- BM25：稀疏精确匹配，对专名/编号/条款号类查询强；
- 向量：字符 bigram 的 256 维 Hash Embedding（stable hash，跨进程一致、可持久化），
  捕捉语义近似；生产环境替换为 bge-m3 等真实 Embedding + Milvus/Weaviate，接口不变；
- RRF（Reciprocal Rank Fusion）：score = Σ 1/(k + rank_i)，只依赖排名、免调参，
  对两路异构分数天然公平。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils import stable_hash, tokenize_cn

EMBED_DIM = 256
RRF_K = 60


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)


class BM25Index:
    """经典 BM25 (k1, b)：构建开销 O(N)，查询开销 O(N·|q|)，演示语料规模完全够用。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._metas: list[dict] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def add(self, doc_id: str, text: str, meta: dict | None = None) -> None:
        tokens = tokenize_cn(text)
        self._doc_ids.append(doc_id)
        self._tokens.append(tokens)
        self._metas.append(meta or {})
        for t in set(tokens):
            self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = sum(len(d) for d in self._tokens) / len(self._tokens)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self._tokens:
            return []
        q_tokens = tokenize_cn(query)
        n = len(self._tokens)
        hits: list[Hit] = []
        for i, doc_tokens in enumerate(self._tokens):
            tf: dict[str, int] = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in q_tokens:
                freq = tf.get(t, 0)
                if freq == 0:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                norm = self.k1 * (1 - self.b + self.b * len(doc_tokens) / (self._avgdl or 1.0))
                score += idf * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                hits.append(Hit(doc_id=self._doc_ids[i], text="".join(doc_tokens)[:400],
                                score=score, meta=self._metas[i]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


class HashEmbedding:
    """256 维 Hash Trick 向量：字符 bigram/单词 → stable hash 桶 → L2 归一化。

    无需训练、零依赖、跨进程一致；精度低于真实 Embedding，但足以演示
    「向量召回 + 融合排序」的完整链路。
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize_cn(text)
        if not tokens:
            return vec
        for t in tokens:
            idx = stable_hash(t) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """暴力余弦检索（演示规模 O(N) 足够；生产换 HNSW/IVF 等向量索引）。"""

    def __init__(self, embedder: HashEmbedding | None = None) -> None:
        self.embedder = embedder or HashEmbedding()
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []
        self._metas: list[dict] = []

    def add(self, doc_id: str, text: str, meta: dict | None = None) -> None:
        self._ids.append(doc_id)
        self._vecs.append(self.embedder.embed(text))
        self._metas.append(meta or {})

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self._ids:
            return []
        qv = self.embedder.embed(query)
        scored = [(self.embedder.similarity(qv, v), i) for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Hit(doc_id=self._ids[i], text="", score=s, meta=self._metas[i])
                for s, i in scored[:top_k] if s > 0.01]


def rrf_fuse(result_lists: list[list[Hit]], top_k: int = 5) -> list[Hit]:
    """Reciprocal Rank Fusion：对多路检索结果按排名融合。

    score(d) = Σ_lists 1/(RRF_K + rank_in_list)，RRF_K=60 为论文推荐值。
    """
    fused: dict[str, tuple[float, Hit]] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            gain = 1.0 / (RRF_K + rank + 1)
            key = hit.doc_id
            if key in fused:
                fused[key] = (fused[key][0] + gain, fused[key][1])
            else:
                fused[key] = (gain, hit)
    ordered = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
    out: list[Hit] = []
    for _doc_id, (score, hit) in ordered[:top_k]:
        out.append(Hit(doc_id=hit.doc_id, text=hit.text, score=score, meta=hit.meta))
    return out
