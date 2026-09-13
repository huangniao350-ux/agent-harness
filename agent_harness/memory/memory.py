"""三层 Memory 与 Context Engineering（对应简历：短期/长期/当前上下文三层记忆机制）。

- LongTermStore：跨会话共享的长期事实库（subject/key/value + 自然语言文本），
  写入时做去重、合并与冲突处理（新值覆盖旧值并记录冲突），检索复用 BM25+向量+RRF；
- MemoryManager：会话级实例，管理短期记忆（超预算触发 LLM 压缩）、工作记忆，
  以及按 ContextPolicy 动态组装注入上下文（优先级 + Token 预算控制）。

同一 SQLite 文件中持久化长期记忆，会话隔离与跨会话沉淀同时满足。
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..events import MEMORY_EVENT, EventBus, Event
from ..llm.base import LLMClient, LLMMessage
from ..memory.retrieval import BM25Index, HashEmbedding, VectorIndex, rrf_fuse
from ..utils import estimate_tokens, now_iso


@dataclass
class MemoryRecord:
    id: str
    subject: str        # 归属主体，如 "user"
    key: str            # 属性名，如 name / dept / preference
    value: str          # 属性值
    text: str           # 自然语言形式（用于检索）
    ts: str = field(default_factory=now_iso)
    hits: int = 0       # 命中次数


@dataclass
class ContextPolicy:
    """Context Configuration：动态控制注入范围、优先级与预算。"""

    include_working: bool = True
    include_longterm: bool = True
    include_recent_turns: int = 4
    longterm_top_k: int = 3
    token_budget: int = 1500
    priorities: tuple = ("working", "longterm", "recent", "summary")


class LongTermStore:
    """长期记忆库：SQLite 持久化 + 混合检索 + 去重/合并/冲突。"""

    def __init__(self, db_path: Path, config=None, event_bus: EventBus | None = None) -> None:
        self.db_path = db_path
        self.config = config
        self.event_bus = event_bus
        self._embedder = HashEmbedding()
        self._bm25 = BM25Index()
        self._vector = VectorIndex(self._embedder)
        self._records: dict[str, MemoryRecord] = {}
        self._init_db()
        self._load()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS longterm_memory(
            id TEXT PRIMARY KEY, subject TEXT, key TEXT, value TEXT,
            text TEXT, ts TEXT, hits INTEGER DEFAULT 0)""")
        conn.commit()
        conn.close()

    def _load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, subject, key, value, text, ts, hits FROM longterm_memory").fetchall()
        conn.close()
        for rid, subject, key, value, text, ts, hits in rows:
            rec = MemoryRecord(id=rid, subject=subject, key=key, value=value, text=text, ts=ts, hits=hits)
            self._records[rid] = rec
            self._index(rec)

    def _index(self, rec: MemoryRecord) -> None:
        self._bm25.add(rec.id, rec.text)
        self._vector.add(rec.id, rec.text)

    def _persist(self, rec: MemoryRecord) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""INSERT INTO longterm_memory(id, subject, key, value, text, ts, hits)
                        VALUES(?,?,?,?,?,?,?)
                        ON CONFLICT(id) DO UPDATE SET value=excluded.value,
                        text=excluded.text, ts=excluded.ts, hits=excluded.hits""",
                     (rec.id, rec.subject, rec.key, rec.value, rec.text, rec.ts, rec.hits))
        conn.commit()
        conn.close()

    @property
    def size(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------ 写入：去重 / 合并 / 冲突
    def add(self, subject: str, key: str, value: str) -> str:
        """写入长期事实，返回动作: added | refreshed | conflicted。"""
        text = f"{subject}的{key}是{value}"
        vid = hashlib.sha1(f"{subject}::{key}::{value}".encode()).hexdigest()[:16]
        threshold = (getattr(self.config.memory, "dedup_similarity_threshold", 0.86)
                     if self.config else 0.86)
        query_vec = self._embedder.embed(text)
        for rec in self.search(text, top_k=3):
            sim = self._embedder.similarity(query_vec, self._embedder.embed(rec.text))
            if sim >= threshold and rec.value == value:
                return "refreshed"  # 完全重复：幂等
            if sim >= threshold or (rec.key == key and rec.subject == subject):
                # 同一属性出现不一致的新值 → 冲突：新值覆盖（时间戳更新者胜），记录冲突
                old_value = rec.value
                rec.value, rec.ts, rec.text = value, now_iso(), text
                self._persist(rec)
                self._emit("conflict", {"key": key, "old": old_value, "new": value,
                                        "action": "newer_wins"})
                return "conflicted"
        rec = MemoryRecord(id=vid, subject=subject, key=key, value=value, text=text)
        self._records[rec.id] = rec
        self._index(rec)
        self._persist(rec)
        self._emit("added", {"key": key, "value": value})
        return "added"

    def search(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        if not self._records:
            return []
        bm25_hits = self._bm25.search(query, top_k=top_k * 2)
        vec_hits = self._vector.search(query, top_k=top_k * 2)
        fused = rrf_fuse([bm25_hits, vec_hits], top_k=top_k)
        out: list[MemoryRecord] = []
        for hit in fused:
            rec = self._records.get(hit.doc_id)
            if rec:
                rec.hits += 1
                out.append(rec)
        return out

    def recent(self, top_k: int = 3) -> list[MemoryRecord]:
        """按时间倒序返回最近记忆（检索未命中时的回退通道）。"""
        return sorted(self._records.values(), key=lambda r: r.ts, reverse=True)[:top_k]

    def _emit(self, action: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.event_bus.emit(Event(
            type=MEMORY_EVENT, payload={"action": action, **payload})))


class MemoryManager:
    """会话级记忆管理器：短期记忆 + 工作记忆 + 上下文组装。"""

    def __init__(self, session_id: str, store: LongTermStore, config=None,
                 event_bus: EventBus | None = None) -> None:
        self.session_id = session_id
        self.store = store
        self.config = config
        self.event_bus = event_bus
        self.short_term: list[dict[str, str]] = []
        self.summary: str = ""
        self.working: dict[str, str] = {}
        self._pending_compress: list[dict[str, str]] | None = None

    # ------------------------------------------------------------ 短期记忆
    def add_message(self, role: str, content: str) -> None:
        self.short_term.append({"role": role, "content": content})
        budget = (getattr(self.config.memory, "short_term_token_budget", 900)
                  if self.config else 900)
        total = sum(estimate_tokens(m["content"]) for m in self.short_term)
        if total > budget and len(self.short_term) > 4:
            self._pending_compress = self.short_term[:-2]   # 保留近 2 条原文
            self.short_term = self.short_term[-2:]

    async def compress_if_needed(self, llm: LLMClient) -> bool:
        """将待压缩历史交给 LLM 摘要，返回是否发生了压缩。"""
        pending = self._pending_compress
        if not pending:
            return False
        text = "\n".join(f"{m['role']}: {m['content']}" for m in pending)
        resp = await llm.chat([
            LLMMessage(role="system", content="你是摘要助手。[MODE: COMPRESS] 将对话历史压缩为要点摘要，保留人名、数字与结论。"),
            LLMMessage(role="user", content=text)])
        self.summary = (self.summary + "\n" + resp.content).strip()
        self._pending_compress = None
        return True

    # ------------------------------------------------------------ 工作记忆
    def set_working(self, key: str, value: str) -> None:
        self.working[key] = value

    # ------------------------------------------------------------ 上下文组装
    def build_context(self, query: str, policy: ContextPolicy | None = None) -> str:
        policy = policy or ContextPolicy(
            token_budget=getattr(self.config.memory, "context_token_budget", 1500)
            if self.config else 1500)
        blocks: dict[str, list[str]] = {"working": [], "longterm": [], "recent": [], "summary": []}

        if policy.include_working and self.working:
            blocks["working"].append("[工作记忆]")
            blocks["working"] += [f"- {k}={v}" for k, v in self.working.items()]

        if policy.include_longterm and self.store.size:
            records = self.store.search(query or "用户信息 偏好 部门", top_k=policy.longterm_top_k)
            if not records:
                records = self.store.recent(top_k=policy.longterm_top_k)  # 检索未命中回退最近记忆
            if records:
                blocks["longterm"].append("[长期记忆]")
                blocks["longterm"] += [f"- key={r.key} value={r.value}" for r in records]

        if self.short_term:
            recent = self.short_term[-policy.include_recent_turns:]
            if recent:
                blocks["recent"].append("[近期对话]")
                blocks["recent"] += [f"{m['role']}: {m['content'][:120]}" for m in recent]
        if self.summary:
            blocks["summary"].append("[历史摘要]")
            blocks["summary"].append(self.summary[:400])

        parts: list[str] = []
        used = 0
        for name in policy.priorities:
            text = "\n".join(blocks[name])
            if not text:
                continue
            cost = estimate_tokens(text)
            if used + cost > policy.token_budget:
                continue  # 超预算的低优先级块整体放弃（高优先级已注入）
            parts.append(text)
            used += cost
        return "\n\n".join(parts)

    def token_usage(self) -> dict[str, int]:
        return {"short_term_msgs": len(self.short_term),
                "short_term_tokens": sum(estimate_tokens(m["content"]) for m in self.short_term),
                "longterm_records": self.store.size}
