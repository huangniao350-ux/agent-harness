"""Trace 全链路追踪（对应简历：Trace/Replay 评估体系与可观测平台）。

- TraceSubscriber 订阅事件总线，把执行过程落为 SQLite 中的 trace + span 两级结构；
- LLM span 记录输出内容，供 Replay 确定性重放；
- 支持导出 JSON 供外部分析（生产可对接 Langfuse/OpenTelemetry，事件结构已对齐思路）。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..events import (ERROR, FINAL_ANSWER, GUARDRAIL, LLM_CALL, MEMORY_EVENT, PLAN_CREATED,
                      REFLECT, STEP_FINISHED, STEP_STARTED, TASK_FINISHED, TASK_STARTED,
                      TOOL_CALL, Event)
from ..utils import now_iso

_SPAN_TYPES = {LLM_CALL: "llm", TOOL_CALL: "tool", PLAN_CREATED: "plan",
               STEP_STARTED: "step", STEP_FINISHED: "step", REFLECT: "reflect",
               GUARDRAIL: "guardrail", MEMORY_EVENT: "memory", ERROR: "error"}


class TraceDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS traces(
            trace_id TEXT PRIMARY KEY, session_id TEXT, goal TEXT, mode TEXT, status TEXT,
            answer TEXT, tokens INTEGER, latency_ms REAL, started_at TEXT, finished_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS spans(
            id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, span_type TEXT, name TEXT,
            status TEXT, latency_ms REAL, detail TEXT, ts TEXT)""")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------- 写入
    def start_trace(self, trace_id: str, session_id: str, goal: str, mode: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO traces VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (trace_id, session_id, goal, mode, "running", "", 0, 0.0, now_iso(), ""))
        conn.commit()
        conn.close()

    def add_span(self, trace_id: str, span_type: str, name: str, status: str = "ok",
                 latency_ms: float = 0.0, detail: dict | None = None) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO spans(trace_id, span_type, name, status, latency_ms, detail, ts) "
                     "VALUES(?,?,?,?,?,?,?)",
                     (trace_id, span_type, name, status, latency_ms,
                      json.dumps(detail or {}, ensure_ascii=False), now_iso()))
        conn.commit()
        conn.close()

    def finish_trace(self, trace_id: str, status: str, answer: str, tokens: int,
                     latency_ms: float) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE traces SET status=?, answer=?, tokens=?, latency_ms=?, finished_at=? "
                     "WHERE trace_id=?",
                     (status, answer[:2000], tokens, latency_ms, now_iso(), trace_id))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------- 查询
    def get(self, trace_id: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        if not row:
            conn.close()
            return None
        cols = [c[0] for c in conn.execute("SELECT * FROM traces LIMIT 1").description]
        spans = conn.execute("SELECT id, span_type, name, status, latency_ms, detail, ts "
                             "FROM spans WHERE trace_id=? ORDER BY id", (trace_id,)).fetchall()
        conn.close()
        trace = dict(zip(cols, row))
        trace["spans"] = [{"id": s[0], "type": s[1], "name": s[2], "status": s[3],
                           "latency_ms": s[4], "detail": json.loads(s[5] or "{}"), "ts": s[6]}
                          for s in spans]
        return trace

    def list_traces(self, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT trace_id, session_id, goal, mode, status, tokens, latency_ms, "
                            "started_at FROM traces ORDER BY started_at DESC LIMIT ?",
                            (limit,)).fetchall()
        conn.close()
        return [{"trace_id": r[0], "session_id": r[1], "goal": r[2], "mode": r[3], "status": r[4],
                 "tokens": r[5], "latency_ms": r[6], "started_at": r[7]} for r in rows]

    def export_json(self, trace_id: str, out_dir: Path) -> Path | None:
        data = self.get(trace_id)
        if data is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"trace_{trace_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class TraceSubscriber:
    """事件总线订阅者：事件 → trace/span 落库（对主链路零侵入）。"""

    def __init__(self, db: TraceDB) -> None:
        self.db = db

    async def __call__(self, event: Event) -> None:
        payload = event.payload or {}
        etype = event.type
        if etype == TASK_STARTED:
            self.db.start_trace(event.trace_id, event.session_id,
                                payload.get("goal", ""), payload.get("mode", ""))
        elif etype == TASK_FINISHED:
            self.db.finish_trace(event.trace_id, payload.get("status", "done"),
                                 payload.get("answer", ""), payload.get("tokens", 0),
                                 payload.get("latency_ms", 0.0))
        elif etype in _SPAN_TYPES:
            detail: dict[str, Any] = {k: v for k, v in payload.items() if k not in ("args",)}
            if "args" in payload:
                detail["args"] = payload["args"]
            status = "ok"
            if etype == TOOL_CALL:
                status = "ok" if payload.get("ok") else "error"
            self.db.add_span(event.trace_id, _SPAN_TYPES[etype],
                             str(payload.get("role") or payload.get("tool")
                                 or payload.get("step_id") or payload.get("stage") or etype),
                             status=status, latency_ms=payload.get("latency_ms", 0.0),
                             detail=detail)
