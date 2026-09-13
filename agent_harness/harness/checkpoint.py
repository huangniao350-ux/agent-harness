"""Checkpoint 持久化（对应简历：状态管理与 Checkpoint 续跑）。

设计要点：
- 以 session_id 为主键保存完整 AgentState 快照（JSON），ReAct 每步后、Plan-Execute
  每层后落盘，进程崩溃/主动中断后可从最近快照恢复，已完成的工作不重做；
- 演示用 SQLite 单表实现；生产可替换为 Redis/Postgres 后端，接口不变。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .state import AgentState


class CheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS checkpoints(
            session_id TEXT PRIMARY KEY, goal TEXT, mode TEXT, status TEXT,
            state_json TEXT, updated_at TEXT)""")
        conn.commit()
        conn.close()

    def save(self, state: AgentState, updated_at: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""INSERT INTO checkpoints(session_id, goal, mode, status, state_json, updated_at)
                        VALUES(?,?,?,?,?,?)
                        ON CONFLICT(session_id) DO UPDATE SET status=excluded.status,
                        state_json=excluded.state_json, updated_at=excluded.updated_at""",
                     (state.session_id, state.goal, state.mode, state.status,
                      json.dumps(state.to_dict(), ensure_ascii=False), updated_at))
        conn.commit()
        conn.close()

    def load(self, session_id: str) -> AgentState | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT state_json FROM checkpoints WHERE session_id=?",
                           (session_id,)).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return AgentState.from_dict(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            return None

    def delete(self, session_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM checkpoints WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()

    def list_sessions(self, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT session_id, goal, mode, status, updated_at FROM checkpoints "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [{"session_id": r[0], "goal": r[1], "mode": r[2], "status": r[3], "updated_at": r[4]}
                for r in rows]
