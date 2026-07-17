"""
Lightweight SQLite-backed conversation memory, keyed by session_id.
Survives process restarts, unlike Streamlit's in-memory session state,
and works fine for a single-instance production deployment.
"""
import sqlite3
from contextlib import contextmanager

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(config.SESSION_DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute(_SCHEMA)


def add_message(session_id: str, role: str, content: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_history(session_id: str, max_turns: int = None) -> list[dict]:
    max_turns = max_turns or config.MAX_HISTORY_TURNS
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, max_turns * 2),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def clear_session(session_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
