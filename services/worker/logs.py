"""
Per-job structured logger backed by the same SQLite DB as the job queue.
Log entries are written to a `job_logs` table and returned with each job poll
so the frontend gets live updates without a separate endpoint.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Literal, Optional

from config import DB_PATH

logger = logging.getLogger(__name__)

LogLevel = Literal["info", "warning", "error"]

_local = threading.local()

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS job_logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT    NOT NULL,
    ts      TEXT    NOT NULL,
    level   TEXT    NOT NULL,
    stage   TEXT,
    message TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_logs_job ON job_logs(job_id);
"""


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "log_conn"):
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(CREATE_SQL)
        conn.commit()
        _local.log_conn = conn
    return _local.log_conn


class JobLogger:
    """Writes structured log entries for a single job and forwards to Python logging."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._py_logger = logging.getLogger(f"worker.job.{job_id[:8]}")
        # Ensure table exists on first use
        _conn()

    def _write(self, level: LogLevel, message: str, stage: Optional[str]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            conn = _conn()
            conn.execute(
                "INSERT INTO job_logs (job_id, ts, level, stage, message) VALUES (?,?,?,?,?)",
                (self.job_id, ts, level, stage, message),
            )
            conn.commit()
        except Exception as exc:
            self._py_logger.error("Failed to write job log: %s", exc)

        # Also forward to Python root logger
        py_level = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
        self._py_logger.log(py_level.get(level, logging.INFO), "[%s] %s", stage or "—", message)

    def info(self, message: str, stage: Optional[str] = None) -> None:
        self._write("info", message, stage)

    def warning(self, message: str, stage: Optional[str] = None) -> None:
        self._write("warning", message, stage)

    def error(self, message: str, stage: Optional[str] = None) -> None:
        self._write("error", message, stage)


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_recent_logs(job_id: str, limit: int = 40) -> list[dict]:
    """Return the most recent log entries for a job (newest last)."""
    conn = _conn()
    rows = conn.execute(
        """SELECT ts, level, stage, message
           FROM job_logs
           WHERE job_id = ?
           ORDER BY id DESC
           LIMIT ?""",
        (job_id, limit),
    ).fetchall()
    # Reverse so oldest-first in the list
    return [
        {"ts": r["ts"], "level": r["level"], "stage": r["stage"], "message": r["message"]}
        for r in reversed(rows)
    ]


def get_best_logs(youtube_id: str, limit: int = 5000) -> list[dict]:
    """
    Return logs from the most log-rich completed job for a youtube_id.

    Bypasses currentJobId — useful when the user re-submitted an instant-cache
    job after a full run, which would otherwise hide the full diagnostic log.
    """
    conn = _conn()
    try:
        row = conn.execute(
            """SELECT l.job_id, COUNT(l.id) as cnt
               FROM job_logs l
               JOIN jobs j ON j.id = l.job_id
               WHERE j.youtube_id = ?
               GROUP BY l.job_id
               ORDER BY cnt DESC
               LIMIT 1""",
            (youtube_id,),
        ).fetchone()
    except Exception:
        return []
    if not row:
        return []
    return get_recent_logs(row[0], limit=limit)


def purge_job_logs(job_id: str) -> None:
    conn = _conn()
    conn.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
    conn.commit()
