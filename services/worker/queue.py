"""
SQLite-backed job queue.
Thread-safe via WAL mode + per-connection pragmas.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from config import DB_PATH
from models import Job, JobStatus, JobStepInfo


_local = threading.local()

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("download",   "Download audio"),
    ("separate",   "Separate vocals"),
    ("transcribe", "Transcribe speech"),
    ("align",      "Align timestamps"),
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    song_id       TEXT NOT NULL,
    youtube_id    TEXT NOT NULL,
    youtube_url   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    progress      INTEGER NOT NULL DEFAULT 0,
    current_step  TEXT,
    steps_json    TEXT NOT NULL,
    error         TEXT,
    result_path   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_youtube ON jobs(youtube_id);
"""


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    if not hasattr(_local, "conn"):
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(CREATE_SQL)
        conn.commit()
        _local.conn = conn
    yield _local.conn


def _default_steps() -> list[JobStepInfo]:
    return [
        JobStepInfo(name=name, label=label, status=JobStatus.queued, progress=0)
        for name, label in PIPELINE_STEPS
    ]


def _row_to_job(row: sqlite3.Row) -> Job:
    steps_raw = json.loads(row["steps_json"])
    steps = [
        JobStepInfo(
            name=s["name"],
            label=s["label"],
            status=JobStatus(s["status"]),
            progress=s["progress"],
            error=s.get("error"),
        )
        for s in steps_raw
    ]
    return Job(
        id=row["id"],
        song_id=row["song_id"],
        youtube_id=row["youtube_id"],
        youtube_url=row["youtube_url"],
        status=JobStatus(row["status"]),
        progress=row["progress"],
        current_step=row["current_step"],
        steps=steps,
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def create_job(song_id: str, youtube_id: str, youtube_url: str) -> Job:
    now = datetime.now(timezone.utc).isoformat()
    job_id = str(uuid.uuid4())
    steps = _default_steps()
    steps_json = json.dumps([s.model_dump() for s in steps])

    with _conn() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, song_id, youtube_id, youtube_url, status, progress,
                current_step, steps_json, error, result_path, created_at, updated_at)
               VALUES (?,?,?,?,'queued',0,NULL,?,NULL,NULL,?,?)""",
            (job_id, song_id, youtube_id, youtube_url, steps_json, now, now),
        )
        conn.commit()

    return get_job(job_id)  # type: ignore[return-value]


def get_job(job_id: str) -> Optional[Job]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def get_queue_depth() -> int:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')"
        ).fetchone()
    return row[0] if row else 0


def update_job_step(
    job_id: str,
    step_name: str,
    step_status: JobStatus,
    step_progress: int,
    overall_progress: int,
    error: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        row = conn.execute("SELECT steps_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return
        steps = json.loads(row["steps_json"])
        for s in steps:
            if s["name"] == step_name:
                s["status"]   = step_status.value
                s["progress"] = step_progress
                s["error"]    = error
        conn.execute(
            """UPDATE jobs SET
               status        = CASE WHEN ? = 'failed' THEN 'failed' ELSE status END,
               current_step  = ?,
               steps_json    = ?,
               progress      = ?,
               error         = COALESCE(?, error),
               updated_at    = ?
               WHERE id = ?""",
            (
                step_status.value,
                step_name,
                json.dumps(steps),
                overall_progress,
                error,
                now,
                job_id,
            ),
        )
        conn.commit()


def set_job_status(
    job_id: str,
    status: JobStatus,
    result_path: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """UPDATE jobs SET status=?, result_path=?, error=?, updated_at=?
               WHERE id=?""",
            (status.value, result_path, error, now, job_id),
        )
        conn.commit()


def get_result_path(job_id: str) -> Optional[str]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT result_path FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
    return row["result_path"] if row else None
