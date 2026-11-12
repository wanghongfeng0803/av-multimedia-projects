"""SQLite-backed job state machine with worker leases and resumable progress.

State machine::

    pending -> queued -> running -> completed
                              \\-> failed -> queued   (manual/auto retry)
        queued/running (dead lease) -> queued         (crash recovery)

Durability contract for "kill -9 and restart":

* Every state transition, heartbeat and segment completion is committed before
  the expensive ffmpeg work it describes is considered done.
* Video is encoded in fixed-length MPEG-TS segments. A finished segment file
  plus its ``segments`` row means it never has to be re-encoded; a missing or
  partial segment is simply overwritten on resume.
* Audio is one whole-file extraction with a temp name + atomic rename.
* The final mux writes to ``output.tmp`` and atomically renames on success, so
  a kill during mux leaves no half-finished deliverable.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- states

class JobState(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def active(cls) -> tuple["JobState", ...]:
        return (cls.QUEUED, cls.RUNNING)


TERMINAL_STATES = (JobState.COMPLETED, JobState.FAILED)

# Allowed transitions; anything outside this raises.
_TRANSITIONS: dict[JobState, tuple[JobState, ...]] = {
    JobState.PENDING: (JobState.QUEUED, JobState.FAILED),
    JobState.QUEUED: (JobState.RUNNING, JobState.FAILED),
    JobState.RUNNING: (JobState.COMPLETED, JobState.FAILED, JobState.QUEUED),
    JobState.FAILED: (JobState.QUEUED,),
    # a finished job may be manually re-run from scratch or after a retry call
    JobState.COMPLETED: (JobState.QUEUED,),
}


class IllegalTransition(RuntimeError):
    pass


# ---------------------------------------------------------------- data

@dataclass
class Job:
    id: str
    source: str
    output: str
    target_fps: float | None
    options: dict[str, Any]
    state: JobState
    progress: float
    worker_id: str | None
    lease_expires_at: float | None
    heartbeat_at: float | None
    attempts: int
    error: str | None
    created_at: float
    updated_at: float
    origin_pts: float | None = None
    segments_done: int = 0
    segments_total: int = 0
    audio_done: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"],
            source=row["source"],
            output=row["output"],
            target_fps=row["target_fps"],
            options=json.loads(row["options"]),
            state=JobState(row["state"]),
            progress=row["progress"],
            worker_id=row["worker_id"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            attempts=row["attempts"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            origin_pts=row["origin_pts"],
            segments_done=row["segments_done"],
            segments_total=row["segments_total"],
            audio_done=bool(row["audio_done"]),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "output": self.output,
            "target_fps": self.target_fps,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "segments_done": self.segments_done,
            "segments_total": self.segments_total,
            "audio_done": self.audio_done,
            "origin_pts": self.origin_pts,
            "attempts": self.attempts,
            "error": self.error,
            "worker_id": self.worker_id,
            "heartbeat_at": self.heartbeat_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------- store

class JobStore:
    """Thread-safe SQLite store. One file per deployment; safe for N workers."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        output TEXT NOT NULL,
        target_fps REAL,
        options TEXT NOT NULL,
        state TEXT NOT NULL,
        progress REAL NOT NULL DEFAULT 0,
        worker_id TEXT,
        lease_expires_at REAL,
        heartbeat_at REAL,
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        origin_pts REAL,
        segments_done INTEGER NOT NULL DEFAULT 0,
        segments_total INTEGER NOT NULL DEFAULT 0,
        audio_done INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, lease_expires_at);

    CREATE TABLE IF NOT EXISTS segments (
        job_id TEXT NOT NULL,
        segment_index INTEGER NOT NULL,
        path TEXT NOT NULL,
        start_time REAL NOT NULL,
        duration REAL NOT NULL,
        PRIMARY KEY (job_id, segment_index)
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(self.SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------- create / read

    def create_job(
        self,
        source: str,
        output: str,
        target_fps: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> Job:
        now = time.time()
        job_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, source, output, target_fps, options, state, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, source, output, target_fps,
                 json.dumps(options or {}), JobState.QUEUED.value, now, now))
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [Job.from_row(r) for r in rows]

    # -------------------------------------------------- transitions

    def _set_state(self, job_id: str, new: JobState, **fields: Any) -> None:
        with self._lock:
            row = self._conn.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = JobState(row["state"])
            if new != current and new not in _TRANSITIONS[current]:
                raise IllegalTransition(f"{job_id}: {current.value} -> {new.value}")
            cols = ["state=?", "updated_at=?"]
            vals: list[Any] = [new.value, time.time()]
            for k, v in fields.items():
                cols.append(f"{k}=?")
                vals.append(v)
            vals.append(job_id)
            self._conn.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?", vals)

    def claim_next(self, worker_id: str, lease_seconds: float) -> Job | None:
        """Atomically claim one queued job, or reclaim an expired lease."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE state=? ORDER BY created_at LIMIT 1",
                (JobState.QUEUED.value,)).fetchone()
            if row is None:
                # crash recovery: running job whose lease expired
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE state=? AND lease_expires_at < ? "
                    "ORDER BY lease_expires_at LIMIT 1",
                    (JobState.RUNNING.value, now)).fetchone()
                if row is None:
                    return None
            job = Job.from_row(row)
            self._conn.execute(
                "UPDATE jobs SET state=?, worker_id=?, lease_expires_at=?, heartbeat_at=?, "
                "attempts=attempts+1, updated_at=? WHERE id=?",
                (JobState.RUNNING.value, worker_id, now + lease_seconds, now,
                 time.time(), job.id))
        return self.get_job(job.id)

    def heartbeat(self, job_id: str, lease_seconds: float,
                  progress: float | None = None, **extra: Any) -> None:
        now = time.time()
        with self._lock:
            sets = ["lease_expires_at=?", "heartbeat_at=?", "updated_at=?"]
            vals: list[Any] = [now + lease_seconds, now, now]
            if progress is not None:
                sets.append("progress=?")
                vals.append(progress)
            for k, v in extra.items():
                sets.append(f"{k}=?")
                vals.append(v)
            vals.append(job_id)
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals)

    def mark_completed(self, job_id: str, output: str) -> None:
        self._set_state(job_id, JobState.COMPLETED, progress=1.0,
                        lease_expires_at=None, error=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._set_state(job_id, JobState.FAILED, error=error[:2000],
                        lease_expires_at=None)

    def requeue(self, job_id: str) -> Job:
        self._set_state(job_id, JobState.QUEUED, worker_id=None,
                        lease_expires_at=None, error=None)
        return self.get_job(job_id)  # type: ignore[return-value]

    def requeue_expired(self, lease_seconds: float) -> int:
        """Move stale running leases back to queued. Returns count."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET state=?, worker_id=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE state=? AND lease_expires_at < ?",
                (JobState.QUEUED.value, now, JobState.RUNNING.value, now))
            return cur.rowcount

    # -------------------------------------------------- segment progress

    def init_segment_plan(self, job_id: str, total: int,
                          origin_pts: float | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET segments_total=?, origin_pts=?, updated_at=? WHERE id=?",
                (total, origin_pts, time.time(), job_id))

    def record_segment(self, job_id: str, index: int, path: str,
                       start_time: float, duration: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments "
                "(job_id, segment_index, path, start_time, duration) VALUES (?,?,?,?,?)",
                (job_id, index, path, start_time, duration))
            done = self._conn.execute(
                "SELECT COUNT(*) c FROM segments WHERE job_id=?", (job_id,)).fetchone()["c"]
            total = self._conn.execute(
                "SELECT segments_total t FROM jobs WHERE id=?", (job_id,)).fetchone()["t"]
            self._conn.execute(
                "UPDATE jobs SET segments_done=?, updated_at=? WHERE id=?",
                (done, time.time(), job_id))

    def get_segments(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE job_id=? ORDER BY segment_index",
                (job_id,)).fetchall()
        return [dict(r) for r in rows]

    def mark_audio_done(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET audio_done=1, updated_at=? WHERE id=?",
                (time.time(), job_id))

    def reset_progress(self, job_id: str) -> None:
        """Clear all resumable progress so the next run starts from scratch.
        Segment/audio FILES are removed by the caller; the store stays
        filesystem-agnostic and only resets its own rows/columns."""
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
            self._conn.execute(
                "UPDATE jobs SET segments_done=0, segments_total=0, audio_done=0, "
                "origin_pts=NULL, error=NULL, updated_at=? WHERE id=?",
                (time.time(), job_id))
