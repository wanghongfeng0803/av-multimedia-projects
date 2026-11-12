"""FastAPI HTTP surface.

The API process only owns the SQLite state; actual transcoding happens in
separate worker processes (see :mod:`vfr_transcoder.worker`). Any number of
each can run against the same work directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, settings as default_settings
from .state import IllegalTransition, JobState, JobStore


class JobCreate(BaseModel):
    source: str = Field(..., description="absolute or server-relative input path")
    output: str = Field(..., description="destination media path")
    target_fps: float | None = Field(None, gt=0, le=240)
    options: dict[str, Any] | None = None


class JobRetry(BaseModel):
    reset_progress: bool = Field(
        False, description="if true, delete segment progress and re-encode from scratch")


def create_app(store: JobStore | None = None, cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or default_settings
    store = store or JobStore(cfg.db_path)
    app = FastAPI(title="VFR Transcoder", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "db": str(Path(cfg.db_path))}

    @app.post("/jobs", status_code=201)
    def submit_job(body: JobCreate) -> dict:
        if not Path(body.source).exists():
            raise HTTPException(400, f"source does not exist: {body.source}")
        job = store.create_job(body.source, body.output, body.target_fps, body.options)
        return job.to_public_dict()

    @app.get("/jobs")
    def list_jobs(limit: int = 100) -> dict:
        jobs = store.list_jobs(limit=min(max(1, limit), 500))
        return {"jobs": [j.to_public_dict() for j in jobs]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        data = job.to_public_dict()
        data["segments"] = store.get_segments(job_id)
        return data

    @app.post("/jobs/{job_id}/retry")
    def retry_job(job_id: str, body: JobRetry | None = None) -> dict:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.state not in (JobState.FAILED, JobState.COMPLETED, JobState.QUEUED):
            raise HTTPException(409, f"cannot retry job in state {job.state.value}")
        if body and body.reset_progress:
            for seg in store.get_segments(job_id):
                Path(seg["path"]).unlink(missing_ok=True)
            (cfg.work_root / "jobs" / job_id / "audio.m4a").unlink(missing_ok=True)
            store.reset_progress(job_id)
        try:
            job = store.requeue(job_id)
        except IllegalTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.to_public_dict()

    return app
