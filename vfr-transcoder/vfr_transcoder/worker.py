"""Worker process: claims queued jobs and drives the transcode pipeline.

Horizontal scaling: any number of workers share one SQLite file. ``claim_next``
performs the queued->running transition atomically, and an expired lease (no
heartbeat because the OS killed the process) makes a running job claimable
again -- this is the kill-and-restart guarantee.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid

from .config import Settings, settings as default_settings
from .pipeline import Transcoder
from .state import JobState, JobStore

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, store: JobStore, cfg: Settings | None = None,
                 worker_id: str | None = None):
        self.store = store
        self.cfg = cfg or default_settings
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:10]}-{os.getpid()}"
        self._shutdown = threading.Event()
        self._cancel_current = threading.Event()

    # -------------------------------------------------- signals

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            log.warning("worker %s received signal %s, shutting down",
                        self.worker_id, signum)
            self._shutdown.set()
            self._cancel_current.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # not in main thread (e.g. tests): skip
                pass

    def request_stop(self) -> None:
        self._shutdown.set()
        self._cancel_current.set()

    # -------------------------------------------------- loop

    def run_forever(self, max_jobs: int | None = None) -> int:
        self._install_signal_handlers()
        transcoder = Transcoder(self.store, self.cfg)
        processed = 0
        log.info("worker %s started, db=%s", self.worker_id, self.cfg.db_path)
        while not self._shutdown.is_set():
            # crash recovery sweep before every claim attempt
            try:
                recovered = self.store.requeue_expired(self.cfg.lease_seconds)
                if recovered:
                    log.warning("requeued %d job(s) with expired leases", recovered)
                job = self.store.claim_next(self.worker_id, self.cfg.lease_seconds)
            except Exception:
                log.exception("claim failed; retrying")
                self._wait_poll()
                continue

            if job is None:
                if max_jobs is not None and processed >= max_jobs:
                    break
                self._wait_poll()
                continue

            self._cancel_current.clear()
            processed += 1
            self._execute(transcoder, job)
            if max_jobs is not None and processed >= max_jobs:
                break
        log.info("worker %s stopping after %d job(s)", self.worker_id, processed)
        return processed

    def run_one(self, job_id: str) -> bool:
        """Used by the CLI and tests: run a specific job in-process.

        A failed or completed job is re-queued first so the state machine stays
        valid; its segment progress is preserved (use the API's reset_progress
        to deliberately start from scratch).
        """
        transcoder = Transcoder(self.store, self.cfg)
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state in (JobState.FAILED, JobState.COMPLETED):
            job = self.store.requeue(job_id)
        if job.state == JobState.QUEUED:
            job = self.store.claim_next(self.worker_id, self.cfg.lease_seconds) or job
        self._execute(transcoder, job)
        final = self.store.get_job(job_id)
        return final is not None and final.state == JobState.COMPLETED

    # -------------------------------------------------- internals

    def _execute(self, transcoder: Transcoder, job) -> None:
        log.info("worker %s claimed job %s (%s -> %s)",
                 self.worker_id, job.id, job.source, job.output)

        def on_progress(frac: float, msg: str) -> None:
            self.store.heartbeat(job.id, self.cfg.lease_seconds, progress=frac)

        try:
            transcoder.run(job, on_progress=on_progress,
                           should_cancel=self._cancel_current.is_set)
            self.store.mark_completed(job.id, job.output)
            log.info("job %s completed", job.id)
        except InterruptedError:
            # graceful stop: hand the job back to the queue, keeping all
            # segment progress; another worker (or a restart) continues it
            log.warning("job %s interrupted, requeueing with progress kept", job.id)
            self.store.requeue(job.id)
        except Exception as exc:  # noqa: BLE001 - recorded as job failure
            log.exception("job %s failed", job.id)
            try:
                self.store.mark_failed(job.id, f"{type(exc).__name__}: {exc}")
            except Exception:
                log.exception("could not mark job %s failed", job.id)

    def _wait_poll(self) -> None:
        self._shutdown.wait(self.cfg.poll_seconds)
