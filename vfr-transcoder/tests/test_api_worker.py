"""HTTP API + worker integration tests."""
from __future__ import annotations

import threading
import time
import unittest

from fastapi.testclient import TestClient

from tests.helpers import ffmpeg_available, make_vfr_sample, make_workdir
from vfr_transcoder.api import create_app
from vfr_transcoder.config import Settings
from vfr_transcoder.state import JobState, JobStore
from vfr_transcoder.worker import Worker


@unittest.skipUnless(ffmpeg_available(), "ffmpeg/ffprobe not installed")
class ApiWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = make_workdir()
        self.cfg = Settings(work_root=self.work / "w", segment_seconds=3.0,
                            poll_seconds=0.1, lease_seconds=30.0)
        self.store = JobStore(self.cfg.db_path)
        self.app = create_app(store=self.store, cfg=self.cfg)
        self.client = TestClient(self.app)
        self.src = make_vfr_sample(self.work / "src.mkv", duration=8.0,
                                   marker_interval=4.0)
        self.out = self.work / "api_out.mp4"
        self.worker = None
        self.worker_thread = None

    def tearDown(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker.request_stop()
            self.worker_thread.join(timeout=10)
        self.store.close()

    def _start_worker(self) -> None:
        self.worker = Worker(self.store, self.cfg)
        self.worker_thread = threading.Thread(
            target=self.worker.run_forever, daemon=True)
        self.worker_thread.start()

    def test_submit_status_and_completion(self) -> None:
        self._start_worker()
        r = self.client.post("/jobs", json={
            "source": str(self.src), "output": str(self.out), "target_fps": 25})
        self.assertEqual(r.status_code, 201)
        job_id = r.json()["id"]
        self.assertEqual(r.json()["state"], JobState.QUEUED.value)

        # poll until terminal
        deadline = time.time() + 90
        final = None
        while time.time() < deadline:
            j = self.client.get(f"/jobs/{job_id}").json()
            if j["state"] in (JobState.COMPLETED.value, JobState.FAILED.value):
                final = j
                break
            time.sleep(0.2)
        self.assertIsNotNone(final, "job never reached a terminal state")
        self.assertEqual(final["state"], JobState.COMPLETED, final.get("error"))
        self.assertEqual(final["progress"], 1.0)
        self.assertTrue(self.out.exists())
        # list endpoint
        self.assertEqual(len(self.client.get("/jobs").json()["jobs"]), 1)
        # health
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_missing_source_rejected(self) -> None:
        r = self.client.post("/jobs", json={
            "source": str(self.work / "nope.mp4"), "output": str(self.out)})
        self.assertEqual(r.status_code, 400)

    def test_failed_job_retry_flow(self) -> None:
        # submit against a source that will be deleted -> failure
        bad_src = self.work / "missing.mkv"
        bad_src.write_bytes(b"x")
        r = self.client.post("/jobs", json={
            "source": str(bad_src), "output": str(self.work / "x.mp4")})
        job_id = r.json()["id"]
        bad_src.unlink()
        self._start_worker()
        deadline = time.time() + 30
        while time.time() < deadline:
            j = self.client.get(f"/jobs/{job_id}").json()
            if j["state"] == JobState.FAILED.value:
                break
            time.sleep(0.1)
        self.assertEqual(
            self.client.get(f"/jobs/{job_id}").json()["state"],
            JobState.FAILED.value)
        # retry returns it to the queue (it will fail again, but state machine is right)
        rr = self.client.post(f"/jobs/{job_id}/retry", json={})
        self.assertEqual(rr.status_code, 200)
        self.assertEqual(rr.json()["state"], JobState.QUEUED.value)

    def test_completed_job_retry_reset_progress(self) -> None:
        # run one job to completion with real segments, then reset + requeue
        self._start_worker()
        r = self.client.post("/jobs", json={
            "source": str(self.src), "output": str(self.out), "target_fps": 25})
        job_id = r.json()["id"]
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.client.get(f"/jobs/{job_id}").json()["state"] == \
                    JobState.COMPLETED.value:
                break
            time.sleep(0.2)
        segs_before = len(self.store.get_segments(job_id))
        self.assertGreater(segs_before, 0)
        # stop the pool worker so it cannot immediately re-claim the retry
        self.worker.request_stop()
        self.worker_thread.join(timeout=10)
        rr = self.client.post(f"/jobs/{job_id}/retry",
                              json={"reset_progress": True})
        self.assertEqual(rr.status_code, 200)
        body = rr.json()
        self.assertEqual(body["state"], JobState.QUEUED.value)
        self.assertEqual(body["segments_done"], 0)
        self.assertEqual(len(self.store.get_segments(job_id)), 0)

    def test_worker_pool_two_workers_no_double_claim(self) -> None:
        # two workers sharing one store must each take distinct jobs
        w1 = Worker(self.store, self.cfg, worker_id="pool-1")
        w2 = Worker(self.store, self.cfg, worker_id="pool-2")
        outs = []
        for i in range(2):
            out = self.work / f"pool{i}.mp4"
            outs.append(out)
            self.client.post("/jobs", json={
                "source": str(self.src), "output": str(out)})
        t1 = threading.Thread(target=w1.run_forever, kwargs={"max_jobs": 1})
        t2 = threading.Thread(target=w2.run_forever, kwargs={"max_jobs": 1})
        t1.start(); t2.start()
        t1.join(timeout=90); t2.join(timeout=90)
        jobs = self.client.get("/jobs").json()["jobs"]
        workers = {j["worker_id"] for j in jobs}
        self.assertEqual(workers, {"pool-1", "pool-2"})
        self.assertTrue(all(o.exists() for o in outs))


if __name__ == "__main__":
    unittest.main()
