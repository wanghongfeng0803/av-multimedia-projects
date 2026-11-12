"""Unit tests for the job state machine, leases and segment persistence."""
from __future__ import annotations

import time
import unittest

from tests.helpers import ffmpeg_available, make_workdir
from vfr_transcoder.config import Settings
from vfr_transcoder.state import IllegalTransition, JobState, JobStore


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = make_workdir()
        cfg = Settings(work_root=self.work)
        self.store = JobStore(cfg.db_path)

    def tearDown(self) -> None:
        self.store.close()

    def test_lifecycle_queued_running_completed(self) -> None:
        job = self.store.create_job("/in.mp4", "/out.mp4", 25.0)
        self.assertEqual(job.state, JobState.QUEUED)
        claimed = self.store.claim_next("w1", lease_seconds=30)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.id, job.id)
        self.assertEqual(claimed.state, JobState.RUNNING)
        self.assertEqual(claimed.worker_id, "w1")
        # a second claim finds nothing while the lease is live
        self.assertIsNone(self.store.claim_next("w2", lease_seconds=30))
        self.store.mark_completed(job.id, job.output)
        self.assertEqual(self.store.get_job(job.id).state, JobState.COMPLETED)

    def test_illegal_transition_rejected(self) -> None:
        job = self.store.create_job("/in.mp4", "/out.mp4")
        self.store.claim_next("w1", 30)
        self.store.mark_completed(job.id, job.output)
        with self.assertRaises(IllegalTransition):
            self.store.mark_failed(job.id, "late failure")

    def test_failed_job_can_be_requed_and_rerun(self) -> None:
        job = self.store.create_job("/in.mp4", "/out.mp4")
        self.store.claim_next("w1", 30)
        self.store.mark_failed(job.id, "boom")
        self.assertEqual(self.store.get_job(job.id).state, JobState.FAILED)
        self.store.requeue(job.id)
        self.assertEqual(self.store.get_job(job.id).state, JobState.QUEUED)
        again = self.store.claim_next("w2", 30)
        self.assertEqual(again.attempts, 2)

    def test_expired_lease_is_reclaimed(self) -> None:
        job = self.store.create_job("/in.mp4", "/out.mp4")
        claimed = self.store.claim_next("w1", lease_seconds=0.2)
        self.assertEqual(claimed.state, JobState.RUNNING)
        time.sleep(0.3)
        # sweep + reclaim models a killed worker being replaced after restart
        n = self.store.requeue_expired(lease_seconds=0.2)
        self.assertEqual(n, 1)
        stolen = self.store.claim_next("w2", 30)
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen.id, job.id)
        self.assertEqual(stolen.worker_id, "w2")

    def test_segment_plan_and_resume_records(self) -> None:
        job = self.store.create_job("/in.mp4", "/out.mp4")
        self.store.init_segment_plan(job.id, total=3, origin_pts=0.12)
        for i in range(2):
            self.store.record_segment(job.id, i, f"/seg{i}.ts", i * 10, 10)
        fresh = self.store.get_job(job.id)
        self.assertEqual(fresh.segments_total, 3)
        self.assertEqual(fresh.segments_done, 2)
        self.assertAlmostEqual(fresh.origin_pts, 0.12)
        segs = self.store.get_segments(job.id)
        self.assertEqual([s["segment_index"] for s in segs], [0, 1])
        # re-inserting is idempotent (resume must not double count)
        self.store.record_segment(job.id, 1, "/seg1.ts", 10, 10)
        self.assertEqual(self.store.get_job(job.id).segments_done, 2)

    def test_persistence_across_reopen(self) -> None:
        job = self.store.create_job("/in.mp4", "/out.mp4")
        self.store.init_segment_plan(job.id, 2)
        self.store.record_segment(job.id, 0, "/seg0.ts", 0, 10)
        self.store.close()
        # a brand new process opens the same db file
        cfg = Settings(work_root=self.work)
        store2 = JobStore(cfg.db_path)
        reopened = store2.get_job(job.id)
        self.assertEqual(reopened.state, JobState.QUEUED)
        self.assertEqual(store2.get_segments(job.id)[0]["path"], "/seg0.ts")
        store2.close()


if __name__ == "__main__":
    unittest.main()
