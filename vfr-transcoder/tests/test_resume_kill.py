"""Kill-and-restart resume tests.

A transcode is launched in a SEPARATE process and hard-killed (TerminateProcess
on Windows, SIGKILL elsewhere) while it is encoding. A fresh process then runs
the same job id and must finish without redoing completed segments.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from tests.helpers import (ROOT, ffmpeg_available, is_process_alive, kill_pid,
                          make_vfr_sample, make_workdir, video_frame_pts)
from vfr_transcoder.config import Settings
from vfr_transcoder.state import JobState, JobStore


@unittest.skipUnless(ffmpeg_available(), "ffmpeg/ffprobe not installed")
class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = make_workdir()
        # long enough that several segments exist when we kill it
        self.src = make_vfr_sample(self.work / "src.mkv", duration=24.0,
                                   marker_interval=4.0)
        self.out = self.work / "out.mp4"
        self.workroot = self.work / "w"
        self.cfg = Settings(work_root=self.workroot, segment_seconds=2.0)

    def _only_job_id(self, store: JobStore) -> str:
        jobs = store.list_jobs()
        self.assertEqual(len(jobs), 1)
        return jobs[0].id

    def test_hard_kill_then_resume_completes_and_keeps_progress(self) -> None:
        store = JobStore(self.cfg.db_path)
        job = store.create_job(str(self.src), str(self.out), 25.0)
        job_id = job.id
        store.close()

        env = dict(os.environ)
        proc = subprocess.Popen(
            [sys.executable, "-m", "vfr_transcoder",
             "--work-root", str(self.workroot),
             "transcode", str(self.src), str(self.out),
             "--segment-seconds", "2", "--job-id", job_id],
            cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # wait until at least one segment file exists, then hard kill
        job_dir = self.workroot / "jobs" / job_id
        deadline = time.time() + 60
        seg_seen = None
        while time.time() < deadline:
            segs = sorted(job_dir.glob("seg_*.ts"))
            if segs:
                seg_seen = segs
                break
            if proc.poll() is not None:
                self.fail(f"transcode exited early: {proc.stderr.read()[-1000:]}")
            time.sleep(0.1)
        self.assertIsNotNone(seg_seen, "no segment produced before kill")
        kept_segments = {p.name for p in job_dir.glob("seg_*.ts")}

        kill_pid(proc.pid)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.fail("killed process did not terminate")
        self.assertFalse(is_process_alive(proc.pid))

        # the lease is still "running" with an expiry in the past after restart
        store = JobStore(self.cfg.db_path)
        killed_job = store.get_job(job_id)
        self.assertIn(killed_job.state, (JobState.RUNNING, JobState.QUEUED))
        done_before = len(store.get_segments(job_id))
        self.assertGreaterEqual(done_before, 1)
        store.close()

        # fresh process resumes the same job; short lease so the stale one is
        # reclaimed immediately
        env["VFR_LEASE_SECONDS"] = "1"
        resume = subprocess.run(
            [sys.executable, "-m", "vfr_transcoder",
             "--work-root", str(self.workroot),
             "transcode", str(self.src), str(self.out),
             "--segment-seconds", "2", "--job-id", job_id],
            cwd=str(ROOT), env=env, capture_output=True, text=True)
        self.assertEqual(resume.returncode, 0, resume.stderr)
        self.assertTrue(self.out.exists())

        store = JobStore(self.cfg.db_path)
        final = store.get_job(job_id)
        self.assertEqual(final.state, JobState.COMPLETED)
        # every segment present before the kill survived and was reused
        for name in kept_segments:
            self.assertTrue((job_dir / name).exists(), f"{name} lost on resume")
        # output is valid CFR
        vpts = video_frame_pts(self.out)
        deltas = {round(vpts[i + 1] - vpts[i], 6) for i in range(len(vpts) - 1)}
        self.assertEqual(deltas, {0.04})
        store.close()

    def test_second_resume_is_cheap_idempotent(self) -> None:
        # running a completed job's pipeline again must not raise and the
        # existing output stays valid (all segments recognized as done)
        store = JobStore(self.cfg.db_path)
        job = store.create_job(str(self.src), str(self.work / "o2.mp4"), 25.0)
        from vfr_transcoder.worker import Worker
        self.assertTrue(Worker(store, self.cfg).run_one(job.id))
        segs_first = len(store.get_segments(job.id))
        # requeue and rerun: segments are skipped, output is rewritten
        store.requeue(job.id)
        self.assertTrue(Worker(store, self.cfg).run_one(job.id))
        self.assertEqual(len(store.get_segments(job.id)), segs_first)
        store.close()


if __name__ == "__main__":
    unittest.main()
