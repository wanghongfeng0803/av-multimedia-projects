"""End-to-end pipeline tests: transcode output properties and sync."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tests.helpers import (ROOT, ffmpeg_available, make_vfr_sample, make_workdir,
                          probe_json, video_frame_pts)
from vfr_transcoder.config import Settings
from vfr_transcoder.media import analyze_source
from vfr_transcoder.state import JobState, JobStore
from vfr_transcoder.worker import Worker


@unittest.skipUnless(ffmpeg_available(), "ffmpeg/ffprobe not installed")
class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = make_workdir()
        self.src = make_vfr_sample(self.work / "src.mkv", duration=10.0,
                                   marker_interval=4.0)
        self.out = self.work / "out.mp4"
        self.cfg = Settings(work_root=self.work / "w", segment_seconds=3.0)

    def _run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vfr_transcoder",
             "--work-root", str(self.cfg.work_root),
             "transcode", str(self.src), str(self.out), *extra],
            cwd=str(ROOT), capture_output=True, text=True)

    def test_cli_produces_cfr_with_grid_accuracy_and_copyts(self) -> None:
        proc = self._run_cli("--segment-seconds", "3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(self.out.exists())

        info = analyze_source(self.src, self.cfg)
        fps = info.dominant_fps
        vpts = video_frame_pts(self.out)
        # CFR: every consecutive delta equals 1/fps
        deltas = {round(vpts[i + 1] - vpts[i], 6) for i in range(len(vpts) - 1)}
        self.assertEqual(deltas, {round(1.0 / fps, 6)})
        # grid error essentially zero, and identical at head and tail
        grid = [vpts[i] - (vpts[0] + i / fps) for i in range(len(vpts))]
        self.assertLess(max(abs(g) for g in grid), 0.002)
        self.assertLess(abs(grid[-1] - grid[0]), 0.002)
        # copyts: output origin equals source origin
        streams = probe_json(self.out, "-show_entries", "stream=start_time")["streams"]
        out_start = float(streams[0]["start_time"])
        self.assertAlmostEqual(out_start, info.start_origin, delta=0.005)

    def test_explicit_target_fps_overrides_dominant(self) -> None:
        out30 = self.work / "out30.mp4"
        proc = subprocess.run(
            [sys.executable, "-m", "vfr_transcoder",
             "--work-root", str(self.cfg.work_root),
             "transcode", str(self.src), str(out30), "--target-fps", "30",
             "--segment-seconds", "3"],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # 30 fps on a 1/1000 time base quantizes 33.333 ms to 33/34 ms, so
        # assert the declared rate and a tight grid error rather than a single
        # exact delta (that +/-1 ms is inside the 40 ms budget by design).
        streams = probe_json(out30, "-show_entries", "stream=r_frame_rate")["streams"]
        self.assertEqual(streams[0]["r_frame_rate"], "30/1")
        vpts = video_frame_pts(out30)
        grid = [vpts[i] - (vpts[0] + i / 30.0) for i in range(len(vpts))]
        self.assertLess(max(abs(g) for g in grid), 0.002)
        self.assertLess(abs(grid[-1] - grid[0]), 0.002)

    def test_dts_strictly_increasing_across_segment_boundaries(self) -> None:
        proc = self._run_cli("--segment-seconds", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = probe_json(self.out, "-select_streams", "v:0",
                          "-show_entries", "packet=pts_time,dts_time")
        dts = [float(p["dts_time"]) for p in data["packets"]
               if p.get("dts_time") not in (None, "N/A")]
        self.assertTrue(all(dts[i] < dts[i + 1] for i in range(len(dts) - 1)))

    def test_completed_job_recorded_in_store(self) -> None:
        store = JobStore(self.cfg.db_path)
        job = store.create_job(str(self.src), str(self.out), 25.0)
        ok = Worker(store, self.cfg).run_one(job.id)
        self.assertTrue(ok)
        final = store.get_job(job.id)
        self.assertEqual(final.state, JobState.COMPLETED)
        self.assertEqual(final.progress, 1.0)
        self.assertEqual(final.segments_done, final.segments_total)
        self.assertTrue(final.audio_done)
        store.close()

    def test_relative_paths_concat_resolution(self) -> None:
        # regression: concat demuxer resolves list entries relative to the
        # list file; relative work roots must not double the path
        import os
        cwd = os.getcwd()
        relroot = self.work / "rel"
        relroot.mkdir()
        try:
            os.chdir(relroot)
            cfg = Settings(work_root="work", segment_seconds=2.0)
            src_rel = Path("src.mkv")
            import shutil
            shutil.copy(self.src, relroot / src_rel)
            store = JobStore(cfg.db_path)
            job = store.create_job(str(src_rel), "out.mp4", 25.0)
            self.assertTrue(Worker(store, cfg).run_one(job.id))
            self.assertTrue((relroot / "out.mp4").exists())
            store.close()
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
