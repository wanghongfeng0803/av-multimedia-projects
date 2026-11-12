"""Tests for source analysis against real generated VFR media."""
from __future__ import annotations

import unittest

from tests.helpers import ffmpeg_available, make_vfr_sample, make_workdir
from vfr_transcoder.media import analyze_source


@unittest.skipUnless(ffmpeg_available(), "ffmpeg/ffprobe not installed")
class SourceAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = make_workdir()
        # 25 fps nominal, symmetric 30/50 ms VFR jitter, B-frames, 1/1000 tb
        self.src = make_vfr_sample(self.work / "src.mkv", duration=8.0,
                                   fps=25.0, marker_interval=4.0)

    def test_detects_vfr_bframes_timebase_and_fps(self) -> None:
        info = analyze_source(self.src)
        self.assertTrue(info.is_vfr, "source must be detected as VFR")
        self.assertTrue(info.has_b_frames, "B-frame reorder must be detected")
        self.assertEqual(info.video_time_base, "1/1000")
        self.assertEqual(info.dominant_fps, 25.0)
        self.assertGreater(info.duration, 7.0)
        # non-zero origin from the generator's start offset
        self.assertGreater(info.start_origin, 0.0)

    def test_30fps_source_detected(self) -> None:
        src30 = make_vfr_sample(self.work / "src30.mkv", duration=6.0, fps=30.0)
        info = analyze_source(src30)
        self.assertEqual(info.dominant_fps, 30.0)
        self.assertTrue(info.is_vfr)


if __name__ == "__main__":
    unittest.main()
