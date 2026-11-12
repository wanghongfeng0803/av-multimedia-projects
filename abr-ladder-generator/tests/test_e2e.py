"""End-to-end test: build a short ladder and verify the deliverables."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr_ladder import ff
from abr_ladder.config import DEFAULT_RUNGS, Settings
from abr_ladder.pipeline import run_pipeline


def _make_source(path: str) -> None:
    """8-second 720p source with two scenes and non-AAC audio.

    Two different lavfi patterns give a real scene cut at 4 s; PCM audio
    exercises the AAC transcode path.
    """
    cmd = [
        ff.ffmpeg_bin(), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "smptebars=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-filter_complex",
        ("[0:v]trim=start=0:duration=4,setpts=PTS-STARTPTS[v0];"
         "[1:v]trim=start=0:duration=4,setpts=PTS-STARTPTS[v1];"
         "[v0][v1]concat=n=2:v=1:a=0[v];"
         "[2:a]atrim=0:8,asetpts=PTS-STARTPTS[a]"),
        "-map", "[v]", "-map", "[a]",
        "-t", "8",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le",
        path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg"))
                or os.environ.get("FFMPEG_BIN")):
            raise unittest.SkipTest("ffmpeg not found")
        if not ff.has_encoder("libx264"):
            raise unittest.SkipTest("libx264 not available")

    def test_ladder_and_verification(self):
        with tempfile.TemporaryDirectory(prefix="abr_e2e_") as tmp:
            source = os.path.join(tmp, "source.mkv")
            _make_source(source)
            out_dir = os.path.join(tmp, "out")
            settings = Settings(
                preset="veryfast",
                threads=4,
                gop_seconds=2.0,
                metric="vmaf" if ff.has_filter("libvmaf") else "ssim",
                overwrite=True,
                time_budget_seconds=600,
                rungs=list(DEFAULT_RUNGS),
            )
            report = run_pipeline(source, out_dir, settings,
                                  log=lambda _m: None, verbose=False)

            with open(os.path.join(out_dir, "report.json"),
                      "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["schema_version"], "1.0")
            self.assertEqual(len(saved["rungs"]), 4)
            self.assertTrue(saved["verification"]["passed"])

            for rung, spec in zip(saved["rungs"], DEFAULT_RUNGS):
                self.assertEqual(rung["name"], spec.name)
                self.assertEqual((rung["width"], rung["height"]),
                                 (spec.width, spec.height))
                self.assertLessEqual(abs(rung["deviation_pct"]), 3.0)
                self.assertIsNotNone(rung["quality"])
                self.assertGreater(rung["quality"]["sample_count"], 0)
                self.assertTrue(os.path.exists(
                    os.path.join(out_dir, rung["output"])))

            kf = saved["keyframes"]["frame_indices"]
            self.assertGreater(len(kf), 2)
            self.assertEqual(saved["keyframes"]["max_gap_frames"],
                             max(b - a for a, b in zip(kf, kf[1:])))
            self.assertTrue(saved["verification"]["keyframe_alignment"]["aligned"])

            audio = saved["verification"]["audio"]
            self.assertTrue(audio["present"])
            self.assertTrue(audio["identical"])

            self.assertEqual(len(saved["adjacent_gaps"]), 3)
            for gap in saved["adjacent_gaps"]:
                self.assertIn("visible_judgement", gap)
                self.assertIn("bandwidth_saving_pct", gap)

            md_path = os.path.join(out_dir, "report.md")
            self.assertTrue(os.path.exists(md_path))
            with open(md_path, "r", encoding="utf-8") as f:
                md = f.read()
            self.assertIn("ABR 阶梯质量报告", md)
            self.assertIn("相邻档质量差", md)
            self.assertIn("关键帧对齐与切档验证", md)


if __name__ == "__main__":
    unittest.main()
