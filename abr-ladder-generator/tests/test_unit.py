"""Unit tests for scheduling, keyframe planning and report judgements."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr_ladder.config import (DEFAULT_RUNGS, bpp, fit_dimensions,
                               load_ladder)
from abr_ladder.keyframes import SceneCut, build_keyframe_plan
from abr_ladder.pipeline import allocate_encode_slots
from abr_ladder.report import visible_judgement


class TestFitDimensions(unittest.TestCase):
    def test_16x9_fits_exactly(self):
        self.assertEqual(fit_dimensions(16 / 9, 854, 480), (854, 480))
        self.assertEqual(fit_dimensions(16 / 9, 640, 360), (640, 360))
        self.assertEqual(fit_dimensions(16 / 9, 1920, 1080), (1920, 1080))

    def test_4x3_is_pillarboxed(self):
        w, h = fit_dimensions(4 / 3, 1920, 1080)
        self.assertEqual(h, 1080)
        self.assertEqual(w % 2, 0)
        self.assertLess(w, 1920)

    def test_ultrawide_is_letterboxed(self):
        w, h = fit_dimensions(2.39, 1920, 1080)
        self.assertEqual(w, 1920)
        self.assertEqual(h % 2, 0)
        self.assertLess(h, 1080)

    def test_bad_aspect_falls_back(self):
        self.assertEqual(fit_dimensions(float("nan"), 640, 360), (640, 360))


class TestKeyframePlan(unittest.TestCase):
    def test_grid_only(self):
        plan = build_keyframe_plan([], 30.0, 600, gop_seconds=2.0)
        self.assertEqual(plan.grid_step_frames, 60)
        self.assertEqual(plan.frame_indices[:3], [0, 60, 120])
        self.assertLessEqual(plan.max_gap_frames, 60)

    def test_scene_cut_becomes_keyframe(self):
        plan = build_keyframe_plan(
            [SceneCut(time=5.0, score=0.8)], 30.0, 600, gop_seconds=2.0
        )
        self.assertIn(150, plan.frame_indices)
        self.assertEqual(len(plan.scene_cuts), 1)

    def test_cut_on_grid_is_deduplicated(self):
        plan = build_keyframe_plan(
            [SceneCut(time=8.0, score=0.8)], 30.0, 600, gop_seconds=2.0
        )
        self.assertNotIn(240, [c.frame_index for c in plan.scene_cuts])
        self.assertEqual(plan.frame_indices,
                         sorted(set(plan.frame_indices)))

    def test_rapid_cuts_are_thinned(self):
        cuts = [SceneCut(time=50.2, score=0.8),
                SceneCut(time=50.3, score=0.9),
                SceneCut(time=55.0, score=0.7)]
        plan = build_keyframe_plan(cuts, 30.0, 3600, gop_seconds=2.0)
        self.assertEqual(len(plan.scene_cuts), 2)
        self.assertIn(1506, plan.frame_indices)
        self.assertIn(1650, plan.frame_indices)

    def test_expr_uses_frame_numbers(self):
        plan = build_keyframe_plan(
            [SceneCut(time=5.0, score=0.8)], 30.0, 600, gop_seconds=2.0
        )
        expr = plan.expr()
        self.assertTrue(expr.startswith("expr:"))
        self.assertIn("eq(n,150)", expr)
        self.assertIn("not(mod(n,60))", expr)


class TestSlotAllocation(unittest.TestCase):
    def test_eight_threads(self):
        self.assertEqual(allocate_encode_slots(DEFAULT_RUNGS, 8), [4, 2, 1, 1])

    def test_four_threads_min_one_each(self):
        self.assertEqual(allocate_encode_slots(DEFAULT_RUNGS, 4), [1, 1, 1, 1])

    def test_six_threads(self):
        alloc = allocate_encode_slots(DEFAULT_RUNGS, 6)
        self.assertEqual(sum(alloc), 6)
        self.assertTrue(all(a >= 1 for a in alloc))
        self.assertGreaterEqual(alloc[0], alloc[1])

    def test_twelve_threads(self):
        alloc = allocate_encode_slots(DEFAULT_RUNGS, 12)
        self.assertEqual(sum(alloc), 12)
        self.assertGreaterEqual(alloc[0], alloc[1])


class TestBpp(unittest.TestCase):
    def test_bpp_typical_ladder(self):
        value = bpp(5000, 1920, 1080, 30)
        self.assertAlmostEqual(value, 0.0804, places=3)
        self.assertAlmostEqual(bpp(2500, 1280, 720, 30), 0.0903, places=3)


class TestLoadLadder(unittest.TestCase):
    def test_default_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ladder.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"rungs": [
                    {"name": "hi", "width": 1280, "height": 720, "kbps": 2000},
                    {"name": "lo", "width": 640, "height": 360, "kbps": 500},
                ]}, f)
            rungs = load_ladder(path)
            self.assertEqual([r.name for r in rungs], ["hi", "lo"])
            self.assertEqual(rungs[0].nominal_kbps, 2000)

    def test_flat_list_and_nominal_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ladder.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"width": 640, "height": 360,
                            "nominal_kbps": 700}], f)
            rungs = load_ladder(path)
            self.assertEqual(rungs[0].name, "360p")
            self.assertEqual(rungs[0].nominal_kbps, 700)

    def test_odd_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ladder.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"rungs": [{"width": 641, "height": 361,
                                      "kbps": 500}]}, f)
            with self.assertRaises(ValueError):
                load_ladder(path)


class TestReportJudgement(unittest.TestCase):
    def test_vmaf_bands(self):
        self.assertEqual(visible_judgement(2.0, None)[0], "几乎无法分辨")
        self.assertEqual(visible_judgement(4.0, None)[0], "仔细对比可分辨")
        self.assertEqual(visible_judgement(9.0, None)[0], "可察觉")
        self.assertEqual(visible_judgement(15.0, None)[0], "明显可察觉")

    def test_ssim_fallback(self):
        self.assertEqual(visible_judgement(None, 0.003)[0], "几乎无法分辨")
        self.assertEqual(visible_judgement(None, 0.03)[0], "明显可察觉")

    def test_unknown(self):
        self.assertEqual(visible_judgement(None, None)[0], "无法判定")

    def test_bpp_typical_ladder(self):
        value = bpp(5000, 1920, 1080, 30)
        self.assertAlmostEqual(value, 0.0804, places=3)
        self.assertAlmostEqual(bpp(2500, 1280, 720, 30), 0.0903, places=3)


if __name__ == "__main__":
    unittest.main()
