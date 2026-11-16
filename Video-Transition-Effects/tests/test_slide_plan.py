"""滑动转场规划测试：FrameJob 位移、运动轴选择、参数白名单与帧恒等式。

淡化旧路径的全部断言仍在 test_plan.py；本文件只覆盖 slide 增量。
"""

from __future__ import annotations

import pytest

from video_transition.easing import (
    EASING_EASE_IN,
    EASING_EASE_IN_OUT,
    EASING_EASE_OUT,
    EASING_LINEAR,
)
from video_transition.errors import PlanError
from video_transition.plan import (
    KIND_OVERLAP,
    TRANSITION_FADE,
    TRANSITION_SLIDE,
    build_plan,
    frame_jobs,
    transition_gain_fraction,
)
from video_transition.probe import ClipMeta
from video_transition.slide import (
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
)


def _meta(name: str, *, frames: int, fps=(25, 1), size=(320, 240)) -> ClipMeta:
    num, den = fps
    return ClipMeta(
        path=name, width=size[0], height=size[1],
        fps_num=num, fps_den=den, frame_count=frames,
        duration=frames * den / num, has_audio=True,
    )


class TestSlidePlan:
    def test_slide_keeps_frame_identity(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_SLIDE, direction=DIRECTION_RIGHT,
        )
        assert plan.is_slide
        assert plan.output_frames == 50 + 40 - 10
        jobs = frame_jobs(plan)
        assert len(jobs) == plan.output_frames

    def test_shift_anchors_zero_and_full_width_horizontal(self):
        a, b = _meta("a", frames=50, size=(320, 240)), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_SLIDE, direction=DIRECTION_LEFT,
        )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        assert ov[0].shift == 0
        assert ov[-1].shift == 320  # 水平轴长 = 宽
        assert all(0 <= j.shift <= 320 for j in ov)

    def test_shift_anchors_zero_and_full_height_vertical(self):
        a, b = _meta("a", frames=50, size=(320, 240)), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_SLIDE, direction=DIRECTION_UP,
        )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        assert ov[0].shift == 0
        assert ov[-1].shift == 240  # 垂直轴长 = 高

    def test_linear_shift_matches_linear_progress(self):
        a, b = _meta("a", frames=50, size=(100, 50)), _meta(
            "b", frames=40, size=(100, 50)
        )
        plan = build_plan(
            a, b, overlap_frames=5,
            transition=TRANSITION_SLIDE, direction=DIRECTION_RIGHT,
            easing=EASING_LINEAR,
        )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        # round(k*100/4)：0,25,50,75,100
        assert [j.shift for j in ov] == [0, 25, 50, 75, 100]

    def test_eased_shift_monotonic_for_all_curves(self):
        a, b = (
            _meta("a", frames=60, size=(100, 80)),
            _meta("b", frames=60, size=(100, 80)),
        )
        for easing in (EASING_LINEAR, EASING_EASE_IN,
                       EASING_EASE_OUT, EASING_EASE_IN_OUT):
            plan = build_plan(
                a, b, overlap_frames=21,
                transition=TRANSITION_SLIDE, direction=DIRECTION_DOWN,
                easing=easing,
            )
            shifts = [j.shift for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
            assert shifts == sorted(shifts), easing
            assert shifts[0] == 0 and shifts[-1] == 80

    def test_non_slide_jobs_have_zero_shift(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_SLIDE, direction=DIRECTION_RIGHT,
        )
        for job in frame_jobs(plan):
            if job.kind != KIND_OVERLAP:
                assert job.shift == 0

    def test_fade_plan_has_zero_shift_everywhere(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        assert not plan.is_slide
        assert all(j.shift == 0 for j in frame_jobs(plan))

    def test_invalid_transition_rejected(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=5, transition="dissolve")

    def test_invalid_direction_rejected(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(
                a, b, overlap_frames=5,
                transition=TRANSITION_SLIDE, direction="north",
            )

    def test_invalid_easing_rejected(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(
                a, b, overlap_frames=5,
                transition=TRANSITION_SLIDE, easing="bounce",
            )


class TestTransitionGain:
    def test_fade_gain_unchanged_linear_fraction(self):
        assert transition_gain_fraction(0, 10, TRANSITION_FADE) == (0, 9)
        assert transition_gain_fraction(9, 10, TRANSITION_FADE) == (9, 9)
        assert transition_gain_fraction(0, 1, TRANSITION_FADE) == (1, 2)

    def test_slide_gain_uses_easing_curve(self):
        num, den = transition_gain_fraction(
            2, 5, TRANSITION_SLIDE, EASING_EASE_IN
        )
        assert num == 4 and den == 16

    def test_slide_gain_anchors(self):
        for k, expected in ((0, 0), (9, 1)):
            num, den = transition_gain_fraction(
                k, 10, TRANSITION_SLIDE, EASING_EASE_IN_OUT
            )
            assert num / den == expected
