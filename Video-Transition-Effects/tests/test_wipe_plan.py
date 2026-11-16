"""擦除转场规划测试：帧恒等式、掩码时间线、软边钳制告警与参数白名单。

淡化/滑动旧路径的全部断言仍在 test_plan.py / test_slide_plan.py；
本文件只覆盖 wipe 增量，且重点核对「擦除掩码与转场时长共用同一条
时间线」这条承诺。
"""

from __future__ import annotations

import warnings

import pytest

from video_transition.easing import (
    EASING_EASE_IN_OUT,
    EASING_EASE_IN,
    EASING_LINEAR,
)
from video_transition.errors import PlanError
from video_transition.plan import (
    KIND_HEAD,
    KIND_OVERLAP,
    KIND_TAIL,
    TRANSITION_FADE,
    TRANSITION_SLIDE,
    TRANSITION_WIPE,
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
from video_transition.wipe import WipeSoftWidthWarning, wipe_edge


def _meta(name: str, *, frames: int, fps=(25, 1), size=(320, 240)) -> ClipMeta:
    num, den = fps
    return ClipMeta(
        path=name, width=size[0], height=size[1],
        fps_num=num, fps_den=den, frame_count=frames,
        duration=frames * den / num, has_audio=True,
    )


class TestWipePlan:
    def test_wipe_keeps_frame_identity(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_WIPE, direction=DIRECTION_RIGHT,
            soft_width=8,
        )
        assert plan.is_wipe
        assert plan.output_frames == 50 + 40 - 10
        jobs = frame_jobs(plan)
        assert len(jobs) == plan.output_frames
        assert plan.soft_width == 8

    def test_wipe_is_registered_transition(self):
        from video_transition.plan import TRANSITIONS

        assert TRANSITION_WIPE in TRANSITIONS

    @pytest.mark.parametrize("direction", [
        DIRECTION_RIGHT, DIRECTION_LEFT, DIRECTION_DOWN, DIRECTION_UP,
    ])
    def test_edge_anchors_off_canvas_pure_a_pure_b(self, direction):
        size = (320, 240)
        axis = size[0] if direction in (DIRECTION_RIGHT, DIRECTION_LEFT) else size[1]
        soft = 12
        a, b = _meta("a", frames=50, size=size), _meta("b", frames=40, size=size)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_WIPE, direction=direction, soft_width=soft,
        )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        assert ov[0].edge == -soft          # 首帧渐变带全在画布外 -> 纯 A
        assert ov[-1].edge == axis          # 末帧渐变带移出画布 -> 纯 B
        assert all(-soft <= j.edge <= axis for j in ov)
        assert [j.edge for j in ov] == sorted(j.edge for j in ov)

    def test_edge_uses_same_timeline_as_eased_progress(self):
        # job.edge 必须等于 wipe_edge(k, ...) 的独立求值，即掩码边界
        # 与转场时长（帧号 k）走同一条进度曲线
        a, b = _meta("a", frames=60, size=(100, 80)), _meta("b", frames=60, size=(100, 80))
        plan = build_plan(
            a, b, overlap_frames=21,
            transition=TRANSITION_WIPE, direction=DIRECTION_DOWN,
            easing=EASING_EASE_IN, soft_width=7,
        )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        for k, job in enumerate(ov):
            assert job.edge == wipe_edge(k, 21, plan.axis_length, 7, EASING_EASE_IN)
            assert job.soft_width == 7

    def test_non_overlap_jobs_carry_no_mask(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_WIPE, soft_width=4,
        )
        for job in frame_jobs(plan):
            if job.kind != KIND_OVERLAP:
                assert job.edge == 0 and job.soft_width == 0
                assert job.shift == 0

    def test_fade_and_slide_plans_ignore_soft_width(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        fade = build_plan(a, b, overlap_frames=10, soft_width=99)
        slide = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_SLIDE, soft_width=99,
        )
        assert fade.soft_width == 0
        assert slide.soft_width == 0
        assert all(j.soft_width == 0 and j.edge == 0 for j in frame_jobs(fade))
        assert all(j.soft_width == 0 for j in frame_jobs(slide))

    def test_slide_jobs_unaffected_by_wipe_fields(self):
        # 旧滑动模式的位移字段逐帧不变
        a, b = _meta("a", frames=50, size=(100, 50)), _meta("b", frames=40, size=(100, 50))
        plan = build_plan(
            a, b, overlap_frames=5,
            transition=TRANSITION_SLIDE, direction=DIRECTION_RIGHT,
        )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        assert [j.shift for j in ov] == [0, 25, 50, 75, 100]


class TestSoftWidthClampAtPlan:
    def test_horizontal_overflow_clamped_to_width_with_warning(self):
        a, b = _meta("a", frames=50, size=(320, 240)), _meta("b", frames=40, size=(320, 240))
        with pytest.warns(WipeSoftWidthWarning) as record:
            plan = build_plan(
                a, b, overlap_frames=10,
                transition=TRANSITION_WIPE, direction=DIRECTION_LEFT,
                soft_width=1000,
            )
        assert plan.soft_width == 320  # 水平轴长 = 宽
        msg = str(record[0].message)
        assert "1000" in msg and "320" in msg

    def test_vertical_overflow_clamped_to_height(self):
        a, b = _meta("a", frames=50, size=(320, 240)), _meta("b", frames=40, size=(320, 240))
        with pytest.warns(WipeSoftWidthWarning):
            plan = build_plan(
                a, b, overlap_frames=10,
                transition=TRANSITION_WIPE, direction=DIRECTION_UP,
                soft_width=500,
            )
        assert plan.soft_width == 240  # 垂直轴长 = 高

    def test_equal_to_axis_is_valid_without_warning(self):
        a, b = _meta("a", frames=50, size=(320, 240)), _meta("b", frames=40, size=(320, 240))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = build_plan(
                a, b, overlap_frames=10,
                transition=TRANSITION_WIPE, soft_width=320,
            )
        assert plan.soft_width == 320

    def test_negative_soft_width_rejected(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(
                a, b, overlap_frames=10,
                transition=TRANSITION_WIPE, soft_width=-1,
            )

    def test_non_int_soft_width_rejected(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        for bad in (1.5, True, "8"):
            with pytest.raises(PlanError):
                build_plan(
                    a, b, overlap_frames=10,
                    transition=TRANSITION_WIPE, soft_width=bad,
                )

    def test_clamped_plan_jobs_stay_in_extended_travel(self):
        a, b = _meta("a", frames=50, size=(40, 30)), _meta("b", frames=40, size=(40, 30))
        with pytest.warns(WipeSoftWidthWarning):
            plan = build_plan(
                a, b, overlap_frames=12,
                transition=TRANSITION_WIPE, direction=DIRECTION_RIGHT,
                soft_width=999,
            )
        ov = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        soft = plan.soft_width
        assert all(-soft <= j.edge <= plan.width for j in ov)
        assert ov[0].edge == -soft and ov[-1].edge == plan.width


class TestWipeGain:
    def test_wipe_gain_shares_easing_curve_with_slide(self):
        # 擦除与滑动的音频增益是同一条曲线（同时间线承诺）
        for k in range(5):
            assert (
                transition_gain_fraction(k, 5, TRANSITION_WIPE, EASING_EASE_IN)
                == transition_gain_fraction(k, 5, TRANSITION_SLIDE, EASING_EASE_IN)
            )

    def test_wipe_gain_anchors_zero_and_one(self):
        assert transition_gain_fraction(0, 10, TRANSITION_WIPE, EASING_LINEAR) == (0, 9)
        assert transition_gain_fraction(9, 10, TRANSITION_WIPE, EASING_LINEAR) == (9, 9)

    def test_wipe_single_frame_overlap_half_gain(self):
        assert transition_gain_fraction(0, 1, TRANSITION_WIPE) == (1, 2)


class TestWipePlanSegments:
    def test_three_segments_unchanged_and_job_kinds(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(
            a, b, overlap_frames=10,
            transition=TRANSITION_WIPE, soft_width=4,
        )
        kinds = [j.kind for j in frame_jobs(plan)]
        assert kinds.count(KIND_HEAD) == 40
        assert kinds.count(KIND_OVERLAP) == 10
        assert kinds.count(KIND_TAIL) == 30
