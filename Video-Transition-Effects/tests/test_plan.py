"""plan.py 纯逻辑测试：时长恒等式、帧映射、权重锚点、参数校验。"""

from __future__ import annotations

import pytest

from video_transition.errors import PlanError
from video_transition.plan import (
    KIND_HEAD,
    KIND_OVERLAP,
    KIND_TAIL,
    build_plan,
    fade_weight_b,
    frame_jobs,
)
from video_transition.probe import ClipMeta


def _meta(name: str, *, frames: int, fps=(25, 1), size=(320, 240), audio=True) -> ClipMeta:
    num, den = fps
    return ClipMeta(
        path=name, width=size[0], height=size[1],
        fps_num=num, fps_den=den, frame_count=frames,
        duration=frames * den / num, has_audio=audio,
    )


class TestBuildPlan:
    def test_duration_identity(self):
        # Na + Nb - No：成片比拼接短恰好一个重叠区
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        assert plan.output_frames == 50 + 40 - 10
        assert plan.output_duration == pytest.approx(80 / 25)

    def test_three_segments_partition_without_gap_or_overlap(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        head, overlap, tail = plan.segments
        # 首尾相接、无交叠
        assert head.out_end == overlap.out_start
        assert overlap.out_end == tail.out_start
        assert tail.out_end == plan.output_frames
        # 各段长度
        assert (head.length, overlap.length, tail.length) == (40, 10, 30)

    def test_no_frame_outside_overlap_is_lost(self):
        # head 覆盖 A 的全部非重叠帧，tail 覆盖 B 的全部非重叠帧
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        a_used = set()
        b_used = set()
        for job in frame_jobs(plan):
            if job.index_a >= 0:
                a_used.add(job.index_a)
            if job.index_b >= 0:
                b_used.add(job.index_b)
        # A 的每帧、B 的每帧都恰好被使用一次
        assert a_used == set(range(50))
        assert b_used == set(range(40))

    def test_overlap_seconds_conversion(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        # 0.4s @ 25fps = 10 帧
        plan = build_plan(a, b, overlap_seconds=0.4)
        assert plan.overlap_frames == 10

    def test_overlap_seconds_ntsc_rounding(self):
        # 30000/1001：0.5s 应换算为 round(14.985) = 15 帧
        a = _meta("a", frames=60, fps=(30000, 1001))
        b = _meta("b", frames=60, fps=(30000, 1001))
        plan = build_plan(a, b, overlap_seconds=0.5)
        assert plan.overlap_frames == 15

    def test_mutually_exclusive_args(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=5, overlap_seconds=0.2)

    def test_missing_overlap_arg(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(a, b)

    def test_overlap_must_be_shorter_than_both_clips(self):
        a, b = _meta("a", frames=10), _meta("b", frames=20)
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=10)
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=15)

    def test_non_positive_overlap_rejected(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=0)
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_seconds=0)

    def test_fps_mismatch_rejected(self):
        a = _meta("a", frames=50, fps=(25, 1))
        b = _meta("b", frames=40, fps=(30, 1))
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=5)

    def test_rational_fps_equality(self):
        # 30000/1001 与 30/1 不应被当作相等
        a = _meta("a", frames=50, fps=(30000, 1001))
        b = _meta("b", frames=40, fps=(30, 1))
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=5)

    def test_size_mismatch_rejected(self):
        a = _meta("a", frames=50, size=(320, 240))
        b = _meta("b", frames=40, size=(640, 480))
        with pytest.raises(PlanError):
            build_plan(a, b, overlap_frames=5)


class TestFrameJobs:
    def test_job_count_and_monotonic_output_index(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        jobs = frame_jobs(plan)
        assert [j.out_index for j in jobs] == list(range(plan.output_frames))

    def test_head_jobs_are_pure_a(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        head_jobs = [j for j in frame_jobs(plan) if j.kind == KIND_HEAD]
        assert len(head_jobs) == 40
        assert all(j.index_b == -1 and j.weight_b_num == 0 for j in head_jobs)

    def test_tail_jobs_are_pure_b(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        tail_jobs = [j for j in frame_jobs(plan) if j.kind == KIND_TAIL]
        assert len(tail_jobs) == 30
        # tail 首帧对应 B 的第 10 帧（No 之后一帧不丢）
        assert tail_jobs[0].index_b == 10
        assert all(j.index_a == -1 and j.weight_b_num == j.weight_b_den for j in tail_jobs)

    def test_overlap_jobs_pair_correct_source_frames(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        ov_jobs = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        assert [j.index_a for j in ov_jobs] == list(range(40, 50))
        assert [j.index_b for j in ov_jobs] == list(range(0, 10))
        assert all(j.is_overlap for j in ov_jobs)

    def test_boundary_anchors_are_exact_zero_and_one(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        ov_jobs = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        # 首帧 B 权重 0（纯 A），末帧 B 权重 1（纯 B）
        assert (ov_jobs[0].weight_b_num, ov_jobs[0].weight_b_den) == (0, 9)
        assert (ov_jobs[-1].weight_b_num, ov_jobs[-1].weight_b_den) == (9, 9)

    def test_single_frame_overlap_uses_half_weight(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=1)
        ov_jobs = [j for j in frame_jobs(plan) if j.kind == KIND_OVERLAP]
        assert len(ov_jobs) == 1
        assert (ov_jobs[0].weight_b_num, ov_jobs[0].weight_b_den) == (1, 2)


class TestFadeWeight:
    def test_linear_endpoints(self):
        assert fade_weight_b(0, 5) == 0.0
        assert fade_weight_b(4, 5) == 1.0
        assert fade_weight_b(2, 5) == 0.5

    def test_single_frame_is_half(self):
        assert fade_weight_b(0, 1) == 0.5

    def test_out_of_range_rejected(self):
        with pytest.raises(PlanError):
            fade_weight_b(5, 5)
        with pytest.raises(PlanError):
            fade_weight_b(0, 0)

    def test_weight_b_at_helper(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        assert plan.weight_b_at(0) == 0.0          # head
        assert plan.weight_b_at(plan.output_frames - 1) == 1.0  # tail
        assert plan.weight_b_at(40) == 0.0        # overlap 首帧
        assert plan.weight_b_at(49) == 1.0        # overlap 末帧
