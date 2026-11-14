"""规划器纯逻辑测试：不依赖 ffmpeg，直接验证 GOP 对齐算法。"""

from __future__ import annotations

import pytest

from hls_packager.planner import plan_segments


def test_boundaries_only_on_keyframes():
    # 关键帧每 2s，目标 3s：所有切点必须是关键帧时刻
    keyframes = tuple(float(t) for t in range(0, 13, 2))  # 0,2,...,12
    segments = plan_segments(12.0, 3.0, keyframes)

    source_starts = [s.source_start for s in segments]
    for start in source_starts:
        assert start in keyframes, f"切点 {start} 不在关键帧上"
    assert source_starts[0] == 0.0
    # 末片结束时间应覆盖整个源
    assert segments[-1].source_start + segments[-1].duration == pytest.approx(
        12.0, abs=1e-6
    )


def test_nearest_keyframe_to_ideal_cut():
    # 目标 5s，关键帧每 2s：理想切点 5，最近关键帧为 4（|4-5|=1）
    keyframes = tuple(float(t) for t in range(0, 11, 2))  # 0..10
    segments = plan_segments(10.0, 5.0, keyframes)
    assert segments[0].source_start == 0.0
    assert segments[1].source_start == 4.0


def test_tie_prefers_earlier_keyframe():
    # 理想切点 5，关键帧 4 与 6 等距：应取靠前的 4
    keyframes = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    segments = plan_segments(10.0, 5.0, keyframes)
    assert segments[1].source_start == 4.0


def test_large_gop_segments_are_longer_but_aligned():
    # 关键帧间隔 10s > 目标 4s：分片只能 ~10s，切点仍是关键帧
    keyframes = (0.0, 10.0, 20.0)
    segments = plan_segments(25.0, 4.0, keyframes)
    starts = [s.source_start for s in segments]
    assert starts == [0.0, 10.0, 20.0]
    assert segments[0].duration == pytest.approx(10.0, abs=1e-6)
    assert segments[-1].duration == pytest.approx(5.0, abs=1e-6)


def test_single_keyframe_produces_one_segment():
    segments = plan_segments(8.0, 2.0, (0.0,))
    assert len(segments) == 1
    assert segments[0].source_start == 0.0
    assert segments[0].duration == pytest.approx(8.0, abs=1e-6)


def test_keyframes_outside_duration_are_clamped():
    # 含负时间戳与超出 duration 的噪声关键帧，应被规整
    keyframes = (-1.0, 0.0, 3.0, 9.0, 100.0)
    segments = plan_segments(9.0, 3.0, keyframes)
    starts = [s.source_start for s in segments]
    assert starts == [0.0, 3.0]
    assert all(0.0 <= s.source_start <= 9.0 for s in segments)


def test_audio_only_uniform_planning():
    # 无关键帧：等长规划，10s / 目标 3s
    segments = plan_segments(10.0, 3.0, ())
    starts = [s.source_start for s in segments]
    assert starts == [0.0, 3.0, 6.0]
    # 末片 4s（尾部 10-6=4，不触发短尾合并）
    assert segments[-1].duration == pytest.approx(4.0, abs=1e-6)


def test_audio_only_short_tail_merged():
    # 9s / 目标 4s：切点 0,4,8，尾片仅 1s（< 4*0.5=2s），并入前一片
    segments = plan_segments(9.0, 4.0, ())
    starts = [s.source_start for s in segments]
    assert starts == [0.0, 4.0]
    assert segments[-1].duration == pytest.approx(5.0, abs=1e-6)


def test_audio_only_normal_tail_kept():
    # 10s / 目标 4s：尾片 2s 恰好等于阈值（不小于），保留为独立片
    segments = plan_segments(10.0, 4.0, ())
    starts = [s.source_start for s in segments]
    assert starts == [0.0, 4.0, 8.0]
    assert segments[-1].duration == pytest.approx(2.0, abs=1e-6)


def test_sequence_numbers_and_filenames():
    segments = plan_segments(10.0, 2.0, (), sequence_start=7, filename_prefix="chunk")
    assert [s.sequence_number for s in segments] == [7, 8, 9, 10, 11]
    assert segments[0].filename == "chunk_0007.ts"


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        plan_segments(0.0, 2.0, ())
    with pytest.raises(ValueError):
        plan_segments(10.0, 0.0, ())
    with pytest.raises(ValueError):
        plan_segments(10.0, 2.0, (), sequence_start=-1)


def test_segment_durations_sum_to_source_duration():
    keyframes = (0.0, 2.5, 5.0, 7.5)
    segments = plan_segments(9.0, 3.0, keyframes)
    total = sum(s.duration for s in segments)
    assert total == pytest.approx(9.0, abs=1e-6)
    # 时间轴上的 start_time 与 source_start 在规划阶段一致
    for seg in segments:
        assert seg.start_time == seg.source_start
