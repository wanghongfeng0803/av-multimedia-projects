"""规划器纯逻辑测试：不依赖 ffmpeg，直接验证 GOP 对齐算法。"""

from __future__ import annotations

import pytest

from hls_packager.planner import (
    ALIGN_AFTER,
    ALIGN_BEFORE,
    ALIGN_NEAREST,
    plan_segments,
)


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


# ---------------------------------------------------------------- 对齐模式
def test_align_before_never_exceeds_target_duration():
    # 关键帧每 2s，目标 5s：理想切点 5，before 只能取 4（不超过 5）
    keyframes = tuple(float(t) for t in range(0, 13, 2))
    segments = plan_segments(12.0, 5.0, keyframes, align_mode=ALIGN_BEFORE)
    starts = [s.source_start for s in segments]
    assert starts[0] == 0.0
    assert all(st != 6.0 for st in starts)
    # 非末片时长都不应超过目标值
    for seg in segments[:-1]:
        assert seg.duration <= 5.0 + 1e-6
    # 末片自然收尾，覆盖到源末尾
    assert segments[-1].source_start + segments[-1].duration == pytest.approx(
        12.0, abs=1e-6
    )


def test_align_after_never_shorter_than_target():
    # 理想切点 5，after 取 6（第一个不小于 5 的关键帧）
    keyframes = tuple(float(t) for t in range(0, 13, 2))
    segments = plan_segments(12.0, 5.0, keyframes, align_mode=ALIGN_AFTER)
    assert segments[1].source_start == 6.0
    for seg in segments[:-1]:
        assert seg.duration >= 5.0 - 1e-6


def test_align_before_falls_forward_when_no_earlier_keyframe():
    # 首个 GOP 长 10s > 目标 4s：before 无前侧候选，应退取之后的 10s
    keyframes = (0.0, 10.0, 20.0)
    segments = plan_segments(25.0, 4.0, keyframes, align_mode=ALIGN_BEFORE)
    assert [s.source_start for s in segments] == [0.0, 10.0, 20.0]


def test_align_modes_coincide_when_ideal_hits_keyframe():
    # 理想切点恰好落在关键帧上：三种模式结果一致
    keyframes = tuple(float(t) for t in range(0, 13, 2))
    results = {
        mode: [s.source_start for s in plan_segments(
            12.0, 4.0, keyframes, align_mode=mode
        )]
        for mode in (ALIGN_NEAREST, ALIGN_BEFORE, ALIGN_AFTER)
    }
    assert results[ALIGN_BEFORE] == results[ALIGN_NEAREST]
    assert results[ALIGN_AFTER] == results[ALIGN_NEAREST]


def test_nearest_remains_default_mode():
    keyframes = tuple(float(t) for t in range(0, 11, 2))
    default_segments = plan_segments(10.0, 5.0, keyframes)
    explicit_segments = plan_segments(
        10.0, 5.0, keyframes, align_mode=ALIGN_NEAREST
    )
    assert [s.filename for s in default_segments] == [
        s.filename for s in explicit_segments
    ]
    assert default_segments[1].source_start == 4.0


def test_invalid_align_mode_raises():
    with pytest.raises(ValueError):
        plan_segments(10.0, 2.0, (0.0, 2.0), align_mode="strict")


def test_align_mode_ignored_for_audio_only():
    # 无关键帧时三种模式走同一条等长规划路径，结果一致
    for mode in (ALIGN_NEAREST, ALIGN_BEFORE, ALIGN_AFTER):
        segments = plan_segments(10.0, 3.0, (), align_mode=mode)
        assert [s.source_start for s in segments] == [0.0, 3.0, 6.0]


# ------------------------------------------------------------- 命名模板接入
def test_custom_name_template_renders_filenames():
    segments = plan_segments(
        10.0, 2.0, (),
        sequence_start=100,
        name_template="{name}-{sequence:05d}.{ext}",
        extension="m4s",
    )
    assert segments[0].filename == "seg-00100.m4s"
    assert segments[1].filename == "seg-00101.m4s"


def test_index_template_numbers_from_zero_regardless_of_sequence():
    segments = plan_segments(
        10.0, 2.0, (),
        sequence_start=100,
        name_template="{name}_{index:03d}.{ext}",
    )
    assert segments[0].filename == "seg_000.ts"
    assert segments[1].filename == "seg_001.ts"
    # 绝对序号仍按 MEDIA-SEQUENCE 递增，不受文件名影响
    assert [s.sequence_number for s in segments[:2]] == [100, 101]


def test_extension_follows_container():
    segments = plan_segments(10.0, 2.0, (), extension="mkv")
    assert all(s.filename.endswith(".mkv") for s in segments)


def test_legacy_default_template_still_produces_ts_names():
    segments = plan_segments(10.0, 2.0, (), sequence_start=7,
                             filename_prefix="chunk")
    assert segments[0].filename == "chunk_0007.ts"

