"""时间轴规划测试：帧边界切点、整数毫秒无漂移、短尾片标记。

全部用构造的 SourceInfo 直接驱动，不依赖 ffmpeg，保证时间轴逻辑
可在任意环境离线验证。
"""

import pytest

from vhls.probe import FrameInfo, SourceInfo
from vhls.timeline import (
    MS_PER_SEC,
    TimelineError,
    build_timeline,
    ceil_div,
    format_extinf,
    ms_to_seconds,
)


def make_source(frame_durations_ms, pts_offsets_ms=None, fps=25.0):
    """按给定每帧毫秒时长构造测试用 SourceInfo。

    :param frame_durations_ms: 每帧持续时长（毫秒）。
    :param pts_offsets_ms: 每帧 PTS（毫秒）；None 时按帧时长累积，
        首帧 PTS=0。
    """
    if pts_offsets_ms is None:
        pts_offsets_ms = []
        acc = 0
        for d in frame_durations_ms:
            pts_offsets_ms.append(acc)
            acc += d
    frames = tuple(
        FrameInfo(
            index=i,
            pts_time=pts_offsets_ms[i] / MS_PER_SEC,
            duration_time=frame_durations_ms[i] / MS_PER_SEC,
        )
        for i in range(len(frame_durations_ms))
    )
    return SourceInfo(
        path=Path_stub("in.mp4"), width=64, height=48, fps=fps,
        frames=frames, has_audio=False,
    )


class _PathStub:
    """测试里无需真实文件，路径字段只做承载。"""

    def __init__(self, name):
        self.name = name

    def __fspath__(self):
        return self.name

    def __str__(self):
        return self.name


def Path_stub(name):
    return _PathStub(name)


class TestBuildTimelineCFR:
    def test_exact_division_25fps(self):
        # 25fps、每帧 40ms，目标 1s：每片恰 25 帧，无短尾。
        source = make_source([40] * 150)  # 6.000s
        plan = build_timeline(source, 1.0)
        assert plan.segment_count == 6
        assert [s.frame_count for s in plan.segments] == [25] * 6
        assert all(s.planned_duration_ms == 1000 for s in plan.segments)
        assert not any(s.is_short for s in plan.segments)
        assert plan.source_duration_ms == 6000

    def test_sequence_numbers_contiguous_from_zero(self):
        source = make_source([40] * 100)  # 4s，目标 1s
        plan = build_timeline(source, 1.0)
        assert [s.sequence for s in plan.segments] == list(range(4))
        # 文件名序号与序列号一致且零填充。
        assert plan.segments[0].filename == "seg_00000.ts"
        assert plan.segments[3].filename == "seg_00003.ts"

    def test_frame_ranges_partition_all_frames(self):
        source = make_source([33] * 181)  # 30fps 舍入口径，约 5.973s
        plan = build_timeline(source, 2.0)
        segs = plan.segments
        # 相邻片首尾相接，且覆盖全部帧、无重叠无遗漏。
        for prev, nxt in zip(segs, segs[1:]):
            assert prev.end_frame == nxt.start_frame
        assert segs[0].start_frame == 0
        assert segs[-1].end_frame == source.frame_count

    def test_cut_points_land_on_frame_edges(self):
        # 任何片的起止毫秒都必须精确等于某帧的 PTS（末片终点=源总时长）。
        source = make_source([40] * 137)
        plan = build_timeline(source, 1.0)
        pts_set = {int(f.pts_time * MS_PER_SEC + 0.5) for f in source.frames}
        total = plan.source_duration_ms
        for seg in plan.segments:
            assert seg.start_ms in pts_set
            assert seg.end_ms in pts_set or (
                seg.is_last and seg.end_ms == total
            )
            assert seg.end_ms > seg.start_ms

    def test_short_tail_marked_when_remainder_small(self):
        # 25fps：2s 目标 = 50 帧/片；127 帧 → 50+50+27（尾片 1.08s）。
        source = make_source([40] * 127)
        plan = build_timeline(source, 2.0)
        assert plan.segment_count == 3
        last = plan.segments[-1]
        assert last.is_last and last.is_short
        assert last.frame_count == 27
        assert last.planned_duration_ms == 1080
        # 非末片即使因帧对齐略短也不得标 short。
        assert not any(s.is_short for s in plan.segments[:-1])

    def test_near_full_tail_not_marked_short(self):
        # 尾片 >= 目标 90% 时不算短尾：目标 1s，尾片 24 帧 = 960ms。
        source = make_source([40] * 49)  # 25+24
        plan = build_timeline(source, 1.0)
        assert plan.segment_count == 2
        assert plan.segments[-1].planned_duration_ms == 960
        assert not plan.segments[-1].is_short

    def test_target_duration_is_ceil_of_worst_segment(self):
        source = make_source([40] * 127)  # 含 1080ms 尾片
        plan = build_timeline(source, 2.0)
        worst = max(s.planned_duration_ms for s in plan.segments)
        assert plan.target_duration == ceil_div(worst, MS_PER_SEC)
        assert plan.target_duration == 2

    def test_segment_never_exceeds_target_plus_one_frame(self):
        # 帧边界对齐下，非末片最长只可能比目标多不到一帧。
        source = make_source([33] * 300)
        plan = build_timeline(source, 1.0)
        max_frame = max(f.duration_time for f in source.frames)
        for seg in plan.segments[:-1]:
            assert seg.planned_duration_ms <= 1000 + int(
                max_frame * MS_PER_SEC + 0.5
            )


class TestBuildTimelineVFR:
    def test_vfr_grouping_uses_real_pts(self):
        # 交替 20ms/60ms 帧（平均 40ms），切点必须跟随真实 PTS。
        durations = [20, 60] * 50  # 100 帧，总 4000ms
        source = make_source(durations, fps=25.0)
        plan = build_timeline(source, 1.0)
        # 每片都由真实帧边缘围成，计划时长之和严格等于源总时长。
        assert sum(s.planned_duration_ms for s in plan.segments) == 4000
        for seg in plan.segments:
            assert seg.planned_duration_ms <= 1060  # 至多超一个最长帧


class TestNoAccumulatedDrift:
    def test_boundaries_are_exact_frame_pts_across_many_segments(self):
        # 长序列（100 片量级）下每片起点都必须精确命中帧 PTS，
        # 任何浮点累积漂移都会让边界落不进 PTS 集合。
        source = make_source([40] * 2537)  # 101.48s
        plan = build_timeline(source, 1.0)
        pts_set = {int(f.pts_time * MS_PER_SEC + 0.5) for f in source.frames}
        for seg in plan.segments:
            assert seg.start_ms in pts_set
        # 计划时长之和 == 源总时长，无跨片误差。
        assert (
            sum(s.planned_duration_ms for s in plan.segments)
            == plan.source_duration_ms
        )

    def test_float_input_rounded_once_at_entry(self):
        # 30fps 的 0.0333...s 帧在入口统一舍入为 33ms；
        # 181 帧后末片终点不得出现亚毫秒尾数。
        frames = tuple(
            FrameInfo(index=i, pts_time=i / 30.0, duration_time=1 / 30.0)
            for i in range(181)
        )
        source = SourceInfo(
            path=_PathStub("in.mp4"), width=32, height=32, fps=30.0,
            frames=frames, has_audio=False,
        )
        plan = build_timeline(source, 2.0)
        for seg in plan.segments:
            assert isinstance(seg.start_ms, int)
            assert isinstance(seg.planned_duration_ms, int)
        assert plan.source_duration_ms == int(181 * (1000 / 30.0) + 0.5)

    def test_quantization_error_bounded_half_ms_per_frame(self):
        # 30fps 非整数帧长：计划总时长与源精确时长之差必须有界，
        # 每帧不超过 0.5ms（这里把毫秒量化的口径固化成回归基线）。
        n = 91
        frames = tuple(
            FrameInfo(index=i, pts_time=i / 30.0, duration_time=1 / 30.0)
            for i in range(n)
        )
        source = SourceInfo(
            path=_PathStub("in.mp4"), width=32, height=32, fps=30.0,
            frames=frames, has_audio=False,
        )
        plan = build_timeline(source, 1.0)
        exact_total_ms = n * (1000.0 / 30.0)
        drift = plan.source_duration_ms - exact_total_ms
        assert abs(drift) <= n * 0.5
        # 91 帧的实测基线：16ms（防止量化口径被无意改动）。
        assert plan.source_duration_ms - round(exact_total_ms) == 0

    def test_integer_frame_rate_has_zero_quantization_error(self):
        # 25fps 帧长恰为 40ms 整数，计划总时长必须与源逐毫秒相等。
        source = make_source([40] * 2537)
        exact = sum(source_f.duration_time for source_f in source.frames)
        plan = build_timeline(source, 1.0)
        assert plan.source_duration_ms == int(round(exact * 1000))


class TestTimelineErrors:
    def test_rejects_nonpositive_target(self):
        source = make_source([40] * 10)
        with pytest.raises(TimelineError):
            build_timeline(source, 0.0)
        with pytest.raises(TimelineError):
            build_timeline(source, -1.0)

    def test_rejects_target_smaller_than_one_frame(self):
        source = make_source([100] * 5)  # 10fps，每帧 100ms
        with pytest.raises(TimelineError):
            build_timeline(source, 0.05)  # 50ms 装不下一帧

    def test_rejects_empty_frames(self):
        source = SourceInfo(
            path=_PathStub("in.mp4"), width=32, height=32, fps=25.0,
            frames=(), has_audio=False,
        )
        with pytest.raises(TimelineError):
            build_timeline(source, 1.0)

    def test_rejects_nonpositive_frame_duration(self):
        source = make_source([40, 40, 0, 40])
        with pytest.raises(TimelineError):
            build_timeline(source, 1.0)

    def test_custom_prefix_and_suffix(self):
        source = make_source([40] * 25)
        plan = build_timeline(
            source, 1.0, segment_prefix="chunk", segment_suffix=".m4s"
        )
        assert plan.segments[0].filename == "chunk_00000.m4s"


class TestFormatters:
    def test_format_extinf_three_decimals(self):
        assert format_extinf(2000) == "2.000"
        assert format_extinf(33) == "0.033"
        assert format_extinf(1080) == "1.080"

    def test_format_extinf_no_binary_float_tail(self):
        # 毫秒整数除以 1000 固定 3 位小数，不得吐出二进制尾数。
        text = format_extinf(3003)
        assert text == "3.003"
        assert len(text.split(".")[1]) == 3

    def test_ms_to_seconds(self):
        assert ms_to_seconds(1500) == 1.5
        assert ms_to_seconds(0) == 0.0

    def test_ceil_div(self):
        assert ceil_div(0, 1000) == 0
        assert ceil_div(1, 1000) == 1
        assert ceil_div(1000, 1000) == 1
        assert ceil_div(1001, 1000) == 2
        assert ceil_div(2080, 1000) == 3
