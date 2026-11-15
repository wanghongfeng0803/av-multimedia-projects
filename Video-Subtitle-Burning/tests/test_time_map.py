"""时间轴变换（线性平移 + 非线性重映射）测试。

覆盖边界语义：正负偏移、越界裁剪/截断、负时间戳、毫秒精度、空结果、
零速平台压扁、舍入碰撞、映射后单调不重叠不变量，以及缓动积分正确性。
"""

from __future__ import annotations

import math

import pytest

from subburn.srt import Cue, format_timecode, render_srt
from subburn.time_map import (
    EASINGS,
    Easing,
    SpeedKeyframe,
    TimeWarper,
    build_speed_warper,
    remap_cues,
    shift_cues,
)


def cue(index: int, start_ms: int, end_ms: int, text: str = "x") -> Cue:
    return Cue(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


class TestShiftCues:
    def test_positive_offset(self):
        result = shift_cues([cue(1, 1000, 2000)], 1500)
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [(2500, 3500)]
        assert result.input_count == 1 and result.output_count == 1
        assert result.dropped == 0 and result.truncated == 0

    def test_negative_offset_keeps_in_bounds_part(self):
        # 负偏移：起点越下界被裁到 0，终点保留 -> 截断而非丢弃
        result = shift_cues([cue(1, 500, 2000)], -1500)
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [(0, 500)]
        assert result.truncated == 1 and result.dropped == 0

    def test_negative_offset_drops_whole_cue(self):
        # 整条落到下界之前 -> 丢弃，空结果容错
        result = shift_cues([cue(1, 100, 200)], -1000)
        assert result.cues == []
        assert result.dropped == 1 and result.output_count == 0
        assert result.warnings  # 有诊断信息

    def test_upper_bound_truncates_then_drops(self):
        # 跨上界：截断到上界；完全越上界：裁成零宽后丢弃。
        # 两条都发生过端点裁剪，故 truncated 计数为 2
        result = shift_cues(
            [cue(1, 1000, 3000, "a"), cue(2, 8000, 9000, "b")],
            0, upper_bound_ms=2000,
        )
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [(1000, 2000)]
        assert result.truncated == 2 and result.dropped == 1

    def test_negative_timestamps_preserved_without_lower_bound(self):
        # 不设下界时负时间戳全链路保留（floor 取整方向也要正确）
        result = shift_cues(
            [cue(1, 100, 500)], -300, lower_bound_ms=None,
        )
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [(-200, 200)]

    def test_negative_fractional_offset_floors_toward_minus_infinity(self):
        # 负小数偏移必须 floor（向负无穷），不能向零截断：100-300.7=-200.7 -> -201
        result = shift_cues(
            [cue(1, 100, 500)], -300.7, lower_bound_ms=None,
        )
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [(-201, 199)]

    def test_millisecond_precision_no_drift_on_many_cues(self):
        # 长列表演示无累积误差：每条独立 floor，不做递推
        cues = [cue(i, i * 1000, i * 1000 + 500) for i in range(1, 200)]
        result = shift_cues(cues, 0.333, lower_bound_ms=None)
        for i, c in enumerate(result.cues, start=1):
            assert c.start_ms == math.floor(i * 1000 + 0.333)
            assert c.end_ms == math.floor(i * 1000 + 500 + 0.333)

    def test_empty_input_returns_empty_result(self):
        result = shift_cues([], 1000)
        assert result.cues == [] and result.output_count == 0
        assert result.input_count == 0

    def test_resequence_and_sort_after_shift(self):
        # 输入乱序：变换后按 (起点,终点) 排序并从 1 重新编号
        result = shift_cues(
            [cue(1, 5000, 6000), cue(2, 1000, 2000)], -500,
        )
        assert [c.index for c in result.cues] == [1, 2]
        assert [c.start_ms for c in result.cues] == [500, 4500]

    def test_zero_width_cue_after_clip_dropped(self):
        # 偏移后两端都越下界，裁剪把起点终点都抬到 0 -> 宽度归零丢弃
        result = shift_cues([cue(1, 100, 200)], -250)
        assert result.cues == [] and result.dropped == 1

    def test_fractional_offset_keeps_integer_ms_model(self):
        # 正小数偏移同样 floor：100+0.6=100.6 -> 100，200+0.6=200.6 -> 200
        result = shift_cues([cue(1, 100, 200)], 0.6)
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [(100, 200)]

    def test_bad_bounds_rejected(self):
        with pytest.raises(ValueError):
            shift_cues([cue(1, 0, 100)], 0,
                       lower_bound_ms=1000, upper_bound_ms=500)

    def test_progress_callback_covers_full_range(self):
        cues = [cue(i, i * 100, i * 100 + 50) for i in range(1, 11)]
        seen: list[float] = []
        shift_cues(cues, 10, progress_callback=seen.append)
        # 每条处理完回调一次，收尾再补发 1.0
        assert seen[0] == pytest.approx(0.1)
        assert seen[-1] == 1.0 and len(seen) == len(cues) + 1


class TestSRTWriter:
    def test_format_timecode_roundtrip(self):
        from subburn.srt import parse_timecode
        for ms in (0, 1, 999, 1000, 3_600_000, 3_723_400, 36_000_000):
            assert parse_timecode(format_timecode(ms)) == ms

    def test_format_negative_rejected(self):
        with pytest.raises(ValueError):
            format_timecode(-1)

    def test_render_srt_resequences_and_parses_back(self):
        from subburn.srt import parse_srt
        text = render_srt([cue(9, 1500, 2500, "你好\n世界"),
                           cue(7, 3000, 4000, "bye")])
        reparsed, _ = parse_srt(text)
        assert [c.index for c in reparsed] == [1, 2]
        assert (reparsed[0].start_ms, reparsed[0].end_ms) == (1500, 2500)
        assert reparsed[0].text == "你好\n世界"

    def test_render_empty(self):
        assert render_srt([]) == ""


class TestEasing:
    def test_endpoints_and_monotonicity(self):
        for easing in EASINGS.values():
            assert easing(0.0) == pytest.approx(0.0)
            assert easing(1.0) == pytest.approx(1.0)
            vals = [easing(i / 100.0) for i in range(101)]
            assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_analytic_integrals_match_numerical(self):
        # 解析积分必须与高密度数值积分一致（钉住公式推导）
        for easing in EASINGS.values():
            numeric = sum(easing(i / 100_000) / 100_000
                          for i in range(100_000))
            assert easing.integral() == pytest.approx(numeric, abs=1e-5)

    def test_bad_custom_easing_rejected(self):
        with pytest.raises(ValueError):
            Easing("decreasing", lambda u: 1.0 - u)  # 单调递减
        with pytest.raises(ValueError):
            Easing("bad-end", lambda u: 2.0 * u)    # f(1)=2


class TestTimeWarperSpeedCurve:
    def test_unity_curve_is_identity(self):
        warper = build_speed_warper([(0, 1.0), (10_000, 1.0)])
        for t in (0.0, 1.0, 250.5, 9999.0):
            assert warper.map_ms(t) == pytest.approx(t)

    def test_constant_double_speed(self):
        warper = build_speed_warper([(0, 2.0), (10_000, 2.0)])
        assert warper.map_ms(5000) == pytest.approx(10_000)
        assert warper.map_ms(0) == pytest.approx(0)

    def test_extrapolation_beyond_domain(self):
        # 定义域外按端点速度外推，含负时刻
        warper = build_speed_warper([(0, 1.0), (1000, 1.0)])
        assert warper.map_ms(-500) == pytest.approx(-500)
        assert warper.map_ms(1500) == pytest.approx(1500)

    def test_zero_speed_plateau_is_pause(self):
        # 速度曲线：[0,1000) 由 1 线性减速到 0（积分均值 0.5，新轴走 500），
        # [1000,3000) 两端速度均为 0 -> 干净定格，平台电平 y=500
        warper = build_speed_warper([
            SpeedKeyframe(0, 1.0),
            SpeedKeyframe(1000, 0.0),
            SpeedKeyframe(3000, 0.0),
            SpeedKeyframe(5000, 1.0),
        ])
        # 减速段中点：时间过半但新轴只走 375（速度在下降）
        assert warper.map_ms(500) == pytest.approx(375)
        # 零速平台内任意时刻都映到同一新时刻 500
        assert warper.map_ms(1000) == pytest.approx(500)
        assert warper.map_ms(1500) == pytest.approx(500)
        assert warper.map_ms(2999.9) == pytest.approx(500)
        # 平台结束后进入加速段，末点新时刻为 500 + 2000*0.5 = 1500
        assert warper.map_ms(5000) == pytest.approx(1500)

    def test_eased_segment_continuous_at_joints(self):
        warper = TimeWarper.from_speed_curve([
            SpeedKeyframe(0, 1.0, "ease_in_out"),
            SpeedKeyframe(5000, 2.0, "smoothstep"),
            SpeedKeyframe(10_000, 0.5),
        ])
        # 分段交界处取值必须连续（同一时刻从两侧逼近）
        for joint in (5000.0, 10_000.0):
            assert warper.map_ms(joint - 1e-9) == pytest.approx(
                warper.map_ms(joint), abs=1e-6)
            assert warper.map_ms(joint + 1e-9) == pytest.approx(
                warper.map_ms(joint), abs=1e-6)

    def test_mapping_is_monotonic_nondecreasing(self):
        warper = TimeWarper.from_speed_curve([
            SpeedKeyframe(-2000, 0.5, "ease_in"),
            SpeedKeyframe(0, 2.0, "smoothstep"),
            SpeedKeyframe(4000, 0.0, "ease_out"),
            SpeedKeyframe(8000, 1.5, "linear"),
            SpeedKeyframe(12_000, 1.0),
        ])
        ts = [-3000 + i * 37.5 for i in range(400)]  # 跨定义域、含外推
        ys = [warper.map_ms(t) for t in ts]
        assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))

    def test_need_two_keyframes(self):
        with pytest.raises(ValueError):
            TimeWarper.from_speed_curve([SpeedKeyframe(0, 1.0)])

    def test_negative_speed_rejected(self):
        with pytest.raises(ValueError):
            SpeedKeyframe(0, -0.1)

    def test_non_increasing_keyframe_time_rejected(self):
        with pytest.raises(ValueError):
            TimeWarper.from_speed_curve([
                SpeedKeyframe(1000, 1.0), SpeedKeyframe(1000, 2.0),
            ])

    def test_unknown_easing_rejected(self):
        with pytest.raises(ValueError):
            SpeedKeyframe(0, 1.0, easing="nope")


class TestAnchorWarper:
    def test_piecewise_linear(self):
        warper = TimeWarper.from_anchors([(0, 0), (1000, 2000), (2000, 2000)])
        assert warper.map_ms(500) == pytest.approx(1000)
        # 平台段：新时刻恒定
        assert warper.map_ms(1500) == pytest.approx(2000)

    def test_clean_plateau_with_exact_levels(self):
        # 锚点版定格：[1000,3000) 旧时间在新轴上恒为 1000，前后斜率均为 1
        warper = TimeWarper.from_anchors(
            [(0, 0), (1000, 1000), (3000, 1000), (5000, 3000)]
        )
        assert warper.map_ms(500) == pytest.approx(500)
        assert warper.map_ms(1000) == pytest.approx(1000)
        assert warper.map_ms(2500) == pytest.approx(1000)
        assert warper.map_ms(3500) == pytest.approx(1500)

    def test_decreasing_anchor_rejected(self):
        with pytest.raises(ValueError):
            TimeWarper.from_anchors([(0, 1000), (1000, 900)])


class TestRemapCues:
    def test_identity_remap_preserves_timestamps(self):
        cues = [cue(1, 100, 200), cue(2, 300, 400)]
        warper = build_speed_warper([(0, 1.0), (10_000, 1.0)])
        result = remap_cues(cues, warper)
        assert [(c.start_ms, c.end_ms) for c in result.cues] == [
            (100, 200), (300, 400),
        ]

    def test_constant_slope_scales_timeline(self):
        # speed=2（dT/dt）：新轴拉伸一倍；speed=0.5：压缩一半
        warper2 = build_speed_warper([(0, 2.0), (10_000, 2.0)])
        result2 = remap_cues(
            [cue(1, 1000, 2000), cue(2, 4000, 5000)], warper2,
        )
        assert [(c.start_ms, c.end_ms) for c in result2.cues] == [
            (2000, 4000), (8000, 10_000),
        ]
        warper_half = build_speed_warper([(0, 0.5), (10_000, 0.5)])
        result_half = remap_cues([cue(1, 1000, 2000)], warper_half)
        assert [(c.start_ms, c.end_ms)
                for c in result_half.cues] == [(500, 1000)]

    def test_zero_speed_plateau_flattens_and_drops(self):
        # 锚点构造干净定格：旧轴 [1000,3000) 在新轴恒为 1000。
        # 整条落在平台上的字幕被压成零宽 -> 丢弃；跨越平台末端的截断存活
        warper = TimeWarper.from_anchors(
            [(0, 0), (1000, 1000), (3000, 1000), (5000, 3000)]
        )
        cues = [cue(1, 1200, 1500, "flat"),  # 完全在平台内 -> 丢弃
                cue(2, 2500, 3500, "straddle")]  # 跨平台末端 -> 存活
        result = remap_cues(cues, warper)
        assert result.dropped == 1
        survivor = result.cues[0]
        assert survivor.text == "straddle"
        # 平台电平 1000；[3000,3500] 斜率 1 走 500ms -> [1000,1500)
        assert (survivor.start_ms, survivor.end_ms) == (1000, 1500)

    def test_plateau_collision_resolved_without_overlap(self):
        # 平台压扁使两条不同区间映到同一新时刻，让位机制保证不重叠：
        # 全在平台内的零宽条丢弃；与存活条重叠的条起点让位
        warper = TimeWarper.from_anchors(
            [(0, 0), (2000, 2000), (4000, 2000), (8000, 6000)]
        )
        cues = [
            cue(1, 2200, 2600, "flat-a"),  # 全在平台 -> 零宽丢弃
            cue(2, 3000, 3400, "flat-b"),  # 全在平台 -> 零宽丢弃
            cue(3, 3500, 4500, "c"),       # 跨边界 -> [2000, 2500)
            cue(4, 3800, 4800, "d"),       # 映为 [2000, 2800)，与 c 重叠
        ]
        result = remap_cues(cues, warper)
        assert result.dropped == 2 and result.overlaps_fixed == 1
        for prev, nxt in zip(result.cues, result.cues[1:]):
            assert prev.end_ms <= nxt.start_ms
        texts = [c.text for c in result.cues]
        assert texts == ["c", "d"]
        # d 的起点被让位到 c 的终点
        assert (result.cues[1].start_ms, result.cues[1].end_ms) == (2500, 2800)

    def test_output_invariant_monotonic_nonoverlapping(self):
        # 复杂曲线（含零速平台与多种缓动）+ 密集字幕，统一钉住不变量：
        # 每条区间有正宽度、起点有序、相邻区间互不重叠
        warper = TimeWarper.from_speed_curve([
            SpeedKeyframe(-1000, 0.2, "ease_in"),
            SpeedKeyframe(0, 3.0, "smoothstep"),
            SpeedKeyframe(3000, 0.0, "ease_in_out"),
            SpeedKeyframe(6000, 2.0, "ease_out"),
            SpeedKeyframe(9000, 0.5),
        ])
        # 输入本身互相重叠（宽度 900 > 间距 500），进一步压测让位逻辑
        cues = [cue(i, i * 500, i * 500 + 900) for i in range(40)]
        result = remap_cues(cues, warper, lower_bound_ms=None)
        assert result.cues  # 至少有存活
        for c in result.cues:
            assert c.end_ms > c.start_ms
        for prev, nxt in zip(result.cues, result.cues[1:]):
            assert prev.start_ms <= nxt.start_ms
            assert prev.end_ms <= nxt.start_ms
        assert result.overlaps_fixed > 0  # 输入自带重叠，确实触发了修复

    def test_remap_with_lower_bound_clips_negative(self):
        # 速度曲线定义域含负轴，映射出负值时由下界裁剪
        warper = TimeWarper.from_anchors([(-1000, -500), (1000, 1500)])
        result = remap_cues([cue(1, -800, 500)], warper)
        c = result.cues[0]
        assert c.start_ms == 0 and c.end_ms > 0
        assert result.truncated == 1

    def test_remap_empty_input(self):
        warper = build_speed_warper([(0, 1.0), (1000, 1.0)])
        result = remap_cues([], warper)
        assert result.cues == [] and result.output_count == 0

    def test_progress_emitted(self):
        warper = build_speed_warper([(0, 1.0), (1000, 1.0)])
        seen: list[float] = []
        remap_cues([cue(1, 0, 10), cue(2, 20, 30)], warper,
                   progress_callback=seen.append)
        assert seen[-1] == 1.0

    def test_resequence_after_remap(self):
        warper = build_speed_warper([(0, 1.0), (10_000, 1.0)])
        result = remap_cues(
            [cue(1, 9000, 9500), cue(2, 100, 200)], warper,
        )
        assert [c.index for c in result.cues] == [1, 2]
        assert result.cues[0].start_ms == 100
