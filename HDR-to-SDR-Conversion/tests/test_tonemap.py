"""色调映射单测：膝盖下恒等、亮度加权不偏色、单调有界不过曝。"""

from __future__ import annotations

import warnings

import pytest

from hdr_to_sdr.tonemap import (
    TONE_REINHARD,
    TONE_REINHARD_CLASSIC,
    TONE_BT2390,
    TONE_CLIP,
    DEFAULT_TONE_METHOD,
    DEFAULT_REINHARD_WHITE,
    REINHARD_WHITE_MIN,
    REINHARD_WHITE_MAX,
    clamp_reinhard_white,
    reinhard_scale,
    reinhard_classic_scale,
    bt2390_scale,
    tone_map_pixel,
    resolve_tone_method,
)


class TestReinhard:
    def test_below_knee_identity(self):
        # 膝盖 0.75·peak 以下完全恒等
        assert reinhard_scale(0.5, 1.0) == 1.0
        assert reinhard_scale(0.74, 1.0) == 1.0

    def test_above_knee_compressed(self):
        assert reinhard_scale(2.0, 1.0) < 1.0

    def test_output_monotonic_and_bounded(self):
        # 核心保证：单调（高光不反转）且输出上界 <= 1（不过曝）
        ys = [0.01, 0.1, 0.5, 0.75, 1.0, 2.0, 10.0, 50.0]
        outs = [y * reinhard_scale(y, 1.0) for y in ys]
        assert all(b >= a for a, b in zip(outs, outs[1:]))
        assert max(outs) <= 1.0 + 1e-12

    def test_larger_peak_means_less_compression(self):
        # 同一信号 y=0.5：peak 越大越远离膝盖，压缩越轻（系数越接近 1）
        assert reinhard_scale(0.5, 1.0) > reinhard_scale(0.5, 0.1)

    def test_degenerate_inputs(self):
        assert reinhard_scale(0.0, 1.0) == 1.0
        assert reinhard_scale(0.5, 0.0) == 1.0


class TestReinhardClassic:
    def test_textbook_form_at_white_one(self):
        # white=1 即教科书 L/(1+L)：peak(x=1) 处输出 0.5
        assert reinhard_classic_scale(1.0, 1.0, 1.0) == pytest.approx(0.5)

    def test_no_identity_segment_compresses_everywhere(self):
        # 与膝盖式不同：任意正亮度都有压缩（系数严格 < 1）
        assert reinhard_classic_scale(0.1, 1.0) < 1.0
        assert reinhard_classic_scale(0.5, 1.0) == pytest.approx(2.0 / 3.0)

    def test_monotonic_and_bounded(self):
        ys = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 50.0, 1000.0]
        outs = [y * reinhard_classic_scale(y, 1.0) for y in ys]
        assert all(b >= a for a, b in zip(outs, outs[1:]))
        # 硬不变量：任意高的高光输出都不超过 1（不过曝）
        assert max(outs) <= 1.0
        assert outs[-1] < 1.0  # 渐近但不达到 1

    def test_white_larger_means_gentler(self):
        # white 越大肩部越靠后，同一亮度压缩越轻（系数越大）
        assert (
            reinhard_classic_scale(2.0, 1.0, 4.0)
            > reinhard_classic_scale(2.0, 1.0, 1.0)
        )
        assert (
            reinhard_classic_scale(2.0, 1.0, 1.0)
            > reinhard_classic_scale(2.0, 1.0, 0.2)
        )

    def test_dark_is_near_identity(self):
        # 暗部压缩很轻：x=0.01 时输出≈0.0099
        out = 0.01 * reinhard_classic_scale(0.01, 1.0)
        assert out == pytest.approx(0.01, rel=2e-2)

    def test_black_and_degenerate(self):
        assert reinhard_classic_scale(0.0, 1.0) == 1.0
        assert reinhard_classic_scale(0.5, 0.0) == 1.0

    def test_deterministic_repeated_calls(self):
        # 同输入两次结果完全一致（纯函数、无随机/全局状态）
        first = [reinhard_classic_scale(y, 1.0, 2.5) for y in (0.1, 1.0, 7.0)]
        second = [reinhard_classic_scale(y, 1.0, 2.5) for y in (0.1, 1.0, 7.0)]
        assert first == second


class TestReinhardWhiteClamp:
    def test_in_range_passthrough(self):
        assert clamp_reinhard_white(1.0) == 1.0
        assert clamp_reinhard_white(3.5) == 3.5

    def test_below_min_clamped_and_warns(self):
        with pytest.warns(UserWarning, match="下限"):
            assert clamp_reinhard_white(-2.0) == REINHARD_WHITE_MIN
        with pytest.warns(UserWarning):
            assert clamp_reinhard_white(0.0) == REINHARD_WHITE_MIN

    def test_above_max_clamped_and_warns(self):
        with pytest.warns(UserWarning, match="上限"):
            assert clamp_reinhard_white(500.0) == REINHARD_WHITE_MAX

    def test_nan_falls_back_to_default(self):
        with pytest.warns(UserWarning):
            assert clamp_reinhard_white(float("nan")) == DEFAULT_REINHARD_WHITE

    def test_non_numeric_falls_back_to_default(self):
        with pytest.warns(UserWarning):
            assert clamp_reinhard_white("oops") == DEFAULT_REINHARD_WHITE

    def test_clamped_value_never_breaks_math(self):
        # 钳制后的值用于实际计算，不得产生 NaN/inf 或 >1 的输出
        for bad in (-10.0, 0.0, 1e9, float("inf")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = reinhard_classic_scale(5.0, 1.0, bad)
            assert 0.0 < s <= 1.0
            assert s == s  # 非 NaN


class TestBT2390:
    def test_below_knee_identity(self):
        # 膝盖 0.7·peak 以下完全恒等
        assert bt2390_scale(0.3, 1.0) == 1.0
        assert bt2390_scale(0.69, 1.0) == 1.0

    def test_above_knee_compressed(self):
        assert bt2390_scale(2.0, 1.0) < 1.0

    def test_monotonic_and_bounded(self):
        ys = [0.01, 0.1, 0.7, 1.0, 2.0, 10.0, 50.0]
        outs = [y * bt2390_scale(y, 1.0) for y in ys]
        assert all(b >= a for a, b in zip(outs, outs[1:]))
        assert max(outs) <= 1.0 + 1e-12

    def test_reinhard_knee_higher_than_bt2390(self):
        # Reinhard 膝盖(0.75)高于 BT2390(0.7)：0.72 处前者恒等、后者已压缩
        assert reinhard_scale(0.72, 1.0) == 1.0
        assert bt2390_scale(0.72, 1.0) < 1.0


class TestPixelToneMap:
    def test_luma_weighted_keeps_ratios(self):
        # 同系数缩放三通道：输出通道比例 = 输入通道比例（不偏色）
        r, g, b = tone_map_pixel(0.8, 0.4, 0.2, TONE_REINHARD, 1.0)
        assert r / 0.8 == pytest.approx(g / 0.4, rel=1e-9)
        assert g / 0.4 == pytest.approx(b / 0.2, rel=1e-9)

    def test_classic_luma_weighted_keeps_ratios(self):
        r, g, b = tone_map_pixel(
            0.8, 0.4, 0.2, TONE_REINHARD_CLASSIC, 1.0, 1.0
        )
        assert r / 0.8 == pytest.approx(g / 0.4, rel=1e-9)
        assert g / 0.4 == pytest.approx(b / 0.2, rel=1e-9)

    def test_classic_white_changes_result(self):
        # 同一像素、不同白点应得到不同压缩强度
        a = tone_map_pixel(2.0, 2.0, 2.0, TONE_REINHARD_CLASSIC, 1.0, 1.0)
        b = tone_map_pixel(2.0, 2.0, 2.0, TONE_REINHARD_CLASSIC, 1.0, 4.0)
        assert a != b
        assert b[0] > a[0]  # white 大、压缩轻、输出亮

    def test_gray_stays_gray(self):
        for method in (TONE_REINHARD, TONE_REINHARD_CLASSIC, TONE_BT2390):
            r, g, b = tone_map_pixel(0.5, 0.5, 0.5, method, 1.0)
            assert r == pytest.approx(g, abs=1e-12)
            assert g == pytest.approx(b, abs=1e-12)

    def test_black(self):
        for method in (
            TONE_REINHARD,
            TONE_REINHARD_CLASSIC,
            TONE_BT2390,
            TONE_CLIP,
        ):
            assert tone_map_pixel(0.0, 0.0, 0.0, method, 1.0) == (0.0, 0.0, 0.0)

    def test_clip_hard_clamps(self):
        r, g, b = tone_map_pixel(1.5, -0.5, 0.5, TONE_CLIP, 1.0)
        assert (r, g, b) == (1.0, 0.0, 0.5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            tone_map_pixel(0.1, 0.1, 0.1, "nonsense", 1.0)

    def test_resolve(self):
        assert resolve_tone_method(" REINHARD ") == TONE_REINHARD
        assert resolve_tone_method("Reinhard_Classic") == TONE_REINHARD_CLASSIC
        with pytest.raises(ValueError):
            resolve_tone_method("nope")

    def test_default_is_reinhard(self):
        assert DEFAULT_TONE_METHOD == TONE_REINHARD
