"""色调映射单测：膝盖下恒等、亮度加权不偏色、单调有界不过曝。"""

from __future__ import annotations

import math
import warnings

import pytest

from hdr_to_sdr.tonemap import (
    TONE_REINHARD,
    TONE_REINHARD_CLASSIC,
    TONE_HABLE,
    TONE_BT2390,
    TONE_CLIP,
    DEFAULT_TONE_METHOD,
    DEFAULT_REINHARD_WHITE,
    DEFAULT_HABLE_SHARPNESS,
    REINHARD_WHITE_MIN,
    REINHARD_WHITE_MAX,
    HABLE_SHARPNESS_MIN,
    HABLE_SHARPNESS_MAX,
    HABLE_KNEE_RATIO,
    clamp_reinhard_white,
    clamp_hable_sharpness,
    reinhard_scale,
    reinhard_classic_scale,
    hable_scale,
    hable_polynomial,
    bt2390_scale,
    build_tone_curve_lut,
    tone_curve_values,
    tone_map_pixel,
    registered_tone_methods,
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
        for method in (TONE_REINHARD, TONE_REINHARD_CLASSIC, TONE_HABLE,
                       TONE_BT2390):
            r, g, b = tone_map_pixel(0.5, 0.5, 0.5, method, 1.0)
            assert r == pytest.approx(g, abs=1e-12)
            assert g == pytest.approx(b, abs=1e-12)

    def test_black(self):
        for method in (
            TONE_REINHARD,
            TONE_REINHARD_CLASSIC,
            TONE_HABLE,
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
        assert resolve_tone_method(" HABLE ") == TONE_HABLE
        with pytest.raises(ValueError):
            resolve_tone_method("nope")

    def test_default_is_reinhard(self):
        assert DEFAULT_TONE_METHOD == TONE_REINHARD


# ---- Hable/Filmic 映射器 ---------------------------------------------------
# 高精度 golden 值：默认 c=8.0、peak=1.0，由 hable_scale 公式独立算出，
# 任何对公式的意外改动都会在这里被抓住。
_HABLE_GOLDEN = {
    #  x : (缩放系数 s, 映射输出 h(x)=x·s)
    1.00: (0.845740103270, 0.845740103270),
    1.50: (0.608493752111, 0.912740628166),
    2.00: (0.469582324455, 0.939164648910),
    4.00: (0.243122207384, 0.972488829535),
    8.00: (0.123358957148, 0.986871657181),
}


class TestHablePolynomial:
    def test_zero_at_origin(self):
        assert hable_polynomial(0.0) == 0.0

    def test_strict_monotonic(self):
        xs = [0.01 * i for i in range(1, 2000)]
        vals = [hable_polynomial(x) for x in xs]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_approaches_asymptote_never_reaches(self):
        # 渐近线 W_INF=14/15：远端任意逼近但严格小于
        w_inf = 14.0 / 15.0
        for x in (10.0, 100.0, 1000.0, 1e6):
            assert hable_polynomial(x) < w_inf
        assert hable_polynomial(1e6) == pytest.approx(w_inf, rel=1e-5)

    def test_negative_input_is_zero(self):
        assert hable_polynomial(-3.0) == 0.0


class TestHableScale:
    def test_below_knee_identity(self):
        # 与 Reinhard 同膝盖 0.75·peak：膝下恒等、暗部零损失
        for y in (0.0, 0.1, 0.5, 0.74 * 1.0, 0.75):
            assert hable_scale(y, 1.0) == 1.0

    def test_above_knee_compressed(self):
        assert hable_scale(2.0, 1.0) < 1.0
        assert hable_scale(2.0, 1.0) > 0.0

    def test_golden_anchor_values(self):
        for x, (s_gold, h_gold) in _HABLE_GOLDEN.items():
            s = hable_scale(x, 1.0)
            assert s == pytest.approx(s_gold, rel=1e-9)
            assert x * s == pytest.approx(h_gold, rel=1e-9)

    def test_peak_output_matches_reinhard_level(self):
        # 峰值同量级硬指标：h(1) 不低于 Reinhard 峰值输出的 95%
        h_hable = hable_scale(1.0, 1.0)
        h_reinhard = reinhard_scale(1.0, 1.0)
        assert h_hable >= 0.95 * h_reinhard
        assert h_hable <= h_reinhard  # 不更曝

    def test_strict_monotonic_output_over_huge_range(self):
        xs = [10 ** (-4 + 6 * i / 3000) for i in range(3001)]
        outs = [x * hable_scale(x, 1.0) for x in xs]
        assert all(b >= a - 1e-15 for a, b in zip(outs, outs[1:]))

    def test_output_strictly_bounded_under_one(self):
        # 不过曝：任意高的高光输出都严格 < 1（渐近不到达）
        for y in (1.0, 2.0, 10.0, 100.0, 1e4, 1e8):
            out = y * hable_scale(y, 1.0)
            assert 0.0 < out < 1.0

    def test_scale_factor_always_in_unit_interval(self):
        for y in (1e-6, 0.1, 1.0, 100.0, 1e8):
            s = hable_scale(y, 1.0)
            assert 0.0 < s <= 1.0
            assert math.isfinite(s)

    def test_degenerate_inputs(self):
        assert hable_scale(0.0, 1.0) == 1.0
        assert hable_scale(0.5, 0.0) == 1.0
        assert hable_scale(-2.0, 1.0) == 1.0

    def test_deterministic(self):
        first = [hable_scale(y, 1.0) for y in (0.8, 2.0, 9.0)]
        second = [hable_scale(y, 1.0) for y in (0.8, 2.0, 9.0)]
        assert first == second

    def test_sharpness_larger_means_steeper_shoulder(self):
        # c 越大肩部越陡：同一高光亮度输出更高（压缩更轻）
        low = hable_scale(2.0, 1.0, sharpness=2.0)
        high = hable_scale(2.0, 1.0, sharpness=32.0)
        assert high > low
        # 两者都合法
        assert 0.0 < low < high <= 1.0

    def test_sharpness_changes_result_but_knee_unchanged(self):
        # 锐度只影响肩部：膝下任何 c 都是恒等
        for c in (HABLE_SHARPNESS_MIN, 8.0, HABLE_SHARPNESS_MAX):
            assert hable_scale(0.5, 1.0, sharpness=c) == 1.0
        assert hable_scale(2.0, 1.0, sharpness=2.0) != \
            hable_scale(2.0, 1.0, sharpness=16.0)

    def test_peak_scaling_equivalence(self):
        # peak 语义：同相对位置 y/peak 的缩放系数一致
        assert hable_scale(2.0, 1.0) == pytest.approx(
            hable_scale(0.2, 0.1), rel=1e-12
        )
        assert hable_scale(0.05, 0.1) == 1.0  # 相对 0.5 < 膝盖


class TestHableVsReinhardDetail:
    """可复算的高光细节验收：同素材同帧号区间的数学等价口径。"""

    @staticmethod
    def _deriv(g, x):
        h = 1e-7
        return (g(x + h) - g(x - h)) / (2 * h)

    @staticmethod
    def _geo_mean_slope(g, lo, hi, n=801):
        xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        return math.exp(
            sum(math.log(max(TestHableVsReinhardDetail._deriv(g, x), 1e-12))
                for x in xs) / n
        )

    @staticmethod
    def _count_levels(g, lo, hi, n=1001):
        xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        return len({round(g(x) * 255) for x in xs})

    def test_highlight_geometric_slope_not_weaker(self):
        # 几何均值局部斜率：两档高光带 Hable 均不弱于 Reinhard
        r_curve = lambda x: x * reinhard_scale(x, 1.0)
        h_curve = lambda x: x * hable_scale(x, 1.0)
        for lo, hi, margin in ((1.0, 4.0, 1.25), (1.0, 10.0, 1.35)):
            ratio = (self._geo_mean_slope(h_curve, lo, hi)
                     / self._geo_mean_slope(r_curve, lo, hi))
            assert ratio >= margin, f"x∈[{lo},{hi}) 斜率比 {ratio} 低于预期"

    def test_highlight_distinguishable_levels_not_weaker(self):
        # 8bit 码级可分辨级数：Hable 不少于 Reinhard
        r_curve = lambda x: x * reinhard_scale(x, 1.0)
        h_curve = lambda x: x * hable_scale(x, 1.0)
        for lo, hi in ((1.0, 4.0), (1.0, 10.0), (2.0, 8.0)):
            assert self._count_levels(h_curve, lo, hi) >= \
                self._count_levels(r_curve, lo, hi)

    def test_identical_below_knee_pixel_for_pixel(self):
        # 公平对比的前提：膝下两曲线逐像素完全相同
        for x in (0.01, 0.1, 0.5, 0.749):
            assert hable_scale(x, 1.0) == reinhard_scale(x, 1.0) == 1.0

    def test_luma_weighted_keeps_ratios(self):
        r, g, b = tone_map_pixel(0.8, 0.4, 0.2, TONE_HABLE, 1.0)
        assert r / 0.8 == pytest.approx(g / 0.4, rel=1e-9)
        assert g / 0.4 == pytest.approx(b / 0.2, rel=1e-9)


class TestHableSharpnessClamp:
    def test_in_range_passthrough(self):
        assert clamp_hable_sharpness(8.0) == 8.0
        assert clamp_hable_sharpness(33.25) == 33.25

    def test_below_min_clamped_and_warns(self):
        with pytest.warns(UserWarning, match="下限"):
            assert clamp_hable_sharpness(0.0) == HABLE_SHARPNESS_MIN
        with pytest.warns(UserWarning):
            assert clamp_hable_sharpness(-9.0) == HABLE_SHARPNESS_MIN

    def test_above_max_clamped_and_warns(self):
        with pytest.warns(UserWarning, match="上限"):
            assert clamp_hable_sharpness(999.0) == HABLE_SHARPNESS_MAX

    def test_nan_falls_back_to_default(self):
        with pytest.warns(UserWarning):
            assert clamp_hable_sharpness(float("nan")) == \
                DEFAULT_HABLE_SHARPNESS

    def test_non_numeric_falls_back_to_default(self):
        with pytest.warns(UserWarning):
            assert clamp_hable_sharpness("oops") == DEFAULT_HABLE_SHARPNESS

    def test_clamped_value_never_breaks_math(self):
        for bad in (-10.0, 0.0, 1e9, float("inf")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = hable_scale(5.0, 1.0, bad)
            assert 0.0 < s <= 1.0
            assert s == s  # 非 NaN


class TestToneRegistry:
    def test_registry_contains_all_compression_mappers(self):
        names = registered_tone_methods()
        assert TONE_REINHARD in names
        assert TONE_REINHARD_CLASSIC in names
        assert TONE_HABLE in names
        assert TONE_BT2390 in names
        # clip 是硬截断分支，不进压缩曲线注册表
        assert TONE_CLIP not in names

    def test_resolve_accepts_every_registered_name(self):
        for name in registered_tone_methods():
            assert resolve_tone_method(f" {name.upper()} ") == name

    def test_resolve_rejects_unknown(self):
        with pytest.raises(ValueError):
            resolve_tone_method("filmic_pro_max")

    def test_hable_param_wired_through_registry(self):
        # 注册契约：hable spec 的参数键与 FrameProcessor 持有的属性同名
        from hdr_to_sdr.tonemap import _TONE_REGISTRY
        spec = _TONE_REGISTRY[TONE_HABLE]
        assert spec.param_key == "hable_sharpness"
        assert spec.default == DEFAULT_HABLE_SHARPNESS
        assert spec.clamp_fn is clamp_hable_sharpness
        # 无参曲线的 spec 形态
        assert _TONE_REGISTRY[TONE_REINHARD].param_key is None


class TestToneCurveLUT:
    def test_lut_matches_analytic_formula(self):
        # LUT 与解析公式同源：网格点上逐点一致
        size = 257
        lut = build_tone_curve_lut(TONE_HABLE, size=size, x_max=4.0)
        grid, values = tone_curve_values(TONE_HABLE, size, 1.0,
                                         1.0, DEFAULT_HABLE_SHARPNESS, 4.0)
        assert lut == pytest.approx(values)
        for x, v in zip(grid, values):
            expect = x * hable_scale(x, 1.0)
            assert v == pytest.approx(expect, rel=1e-12)

    def test_lut_endpoints_and_monotonicity(self):
        lut = build_tone_curve_lut(TONE_HABLE, size=512)
        assert lut[0] == 0.0
        assert all(b >= a for a, b in zip(lut, lut[1:]))
        assert all(v < 1.0 for v in lut)

    def test_lut_works_for_every_registered_mapper(self):
        for name in registered_tone_methods():
            lut = build_tone_curve_lut(name, size=64)
            assert len(lut) == 64 and math.isfinite(lut[-1])

    def test_clip_has_no_curve_table(self):
        with pytest.raises(ValueError):
            tone_curve_values(TONE_CLIP)

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            build_tone_curve_lut(TONE_HABLE, size=1)
        with pytest.raises(ValueError):
            build_tone_curve_lut(TONE_HABLE, x_max=0.0)


class TestToneMapFallback:
    def test_invalid_scale_falls_back_to_reinhard_without_raising(self):
        # 模拟映射器产出非法系数：monkeypatch 注册表分发函数，
        # 验证 tone_map_pixel 不抛异常、回退到 Reinhard 的确定结果
        import hdr_to_sdr.tonemap as tm
        original = tm._resolve_scale

        def fake_resolve(method, y, peak, params):
            if method == TONE_HABLE:
                return float("nan")
            return original(method, y, peak, params)

        tm._resolve_scale = fake_resolve
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r, g, b = tm.tone_map_pixel(2.0, 1.0, 0.5, TONE_HABLE, 1.0)
            assert any("回退" in str(w.message) for w in caught)
        finally:
            tm._resolve_scale = original

        # 回退结果 == 同输入 Reinhard 结果
        rr, rg, rb = tone_map_pixel(2.0, 1.0, 0.5, TONE_REINHARD, 1.0)
        assert (r, g, b) == pytest.approx((rr, rg, rb))
        assert all(math.isfinite(v) for v in (r, g, b))

    def test_non_finite_inputs_stay_finite(self):
        # 极端防御：NaN/inf 像素输入也不能炸掉逐像素循环，
        # 边界防护夹取到有限 [0,1] 值并发出 warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r, g, b = tone_map_pixel(
                float("nan"), 0.4, 0.2, TONE_HABLE, 1.0
            )
        assert all(math.isfinite(v) for v in (r, g, b))
        assert all(0.0 <= v <= 1.0 for v in (r, g, b))
        assert any("非有限" in str(w.message) for w in caught)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2, g2, b2 = tone_map_pixel(
                float("inf"), 0.4, 0.2, TONE_REINHARD, 1.0
            )
        assert all(math.isfinite(v) for v in (r2, g2, b2))
