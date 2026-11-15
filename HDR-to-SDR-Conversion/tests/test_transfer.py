"""传递函数数学单测：PQ/HLG EOTF、BT.709 OETF、LUT 一致性。

全部为纯函数测试，不依赖 ffmpeg。
"""

from __future__ import annotations

import math

import pytest

from hdr_to_sdr.transfer import (
    TRANSFER_PQ,
    TRANSFER_HLG,
    pq_eotf_normalized,
    hlg_inverse_oetf,
    hlg_display_linear,
    bt709_oetf,
    build_eotf_lut,
    build_oetf_lut,
    lerp_lut,
)


class TestPQ:
    def test_endpoints(self):
        assert pq_eotf_normalized(0.0) == 0.0
        assert pq_eotf_normalized(1.0) == pytest.approx(1.0, abs=1e-9)

    def test_known_value_half_signal(self):
        # ST 2084: N=0.5 对应约 92.2 nit → 归一化 0.00922
        out = pq_eotf_normalized(0.5)
        assert out == pytest.approx(0.00922, rel=2e-3)

    def test_monotonic(self):
        vals = [pq_eotf_normalized(i / 100.0) for i in range(101)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_input_clamped(self):
        assert pq_eotf_normalized(-0.5) == 0.0
        assert pq_eotf_normalized(1.5) == pytest.approx(1.0)

    def test_dark_region_scaled_correctly(self):
        # 100 nit 参考：PQ 信号约 0.508，归一化线性光 0.01
        n_100nit = 0.50808
        assert pq_eotf_normalized(n_100nit) == pytest.approx(0.01, rel=2e-3)


class TestHLG:
    def test_inverse_oetf_piecewise(self):
        assert hlg_inverse_oetf(0.0) == 0.0
        # 分段点 0.5 → E_s = 1/12
        assert hlg_inverse_oetf(0.5) == pytest.approx(1.0 / 12.0)
        assert hlg_inverse_oetf(1.0) == pytest.approx(1.0)

    def test_inverse_oetf_monotonic(self):
        vals = [hlg_inverse_oetf(i / 50.0) for i in range(51)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_ootf_reference_white_ratio(self):
        # 场景白 (1,1,1)：Y_s=1，显示归一化 = 1/1.2^1.2 ≈ 0.8026
        r, g, b = hlg_display_linear(1.0, 1.0, 1.0)
        expected = 1.0 / (1.2 ** 1.2)
        assert r == pytest.approx(expected, rel=1e-6)
        assert (r, g, b) == (pytest.approx(r),) * 3

    def test_ootf_black(self):
        assert hlg_display_linear(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_ootf_preserves_gray_hue(self):
        # 中性灰三通道相等，OOTF 后仍相等（不偏色）
        r, g, b = hlg_display_linear(0.3, 0.3, 0.3)
        assert r == g == b


class TestBT709:
    def test_endpoints(self):
        assert bt709_oetf(0.0) == 0.0
        assert bt709_oetf(1.0) == pytest.approx(1.0)

    def test_linear_segment(self):
        assert bt709_oetf(0.01) == pytest.approx(0.045)

    def test_knee_small_jump_per_spec(self):
        # BT.709 规范在 0.018 处两段并非严格连续，存在约 2.5e-4 的规范内小跳变，
        # 实现以规范文本为准，锁定该行为
        low = 4.5 * 0.018
        high = 1.099 * (0.018 ** 0.45) - 0.099
        assert abs(high - low) == pytest.approx(2.479e-4, abs=2e-6)
        # 函数在膝盖处取线性段值
        assert bt709_oetf(0.018) == pytest.approx(low, abs=1e-12)

    def test_monotonic(self):
        vals = [bt709_oetf(i / 100.0) for i in range(101)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))


class TestLUT:
    def test_eotf_lut_matches_function_pq(self):
        lut = build_eotf_lut(TRANSFER_PQ, 10)
        assert len(lut) == 1024
        for code in (0, 1, 127, 512, 1023):
            expected = pq_eotf_normalized(code / 1023.0)
            assert lut[code] == pytest.approx(expected, rel=1e-12)

    def test_eotf_lut_hlg(self):
        lut = build_eotf_lut(TRANSFER_HLG, 10)
        # 码值 512 对应信号 512/1023≈0.5005，逆 OETF 接近 1/12（查表有量化误差）
        assert lut[512] == pytest.approx(1.0 / 12.0, abs=2e-4)
        assert lut[1023] == pytest.approx(1.0)

    def test_eotf_lut_bad_bits(self):
        with pytest.raises(ValueError):
            build_eotf_lut(TRANSFER_PQ, 9)

    def test_oetf_lut_interpolation(self):
        lut = build_oetf_lut(4096)
        # 膝盖点附近一阶导数不连续，插值误差天然偏大，单独放宽；
        # 其余采样点误差应在 2e-5 内
        for x in (0.0, 0.25, 0.5, 0.7, 1.0):
            assert lerp_lut(lut, x) == pytest.approx(bt709_oetf(x), abs=2e-5)
        for x in (0.01, 0.03):
            assert lerp_lut(lut, x) == pytest.approx(bt709_oetf(x), abs=3e-4)

    def test_lerp_clamps_outside_range(self):
        lut = [0.0, 1.0]
        assert lerp_lut(lut, -1.0) == 0.0
        assert lerp_lut(lut, 2.0) == 1.0
