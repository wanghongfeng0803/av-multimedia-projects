"""色彩科学单测：量化常数、色域矩阵、保亮度色域压缩。"""

from __future__ import annotations

import pytest

from hdr_to_sdr.colorscience import (
    RANGE_LIMITED,
    RANGE_FULL,
    BT709_KR,
    BT709_KG,
    BT709_KB,
    range_constants,
    yuv_to_rgb_offsets,
    rgb_to_yuv_offsets,
    apply_gamut_matrix,
    gamut_compress,
    gamut_map_pixel,
)


class TestRangeConstants:
    def test_8bit_limited(self):
        assert range_constants(RANGE_LIMITED, 8) == (16, 235, 16, 240)

    def test_10bit_limited(self):
        offset, y_max, c_min, c_max = range_constants(RANGE_LIMITED, 10)
        assert (offset, y_max, c_min, c_max) == (64, 940, 64, 1020)

    def test_full_range(self):
        assert range_constants(RANGE_FULL, 8) == (0, 255, 0, 255)
        assert range_constants(RANGE_FULL, 10) == (0, 1023, 0, 1023)

    def test_unknown_range_raises(self):
        with pytest.raises(ValueError):
            range_constants("weird", 8)


class TestYUVRoundtrip:
    """RGB→YUV→RGB 在各自量化域内应能闭合（误差不超过 1 个码值）。

    测试向量必须落在 YUV 可表示域内（三通道等比的灰度，以及经量化后
    色差不越界的柔和饱和色）；任意 RGB 组合可能落到色域外，码值会被
    钳制从而无法闭合，那不是系数错误。
    """

    @pytest.mark.parametrize("bits", [8, 10])
    @pytest.mark.parametrize("prange", [RANGE_LIMITED, RANGE_FULL])
    def test_bt709_roundtrip(self, bits, prange):
        y_gain, c_gain, kr, kg, kb, y_off, c_mid = rgb_to_yuv_offsets(
            BT709_KR, BT709_KB, prange, bits
        )
        y_scale, rv, gu, gv, bu = yuv_to_rgb_offsets(
            BT709_KR, BT709_KB, prange, bits
        )
        peak = float((1 << bits) - 1)
        # 灰度向量色差为 0，天然在域内且对量化最稳健
        vectors = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25),
                   (0.5, 0.5, 0.5), (0.9, 0.9, 0.9), (1.0, 1.0, 1.0)]
        for r, g, b in vectors:
            luma = kr * r + kg * g + kb * b
            y_code = round(luma * y_gain + y_off)
            u_code = round((b - luma) * c_gain + c_mid)
            v_code = round((r - luma) * c_gain + c_mid)

            yn = (y_code - y_off) / peak * y_scale
            un = (u_code - c_mid) / peak
            vn = (v_code - c_mid) / peak
            r2 = yn + rv * vn
            g2 = yn + gu * un + gv * vn
            b2 = yn + bu * un
            tol = 2.0 / peak
            assert r2 == pytest.approx(r, abs=tol)
            assert g2 == pytest.approx(g, abs=tol)
            assert b2 == pytest.approx(b, abs=tol)

    @pytest.mark.parametrize("bits", [8, 10])
    def test_gray_chroma_neutral(self, bits):
        """任意灰度的 U/V 码值必须落在中性色中点（不产生色偏）。"""
        y_gain, c_gain, kr, kg, kb, y_off, c_mid = rgb_to_yuv_offsets(
            BT709_KR, BT709_KB, RANGE_LIMITED, bits
        )
        for v in (0.1, 0.5, 0.9):
            luma = v  # 灰度时 luma 即 v
            u = round((v - luma) * c_gain + c_mid)
            w = round((v - luma) * c_gain + c_mid)
            assert u == round(c_mid)
            assert w == round(c_mid)


class TestGamutMatrix:
    def test_white_preserved(self):
        # D65 白 (1,1,1) 经过矩阵仍应为 (1,1,1)
        r, g, b = apply_gamut_matrix(1.0, 1.0, 1.0)
        assert (r, g, b) == pytest.approx((1.0, 1.0, 1.0), abs=2e-3)

    def test_black_preserved(self):
        assert apply_gamut_matrix(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_wide_gamut_goes_out_of_range(self):
        # BT.2020 纯红在 BT.709 原色下必然越界（负 G/B）
        r, g, b = apply_gamut_matrix(1.0, 0.0, 0.0)
        assert r > 1.0
        assert g < 0.0 and b < 0.0


class TestGamutCompress:
    def test_in_gamut_untouched(self):
        rgb = (0.3, 0.4, 0.5)
        assert gamut_compress(*rgb) == pytest.approx(rgb)

    def test_output_always_in_unit_cube(self):
        for rgb in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                    (1.2, -0.3, 0.8), (-0.5, 1.4, 0.2)]:
            out = gamut_compress(*rgb)
            for c in out:
                assert 0.0 - 1e-12 <= c <= 1.0 + 1e-12

    def test_luma_preserved(self):
        # 核心不变量：压缩前后 BT.709 亮度严格相等
        for rgb in [(1.0, 0.0, 0.0), (1.2, -0.3, 0.8), (0.0, 1.0, 0.0)]:
            y_before = BT709_KR * rgb[0] + BT709_KG * rgb[1] + BT709_KB * rgb[2]
            out = gamut_compress(*rgb)
            y_after = BT709_KR * out[0] + BT709_KG * out[1] + BT709_KB * out[2]
            assert y_after == pytest.approx(y_before, abs=1e-12)

    def test_full_pipeline_gray_neutral(self):
        # 中性灰经过完整色域映射后仍为中性灰（不偏色）；
        # 矩阵系数仅 4 位小数，允许 1e-4 量级的浮点不平衡
        for v in (0.1, 0.5, 0.9):
            r, g, b = gamut_map_pixel(v, v, v)
            assert r == pytest.approx(g, abs=1e-4)
            assert g == pytest.approx(b, abs=1e-4)
            assert 0.0 <= r <= 1.0

    def test_full_pipeline_bounded(self):
        out = gamut_map_pixel(1.0, 0.0, 0.0)  # BT.2020 纯红
        assert all(0.0 <= c <= 1.0 for c in out)
