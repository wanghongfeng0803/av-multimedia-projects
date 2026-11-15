"""色调映射单测：膝盖下恒等、亮度加权不偏色、单调有界不过曝。"""

from __future__ import annotations

import pytest

from hdr_to_sdr.tonemap import (
    TONE_REINHARD,
    TONE_BT2390,
    TONE_CLIP,
    DEFAULT_TONE_METHOD,
    reinhard_scale,
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

    def test_gray_stays_gray(self):
        for method in (TONE_REINHARD, TONE_BT2390):
            r, g, b = tone_map_pixel(0.5, 0.5, 0.5, method, 1.0)
            assert r == pytest.approx(g, abs=1e-12)
            assert g == pytest.approx(b, abs=1e-12)

    def test_black(self):
        for method in (TONE_REINHARD, TONE_BT2390, TONE_CLIP):
            assert tone_map_pixel(0.0, 0.0, 0.0, method, 1.0) == (0.0, 0.0, 0.0)

    def test_clip_hard_clamps(self):
        r, g, b = tone_map_pixel(1.5, -0.5, 0.5, TONE_CLIP, 1.0)
        assert (r, g, b) == (1.0, 0.0, 0.5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            tone_map_pixel(0.1, 0.1, 0.1, "nonsense", 1.0)

    def test_resolve(self):
        assert resolve_tone_method(" REINHARD ") == TONE_REINHARD
        with pytest.raises(ValueError):
            resolve_tone_method("nope")

    def test_default_is_reinhard(self):
        assert DEFAULT_TONE_METHOD == TONE_REINHARD
