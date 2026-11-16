"""层 1 键控纯逻辑测试：双阈值 alpha、despill、非绿键色、恒等基线。"""

import math

import pytest

from vkeyer.keyer import (
    MAX_DISTANCE,
    Keyer,
    KeyerConfig,
    despill_rgb,
    dominant_channel,
)
from tests.conftest import green_screen_frame, make_rgb_frame


def test_dominant_channel_green_blue_gray():
    assert dominant_channel((0, 200, 0)) == 1      # 绿幕
    assert dominant_channel((0, 40, 230)) == 2     # 蓝幕
    assert dominant_channel((230, 30, 30)) == 0    # 红幕
    assert dominant_channel((128, 128, 128)) == -1  # 灰幕无主导
    assert dominant_channel((100, 100, 50)) == -1   # 并列最大


def test_config_validation():
    KeyerConfig((0, 200, 0))
    with pytest.raises(ValueError):
        KeyerConfig((0, 256, 0))
    with pytest.raises(ValueError):
        KeyerConfig((0, 200, 0), similarity=-1)
    with pytest.raises(ValueError):
        KeyerConfig((0, 200, 0), softness=-1)
    with pytest.raises(ValueError):
        KeyerConfig((0, 0, 0), similarity=MAX_DISTANCE + 1)
    with pytest.raises(ValueError):
        KeyerConfig((0, 200))


def test_exact_key_color_is_fully_transparent():
    """与键色完全相同的像素 alpha=0，且 RGB 被 despill 中和。"""
    green = (0, 200, 0)
    keyer = Keyer(KeyerConfig(green, similarity=32, softness=64))
    frame = make_rgb_frame(2, 2, lambda x, y: green)
    out = keyer.key_frame(frame, 2, 2)
    for i in range(4):
        assert out[i * 4 + 3] == 0


def test_far_color_is_fully_opaque_and_rgb_untouched():
    """距离 >= 外阈值的像素 alpha=255，despill 绝不改不透明像素。"""
    green = (0, 200, 0)
    red = (200, 30, 30)
    keyer = Keyer(KeyerConfig(green, similarity=32, softness=64))
    frame = make_rgb_frame(2, 2, lambda x, y: red)
    out = keyer.key_frame(frame, 2, 2)
    for i in range(4):
        p = i * 4
        assert out[p:p + 3] == bytes(red)
        assert out[p + 3] == 255


def test_soft_band_alpha_monotonic_and_clamped():
    """软化带内 alpha 随距离单调递增，且严格落在 [1, 254]。"""
    green = (0, 200, 0)
    sim, soft = 32.0, 64.0
    keyer = Keyer(KeyerConfig(green, similarity=sim, softness=soft))
    alphas = []
    # 沿 R 通道构造一系列距离落在软化带内的像素
    for dr in range(0, 100):
        px = (dr, 200, 0)
        d = math.sqrt(dr * dr)
        if sim < d < sim + soft:
            frame = make_rgb_frame(1, 1, lambda x, y: px)
            out = keyer.key_frame(frame, 1, 1)
            a = out[3]
            assert 1 <= a <= 254
            alphas.append(a)
    assert alphas == sorted(alphas)
    assert len(alphas) >= 10


def test_hard_key_zero_softness_is_binary():
    """softness=0 退化为硬键：只有 0/255 两种 alpha。"""
    green = (0, 200, 0)
    keyer = Keyer(KeyerConfig(green, similarity=60, softness=0))
    frame = green_screen_frame(width=16, height=16)
    out = keyer.key_frame(frame, 16, 16)
    seen = {out[i * 4 + 3] for i in range(256)}
    assert seen <= {0, 255}


def test_despill_reduces_green_on_semitransparent_edge():
    """半透明绿边像素 despill 后 G 通道下降，R/B 不动。"""
    green = (0, 200, 0)
    keyer = Keyer(KeyerConfig(green, similarity=32, softness=200))
    # 距离适中的像素：G 偏高（150），R=B=60
    frame = make_rgb_frame(1, 1, lambda x, y: (60, 150, 60))
    out = keyer.key_frame(frame, 1, 1)
    r, g, b, a = out
    assert 0 < a < 255
    assert g < 150       # 主导通道被压
    assert r == 60 and b == 60


def test_despill_weight_scales_with_inverse_alpha():
    """全透明权重最大、近不透明权重最小：同颜色 alpha 越低 G 压得越狠。"""
    # 直接验证 despill_rgb 的权重语义
    r, g, b = 60, 150, 60
    full = despill_rgb(r, g, b, 1, 255)
    weak = despill_rgb(r, g, b, 1, 10)
    assert full[1] < weak[1] < g


def test_despill_skips_when_dominant_not_above_others():
    """主导通道并不高于另两通道最大值时原样返回（不误伤）。"""
    assert despill_rgb(200, 100, 200, 1, 255) == (200, 100, 200)
    assert despill_rgb(10, 10, 10, -1, 255) == (10, 10, 10)


def test_blue_screen_uses_blue_channel_for_despill():
    """蓝幕：主导通道为 B，残蓝被压而 G 不动。"""
    blue = (0, 30, 220)
    keyer = Keyer(KeyerConfig(blue, similarity=32, softness=200))
    frame = make_rgb_frame(1, 1, lambda x, y: (40, 40, 160))
    out = keyer.key_frame(frame, 1, 1)
    r, g, b, a = out
    assert 0 < a < 255
    assert b < 160
    assert g == 40


def test_frame_length_validation():
    keyer = Keyer(KeyerConfig((0, 200, 0)))
    with pytest.raises(ValueError):
        keyer.key_frame(b"\x00" * 10, 4, 4)


def test_green_screen_geometry():
    """合成绿幕帧：幕布区全透明、主体矩形区全不透明。"""
    green = (0, 200, 0)
    red = (200, 30, 30)
    keyer = Keyer(KeyerConfig(green, similarity=32, softness=64))
    frame = green_screen_frame(16, 16, subject_box=(4, 4, 11, 11),
                               screen=green, subject=red)
    out = keyer.key_frame(frame, 16, 16)
    for y in range(16):
        for x in range(16):
            a = out[(y * 16 + x) * 4 + 3]
            if 4 <= x < 11 and 4 <= y < 11:
                assert a == 255
            else:
                assert a == 0


def test_no_despill_keeps_screen_rgb():
    """关闭 despill 后，透明像素的 RGB 保持原始幕布色。"""
    green = (0, 200, 0)
    keyer = Keyer(KeyerConfig(green, despill=False))
    frame = make_rgb_frame(1, 1, lambda x, y: green)
    out = keyer.key_frame(frame, 1, 1)
    assert out[0] == 0 and out[1] == 200 and out[2] == 0
    assert out[3] == 0
