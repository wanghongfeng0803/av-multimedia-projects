"""色相 / 饱和度的纯逻辑测试（不依赖 ffmpeg）。

方向核对不依赖肉眼：

- 主色 / 间色色块在色相环上的角度由代码数值断言（红 0°、黄 60°、
  绿 120°、青 180°、蓝 240°、品红 300°）；
- 随机像素与标准库 :mod:`colorsys` 的独立 HSV 实现交叉比对，
  2 万像素零误差；
- 端到端方向再由 test_pipeline 与 ffmpeg 原生 ``hue`` 滤镜对照。

通道顺序：本文件所有输入 / 断言都按 **RGB** 字节序书写，
与全工具统一约定一致（不出现 BGR）。
"""

from __future__ import annotations

import colorsys
import os
import warnings

import pytest

from color_grading.colorspace import (
    HUE_MAX,
    HUE_MIN,
    SATURATION_MAX,
    SATURATION_MIN,
    adjust_pixel,
    clamp_color_params,
    is_identity,
    map_chunk,
)

# 色相环六个主色 / 间色：(名称, RGB, 标准 HSV 角度)
WHEEL = [
    ("红", (255, 0, 0), 0),
    ("黄", (255, 255, 0), 60),
    ("绿", (0, 255, 0), 120),
    ("青", (0, 255, 255), 180),
    ("蓝", (0, 0, 255), 240),
    ("品红", (255, 0, 255), 300),
]


def test_identity_predicate():
    assert is_identity(0, 0)
    assert not is_identity(1, 0)
    assert not is_identity(0, 1)


def test_identity_map_chunk_is_byte_exact():
    """色相 / 饱和度均为 0 时整块逐字节不变（不引入浮点往返）。"""
    data = os.urandom(999)
    assert map_chunk(data, 0, 0) == data


def test_map_chunk_matches_adjust_pixel_reference():
    """内联热点循环必须与可读参考 adjust_pixel 逐像素等价（含固定顺序）。"""
    import random

    rng = random.Random(7)
    # 普通随机像素 + 刻意覆盖的边界像素（灰度 / 纯饱和主色 / 黑白）
    pixels = [(rng.randrange(256), rng.randrange(256), rng.randrange(256))
              for _ in range(3000)]
    pixels += [(v, v, v) for v in (0, 1, 128, 254, 255)]
    pixels += [c for _, c, _ in WHEEL]
    raw = b"".join(bytes(p) for p in pixels)

    for hue, satp in ((30, 20), (-45, -60), (180, 100), (90, -100), (0, 50)):
        sat_factor = (100 + satp) / 100
        got = map_chunk(raw, hue, satp)
        for idx, (r, g, b) in enumerate(pixels):
            assert got[idx * 3:idx * 3 + 3] == bytes(
                adjust_pixel(r, g, b, hue, sat_factor)
            )


def test_map_chunk_requires_triplet_length():
    with pytest.raises(ValueError):
        map_chunk(b"\x00\x01", 0, 0)


# ---------------------------------------------------------------- 色相方向


@pytest.mark.parametrize("name,rgb,angle", WHEEL)
def test_positive_hue_rotates_counterclockwise(name, rgb, angle):
    """正色相沿 红→黄→绿→青→蓝→品红 方向（逆时针为正）旋转 60°。"""
    target = (angle + 60) % 360
    expected = next(rgb2 for _, rgb2, a in WHEEL if a == target)
    assert adjust_pixel(*rgb, 60, 1.0) == expected


@pytest.mark.parametrize("name,rgb,angle", WHEEL)
def test_negative_hue_rotates_clockwise(name, rgb, angle):
    """负色相沿反方向旋转 60°。"""
    target = (angle - 60) % 360
    expected = next(rgb2 for _, rgb2, a in WHEEL if a == target)
    assert adjust_pixel(*rgb, -60, 1.0) == expected


def test_full_180_rotation_swaps_complementary_colors():
    # 红 180° → 青；蓝 180° → 黄
    assert adjust_pixel(255, 0, 0, 180, 1.0) == (0, 255, 255)
    assert adjust_pixel(0, 0, 255, 180, 1.0) == (255, 255, 0)


def test_full_360_equivalent_to_identity_on_pure_colors():
    for _, rgb, _ in WHEEL:
        assert adjust_pixel(*rgb, 360, 1.0) == rgb


def test_gray_has_no_hue_and_is_unchanged_by_hue_rotation():
    """灰度像素色相无定义，旋转任意角度都保持原灰度。"""
    for v in (0, 1, 64, 128, 200, 255):
        assert adjust_pixel(v, v, v, 137, 1.0) == (v, v, v)


# ---------------------------------------------------------------- 饱和度


def test_saturation_min_fully_desaturates():
    """saturation=-100 → 系数 0，任意彩色变成 R=G=B 的灰度。"""
    out = adjust_pixel(200, 100, 50, 0, 0.0)
    assert out[0] == out[1] == out[2]


def test_saturation_plus_100_doubles():
    """saturation=+100 → 系数 2，饱和度翻倍（仍钳制到 1）。"""
    base = adjust_pixel(120, 90, 60, 0, 1.0)
    doubled = adjust_pixel(120, 90, 60, 0, 2.0)
    # 翻倍后通道更分散：max-min 应变大或相等，且结果仍合法。
    assert (max(doubled) - min(doubled)) >= (max(base) - min(base))
    assert all(0 <= c <= 255 for c in doubled)


def test_saturation_zero_factor_keeps_luma_value():
    """去色后的灰度值等于 HSV 的 V（max(R,G,B)），方向可数值核对。"""
    r, g, b = 200, 100, 50
    out = adjust_pixel(r, g, b, 0, 0.0)
    assert out == (max(r, g, b),) * 3


# ---------------------------------------------------------- 固定处理顺序


def test_combined_order_matches_hue_then_saturation_reference():
    """两参数同时生效：结果必须等于「先色相、后饱和度」的参考步骤。"""
    r, g, b, hue, satp = 180, 70, 210, 45, 40
    sat_factor = (100 + satp) / 100
    got = adjust_pixel(r, g, b, hue, sat_factor)

    # 参考：先只转色相，再对其结果只做饱和度——与 colorspace 内顺序一致。
    after_hue = adjust_pixel(r, g, b, hue, 1.0)
    reference = adjust_pixel(*after_hue, 0.0, sat_factor)
    assert got == reference


def test_combined_matches_colorsys_independent_reference():
    """与标准库 colorsys 的独立 HSV 实现在大量随机像素上零误差。

    colorsys 同样按「先色相旋转、后饱和度缩放」组合，从而既验证
    数学正确性，也验证固定顺序与参考实现一致。
    """
    import random

    rng = random.Random(20261116)
    for _ in range(20000):
        r, g, b = (rng.randrange(256) for _ in range(3))
        hue = rng.uniform(-180, 180)
        sat_factor = (100 + rng.uniform(-100, 100)) / 100

        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if max(r, g, b) != min(r, g, b):  # 彩色才旋转色相
            h = (h + hue / 360.0) % 1.0
        s = min(1.0, max(0.0, s * sat_factor))
        rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
        ref = tuple(min(255, max(0, round(c * 255))) for c in (rr, gg, bb))

        assert adjust_pixel(r, g, b, hue, sat_factor) == ref


# ---------------------------------------------------------------- 钳制


def test_clamp_color_params_in_range_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h, s, msgs = clamp_color_params(30, -40)
    assert (h, s) == (30.0, -40.0)
    assert msgs == []
    assert caught == []


def test_clamp_color_params_out_of_range():
    with pytest.warns(UserWarning, match="钳制"):
        h, s, msgs = clamp_color_params(999, -500)
    assert h == HUE_MAX
    assert s == SATURATION_MIN
    assert len(msgs) == 2
    assert all("钳制" in m for m in msgs)


def test_clamped_hue_matches_boundary_value():
    with pytest.warns(UserWarning):
        clamped = clamp_color_params(400, 0)[0]
    assert clamped == HUE_MAX


def test_constant_ranges():
    assert (HUE_MIN, HUE_MAX) == (-180.0, 180.0)
    assert (SATURATION_MIN, SATURATION_MAX) == (-100.0, 100.0)
