"""映射表构建与参数钳制的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import warnings

import pytest

from color_grading.lut import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    CONTRAST_MAX,
    CONTRAST_MIN,
    IDENTITY_LUT,
    build_lut,
    clamp_params,
    is_identity,
)


def test_default_is_exact_identity():
    """无参数调用必须退化为逐字节恒等映射（旧行为回归基线）。"""
    lut = build_lut()
    assert isinstance(lut, bytes)
    assert len(lut) == 256
    assert lut == IDENTITY_LUT
    assert is_identity(lut)


def test_default_call_emits_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_lut()
    assert caught == []


def test_zero_args_equal_default():
    assert build_lut(0.0, 0.0) == build_lut()


def test_clamp_warns_and_matches_boundary():
    """超范围值钳制到边界，结果与直接给边界值一致。"""
    with pytest.warns(UserWarning, match="钳制"):
        clamped = build_lut(brightness=150)
    assert clamped == build_lut(brightness=BRIGHTNESS_MAX)

    with pytest.warns(UserWarning, match="钳制"):
        clamped = build_lut(contrast=-300)
    assert clamped == build_lut(contrast=CONTRAST_MIN)


def test_clamp_params_collects_messages():
    b, c, msgs = clamp_params(-200, 500)
    assert b == BRIGHTNESS_MIN
    assert c == CONTRAST_MAX
    assert len(msgs) == 2
    assert all("钳制" in m for m in msgs)


def test_clamp_params_in_range_is_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b, c, msgs = clamp_params(30, -40)
    assert (b, c) == (30.0, -40.0)
    assert msgs == []
    assert caught == []


def test_brightness_extremes_saturate():
    assert build_lut(brightness=100) == bytes([255] * 256)
    assert build_lut(brightness=-100) == bytes([0] * 256)


def test_contrast_min_flattens_to_midgray():
    """factor=0 时所有像素塌缩到支点 128。"""
    assert build_lut(contrast=-100) == bytes([128] * 256)


def test_contrast_pivot_stays_put():
    """中灰支点不受对比度影响。"""
    for contrast in (-80, -30, 0, 40, 100):
        assert build_lut(contrast=contrast)[128] == 128


def test_spot_values():
    # brightness=20 → offset = 51.0（整数，避开舍入歧义）
    lut = build_lut(brightness=20)
    assert lut[0] == 51
    assert lut[100] == 151
    assert lut[204] == 255  # 255 恰好到顶
    assert lut[205] == 255  # 钳制
    # contrast=100 → factor=2
    lut = build_lut(contrast=100)
    assert lut[0] == 0      # (0-128)*2+128 = -128 → 钳到 0
    assert lut[64] == 0     # 恰好到 0
    assert lut[192] == 255  # 恰好到顶
    assert lut[255] == 255


@pytest.mark.parametrize("contrast", [-100, -50, -10, 0, 10, 50, 100])
def test_lut_is_monotonic_non_decreasing(contrast):
    lut = build_lut(brightness=10, contrast=contrast)
    assert all(lut[i] <= lut[i + 1] for i in range(255))
