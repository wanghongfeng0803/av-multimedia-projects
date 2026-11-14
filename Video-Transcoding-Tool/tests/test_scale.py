"""分辨率规划模块的单元测试（纯逻辑，不启动 ffmpeg）。

覆盖：各类表达式解析、宽高比保持、偶数对齐、放大收敛、最小/最大
尺寸边界、非方形像素（SAR）与非法输入。
"""

from __future__ import annotations

import pytest

from transcode_tool.scale import (
    MAX_DIMENSION,
    MIN_DIMENSION,
    PRESETS,
    ResolutionPlan,
    build_scale_filter,
    even_floor,
    parse_resolution,
)

SRC_W, SRC_H = 1920, 1080


def _plan(spec, **kwargs):
    return parse_resolution(spec, SRC_W, SRC_H, **kwargs)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def test_even_floor():
    assert even_floor(4) == 4
    assert even_floor(3) == 2
    assert even_floor(2.9) == 2
    assert even_floor(0) == 0


def test_build_scale_filter():
    assert build_scale_filter(1280, 720) == \
        "scale=width=1280:height=720:flags=lanczos"


# ---------------------------------------------------------------------------
# 各类表达式
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec,expected",
    [
        ("1280x720", (1280, 720)),
        ("1920x1080", (1920, 1080)),
        ("1280", (1280, 720)),            # 裸整数 = 定宽
        ("1280x0", (1280, 720)),          # Wx0 = 定宽
        ("x720", (1280, 720)),            # 前导 x = 定高
        ("X720", (1280, 720)),
        ("0x720", (1280, 720)),
        ("50%", (960, 540)),
        ("720p", (1280, 720)),
        ("1080P", (1920, 1080)),
        ("fhd", (1920, 1080)),
        (" 1280x720 ", (1280, 720)),
        ("1280×720", (1280, 720)),        # 全角乘号
    ],
)
def test_resolution_expressions(spec, expected):
    plan = _plan(spec)
    assert (plan.width, plan.height) == expected
    assert isinstance(plan, ResolutionPlan)
    assert "lanczos" in plan.scale_filter


def test_explicit_box_keeps_aspect_ratio_without_stretch():
    # 方形盒子容纳 16:9 源：以更紧的一边为准，绝不拉伸
    plan = _plan("1000x1000")
    assert plan.width == 1000
    assert plan.height == 562              # 1080 * (1000/1920) = 562.5 向下取偶数
    assert plan.width / plan.height == pytest.approx(16 / 9, abs=0.01)


def test_preset_table_keys_are_normalized_lowercase():
    # 查询前会做小写归一化，键里若混入大写将永远无法命中
    for key in PRESETS:
        assert key == key.lower()


def test_preset_height_anchor_preserves_portrait_ratio():
    # 竖屏源 1080x1920 选 720p：高度锚定 720，宽度按比例得 404（偶数）
    plan = parse_resolution("720p", 1080, 1920)
    assert plan.height == 720
    assert plan.width == 404
    assert plan.width / plan.height == pytest.approx(1080 / 1920, abs=0.01)


def test_sar_affects_display_aspect_ratio():
    # 非方形像素 sar=2：显示宽高比翻倍，定高推导出的宽度也翻倍
    plan = parse_resolution("x540", 1920, 1080, sar=2.0)
    assert plan.height == 540
    assert plan.width == 1920


def test_invalid_sar_falls_back_to_one():
    plan = parse_resolution("x540", 1920, 1080, sar=0.0)  # type: ignore[arg-type]
    assert (plan.width, plan.height) == (960, 540)


# ---------------------------------------------------------------------------
# 偶数对齐与最小尺寸兜底
# ---------------------------------------------------------------------------

def test_odd_source_scaled_down_to_even():
    plan = parse_resolution("100%", 641, 363)
    assert (plan.width, plan.height) == (640, 362)
    assert plan.width % 2 == 0 and plan.height % 2 == 0
    assert any("偶数" in w for w in plan.warnings)


def test_tiny_result_snaps_up_to_minimum_even():
    # 10x10 缩 10% → 1x1，取整为 0 后必须抬升到最小合法偶数 2
    plan = parse_resolution("10%", 10, 10)
    assert (plan.width, plan.height) == (MIN_DIMENSION, MIN_DIMENSION)
    assert any("最小合法偶数" in w for w in plan.warnings)


def test_small_but_valid_two_pixels():
    plan = parse_resolution("50%", 4, 4)
    assert (plan.width, plan.height) == (2, 2)


# ---------------------------------------------------------------------------
# 放大收敛
# ---------------------------------------------------------------------------

def test_upscale_blocked_by_default():
    plan = _plan("4k")  # 1080p → 2160p 是放大
    assert plan.upscaled is False
    assert (plan.width, plan.height) == (1920, 1080)  # 收敛回源尺寸
    assert any("放大" in w for w in plan.warnings)


def test_upscale_allowed_explicitly():
    plan = _plan("4k", allow_upscale=True)
    assert plan.upscaled is True
    assert (plan.width, plan.height) == (3840, 2160)
    assert not plan.warnings


def test_percent_upscale_blocked_then_allowed():
    blocked = _plan("200%")
    assert (blocked.width, blocked.height) == (1920, 1080)
    allowed = _plan("200%", allow_upscale=True)
    assert (allowed.width, allowed.height) == (3840, 2160)


def test_downscale_is_not_flagged_upscale():
    plan = _plan("720p")
    assert plan.upscaled is False
    assert not plan.warnings


# ---------------------------------------------------------------------------
# 非法输入与最大边界
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "", "   ", "abc", "1920y1080", "0x0", "0%", "-5x5", "10000x",
])
def test_unrecognized_expressions_raise(bad):
    with pytest.raises(ValueError):
        _plan(bad)


def test_dimension_beyond_8k_raises():
    # 定宽 8000 直接推导，超过 8K 上限
    with pytest.raises(ValueError, match="8K"):
        _plan("8000")
    assert MAX_DIMENSION == 7680


def test_invalid_source_dimensions_raise():
    with pytest.raises(ValueError):
        parse_resolution("720p", 0, 1080)
    with pytest.raises(ValueError):
        parse_resolution("720p", -1920, 1080)
    with pytest.raises(ValueError):
        parse_resolution("720p", True, 1080)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_resolution("", 1920, 1080)


def test_requested_string_is_preserved():
    plan = _plan("  1280x720 ")
    assert plan.requested == "1280x720"
