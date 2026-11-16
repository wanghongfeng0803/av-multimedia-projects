"""层 3 纯逻辑测试：编码规格解析、序列识别、source-over 合成端点精确。"""

import pytest

from vkeyer.encoder import (
    ALPHA_VIDEO_SPECS,
    RGB_VIDEO_SPECS,
    composite_over_solid,
    composite_rgba_frame,
    is_image_sequence,
    resolve_video_spec,
)
from vkeyer.errors import OutputFormatError


def test_alpha_specs_cover_expected_containers():
    assert set(ALPHA_VIDEO_SPECS) == {".mov", ".webm", ".mkv"}
    for codec, pix, _ in ALPHA_VIDEO_SPECS.values():
        assert codec and pix  # 都必须显式指定带 alpha 的像素格式


def test_mp4_rejected_for_alpha():
    with pytest.raises(OutputFormatError):
        resolve_video_spec("out.mp4", alpha=True)


def test_mp4_accepted_for_composite():
    codec, pix, extra = resolve_video_spec("out.mp4", alpha=False)
    assert codec == "libx264" and pix == "yuv420p"


def test_resolve_video_spec_case_insensitive_ext():
    assert resolve_video_spec("OUT.MOV", alpha=True)[0] == "qtrle"


def test_unknown_container_rejected():
    with pytest.raises(OutputFormatError):
        resolve_video_spec("out.avi", alpha=False)


def test_image_sequence_detection():
    assert is_image_sequence("frames/out_%06d.png")
    assert is_image_sequence("a.webp") is False
    with pytest.raises(OutputFormatError):
        is_image_sequence("frames/out_%06d.jpg")


def test_composite_endpoints_exact():
    """全透明=背景、全不透明=前景，两端无舍入误差（锁定基线）。"""
    fg = bytes([10, 20, 30, 0,     # 全透明
                40, 50, 60, 255])  # 全不透明
    bg = bytes([90, 100, 110, 200, 210, 220])
    out = composite_rgba_frame(fg, bg, 2, 1)
    assert out[:3] == bytes([90, 100, 110])
    assert out[3:6] == bytes([40, 50, 60])


def test_composite_semitransparent_midpoint():
    """alpha=128 时结果约为前后景均值（允许 ±1 定点舍入）。"""
    fg = bytes([100, 100, 100, 128])
    bg = bytes([200, 200, 200])
    out = composite_rgba_frame(fg, bg, 1, 1)
    for v in out:
        assert abs(v - 150) <= 1


def test_composite_over_solid_color():
    fg = bytes([0, 0, 0, 0, 255, 255, 255, 255])
    out = composite_over_solid(fg, 2, 1, (10, 20, 30))
    assert out[:3] == bytes([10, 20, 30])   # 透明处露出纯色
    assert out[3:6] == bytes([255, 255, 255])


def test_composite_length_validation():
    with pytest.raises(ValueError):
        composite_rgba_frame(b"\x00" * 8, b"\x00" * 6, 3, 1)
    with pytest.raises(ValueError):
        composite_rgba_frame(b"\x00" * 8, b"\x00" * 3, 2, 1)
