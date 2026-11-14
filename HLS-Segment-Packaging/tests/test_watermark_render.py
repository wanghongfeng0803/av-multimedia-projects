"""文本排版栅格化端到端测试（PNG 产物、回退、颜色、异常输入）。"""

from __future__ import annotations

import os
import struct
import zlib

import pytest

from hls_packager.errors import WatermarkError
from hls_packager.watermark.config import WatermarkConfig
from hls_packager.watermark.rasterize import render_text_png


def _read_png(path: str) -> tuple[int, int, bytearray]:
    """最小 PNG 解码（仅测试用）：返回 (宽, 高, 去滤波 RGBA 行数据)。"""
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = height = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", data[pos + 8:pos + 16])
        elif tag == b"IDAT":
            idat += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    # 本工具写出的 PNG 每行都是 None 滤波器（首字节 0）
    pixels = bytearray(width * height * 4)
    for row in range(height):
        line_start = row * (stride + 1)
        assert raw[line_start] == 0
        pixels[row * stride:(row + 1) * stride] = raw[line_start + 1:line_start + 1 + stride]
    return width, height, pixels


def _ink_colors(pixels: bytearray) -> set[tuple[int, int, int]]:
    return {
        (pixels[i], pixels[i + 1], pixels[i + 2])
        for i in range(0, len(pixels), 4)
        if pixels[i + 3] > 0
    }


def test_render_basic_text_writes_valid_png(tmp_path):
    out = tmp_path / "wm.png"
    result = render_text_png(WatermarkConfig(text="Hello 2026", font_size=28), str(out))
    assert os.path.isfile(out)
    width, height, pixels = _read_png(str(out))
    assert (width, height) == (result.width, result.height)
    assert _ink_colors(pixels)  # 存在非透明墨迹


def test_render_cjk_multibyte_text(tmp_path):
    out = tmp_path / "cjk.png"
    result = render_text_png(
        WatermarkConfig(text="中文水印测试", font_size=32), str(out)
    )
    assert result.codepoint_count == 6
    width, height, pixels = _read_png(str(out))
    # 6 个汉字的宽度应明显大于高度
    assert width > height
    assert _ink_colors(pixels)


def test_render_accepts_utf8_bytes(tmp_path):
    out = tmp_path / "bytes.png"
    result = render_text_png(
        WatermarkConfig(text="水印".encode("utf-8"), font_size=24), str(out)
    )
    assert result.codepoint_count == 2
    assert _ink_colors(_read_png(str(out))[2])


def test_render_text_with_spaces(tmp_path):
    out = tmp_path / "spaces.png"
    result = render_text_png(
        WatermarkConfig(text="a b c", font_size=20), str(out)
    )
    # 空格只占步进，整体仍正常出图
    assert result.width > 0 and result.height > 0
    assert _ink_colors(_read_png(str(out))[2])


def test_render_applies_named_and_hex_colors(tmp_path):
    red_out = tmp_path / "red.png"
    render_text_png(WatermarkConfig(text="A", color="red", font_size=24), str(red_out))
    colors = _ink_colors(_read_png(str(red_out))[2])
    assert any(r > 200 and g < 80 and b < 80 for r, g, b in colors)

    hex_out = tmp_path / "hex.png"
    render_text_png(
        WatermarkConfig(text="A", color="#00FF00", font_size=24), str(hex_out)
    )
    colors = _ink_colors(_read_png(hex_out)[2])
    assert any(g > 200 and r < 80 and b < 80 for r, g, b in colors)

    short_out = tmp_path / "short.png"
    render_text_png(
        WatermarkConfig(text="A", color="#f00", font_size=24), str(short_out)
    )
    colors = _ink_colors(_read_png(short_out)[2])
    assert any(r > 200 for r, _, _ in colors)


def test_render_rejects_bad_color(tmp_path):
    with pytest.raises(WatermarkError):
        render_text_png(
            WatermarkConfig(text="A", color="not-a-color"),
            str(tmp_path / "x.png"),
        )


def test_render_rejects_empty_text(tmp_path):
    with pytest.raises(WatermarkError):
        render_text_png(WatermarkConfig(text=""), str(tmp_path / "x.png"))


def test_render_rejects_bad_font_size(tmp_path):
    with pytest.raises(WatermarkError):
        WatermarkConfig(text="A", font_size=0).validate()
    with pytest.raises(WatermarkError):
        WatermarkConfig(text="A", font_size=-5).validate()


def test_render_rejects_bad_opacity_and_anchor(tmp_path):
    with pytest.raises(WatermarkError):
        WatermarkConfig(text="A", opacity=1.5).validate()
    with pytest.raises(WatermarkError):
        WatermarkConfig(text="A", anchor="somewhere").validate()


def test_render_rejects_invalid_utf8_bytes(tmp_path):
    # E6 B0 是"水"的截断序列
    with pytest.raises(WatermarkError):
        render_text_png(
            WatermarkConfig(text=b"\xE6\xB0", font_size=16),
            str(tmp_path / "x.png"),
        )


def test_render_rejects_output_dir_missing(tmp_path):
    with pytest.raises(WatermarkError):
        render_text_png(
            WatermarkConfig(text="A"),
            str(tmp_path / "no-such-dir" / "x.png"),
        )


def test_render_whitespace_only_text_still_produces_canvas(tmp_path):
    out = tmp_path / "blank.png"
    result = render_text_png(WatermarkConfig(text="   ", font_size=16), str(out))
    width, height, _ = _read_png(str(out))
    assert width > 0 and height > 0
    assert (width, height) == (result.width, result.height)


def test_render_with_special_character_directory(tmp_path):
    # 目录含空格、#、分号、中文（Windows 不允许结尾空格，故不以空格结尾）
    special_dir = tmp_path / "wm out#1; 水印"
    special_dir.mkdir()
    out = special_dir / "water mark.png"
    result = render_text_png(WatermarkConfig(text="A; rm -rf /", font_size=20), str(out))
    assert os.path.isfile(out)
    assert result.codepoint_count == len("A; rm -rf /")
    assert _ink_colors(_read_png(str(out))[2])


def test_render_tofu_path_when_no_font_covers_codepoint(tmp_path, monkeypatch):
    # 强制系统字体列表为空，制造"无字体覆盖"环境
    monkeypatch.setattr(
        "hls_packager.watermark.rasterize.FontStack",
        lambda fonts: FontStackStub(),
    )
    out = tmp_path / "tofu.png"
    result = render_text_png(
        WatermarkConfig(text="水", font_size=24), str(out)
    )
    # 豆腐块仍产出可见墨迹，且报告标注 tofu
    assert ("tofu",) == result.used_fallbacks
    assert _ink_colors(_read_png(str(out))[2])


def test_render_bitmap_fallback_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hls_packager.watermark.rasterize.FontStack",
        lambda fonts: FontStackStub(),
    )
    out = tmp_path / "bitmap.png"
    result = render_text_png(
        WatermarkConfig(text="AB", font_size=21), str(out)
    )
    assert set(result.used_fallbacks) == {"bitmap"}
    assert _ink_colors(_read_png(str(out))[2])


class FontStackStub:
    """测试桩：模拟一个完全没有 TrueType 字体的环境。"""

    def resolve(self, codepoint: int):
        from hls_packager.watermark import bitmap_font
        from hls_packager.watermark.font_stack import ResolvedFace

        if bitmap_font.supports(codepoint):
            return ResolvedFace(codepoint, "bitmap")
        return ResolvedFace(codepoint, "tofu")
