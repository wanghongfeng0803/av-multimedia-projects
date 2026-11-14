"""文本排版与栅格化主入口：码点序列 -> RGBA 位图 -> PNG。

两趟排版：
1. 度量趟逐码点走回退链拿字形位图与步进，求整段墨迹包围盒
   （空格只有步进、无墨迹，不参与包围盒）；
2. 绘制趟按包围盒分配透明画布，逐字形以 src-over 写入覆盖率，
   无字体覆盖的码点画豆腐块占位。

str 输入也先重新编码为 UTF-8 再走自研解码器，多字节口径与 bytes 一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..errors import WatermarkError
from . import bitmap_font
from .config import WatermarkConfig
from .font_stack import FontStack, ResolvedFace
from .png_buf import Image, parse_color, write_png
from .rasterizer import GlyphBitmap, rasterize_glyph
from .utf8 import decode_utf8


@dataclass(frozen=True)
class RenderedText:
    path: str
    width: int
    height: int
    codepoint_count: int
    fallback_report: tuple[tuple[int, str], ...]

    @property
    def used_fallbacks(self):
        return tuple(sorted({kind for _, kind in self.fallback_report}))


def _codepoints_from_text(text):
    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    try:
        return decode_utf8(raw)
    except UnicodeDecodeError as exc:
        raise WatermarkError(f"水印文本不是合法 UTF-8：{exc.reason}") from exc


def _bitmap_glyph(codepoint, pixel_size):
    """内置 5x7 点阵按最近邻放大。"""
    scale = max(1, round(pixel_size / bitmap_font.GLYPH_HEIGHT))
    rows = bitmap_font.glyph_rows(codepoint)
    width = bitmap_font.GLYPH_WIDTH * scale
    height = bitmap_font.GLYPH_HEIGHT * scale
    coverage = bytearray(width * height)
    if rows is not None:
        for row, bits in enumerate(rows):
            for col in range(bitmap_font.GLYPH_WIDTH):
                if bits & (1 << (bitmap_font.GLYPH_WIDTH - 1 - col)):
                    for dy in range(scale):
                        line = (row * scale + dy) * width
                        for dx in range(scale):
                            coverage[line + col * scale + dx] = 255
    return GlyphBitmap(width=width, height=height, coverage=coverage,
                       bearing_x=0, bearing_y=6 * scale,
                       advance=bitmap_font.GLYPH_ADVANCE * scale)


def _tofu_glyph(pixel_size):
    """无字体覆盖时的空心方框占位。"""
    box = max(4, int(pixel_size * 0.8))
    coverage = bytearray(box * box)
    border = max(1, box // 8)
    for y in range(box):
        for x in range(box):
            if x < border or x >= box - border or y < border or y >= box - border:
                coverage[y * box + x] = 255
    return GlyphBitmap(width=box, height=box, coverage=coverage,
                       bearing_x=0, bearing_y=int(pixel_size * 0.75),
                       advance=box + max(2, pixel_size // 8))


@dataclass
class _PlacedGlyph:
    face: ResolvedFace
    bitmap: GlyphBitmap
    pen_x: int


def _measure(codepoints, stack, pixel_size):
    placed = []
    pen_x = 0
    min_x = min_y = 10**9
    max_x = max_y = -10**9
    has_ink = False

    for codepoint in codepoints:
        face = stack.resolve(codepoint)
        if face.kind == "ttf":
            bitmap = rasterize_glyph(face.font, face.glyph_id, pixel_size)
        elif face.kind == "bitmap":
            bitmap = _bitmap_glyph(codepoint, pixel_size)
        else:
            bitmap = _tofu_glyph(pixel_size)
        placed.append(_PlacedGlyph(face, bitmap, pen_x))

        if any(bitmap.coverage):
            has_ink = True
            x0 = pen_x + bitmap.bearing_x
            y0 = -bitmap.bearing_y
            min_x, min_y = min(min_x, x0), min(min_y, y0)
            max_x, max_y = max(max_x, x0 + bitmap.width), max(max_y, y0 + bitmap.height)
        pen_x += bitmap.advance

    if not has_ink:
        return placed, 0, 0, max(1, pen_x), max(1, pixel_size)
    return placed, min_x, min_y, max_x, max_y


def _blit_glyph(image, glyph, origin_x, baseline_y, color, global_alpha):
    bitmap = glyph.bitmap
    left = origin_x + glyph.pen_x + bitmap.bearing_x
    top = baseline_y - bitmap.bearing_y
    r, g, b = color
    for y in range(bitmap.height):
        canvas_y = top + y
        if not (0 <= canvas_y < image.height):
            continue
        row_base = y * bitmap.width
        for x in range(bitmap.width):
            coverage = bitmap.coverage[row_base + x]
            if coverage == 0:
                continue
            canvas_x = left + x
            if not (0 <= canvas_x < image.width):
                continue
            image.blend_pixel(canvas_x, canvas_y, r, g, b,
                              coverage * global_alpha // 255)


def render_text_png(config: WatermarkConfig, output_path: str) -> RenderedText:
    """按配置栅格化文字并写出 RGBA PNG。"""
    config.validate()
    codepoints = _codepoints_from_text(config.text)
    if not codepoints:
        raise WatermarkError("水印文本不能为空")

    stack = FontStack(list(config.font_files))
    color = parse_color(config.color)
    global_alpha = round(config.opacity * 255)

    placed, min_x, min_y, max_x, max_y = _measure(
        codepoints, stack, config.font_size
    )

    padding = config.padding
    canvas_w = (max_x - min_x) + 2 * padding
    canvas_h = (max_y - min_y) + 2 * padding
    if canvas_w > 16384 or canvas_h > 16384:
        raise WatermarkError(
            f"水印位图尺寸 {canvas_w}x{canvas_h} 超过 16384 上限，请缩短文本或减小字号"
        )

    image = Image(canvas_w, canvas_h)
    origin_x = padding - min_x
    baseline_y = padding - min_y
    for glyph in placed:
        _blit_glyph(image, glyph, origin_x, baseline_y, color, global_alpha)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        raise WatermarkError(f"水印输出目录不存在：{parent}")
    write_png(output_path, image)

    report = tuple((cp, g.face.kind) for cp, g in zip(codepoints, placed))
    return RenderedText(
        path=os.path.abspath(output_path), width=canvas_w, height=canvas_h,
        codepoint_count=len(codepoints), fallback_report=report,
    )
