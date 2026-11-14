"""扫描线栅格化器：TrueType 轮廓多边形 -> 带抗锯齿的字形位图。

1. 设计单位轮廓统一缩放到 ``像素尺寸 * SUPERSAMPLE`` 的超采样坐标；
2. 每条扫描线求与全部轮廓边的交点，排序后做奇偶环绕（even-odd）填充；
3. 每个目标像素内 SUPERSAMPLE^2 个子采样平均为 0~255 的覆盖率 alpha。
"""

from __future__ import annotations

from dataclasses import dataclass

from .ttf_font import TrueTypeFont

#: 每像素每方向的超采样数（4 => 16 个子像素/像素）
SUPERSAMPLE = 4
_SAMPLES = SUPERSAMPLE * SUPERSAMPLE


@dataclass(frozen=True)
class GlyphBitmap:
    """栅格化后的单字形位图（alpha 单通道，行优先）。"""

    width: int
    height: int
    coverage: bytearray
    bearing_x: int
    bearing_y: int
    advance: int


def _scale_outlines(font, outlines, pixel_size):
    factor = pixel_size * SUPERSAMPLE / font.units_per_em
    return [[(x * factor, -y * factor) for x, y in contour] for contour in outlines]


def _bounds(polygons):
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _scanline_intersections(polygons, y):
    """扫描线 y 与所有多边形边的交点；下闭上开避免水平顶点重复计数。"""
    intersections = []
    for polygon in polygons:
        n = len(polygon)
        for i in range(n):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % n]
            if y0 == y1:
                continue
            ylo, yhi = (y0, y1) if y0 < y1 else (y1, y0)
            if ylo <= y < yhi:
                t = (y - y0) / (y1 - y0)
                intersections.append(x0 + t * (x1 - x0))
    intersections.sort()
    return intersections


def rasterize_glyph(font: TrueTypeFont, glyph_id: int, pixel_size: int) -> GlyphBitmap:
    """把单个字形栅格化为超采样抗锯齿位图；空字形返回全透明位图。"""
    advance_units = font.advance_width(glyph_id)
    advance_px = max(1, round(advance_units * pixel_size / font.units_per_em))

    outlines = font.glyph_outlines(glyph_id)
    if not outlines:
        return GlyphBitmap(
            width=advance_px, height=pixel_size,
            coverage=bytearray(advance_px * pixel_size),
            bearing_x=0, bearing_y=0, advance=advance_px,
        )

    scaled = _scale_outlines(font, outlines, pixel_size)
    min_x, min_y, max_x, max_y = _bounds(scaled)

    ss_x0 = int(min_x) - 1
    ss_y0 = int(min_y) - 1
    ss_x1 = int(max_x) + 2
    ss_y1 = int(max_y) + 2
    ss_w = max(1, ss_x1 - ss_x0)
    ss_h = max(1, ss_y1 - ss_y0)
    ss_grid = bytearray(ss_w * ss_h)

    for row in range(ss_h):
        y = ss_y0 + row + 0.5
        hits = _scanline_intersections(scaled, y)
        for pair in range(0, len(hits) - 1, 2):
            cx0 = max(ss_x0, int(hits[pair]))
            cx1 = min(ss_x1, int(hits[pair + 1]) + 1)
            for x in range(cx0, cx1):
                ss_grid[row * ss_w + (x - ss_x0)] = 1

    out_w = (ss_w + SUPERSAMPLE - 1) // SUPERSAMPLE
    out_h = (ss_h + SUPERSAMPLE - 1) // SUPERSAMPLE
    coverage = bytearray(out_w * out_h)
    for oy in range(out_h):
        for ox in range(out_w):
            hit = 0
            for sy in range(SUPERSAMPLE):
                ss_row = oy * SUPERSAMPLE + sy
                if ss_row >= ss_h:
                    break
                base = ss_row * ss_w + ox * SUPERSAMPLE
                for sx in range(SUPERSAMPLE):
                    ss_col = ox * SUPERSAMPLE + sx
                    if ss_col < ss_w and ss_grid[base + sx]:
                        hit += 1
            coverage[oy * out_w + ox] = round(255 * hit / _SAMPLES)

    return GlyphBitmap(
        width=out_w, height=out_h, coverage=coverage,
        bearing_x=ss_x0 // SUPERSAMPLE,
        bearing_y=max(1, (-ss_y0 + SUPERSAMPLE - 1) // SUPERSAMPLE),
        advance=advance_px,
    )
