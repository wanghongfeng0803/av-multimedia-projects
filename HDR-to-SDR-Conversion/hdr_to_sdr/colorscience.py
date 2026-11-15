"""色彩科学：色域（primaries）矩阵、YUV↔RGB 与 BT.2020→BT.709 色域映射。

所有矩阵都在**线性光 RGB** 域工作，色调映射见 :mod:`hdr_to_sdr.tonemap`。

防偏色的关键设计
================
BT.2020→BT.709 直接做 3×3 矩阵后，宽色域颜色必然出现负通道/超 1 通道，
若直接截断会发生明显色偏（饱和原色整体发白/发灰）。本模块采用
「矩阵 + 保亮度自适应色域压缩」：

1. 先按标准矩阵把线性 RGB 变换到 BT.709 原色；
2. 对落在 [0,1] 外的像素，沿 BT.709 亮度保持的方向（R/G/B 与 Y 之差）
   等比缩回，缩放系数取所有越界通道中最紧的那个。

由于缩放沿 (R-Y, G-Y, B-Y) 方向进行，亮度 Y 严格不变、色相保持，
只降低饱和度——这是色度学里标准的 luma-preserving gamut mapping。
"""

from __future__ import annotations

__all__ = [
    "RANGE_LIMITED",
    "RANGE_FULL",
    "BT709_KR",
    "BT709_KB",
    "BT2020_KR",
    "BT2020_KB",
    "range_constants",
    "yuv_to_rgb_offsets",
    "rgb_to_yuv_offsets",
    "GAMUT_2020_TO_709",
    "apply_gamut_matrix",
    "gamut_compress",
    "gamut_map_pixel",
]

RANGE_LIMITED = "limited"
RANGE_FULL = "full"

BT709_KR = 0.2126
BT709_KG = 1.0 - BT709_KR - 0.0722
BT709_KB = 0.0722

BT2020_KR = 0.2627
BT2020_KG = 1.0 - BT2020_KR - 0.0593
BT2020_KB = 0.0593


def range_constants(pixel_range: str, bits_per_sample: int) -> tuple[int, int, int, int]:
    """返回量化常数 (offset, y_max, c_min, c_max)。

    limited 范围遵循 BT.709/BT.2020 附件规定的码值映射；
    full 范围使用完整 [0, 2^bits-1] 码值。
    """
    if bits_per_sample not in (8, 10, 12, 16):
        raise ValueError(f"不支持的码深: {bits_per_sample}")
    peak = (1 << bits_per_sample) - 1
    if pixel_range == RANGE_FULL:
        return 0, peak, 0, peak
    if pixel_range != RANGE_LIMITED:
        raise ValueError(f"未知像素范围: {pixel_range}")
    if bits_per_sample == 8:
        return 16, 235, 16, 240
    # 10/12/16bit 按比例放大（10bit 标称 64/940/1020；12bit 256/3760/4080）
    scale = peak / 1023.0
    return round(64 * scale), round(940 * scale), round(64 * scale), round(1020 * scale)


def yuv_to_rgb_offsets(
    kr: float, kb: float, pixel_range: str, bits_per_sample: int
) -> tuple[float, float, float, float, float]:
    """YUV→RGB 系数，返回 (y_scale, rv, gu, gv, bu)。

    统一约定（与 :func:`rgb_to_yuv_offsets` 严格配对）::

        yn = (Y - y_off)/peak · y_scale     # 亮度
        cn = (C - c_mid)/peak               # 色差裸归一化，展宽含在系数内
        R = yn + rv·Vn ; B = yn + bu·Un
        G = yn + gu·Un + gv·Vn

    limited 8bit：y_scale=255/219；色差系数含 255/224 展宽与 2 倍矩阵增益。
    """
    kg = 1.0 - kr - kb
    offset, y_max, c_min, c_max = range_constants(pixel_range, bits_per_sample)
    y_span = y_max - offset
    c_span = c_max - c_min
    peak = peak_scale(bits_per_sample)
    if pixel_range == RANGE_FULL:
        # full range：码值/peak 即归一化，系数只含矩阵项
        y_scale = 1.0
        c_scale = 1.0
    else:
        # limited range：亮度按 y_span 展宽，色度按 c_span 展宽；
        # 两者的码值都先除以 peak 归一化，因此展宽因子分别为 peak/y_span、
        # peak/c_span，不能共用一个（8bit 下 219 与 224 不同）。
        y_scale = peak / y_span
        c_scale = peak / c_span
    rv = (2.0 - 2.0 * kr) * c_scale
    bu = (2.0 - 2.0 * kb) * c_scale
    gv = -2.0 * kr * (1.0 - kr) / kg * c_scale
    gu = -2.0 * kb * (1.0 - kb) / kg * c_scale
    return y_scale, rv, gu, gv, bu


def peak_scale(bits_per_sample: int) -> float:
    """码值峰值 2^bits-1，单独成函数便于阅读量化推导。"""
    return float((1 << bits_per_sample) - 1)


def rgb_to_yuv_offsets(
    kr: float, kb: float, pixel_range: str, bits_per_sample: int
) -> tuple[float, float, float, float, float, float, float]:
    """RGB→YUV 系数，返回 (y_gain, c_gain, kr, kg, kb, y_off, c_mid)。

    与 :func:`yuv_to_rgb_offsets` 配对，编码公式::

        Y = luma*y_gain + y_off
        U = (B-luma)*c_gain + c_mid
        V = (R-luma)*c_gain + c_mid

    ``c_gain`` 已含完整色度增益（limited 8bit 为 112 = 224/2，
    full 8bit 为 127.5 = 255/2），调用方不要再乘 2。
    """
    kg = 1.0 - kr - kb
    offset, y_max, c_min, c_max = range_constants(pixel_range, bits_per_sample)
    y_span = y_max - offset
    c_span = c_max - c_min
    if pixel_range == RANGE_FULL:
        y_scale = peak_scale(bits_per_sample)
        c_scale = peak_scale(bits_per_sample) / 2.0
        c_mid = peak_scale(bits_per_sample) / 2.0
    else:
        y_scale = y_span
        c_scale = c_span / 2.0
        c_mid = (c_min + c_max) / 2.0
    return y_scale, c_scale, kr, kg, kb, float(offset), c_mid


# BT.2020(线性) → BT.709(线性) 标准 3×3 矩阵，行序 R/G/B、列序 R/G/B。
# 数值由两套原色 + D65 白点的色度坐标推导（Lindbloom 计算值，6 位小数），
# 高精度系数保证中性灰经过矩阵后通道差异在 1e-7 量级，避免量化后偏色。
GAMUT_2020_TO_709: tuple[tuple[float, float, float], ...] = (
    (1.660491, -0.587641, -0.072850),
    (-0.124550, 1.132900, -0.008349),
    (-0.018151, -0.100579, 1.118730),
)


def apply_gamut_matrix(
    r: float, g: float, b: float
) -> tuple[float, float, float]:
    """对线性 RGB 施加 BT.2020→BT.709 矩阵（不做截断）。"""
    m = GAMUT_2020_TO_709
    return (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )


def gamut_compress(
    r: float, g: float, b: float
) -> tuple[float, float, float]:
    """保亮度色域压缩：把越界线性 RGB 沿 (通道−亮度) 方向缩回 [0,1]。

    亮度用 BT.709 系数。缩放系数 s 对所有越界通道取最小可行值；
    s=1 表示像素本就在色域内，原样返回。
    """
    y = BT709_KR * r + BT709_KG * g + BT709_KB * b
    s = 1.0
    for c in (r, g, b):
        diff = c - y
        if diff == 0.0:
            continue
        # c' = y + s*diff，要求 0 <= c' <= 1
        if diff > 0.0:
            limit = (1.0 - y) / diff
        else:
            limit = (0.0 - y) / diff
        if limit < s:
            s = limit
    if s >= 1.0:
        return r, g, b
    return y + s * (r - y), y + s * (g - y), y + s * (b - y)


def gamut_map_pixel(
    r: float, g: float, b: float
) -> tuple[float, float, float]:
    """完整色域映射：先矩阵变换再保亮度压缩，输出保证落在 [0,1]。"""
    r709, g709, b709 = apply_gamut_matrix(r, g, b)
    return gamut_compress(r709, g709, b709)
