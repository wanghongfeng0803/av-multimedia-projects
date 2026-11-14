"""基于帧内容的自适应蒙版。

思路：水印越靠近“关键区域”（强边缘、明暗突变处，通常是人脸轮廓、文字、
物体边界），透明度压得越低，避免遮挡画面主体。对每个帧像素计算两路信号：

1. **边缘强度**：在 Rec.601 亮度平面上跑 3x3 Sobel，梯度幅值归一化到 0..255；
2. **局部亮度变化**：用积分图求邻域均值，像素与邻域均值的偏差越大，
   说明该位置越不平坦（纹理/光斑/暗角边缘同样会被捕捉）。

两路取最大值得到细节图 ``detail``，再按线性区间映射为透明度系数::

    t        = clamp((detail - detail_low) / (detail_high - detail_low), 0, 1)
    alpha'   = base_alpha * (1 - strength * t)

``strength=0`` 时退化为固定透明度；平坦区域（detail <= detail_low）保持
原始透明度，细节越强的区域透明度越低。
"""

from __future__ import annotations

from .image import Image

# Rec.601 亮度系数（和为 256，右移 8 即可）
_LUMA_R = 77
_LUMA_G = 150
_LUMA_B = 29


def luma(r: int, g: int, b: int) -> int:
    """Rec.601 亮度，返回 0..255。"""
    return (_LUMA_R * r + _LUMA_G * g + _LUMA_B * b) >> 8


def _luma_plane(frame: Image) -> bytearray:
    plane = bytearray(frame.width * frame.height)
    p = frame.pixels
    for i in range(frame.width * frame.height):
        j = i * 4
        plane[i] = luma(p[j], p[j + 1], p[j + 2])
    return plane


def _integral_image(plane: bytearray, width: int, height: int) -> list[int]:
    """标准二维积分图（含一行/一列零边），尺寸 (w+1)*(h+1)。"""
    stride = width + 1
    integral = [0] * (stride * (height + 1))
    for y in range(height):
        row_sum = 0
        src_row = y * width
        dst_row = (y + 1) * stride
        prev_row = y * stride
        for x in range(width):
            row_sum += plane[src_row + x]
            integral[dst_row + x + 1] = integral[prev_row + x + 1] + row_sum
    return integral


def build_detail_map(frame: Image, radius: int = 2) -> bytearray:
    """生成与帧等大的细节强度图，取值 0..255。"""
    width, height = frame.width, frame.height
    plane = _luma_plane(frame)
    integral = _integral_image(plane, width, height)
    detail = bytearray(width * height)
    stride = width + 1

    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height - 1, y + radius)
        for x in range(width):
            idx = y * width + x
            cur = plane[idx]

            # Sobel 3x3（越界按边界像素 clamp 复制，避免平坦图在帧边缘产生假梯度）
            gx = gy = 0
            for ky in (-1, 0, 1):
                sy = y + ky
                if sy < 0:
                    sy = 0
                elif sy >= height:
                    sy = height - 1
                for kx in (-1, 0, 1):
                    if kx == 0:
                        continue
                    sx = x + kx
                    if sx < 0:
                        sx = 0
                    elif sx >= width:
                        sx = width - 1
                    gx += kx * plane[sy * width + sx]
                    if ky != 0:
                        gy += ky * plane[sy * width + sx]
            # 最大可能幅值约为 8*255=2040，除以 8 归一化
            edge = min(255, (abs(gx) + abs(gy)) >> 3)

            # 局部亮度偏差（积分图 O(1) 求邻域均值）
            x0 = max(0, x - radius)
            x1 = min(width - 1, x + radius)
            box_area = (y1 - y0 + 1) * (x1 - x0 + 1)
            total = (
                integral[(y1 + 1) * stride + (x1 + 1)]
                - integral[y0 * stride + (x1 + 1)]
                - integral[(y1 + 1) * stride + x0]
                + integral[y0 * stride + x0]
            )
            local_mean = total // box_area
            variation = min(255, abs(cur - local_mean) * 2)

            detail[idx] = max(edge, variation)

    return detail


def build_alpha_map(
    frame: Image,
    origin_x: int,
    origin_y: int,
    watermark_width: int,
    watermark_height: int,
    base_alpha: int,
    strength: float,
    detail_low: int,
    detail_high: int,
    radius: int = 2,
) -> bytearray:
    """生成帧大小的源 alpha 系数图（0..255），水印矩形外恒为 0。

    :param base_alpha: 全局透明度对应的 0..255 系数
    :param strength: 0.0~1.0，关键区域最大压暗比例
    :param detail_low/high: 细节强度的线性映射区间
    """
    width, height = frame.width, frame.height
    alpha_map = bytearray(width * height)

    x0 = max(0, origin_x)
    y0 = max(0, origin_y)
    x1 = min(width, origin_x + watermark_width)
    y1 = min(height, origin_y + watermark_height)
    if x0 >= x1 or y0 >= y1:
        return alpha_map

    span = detail_high - detail_low
    if span <= 0:
        span = 1  # 配置层本会拒绝，这里防御性兜底避免除零

    if strength <= 0.0:
        # 快速路径：无自适应，矩形内统一系数
        for y in range(y0, y1):
            row = y * width
            for x in range(x0, x1):
                alpha_map[row + x] = base_alpha
        return alpha_map

    detail = build_detail_map(frame, radius=radius)
    for y in range(y0, y1):
        row = y * width
        for x in range(x0, x1):
            d = detail[row + x]
            if d <= detail_low:
                t = 0.0
            elif d >= detail_high:
                t = 1.0
            else:
                t = (d - detail_low) / span
            factor = 1.0 - strength * t
            alpha_map[row + x] = max(0, round(base_alpha * factor))

    return alpha_map
