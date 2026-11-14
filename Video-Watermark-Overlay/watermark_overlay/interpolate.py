"""双线性插值缩放（在预乘 alpha 空间进行）。

直接对**直接 alpha** 的 RGBA 做双线性会在透明边缘混入黑色（透明像素的 RGB
不携带颜色信息），因此这里先把每个采样点预乘，再对 R/G/B/A 四个平面分别
双线性插值，最后反预乘回去。采样坐标使用“像素角点对齐”映射
``src = (dst + 0.5) * scale - 0.5``，越界采样坐标 clamp 到边界。

纯 float 运算 + 手写四邻域加权，不调用任何图像处理库。
"""

from __future__ import annotations

import math

from .errors import InvalidParameterError
from .image import Image


def bilinear_resize(image: Image, dst_width: int, dst_height: int) -> Image:
    """返回缩放到 ``dst_width x dst_height`` 的新 :class:`Image`。"""
    if not isinstance(dst_width, int) or not isinstance(dst_height, int):
        raise TypeError("目标宽高必须为整数")
    if dst_width <= 0 or dst_height <= 0:
        raise InvalidParameterError("目标宽高必须为正整数")
    if dst_width == image.width and dst_height == image.height:
        return image.copy()

    src_w, src_h = image.width, image.height
    src = image.pixels
    out = Image(dst_width, dst_height)
    dst = out.pixels

    scale_x = src_w / dst_width
    scale_y = src_h / dst_height

    for dy in range(dst_height):
        fy = (dy + 0.5) * scale_y - 0.5
        # 负坐标必须用 floor：int() 会向 0 截断，导致权重与邻居错位
        y0 = math.floor(fy)
        wy = fy - y0
        # 边界 clamp：上/下采样都不会读到缓冲外
        y0 = max(0, min(y0, src_h - 1))
        y1 = min(y0 + 1, src_h - 1)

        for dx in range(dst_width):
            fx = (dx + 0.5) * scale_x - 0.5
            x0 = math.floor(fx)
            wx = fx - x0
            x0 = max(0, min(x0, src_w - 1))
            x1 = min(x0 + 1, src_w - 1)

            i00 = (y0 * src_w + x0) * 4
            i10 = (y0 * src_w + x1) * 4
            i01 = (y1 * src_w + x0) * 4
            i11 = (y1 * src_w + x1) * 4

            w00 = (1.0 - wx) * (1.0 - wy)
            w10 = wx * (1.0 - wy)
            w01 = (1.0 - wx) * wy
            w11 = wx * wy

            # alpha 直接插值；RGB 先预乘再插值
            alpha = 0.0
            premul = [0.0, 0.0, 0.0]
            for base, weight in ((i00, w00), (i10, w10), (i01, w01), (i11, w11)):
                a = src[base + 3]
                alpha += a * weight
                for c in range(3):
                    premul[c] += src[base + c] * a / 255.0 * weight

            out_a = max(0, min(255, int(alpha + 0.5)))
            di = (dy * dst_width + dx) * 4
            dst[di + 3] = out_a
            if out_a == 0:
                dst[di] = dst[di + 1] = dst[di + 2] = 0
                continue

            for c in range(3):
                # 反预乘
                value = premul[c] * 255.0 / out_a
                dst[di + c] = max(0, min(255, int(value + 0.5)))

    return out


def resolve_target_size(
    image: Image,
    target_width: int | None,
    target_height: int | None,
) -> tuple[int, int]:
    """根据宽/高配置解析最终尺寸；只给一边时按原宽高比推算另一边。"""
    if target_width is None and target_height is None:
        return image.width, image.height
    if target_width is not None and target_height is not None:
        return target_width, target_height
    if target_width is not None:
        ratio = target_width / image.width
        return target_width, max(1, round(image.height * ratio))
    ratio = target_height / image.height
    return max(1, round(image.width * ratio)), target_height
