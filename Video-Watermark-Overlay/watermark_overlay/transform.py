"""旋转 + 亚像素平移的仿射重采样（预乘 alpha 空间双线性）。

正向映射：源图像素 p 先绕锚点 a 旋转 angle 度，再平移 (dx, dy)::

    p' = a + R · (p - a) + d

渲染采用**逆映射**：对每个目标像素中心 q，求其在源图中的对应点
``p = a + R⁻¹ · (q - a - d)``，再做双线性采样。与缩放一样，RGB 先在
预乘 alpha 空间加权（越界邻居按全透明 (0,0,0,0) 参与，而不是 clamp
到边缘），因此：

- 旋转产生的三角形空白区自然全透明；
- 半透明边缘不会混入黑色（黑边）或产生锯齿状色偏；
- 亚像素平移（dx/dy 的小数部分）被精确烘焙进采样坐标。

画布原点对齐：旋转后的包围盒角点一般不是整数，这里把画布原点取为
角点最小值的 floor（整数），采样时以同一点为参考，因此取整不会
引入任何定位误差——锚点在新画布中的浮点位置恒为
``(anchor_x + dx - offset_x, anchor_y + dy - offset_y)``。
"""

from __future__ import annotations

import math

from .animation import normalize_angle
from .errors import InvalidParameterError
from .image import Image


def _check_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidParameterError(f"{name} 必须为数值")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidParameterError(f"{name} 必须为有限数，收到 {value}")
    return value


def _rotation_trig(angle_deg: float) -> tuple[float, float]:
    """返回 (cos, sin)；90° 的整数倍查表取精确值。

    ``math.cos(math.radians(90))`` 会留下 6e-17 的尾数，足以让
    包围盒角点越过整数边界、画布凭空膨胀 1 像素，因此轴向旋转
    直接使用精确的 0/±1。
    """
    if angle_deg % 90 == 0:
        quadrant = int(angle_deg // 90) % 4
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quadrant]
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def rotated_bounds(
    width: int,
    height: int,
    angle_deg: float,
    anchor_x: float,
    anchor_y: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[int, int, int, int]:
    """返回旋转+平移后的包围盒 ``(offset_x, offset_y, new_width, new_height)``。

    offset 为新画布原点相对原图左上角的整数位移（角点最小值的 floor），
    画布尺寸保证完整覆盖旋转后的内容。
    """
    angle = normalize_angle(_check_finite("angle_deg", angle_deg))
    anchor_x = _check_finite("anchor_x", anchor_x)
    anchor_y = _check_finite("anchor_y", anchor_y)
    dx = _check_finite("dx", dx)
    dy = _check_finite("dy", dy)

    cos_a, sin_a = _rotation_trig(angle)

    xs = []
    ys = []
    for cx, cy in ((0.0, 0.0), (float(width), 0.0), (0.0, float(height)), (float(width), float(height))):
        rx = anchor_x + cos_a * (cx - anchor_x) - sin_a * (cy - anchor_y) + dx
        ry = anchor_y + sin_a * (cx - anchor_x) + cos_a * (cy - anchor_y) + dy
        xs.append(rx)
        ys.append(ry)

    offset_x = math.floor(min(xs))
    offset_y = math.floor(min(ys))
    new_width = max(1, math.ceil(max(xs) - offset_x))
    new_height = max(1, math.ceil(max(ys) - offset_y))
    return offset_x, offset_y, new_width, new_height


def warp_watermark(
    image: Image,
    angle_deg: float,
    anchor_x: float,
    anchor_y: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[Image, int, int]:
    """绕锚点旋转 ``angle_deg`` 度并平移 ``(dx, dy)``，返回 (新图像, ox, oy)。

    :param anchor_x/y: 旋转锚点在源图中的像素坐标（浮点，几何中心为
        ``(width / 2, height / 2)``）
    :param dx/dy: 旋转后的平移量（浮点，小数部分即亚像素位移）
    :return: (变换后的图像, 新画布原点相对原图左上角的整数偏移)。
        角度为 0 且位移为 0 时直接返回原图拷贝与零偏移（退化为静态）。
    """
    angle = normalize_angle(_check_finite("angle_deg", angle_deg))
    anchor_x = _check_finite("anchor_x", anchor_x)
    anchor_y = _check_finite("anchor_y", anchor_y)
    dx = _check_finite("dx", dx)
    dy = _check_finite("dy", dy)

    # 退化快速路径：无旋转、无位移 —— 逐字节等于原静态叠加的输入
    if angle == 0.0 and dx == 0.0 and dy == 0.0:
        return image.copy(), 0, 0

    offset_x, offset_y, new_w, new_h = rotated_bounds(
        image.width, image.height, angle, anchor_x, anchor_y, dx, dy
    )

    cos_a, sin_a = _rotation_trig(angle)

    src_w, src_h = image.width, image.height
    src = image.pixels
    out = Image(new_w, new_h)
    dst = out.pixels

    for dy_px in range(new_h):
        # 目标像素中心在“旋转+平移后”坐标系中的位置
        qy = dy_px + 0.5 + offset_y
        # 逆平移
        ry = qy - anchor_y - dy
        for dx_px in range(new_w):
            qx = dx_px + 0.5 + offset_x
            rx = qx - anchor_x - dx
            # 逆旋转（R⁻¹ = 旋转 -angle）
            px = anchor_x + cos_a * rx + sin_a * ry
            py = anchor_y - sin_a * rx + cos_a * ry

            # 双线性：像素中心位于整数 + 0.5，负坐标必须用 floor
            sx = px - 0.5
            sy = py - 0.5
            x0 = math.floor(sx)
            y0 = math.floor(sy)
            wx = sx - x0
            wy = sy - y0

            alpha = 0.0
            premul_r = premul_g = premul_b = 0.0
            for ky in (0, 1):
                ny = y0 + ky
                if ny < 0 or ny >= src_h:
                    continue  # 越界邻居按全透明参与（预乘空间即全零）
                w_y = (1.0 - wy) if ky == 0 else wy
                row = ny * src_w
                for kx in (0, 1):
                    nx = x0 + kx
                    if nx < 0 or nx >= src_w:
                        continue
                    weight = w_y * ((1.0 - wx) if kx == 0 else wx)
                    base = (row + nx) * 4
                    a = src[base + 3]
                    alpha += a * weight
                    scale = a / 255.0 * weight
                    premul_r += src[base] * scale
                    premul_g += src[base + 1] * scale
                    premul_b += src[base + 2] * scale

            out_a = max(0, min(255, int(alpha + 0.5)))
            di = (dy_px * new_w + dx_px) * 4
            dst[di + 3] = out_a
            if out_a == 0:
                # 全透明像素 RGB 归零，避免脏数据影响后续预乘运算
                dst[di] = dst[di + 1] = dst[di + 2] = 0
                continue
            # 反预乘回直接 alpha
            dst[di] = max(0, min(255, int(premul_r * 255.0 / out_a + 0.5)))
            dst[di + 1] = max(0, min(255, int(premul_g * 255.0 / out_a + 0.5)))
            dst[di + 2] = max(0, min(255, int(premul_b * 255.0 / out_a + 0.5)))

    return out, offset_x, offset_y
