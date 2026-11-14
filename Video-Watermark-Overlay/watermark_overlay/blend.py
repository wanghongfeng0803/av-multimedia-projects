"""预乘 alpha 的 src-over 逐像素混合。

合成公式（Porter-Duff "source over destination"，在**预乘 alpha** 空间）::

    out_a   = a_s + a_d * (1 - a_s)
    out_rgb = rgb_s + rgb_d * (1 - a_s)        # rgb 均为预乘值

输出时再反预乘回直接 alpha 存储。全程使用 0..255 定点整数运算
（除以 255 用 ``(x + 127) // 255`` 四舍五入），不依赖任何合成库。
"""

from __future__ import annotations

from .image import Image

# 定点比例分母：alpha 以 1/255 为最小单位
SCALE = 255
_HALF = 127


def src_over_pixel(
    dst: tuple[int, int, int, int],
    src: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """对单个像素做 src-over 合成，入参/出参均为直接 alpha 的 RGBA 元组。"""
    dr, dg, db, da = dst
    sr, sg, sb, sa = src

    # 全透明源：目标原样保留（最常见的快速路径）
    if sa == 0:
        return dst

    # 预乘
    spr = (sr * sa + _HALF) // SCALE
    spg = (sg * sa + _HALF) // SCALE
    spb = (sb * sa + _HALF) // SCALE
    dpr = (dr * da + _HALF) // SCALE
    dpg = (dg * da + _HALF) // SCALE
    dpb = (db * da + _HALF) // SCALE

    out_a = sa + da - (sa * da + _HALF) // SCALE
    if out_a == 0:
        return 0, 0, 0, 0

    inv_sa = SCALE - sa
    out_pr = spr + (dpr * inv_sa + _HALF) // SCALE
    out_pg = spg + (dpg * inv_sa + _HALF) // SCALE
    out_pb = spb + (dpb * inv_sa + _HALF) // SCALE

    # 反预乘：out_rgb = out_premultiplied / out_a
    out_r = (out_pr * SCALE + _HALF) // out_a
    out_g = (out_pg * SCALE + _HALF) // out_a
    out_b = (out_pb * SCALE + _HALF) // out_a
    return out_r, out_g, out_b, out_a


def apply_watermark(
    frame: Image,
    watermark: Image,
    origin_x: int,
    origin_y: int,
    alpha_map: bytearray,
) -> int:
    """把水印混合到帧上，原位修改 ``frame`` 并返回实际混合的像素数。

    :param origin_x/y: 水印左上角在帧坐标系中的位置，允许为负或超出帧边界，
        只有落在帧内的部分会被合成（逐像素裁剪，无需分配越界缓冲）。
    :param alpha_map: 长度 ``frame.width * frame.height`` 的字节数组，
        每个帧像素一个 0..255 的源 alpha 缩放系数
        （全局透明度与自适应蒙版的乘积在调用前合并好）。
    """
    fw, fh = frame.width, frame.height
    x0 = max(0, origin_x)
    y0 = max(0, origin_y)
    x1 = min(fw, origin_x + watermark.width)
    y1 = min(fh, origin_y + watermark.height)
    if x0 >= x1 or y0 >= y1:
        return 0

    blended = 0
    fp = frame.pixels
    wp = watermark.pixels
    ww = watermark.width

    for y in range(y0, y1):
        sy = y - origin_y
        row_frame = y * fw
        row_wm = sy * ww
        for x in range(x0, x1):
            multiplier = alpha_map[row_frame + x]
            if multiplier == 0:
                continue

            sx = x - origin_x
            wi = (row_wm + sx) * 4
            sa = wp[wi + 3]
            if multiplier != SCALE:
                sa = (sa * multiplier + _HALF) // SCALE
            if sa == 0:
                continue

            fi = (row_frame + x) * 4
            result = src_over_pixel(
                (fp[fi], fp[fi + 1], fp[fi + 2], fp[fi + 3]),
                (wp[wi], wp[wi + 1], wp[wi + 2], sa),
            )
            fp[fi], fp[fi + 1], fp[fi + 2], fp[fi + 3] = result
            blended += 1

    return blended
