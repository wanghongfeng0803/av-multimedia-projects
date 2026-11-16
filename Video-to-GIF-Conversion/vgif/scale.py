"""纯 Python 缩放：RGB 双线性（定点可复现）与索引帧最近邻。

两条路径对应三步解耦里的两种复用方式：

- :func:`bilinear_resize` 在 **RGB 域**工作，缩放后须重新量化，画质好，
  作为默认路径；坐标映射用 1/65536 定点，全整数插值，跨平台逐字节
  可复现（与 transition 项目 resample 的定点口径一致）。
- :func:`nearest_resize_indexed` 在 **索引域**工作，直接对调色板下标
  取最近邻，无需重新量化、零颜色损失且极快，供已量化产物改尺寸时
  复用（适合像素风 / 图表类素材）。
"""

from __future__ import annotations

__all__ = ["bilinear_resize", "nearest_resize_indexed"]

#: 定点小数位数（2^16），坐标与权重都放大到该尺度做整数运算。
SHIFT = 16
SCALE = 1 << SHIFT


def _round_shift2(x: int) -> int:
    """右移 2*SHIFT 位并半值向上取整。

    双线性的两个权重各为 16 位定点，相乘后权重和是 32 位定点
    （SCALE²），故累计值要右移 32 位而非 16 位，否则结果放大 65536
    倍溢出字节。
    """
    return (x + (1 << (2 * SHIFT - 1))) >> (2 * SHIFT)


def bilinear_resize(frame: bytes, src_w: int, src_h: int,
                    dst_w: int, dst_h: int) -> bytes:
    """把一帧 rgb24 字节双线性缩放到 ``dst_w × dst_h``。

    采用「角点对齐」坐标映射：输出像素中心 ``(dx+0.5)/dst_w`` 映回
    源图 ``sx = (dx+0.5)*src_w/dst_w - 0.5``，越界坐标钳到边界。
    权重以 1/65536 定点表示，插值结果四舍五入到整数，恒等尺寸
    （src==dst）时逐字节等于输入。

    :param frame: 长度 ``src_w*src_h*3`` 的 rgb24 字节。
    :return: 长度 ``dst_w*dst_h*3`` 的 rgb24 字节。
    """
    if len(frame) != src_w * src_h * 3:
        raise ValueError(
            f"帧长度 {len(frame)} 与 {src_w}x{src_h} rgb24 不符"
        )
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"目标尺寸必须为正：{dst_w}x{dst_h}")
    if dst_w == src_w and dst_h == src_h:
        return bytes(frame)

    out = bytearray(dst_w * dst_h * 3)
    for dy in range(dst_h):
        # 映回源坐标（定点）。
        sy_num = (2 * dy + 1) * src_h - dst_h       # * dst_h 后的分子
        sy_q = sy_num * SCALE // (2 * dst_h)        # 源 y 定点坐标
        y0 = sy_q >> SHIFT
        wy = sy_q & (SCALE - 1)
        if y0 < 0:
            y0, wy = 0, 0
        y1 = y0 + 1
        if y1 >= src_h:
            y1 = y0
            wy = 0
        row0 = y0 * src_w * 3
        row1 = y1 * src_w * 3

        for dx in range(dst_w):
            sx_num = (2 * dx + 1) * src_w - dst_w
            sx_q = sx_num * SCALE // (2 * dst_w)
            x0 = sx_q >> SHIFT
            wx = sx_q & (SCALE - 1)
            if x0 < 0:
                x0, wx = 0, 0
            x1 = x0 + 1
            if x1 >= src_w:
                x1 = x0
                wx = 0

            i00 = row0 + x0 * 3
            i01 = row0 + x1 * 3
            i10 = row1 + x0 * 3
            i11 = row1 + x1 * 3
            # 双线性权重（定点）：w00=(1-wx)(1-wy) 等，和约为 SCALE。
            w00 = (SCALE - wx) * (SCALE - wy)
            w01 = wx * (SCALE - wy)
            w10 = (SCALE - wx) * wy
            w11 = wx * wy
            op = (dy * dst_w + dx) * 3
            for c in range(3):
                v = (
                    frame[i00 + c] * w00
                    + frame[i01 + c] * w01
                    + frame[i10 + c] * w10
                    + frame[i11 + c] * w11
                )
                out[op + c] = _round_shift2(v)
    return bytes(out)


def nearest_resize_indexed(indices: bytes, src_w: int, src_h: int,
                           dst_w: int, dst_h: int) -> bytes:
    """对调色板索引帧做最近邻缩放（不经过 RGB、不重新量化）。

    索引值是调色板下标而非颜色分量，任何加权混合都会造出不存在的
    下标，因此只做最近邻；角点对齐映射保证恒等尺寸时逐字节不变。

    :param indices: 长度 ``src_w*src_h`` 的索引字节（每像素 1 字节）。
    :return: 长度 ``dst_w*dst_h`` 的索引字节。
    """
    if len(indices) != src_w * src_h:
        raise ValueError(
            f"索引帧长度 {len(indices)} 与 {src_w}x{src_h} 不符"
        )
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"目标尺寸必须为正：{dst_w}x{dst_h}")
    if dst_w == src_w and dst_h == src_h:
        return bytes(indices)

    out = bytearray(dst_w * dst_h)
    for dy in range(dst_h):
        sy_num = (2 * dy + 1) * src_h - dst_h
        sy_q = sy_num // (2 * dst_h)
        y = sy_q if sy_q >= 0 else 0
        if y >= src_h:
            y = src_h - 1
        for dx in range(dst_w):
            sx_num = (2 * dx + 1) * src_w - dst_w
            sx_q = sx_num // (2 * dst_w)
            x = sx_q if sx_q >= 0 else 0
            if x >= src_w:
                x = src_w - 1
            out[dy * dst_w + dx] = indices[y * src_w + x]
    return bytes(out)
