"""GIF89a 编码器：纯 Python 字节封装 + LZW 压缩，零三方依赖。

只实现本项目需要的子集：单逻辑屏、一张全局颜色表、每帧一个图形控制
扩展（GCE，携带厘秒延迟）+ 一张基于全局表的图像。不做透明度、
交错与局部颜色表；每帧全画面重绘（disposal=0），时序完全由调用方
通过 :class:`vgif.timing.DelayPlan` 决定，本模块不改动任何延迟。

字节布局（小端序）：

1. 头部 ``GIF89a`` + 逻辑屏描述符 + 全局颜色表（条目数向上取 2 的幂）；
2. NETSCAPE2.0 应用扩展，开启无限循环；
3. 逐帧：GCE（delay 单位 1/100 秒）→ 图像描述符 → LZW 最小码长 →
   长度分块（每块 ≤255 字节）的 LZW 码流 → 块终止符 ``0x00``；
4.  trailer ``0x3B``。

LZW 采用 GIF 的变体：CLEAR=2**min_code_size、END=CLEAR+1，初始可用
码从 CLEAR+2 起；码表满 4096 前先发 CLEAR 重置；码位按 LSB 打包。
"""

from __future__ import annotations

import struct
from pathlib import Path

from .quantize import QuantizedFrame, next_pow2_colors

__all__ = ["GifEncodeError", "write_gif", "lzw_encode", "pad_palette"]


class GifEncodeError(ValueError):
    """GIF 编码参数非法。"""


#: LZW 码表硬上限（12 位码）。
_MAX_CODE = 4096


def pad_palette(palette: tuple) -> tuple:
    """把调色板补齐为 2 的幂长度并保证至少 2 条目。

    GIF 全局颜色表长度只能是 2/4/8/…/256，不足部分用最后一个条目
    重复填充（这些条目不会被索引引用，仅满足表长约束）。
    """
    if not palette:
        raise GifEncodeError("调色板为空")
    if len(palette) > 256:
        raise GifEncodeError(f"调色板条目超过 256：{len(palette)}")
    target = next_pow2_colors(len(palette))
    out = list(palette)
    while len(out) < target:
        out.append(palette[-1])
    return tuple(out)


def _color_table_size_flag(n_entries: int) -> int:
    """全局颜色表字段的尺寸标记 k：表长 = 2**(k+1)。"""
    k = 0
    while (1 << (k + 1)) < n_entries:
        k += 1
    return k


def lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    """GIF 变体 LZW 压缩。

    :param indices: 索引字节（每个值须 < 2**min_code_size）。
    :param min_code_size: LZW 最小码长（2..8；GIF 不允许小于 2）。
    :return: 已 LSB 打包的码流字节（不含分块头与块终止符）。
    """
    if not 2 <= min_code_size <= 8:
        raise GifEncodeError(f"LZW 最小码长须在 2..8：{min_code_size}")
    clear = 1 << min_code_size
    end = clear + 1
    table_limit = (1 << (min_code_size + 1))  # 当前码宽下的表上限

    def _reset_table():
        # 字符串表：前缀码+新字节 → 输出码。
        return {}, end + 1, min_code_size + 1, table_limit

    table, next_code, code_size, limit = _reset_table()

    # 码流以 CLEAR 开头、END 结尾；位缓冲按 LSB 顺序追加。
    bit_buf = 0
    bit_count = 0
    out = bytearray()

    def _emit(code: int, bit_buf: int, bit_count: int):
        bit_buf |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8
        return bit_buf, bit_count

    bit_buf, bit_count = _emit(clear, bit_buf, bit_count)

    prefix = int(indices[0])
    for b in indices[1:]:
        k = (prefix, b)
        if k in table:
            prefix = table[k]
            continue
        bit_buf, bit_count = _emit(prefix, bit_buf, bit_count)
        # 先登记再判断扩位/重置，保证 next_code 不越过 4096。
        if next_code < _MAX_CODE:
            table[k] = next_code
            next_code += 1
            if next_code > limit and code_size < 12:
                code_size += 1
                limit = 1 << code_size
        else:
            # 表满：发 CLEAR 并重置（当前字符串作为新起点继续）。
            bit_buf, bit_count = _emit(clear, bit_buf, bit_count)
            table, next_code, code_size, limit = _reset_table()
        prefix = b

    bit_buf, bit_count = _emit(prefix, bit_buf, bit_count)
    bit_buf, bit_count = _emit(end, bit_buf, bit_count)
    if bit_count > 0:
        out.append(bit_buf & 0xFF)
    return bytes(out)


def _subblocks(data: bytes) -> bytes:
    """把码流切成每块 ≤255 字节的 GIF 数据子块并追加块终止符。"""
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i:i + 255]
        out.append(len(chunk))
        out.extend(chunk)
    out.append(0x00)
    return bytes(out)


def write_gif(path: str | Path, qframes: list, palette: tuple,
              delays_cs: tuple, width: int, height: int,
              min_code_size: int = 8) -> Path:
    """把量化帧序列与厘秒延迟写成 GIF89a 文件。

    :param qframes: :class:`QuantizedFrame` 列表（几何尺寸须一致）。
    :param palette: 调色板（条目 ≤256，内部自动补齐到 2 的幂）。
    :param delays_cs: 逐帧延迟（厘秒），长度须与 ``qframes`` 相同；
        本函数原样写入，不做任何钳制或重算。
    :param min_code_size: LZW 最小码长，默认 8（256 色表）；调色板
        较小时调用方可按颜色表位宽收窄以提升压缩率。
    :return: 输出文件路径。
    """
    if not qframes:
        raise GifEncodeError("没有可写的帧")
    if len(qframes) != len(delays_cs):
        raise GifEncodeError(
            f"帧数 {len(qframes)} 与延迟数 {len(delays_cs)} 不一致"
        )
    for i, q in enumerate(qframes):
        if not isinstance(q, QuantizedFrame):
            raise GifEncodeError(f"第 {i} 帧不是 QuantizedFrame")
        if q.width != width or q.height != height:
            raise GifEncodeError(
                f"第 {i} 帧尺寸 {q.width}x{q.height} 与逻辑屏 "
                f"{width}x{height} 不一致"
            )
    for d in delays_cs:
        if not 0 <= d <= 0xFFFF:
            raise GifEncodeError(f"延迟 {d} 超出 GIF 厘秒字段范围 0..65535")

    full_palette = pad_palette(palette)
    k = _color_table_size_flag(len(full_palette))
    # min_code_size 须覆盖实际索引位宽，且 GIF 规定不小于 2。
    table_bits = k + 1
    min_code_size = max(2, min(min_code_size, 8))
    if min_code_size < table_bits:
        min_code_size = table_bits

    out = bytearray()
    # 头部 + 逻辑屏描述符。
    out.extend(b"GIF89a")
    packed = 0x80 | (k << 4) | k   # GCT 存在、无色彩排序、底色 0
    out.extend(struct.pack("<HH", width, height))
    out.append(packed)
    out.append(0)                  # 背景色索引
    out.append(0)                  # 像素宽高比（0 表示无指定）

    # 全局颜色表：每色 RGB 三字节。
    for r, g, b in full_palette:
        out.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    # NETSCAPE2.0 应用扩展：无限循环（loop=0）。
    out.extend(b"\x21\xFF\x0BNETSCAPE2.0")
    out.extend(b"\x03\x01\x00\x00\x00")

    for q, delay in zip(qframes, delays_cs):
        if max(q.indices) >= (1 << table_bits):
            raise GifEncodeError(
                "帧索引超出全局颜色表位宽，调色板与量化帧不匹配"
            )
        # 图形控制扩展：disposal=0、无透明色、user input=0。
        out.extend(b"\x21\xF9\x04")
        out.append(0x00)
        out.extend(struct.pack("<H", delay))
        out.append(0x00)           # 透明色索引（透明关闭时占位）
        out.append(0x00)           # 块终止
        # 图像描述符。
        out.append(0x2C)
        out.extend(struct.pack("<HHHH", 0, 0, q.width, q.height))
        out.append(0x00)           # 无局部颜色表、逐行、像素顺序
        # LZW 码流。
        out.append(min_code_size)
        out.extend(_subblocks(lzw_encode(q.indices, min_code_size)))

    out.append(0x3B)               # trailer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path
