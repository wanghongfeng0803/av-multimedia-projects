"""RGBA 像素缓冲与自研 PNG 编解码。

本模块不依赖任何图像处理库：PNG 的 chunk 解析、zlib 解压、
5 种扫描线滤波器（None/Sub/Up/Average/Paeth）的逆滤波均自行实现，
像素以 RGBA8888 紧凑排列在 :class:`bytearray` 中，供后续逐像素运算。

仅覆盖本工具需要的 PNG 子集：8 位采样、非隔行、RGB(类型 2)/RGBA(类型 6)。
"""

from __future__ import annotations

import struct
import zlib

from .errors import ImageDecodeError

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# color_type -> 每像素通道数（仅支持 8 位深度）
_CHANNELS = {2: 3, 6: 4}
_FILTER_NONE = 0
_FILTER_SUB = 1
_FILTER_UP = 2
_FILTER_AVERAGE = 3
_FILTER_PAETH = 4


class Image:
    """一块内存中的 RGBA8888 位图。

    像素按行优先存储，每个像素占 4 字节（R、G、B、A），
    所有分量取值 0~255，alpha 为直接 alpha（非预乘）。
    """

    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int, pixels: bytearray | None = None):
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("宽高必须为整数")
        if width <= 0 or height <= 0:
            raise ValueError("宽高必须为正整数")
        expected = width * height * 4
        if pixels is None:
            pixels = bytearray(expected)
        else:
            if not isinstance(pixels, (bytearray, bytes)):
                raise TypeError("pixels 必须为 bytearray/bytes")
            if len(pixels) != expected:
                raise ValueError(
                    f"像素缓冲长度 {len(pixels)} 与 {width}x{height} 不匹配，"
                    f"应为 {expected}"
                )
            pixels = bytearray(pixels)
        self.width = width
        self.height = height
        self.pixels = pixels

    # ---- 基础访问 ----

    def index(self, x: int, y: int) -> int:
        """返回像素 (x, y) 在缓冲中的起始字节偏移。"""
        return (y * self.width + x) * 4

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int, int]:
        i = self.index(x, y)
        p = self.pixels
        return p[i], p[i + 1], p[i + 2], p[i + 3]

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int, a: int) -> None:
        i = self.index(x, y)
        p = self.pixels
        p[i] = r
        p[i + 1] = g
        p[i + 2] = b
        p[i + 3] = a

    def copy(self) -> "Image":
        return Image(self.width, self.height, bytearray(self.pixels))

    def __repr__(self) -> str:
        return f"Image(width={self.width}, height={self.height})"


# ---------------------------------------------------------------- PNG 解码


def _paeth(a: int, b: int, c: int) -> int:
    """PNG 规范的 Paeth 预测器。"""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> bytearray:
    """对带滤波器字节的原始扫描线数据做逆滤波，输出逐行紧凑像素。"""
    stride = width * channels
    result = bytearray(stride * height)
    pos = 0
    prev_start = -stride  # 第一行的“上一行”视为不存在

    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        line_start = y * stride
        if filter_type > _FILTER_PAETH:
            raise ImageDecodeError(f"未知的 PNG 滤波器类型 {filter_type}")

        for x in range(stride):
            cur = raw[pos]
            pos += 1
            left = result[line_start + x - channels] if x >= channels else 0
            up = result[prev_start + x] if y > 0 else 0
            up_left = (
                result[prev_start + x - channels]
                if y > 0 and x >= channels
                else 0
            )

            if filter_type == _FILTER_NONE:
                value = cur
            elif filter_type == _FILTER_SUB:
                value = (cur + left) & 0xFF
            elif filter_type == _FILTER_UP:
                value = (cur + up) & 0xFF
            elif filter_type == _FILTER_AVERAGE:
                value = (cur + ((left + up) >> 1)) & 0xFF
            else:  # _FILTER_PAETH
                value = (cur + _paeth(left, up, up_left)) & 0xFF
            result[line_start + x] = value

        prev_start = line_start

    return result


def _iter_chunks(data: bytes):
    """依次产出 (chunk_type, chunk_data)。"""
    offset = len(_PNG_SIGNATURE)
    length = len(data)
    while offset < length:
        if offset + 8 > length:
            raise ImageDecodeError("PNG 数据被截断：chunk 头不完整")
        (size,) = struct.unpack(">I", data[offset:offset + 4])
        chunk_type = data[offset + 4:offset + 8]
        start = offset + 8
        end = start + size
        if end + 4 > length:
            raise ImageDecodeError(f"PNG chunk {chunk_type!r} 数据被截断")
        chunk_data = data[start:end]
        (stored_crc,) = struct.unpack(">I", data[end:end + 4])
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ImageDecodeError(f"PNG chunk {chunk_type!r} CRC 校验失败")
        yield chunk_type, chunk_data
        offset = end + 4


def read_png(path: str) -> Image:
    """解码 PNG 文件为 :class:`Image`。

    支持 8 位、非隔行的 RGB/RGBA PNG；其余位深、色彩类型、隔行方式
    以及任何结构损坏都抛出 :class:`ImageDecodeError`。
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise ImageDecodeError(f"无法读取文件 {path!r}：{exc}") from exc

    if len(data) < 8 or data[:8] != _PNG_SIGNATURE:
        raise ImageDecodeError("不是有效的 PNG 文件（签名不匹配）")

    width = height = bit_depth = color_type = interlace = -1
    idat = bytearray()
    saw_ihdr = False

    for chunk_type, chunk_data in _iter_chunks(data):
        if chunk_type == b"IHDR":
            saw_ihdr = True
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk_data)
            if compression != 0 or filter_method != 0:
                raise ImageDecodeError("不支持的 PNG 压缩/滤波方式")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if not saw_ihdr:
        raise ImageDecodeError("PNG 缺少 IHDR 块")
    if width <= 0 or height > 0x7FFFFFFF or height <= 0:
        raise ImageDecodeError("PNG 图像尺寸非法")
    if bit_depth != 8:
        raise ImageDecodeError(f"仅支持 8 位 PNG，当前位深为 {bit_depth}")
    channels = _CHANNELS.get(color_type)
    if channels is None:
        raise ImageDecodeError(f"不支持的 PNG 色彩类型 {color_type}（仅 RGB/RGBA）")
    if interlace != 0:
        raise ImageDecodeError("不支持隔行扫描（Adam7）的 PNG")
    if not idat:
        raise ImageDecodeError("PNG 缺少 IDAT 图像数据")

    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise ImageDecodeError(f"PNG 图像数据解压失败：{exc}") from exc

    expected = height * (1 + width * channels)
    if len(raw) != expected:
        raise ImageDecodeError(
            f"PNG 解压后长度 {len(raw)} 与声明尺寸不符（应为 {expected}）"
        )

    planar = _unfilter(raw, width, height, channels)

    if channels == 4:
        return Image(width, height, planar)

    # RGB -> RGBA，不透明
    rgba = bytearray(width * height * 4)
    src = 0
    dst = 0
    for _ in range(width * height):
        rgba[dst] = planar[src]
        rgba[dst + 1] = planar[src + 1]
        rgba[dst + 2] = planar[src + 2]
        rgba[dst + 3] = 255
        src += 3
        dst += 4
    return Image(width, height, rgba)


# ---------------------------------------------------------------- PNG 编码


def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(chunk_data, crc) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", crc)
    )


def write_png(path: str, image: Image) -> None:
    """把 :class:`Image` 编码为 8 位 RGBA PNG（每行使用 None 滤波器）。"""
    width, height = image.width, image.height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)

    stride = width * 4
    raw = bytearray(height * (1 + stride))
    src = 0
    dst = 0
    for _ in range(height):
        raw[dst] = _FILTER_NONE
        dst += 1
        raw[dst:dst + stride] = image.pixels[src:src + stride]
        src += stride
        dst += stride

    compressed = zlib.compress(bytes(raw), level=9)
    payload = (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    try:
        with open(path, "wb") as f:
            f.write(payload)
    except OSError as exc:
        raise ImageDecodeError(f"无法写入文件 {path!r}：{exc}") from exc
