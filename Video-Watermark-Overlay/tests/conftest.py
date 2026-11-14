"""测试公共夹具与手写 PNG 编码辅助（不依赖被测的 write_png，便于构造异常样本）。"""

from __future__ import annotations

import struct
import zlib

import pytest

from watermark_overlay.image import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def solid_image(width: int, height: int, color: tuple[int, int, int, int]) -> Image:
    """生成纯色 RGBA 图像。"""
    img = Image(width, height)
    r, g, b, a = color
    for i in range(width * height):
        j = i * 4
        img.pixels[j:j + 4] = bytes((r, g, b, a))
    return img


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _apply_filter(raw: bytes, width: int, channels: int, filter_type: int) -> bytes:
    """对紧凑扫描线数据施加指定 PNG 滤波器（每行同一种），用于构造测试样本。"""
    stride = width * channels
    out = bytearray()
    for y in range(len(raw) // stride):
        out.append(filter_type)
        line = raw[y * stride:(y + 1) * stride]
        for x, value in enumerate(line):
            left = line[x - channels] if x >= channels else 0
            up = raw[(y - 1) * stride + x] if y > 0 else 0
            up_left = raw[(y - 1) * stride + x - channels] if y > 0 and x >= channels else 0
            if filter_type == 0:
                delta = value
            elif filter_type == 1:
                delta = (value - left) & 0xFF
            elif filter_type == 2:
                delta = (value - up) & 0xFF
            elif filter_type == 3:
                delta = (value - ((left + up) >> 1)) & 0xFF
            else:
                delta = (value - _paeth(left, up, up_left)) & 0xFF
            out.append(delta)
    return bytes(out)


def make_png(
    path,
    width: int,
    height: int,
    channels: int = 4,
    fill: tuple[int, ...] | None = None,
    filter_type: int = 0,
    bit_depth: int = 8,
    color_type: int | None = None,
    interlace: int = 0,
    bad_crc: bool = False,
) -> None:
    """手写一个最小 PNG；可指定滤波器、位深、色彩类型、CRC 损坏等。"""
    if color_type is None:
        color_type = 6 if channels == 4 else 2
    if fill is None:
        fill = (10, 20, 30, 255) if channels == 4 else (10, 20, 30)
    raw = bytes(fill[:channels]) * (width * height)
    filtered = _apply_filter(raw, width, channels, filter_type)
    ihdr = struct.pack(
        ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        if bad_crc and tag == b"IDAT":
            crc ^= 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    payload = (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(filtered))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(payload)


@pytest.fixture
def frame_4x4():
    return solid_image(4, 4, (100, 150, 200, 255))


@pytest.fixture
def red_watermark():
    return solid_image(2, 2, (255, 0, 0, 128))
