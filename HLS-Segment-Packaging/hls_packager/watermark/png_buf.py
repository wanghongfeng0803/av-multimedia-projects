"""内存 RGBA 位图与自研 PNG 编码（零第三方依赖）。"""

from __future__ import annotations

import struct
import zlib

from ..errors import WatermarkError

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FILTER_NONE = 0

_NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "orange": (255, 165, 0),
}


class Image:
    """内存中的 RGBA8888 位图，像素行优先、每像素 4 字节。"""

    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int):
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("宽高必须为整数")
        if width <= 0 or height <= 0:
            raise ValueError("宽高必须为正整数")
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 4)

    def index(self, x: int, y: int) -> int:
        return (y * self.width + x) * 4

    def blend_pixel(self, x, y, r, g, b, alpha) -> None:
        """覆盖率 alpha（0~255）的前景色做 src-over 混合。"""
        if not (0 <= x < self.width and 0 <= y < self.height) or alpha == 0:
            return
        i = self.index(x, y)
        p = self.pixels
        inv = 255 - alpha
        old_a = p[i + 3]
        new_a = alpha + old_a * inv // 255
        if new_a == 0:
            return
        p[i] = (r * alpha + p[i] * old_a * inv // 255) // new_a
        p[i + 1] = (g * alpha + p[i + 1] * old_a * inv // 255) // new_a
        p[i + 2] = (b * alpha + p[i + 2] * old_a * inv // 255) // new_a
        p[i + 3] = new_a


def parse_color(value: str) -> tuple[int, int, int]:
    """解析 #rgb / #rrggbb / 命名颜色，失败抛 WatermarkError。"""
    if not isinstance(value, str) or not value:
        raise WatermarkError(f"颜色值非法：{value!r}")
    text = value.strip().lower()
    if text in _NAMED_COLORS:
        return _NAMED_COLORS[text]
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            try:
                return tuple(int(ch * 2, 16) for ch in digits)  # type: ignore[return-value]
            except ValueError:
                pass
        elif len(digits) == 6:
            try:
                return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
            except ValueError:
                pass
    raise WatermarkError(f"无法识别的颜色 {value!r}（支持 #RGB / #RRGGBB 与命名颜色）")


def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(chunk_data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", crc)


def write_png(path: str, image: Image) -> None:
    """把 Image 编码为 8 位 RGBA PNG（每行 None 滤波器）。"""
    width, height = image.width, image.height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    stride = width * 4
    raw = bytearray(height * (1 + stride))
    src = dst = 0
    for _ in range(height):
        raw[dst] = _FILTER_NONE
        dst += 1
        raw[dst:dst + stride] = image.pixels[src:src + stride]
        src += stride
        dst += stride
    payload = (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + _chunk(b"IEND", b"")
    )
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        raise WatermarkError(f"无法写入水印 PNG {path!r}：{exc}") from exc
