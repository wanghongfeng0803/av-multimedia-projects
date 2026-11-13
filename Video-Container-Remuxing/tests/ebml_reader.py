"""最小 EBML 读取器，仅供测试校验输出 MKV 结构使用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EbmlElement:
    element_id: int
    payload: bytes
    start: int
    end: int

    def children(self) -> List["EbmlElement"]:
        return parse_elements(self.payload)

    def find(self, element_id: int) -> Optional["EbmlElement"]:
        for child in self.children():
            if child.element_id == element_id:
                return child
        return None

    def find_all(self, element_id: int) -> List["EbmlElement"]:
        return [c for c in self.children() if c.element_id == element_id]

    def as_uint(self) -> int:
        return int.from_bytes(self.payload, "big") if self.payload else 0

    def as_float(self) -> float:
        import struct

        if len(self.payload) == 4:
            return struct.unpack(">f", self.payload)[0]
        return struct.unpack(">d", self.payload)[0]

    def as_string(self) -> str:
        return self.payload.decode("utf-8")


def read_vint(buf: bytes, pos: int) -> tuple[int, int]:
    """读取 size VINT，返回 (去掉长度标记后的数值, next_pos)。"""
    first = buf[pos]
    for length in range(1, 9):
        if first & (0x80 >> (length - 1)):
            break
    else:  # pragma: no cover
        raise ValueError("非法 VINT")
    if pos + length > len(buf):
        raise ValueError("VINT 被截断")
    value = first & (0x7F >> (length - 1))
    for i in range(1, length):
        value = (value << 8) | buf[pos + i]
    return value, pos + length


def read_element_id(buf: bytes, pos: int) -> tuple[int, int]:
    """读取 Element ID：长度由标记位决定，但数值保留原始标记位。"""
    first = buf[pos]
    for length in range(1, 5):
        if first & (0x80 >> (length - 1)):
            break
    else:
        raise ValueError("非法 Element ID")
    if pos + length > len(buf):
        raise ValueError("Element ID 被截断")
    return int.from_bytes(buf[pos:pos + length], "big"), pos + length


def parse_elements(buf: bytes) -> List[EbmlElement]:
    elements: List[EbmlElement] = []
    pos = 0
    while pos < len(buf):
        element_id, after_id = read_element_id(buf, pos)
        size, after_size = read_vint(buf, after_id)
        end = after_size + size
        if end > len(buf):
            raise ValueError(
                f"元素 0x{element_id:X} 声明 size={size} 超出边界"
            )
        elements.append(
            EbmlElement(element_id, buf[after_size:end], pos, end)
        )
        pos = end
    return elements


def parse_document(buf: bytes) -> List[EbmlElement]:
    return parse_elements(buf)
