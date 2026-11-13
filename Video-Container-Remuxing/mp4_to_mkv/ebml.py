"""EBML（Extensible Binary Meta Language）编码基础模块。

Matroska(MKV) 是建立在 EBML 之上的容器格式。EBML 元素的通用结构为::

    [Element ID（VINT 形式，本身带长度标记）]
    [数据区大小 size（VINT 形式）]
    [数据区 data]

其中 VINT 是一种变长整数：首字节前导 1 之前 0 的个数决定总长度
（0xxxxxxx → 1 字节，10xxxxxx → 2 字节……），长度标记位之后的位才是
有效数据位。

本模块只负责“编码写出”，不依赖任何第三方库。
"""

from __future__ import annotations

from struct import pack
from typing import Iterable, Sequence

# VINT 各字节长度对应的有效数据位数：1 字节存 7 位，2 字节存 14 位……
_VINT_DATA_WIDTH = (7, 14, 21, 28, 35, 42, 49, 56)
# 各长度首字节的长度标记（10000000、01000000、……、00000001）
_VINT_LENGTH_MARKER = (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)


class EbmlEncodeError(ValueError):
    """待编码的数值超出 EBML 表示范围等编码侧错误。"""


def encode_vint(value: int, *, unknown: bool = False) -> bytes:
    """把非负整数编码为 EBML VINT（最短表示）。

    :param value: 待编码的非负整数。
    :param unknown: 为 True 时编码“未知大小” VINT（全 1 数据位，
        如 8 字节的 ``0xFF FF ... FF``），此时忽略 value。
    """
    if unknown:
        return b"\xff" * 8
    if value < 0:
        raise EbmlEncodeError(f"VINT 不能表示负数: {value}")
    for idx, width in enumerate(_VINT_DATA_WIDTH):
        if value < (1 << width):
            length = idx + 1
            break
    else:  # pragma: no cover - 56 位已覆盖实际所有容器字段
        raise EbmlEncodeError(f"数值超出 EBML VINT 最大表示范围: {value}")

    raw = value.to_bytes(length, "big")
    first = raw[0] | _VINT_LENGTH_MARKER[idx]
    return bytes([first]) + raw[1:]


def encode_element_id(element_id: int) -> bytes:
    """序列化 Element ID。

    Element ID 常量在定义时就已经包含 VINT 长度标记位（例如
    ``0x1A45DFA3`` 首字节 0x1A = 0001 1010，前导 000 + 1 表示 4 字节），
    因此按首字节前导 0 的个数判定长度后，从最短大端表示中原样输出。
    """
    if element_id <= 0 or element_id > 0xFFFFFFFF:
        raise EbmlEncodeError(f"非法 Element ID: 0x{element_id:X}")
    # 先取最短大端表示，再由首字节前导 0 的个数判定 VINT 长度
    byte_length = (element_id.bit_length() + 7) // 8
    raw = element_id.to_bytes(byte_length, "big")
    first = raw[0]
    if first & 0x80:
        length = 1
    elif first & 0x40:
        length = 2
    elif first & 0x20:
        length = 3
    elif first & 0x10:
        length = 4
    else:
        raise EbmlEncodeError(
            f"Element ID 缺少合法 VINT 长度标记: 0x{element_id:X}"
        )
    if length != byte_length:
        # 例如首字节标记称 2 字节、实际却需要 3 字节，说明 ID 常量非法
        raise EbmlEncodeError(
            f"Element ID 长度标记与实际字节数不符: 0x{element_id:X}"
        )
    return raw


def _wrap(element_id: int, payload: bytes) -> bytes:
    """组装完整 EBML 元素：ID + size(VINT) + data。"""
    return encode_element_id(element_id) + encode_vint(len(payload)) + payload


def uint_element(element_id: int, value: int) -> bytes:
    """无符号整数元素，按最短大端字节存储（value=0 时数据区为空）。"""
    if value < 0:
        raise EbmlEncodeError(f"无符号整数元素不能为负: {value}")
    if value == 0:
        payload = b""
    else:
        payload = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return _wrap(element_id, payload)


def sint_element(element_id: int, value: int) -> bytes:
    """有符号整数元素，按最短大端补码存储。"""
    if value == 0:
        payload = b""
    else:
        # 取补码表示所需的最小字节数（含 1 位符号位），再裁掉冗余字节。
        length = (value.bit_length() + 1 + 7) // 8
        payload = value.to_bytes(length, "big", signed=True)
        while len(payload) > 1 and (
            (payload[0] == 0x00 and payload[1] & 0x80 == 0)
            or (payload[0] == 0xFF and payload[1] & 0x80 == 0x80)
        ):
            payload = payload[1:]
    return _wrap(element_id, payload)


def float_element(element_id: int, value: float) -> bytes:
    """浮点元素，统一使用 64 位双精度（Matroska 时长字段即 float64）。"""
    return _wrap(element_id, pack(">d", value))


def binary_element(element_id: int, data: bytes) -> bytes:
    """二进制元素（CodecPrivate、SimpleBlock 数据等）。"""
    return _wrap(element_id, bytes(data))


def string_element(element_id: int, text: str) -> bytes:
    """ASCII 字符串元素。"""
    return _wrap(element_id, text.encode("ascii"))


def utf8_element(element_id: int, text: str) -> bytes:
    """UTF-8 字符串元素。"""
    return _wrap(element_id, text.encode("utf-8"))


def master_element(element_id: int, children: Iterable[bytes]) -> bytes:
    """主元素：把若干已编码的子元素拼接后整体包裹。"""
    payload = b"".join(children)
    return _wrap(element_id, payload)


def concat(elements: Sequence[bytes]) -> bytes:
    """拼接多个已编码元素。"""
    return b"".join(elements)


def unknown_size_header(element_id: int) -> bytes:
    """生成“未知大小”的元素头（ID + 未知 size VINT），用于流式场景。

    基础版本采用一次性确定大小的写法，此函数主要供测试与扩展使用。
    """
    return encode_element_id(element_id) + encode_vint(0, unknown=True)
