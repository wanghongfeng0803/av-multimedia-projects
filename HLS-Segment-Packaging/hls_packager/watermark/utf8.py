"""自研 UTF-8 解码器（RFC 3629）。

不使用 ``bytes.decode``：亲手处理 1~4 字节变长序列、代理区拒绝、
过长编码拒绝、越界码点拒绝与非法首/续字节，保证任何异常输入都抛出
:class:`UnicodeDecodeError` 而不是产出歧义字符。
"""

from __future__ import annotations

_MAX_CODEPOINT = 0x10FFFF
_SURROGATE_START = 0xD800
_SURROGATE_END = 0xDFFF


def _is_continuation(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def decode_utf8(data: bytes) -> list[int]:
    """把 UTF-8 字节串解码为 Unicode 码点列表。

    :raises UnicodeDecodeError: 任何非法编码（截断、续字节缺失、
        过长编码、代理码点、超界码点、非法首字节）。
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("decode_utf8 只接受 bytes/bytearray 输入")

    codepoints: list[int] = []
    length = len(data)
    i = 0

    while i < length:
        first = data[i]

        if first < 0x80:
            codepoints.append(first)
            i += 1
            continue

        if first >> 5 == 0b110:
            seq_len, codepoint = 2, first & 0x1F
        elif first >> 4 == 0b1110:
            seq_len, codepoint = 3, first & 0x0F
        elif first >> 3 == 0b11110:
            seq_len, codepoint = 4, first & 0x07
        else:
            raise UnicodeDecodeError(
                "utf-8", bytes(data), i, i + 1, "非法的 UTF-8 首字节"
            )

        end = i + seq_len
        if end > length:
            raise UnicodeDecodeError(
                "utf-8", bytes(data), i, length, "UTF-8 序列被截断"
            )

        for j in range(i + 1, end):
            byte = data[j]
            if not _is_continuation(byte):
                raise UnicodeDecodeError(
                    "utf-8", bytes(data), j, j + 1, "缺少 UTF-8 续字节"
                )
            codepoint = (codepoint << 6) | (byte & 0x3F)

        if seq_len == 2 and codepoint < 0x80:
            raise UnicodeDecodeError("utf-8", bytes(data), i, end, "过长的 UTF-8 编码")
        if seq_len == 3 and codepoint < 0x800:
            raise UnicodeDecodeError("utf-8", bytes(data), i, end, "过长的 UTF-8 编码")
        if seq_len == 4 and codepoint < 0x10000:
            raise UnicodeDecodeError("utf-8", bytes(data), i, end, "过长的 UTF-8 编码")

        if _SURROGATE_START <= codepoint <= _SURROGATE_END:
            raise UnicodeDecodeError(
                "utf-8", bytes(data), i, end, "UTF-8 不能编码代理区码点"
            )
        if codepoint > _MAX_CODEPOINT:
            raise UnicodeDecodeError(
                "utf-8", bytes(data), i, end, "码点超出 Unicode 范围"
            )

        codepoints.append(codepoint)
        i = end

    return codepoints
