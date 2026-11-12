# -*- coding: utf-8 -*-
"""
ISO-BMFF (MP4/fMP4) 盒子的底层读写原语。

只依赖标准库，提供：
- Reader：大端字节流读取
- iter_boxes / find_child：盒子遍历（支持 64 位 largesize、size=0 伸到末尾）
- box / fullbox：盒子构建
- 一组本项目用到的 box type / flag 常量

上层 fMP4 读取器与写入器都基于这里的原语实现，
保证「切割逻辑、moof/mdat 重组、m3u8 生成都在自己代码里」。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 容器型 box（其 payload 内部继续装 box）。注意 meta 是 FullBox，单独处理。
# ---------------------------------------------------------------------------
CONTAINER_TYPES = {
    b"moov", b"trak", b"edts", b"mdia", b"minf", b"dinf",
    b"stbl", b"mvex", b"moof", b"traf", b"udta",
}

# tfhd flags（ISO 14496-12 8.8.7）
TFHD_BASE_DATA_OFFSET_PRESENT = 0x000001
TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT = 0x000002
TFHD_DEFAULT_SAMPLE_DURATION_PRESENT = 0x000008
TFHD_DEFAULT_SAMPLE_SIZE_PRESENT = 0x000010
TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT = 0x000020
TFHD_DURATION_IS_EMPTY = 0x010000
TFHD_DEFAULT_BASE_IS_MOOF = 0x020000

# trun flags（ISO 14496-12 8.8.8）
TRUN_DATA_OFFSET_PRESENT = 0x000001
TRUN_FIRST_SAMPLE_FLAGS_PRESENT = 0x000004
TRUN_SAMPLE_DURATION_PRESENT = 0x000100
TRUN_SAMPLE_SIZE_PRESENT = 0x000200
TRUN_SAMPLE_FLAGS_PRESENT = 0x000400
TRUN_SAMPLE_CTO_PRESENT = 0x000800

# sample_flags（trun 样本标志 / trex 默认标志）
# sample_depends_on=2(该样本不依赖其他样本，即同步样本) → 0x02000000
SAMPLE_FLAGS_SYNC = 0x02000000
# sample_depends_on=1 + sample_is_non_sync_sample=1 → 0x01010000
SAMPLE_FLAGS_NON_SYNC = 0x01010000

# tkhd flags
TKHD_TRACK_ENABLED = 0x000001
TKHD_TRACK_IN_MOVIE = 0x000002

# 标准矩阵（单位矩阵，16.16 定点）
IDENTITY_MATRIX = (
    0x00010000, 0, 0,
    0, 0x00010000, 0,
    0, 0, 0x40000000,
)

# und 语言码：packed ISO-639-2/T，0x55C4 即 "und"
LANGUAGE_UNDEFINED = 0x55C4


class Reader:
    """对 bytes 的游标式大端读取器。"""

    def __init__(self, data: bytes, pos: int = 0, end: Optional[int] = None):
        self.data = data
        self.pos = pos
        self.end = len(data) if end is None else end

    def remaining(self) -> int:
        return self.end - self.pos

    def read(self, n: int) -> bytes:
        if self.pos + n > self.end:
            raise EOFError("box 数据不完整")
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v

    def u8(self) -> int:
        return self.read(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self.read(2), "big")

    def u24(self) -> int:
        return int.from_bytes(self.read(3), "big")

    def u32(self) -> int:
        return int.from_bytes(self.read(4), "big")

    def u64(self) -> int:
        return int.from_bytes(self.read(8), "big")

    def i32(self) -> int:
        return int.from_bytes(self.read(4), "big", signed=True)

    def fourcc(self) -> bytes:
        return self.read(4)


@dataclass
class Box:
    type: bytes
    start: int          # box 起始偏移（含 header）
    header_size: int
    payload_start: int
    payload_end: int    # box 结束偏移（不含）

    @property
    def size(self) -> int:
        return self.payload_end - self.start

    def payload(self, buf: bytes) -> bytes:
        return buf[self.payload_start:self.payload_end]


def iter_boxes(buf: bytes, start: int, end: Optional[int] = None) -> Iterator[Box]:
    """在 [start, end) 范围内迭代顶层 box；数据不完整时静默结束。"""
    if end is None:
        end = len(buf)
    p = start
    while p + 8 <= end:
        size = int.from_bytes(buf[p:p + 4], "big")
        btype = buf[p + 4:p + 8]
        header = 8
        if size == 1:  # 64 位 largesize
            if p + 16 > end:
                return
            size = int.from_bytes(buf[p + 8:p + 16], "big")
            header = 16
        elif size == 0:  # box 一直延伸到 buf 末尾
            size = end - p
        if size < header or p + size > end:
            return  # 不完整，等更多数据
        yield Box(btype, p, header, p + header, p + size)
        p += size


def find_child(buf: bytes, start: int, end: int, btype: bytes) -> Optional[Box]:
    """在容器 box 的 payload 范围内找第一个指定类型的子 box。"""
    for child in iter_boxes(buf, start, end):
        if child.type == btype:
            return child
    return None


def find_path(buf: bytes, root_start: int, root_end: int, *path: bytes) -> Optional[Box]:
    """按 moov/trak/mdia/... 这样的路径逐级查找。"""
    start, end = root_start, root_end
    for btype in path:
        node = find_child(buf, start, end, btype)
        if node is None:
            return None
        start, end = node.payload_start, node.payload_end
    return Box(path[-1], start - 8, 8, start, end) if path else None


def parse_fullbox(buf: bytes, payload_start: int) -> Tuple[int, int, int]:
    """解析 FullBox 头，返回 (version, flags, payload_after_header_offset)。"""
    version = buf[payload_start]
    flags = int.from_bytes(buf[payload_start + 1:payload_start + 4], "big")
    return version, flags, payload_start + 4


# ---------------------------------------------------------------------------
# 构建侧
# ---------------------------------------------------------------------------
def box(btype: bytes, payload: bytes = b"") -> bytes:
    # size 字段是含 8 字节 box header 的总长度，不是 payload 长度
    total = 8 + len(payload)
    return total.to_bytes(4, "big") + btype + payload


def fullbox(btype: bytes, version: int, flags: int, payload: bytes = b"") -> bytes:
    head = bytes([version]) + flags.to_bytes(3, "big")
    return box(btype, head + payload)


def u16(v: int) -> bytes:
    return v.to_bytes(2, "big")


def u32(v: int) -> bytes:
    return v.to_bytes(4, "big")


def u64(v: int) -> bytes:
    return v.to_bytes(8, "big")


def i32(v: int) -> bytes:
    return v.to_bytes(4, "big", signed=True)


def matrix_bytes(matrix: Tuple[int, ...] = IDENTITY_MATRIX) -> bytes:
    return b"".join(u32(v) for v in matrix)


def children(*boxes: Optional[bytes]) -> bytes:
    return b"".join(b for b in boxes if b)


def find_all_children(buf: bytes, start: int, end: int, btype: bytes) -> List[Box]:
    return [c for c in iter_boxes(buf, start, end) if c.type == btype]
