"""分块并行映射的纯逻辑测试（不依赖 ffmpeg）。

进程池路径在 Windows 上使用 spawn 启动 worker，worker 函数与
initializer 都是 color_grading.chunks 的模块级符号，可安全 pickle。
"""

from __future__ import annotations

import io
import os

import pytest

from color_grading.chunks import frame_chunk_sizes, map_stream
from color_grading.errors import StreamTruncatedError
from color_grading.lut import IDENTITY_LUT, build_lut

W, H = 8, 6  # 小帧：一帧 144 字节


def _random_frames(width: int, height: int, count: int) -> bytes:
    return os.urandom(width * height * 3 * count)


class _ReadSpy:
    """记录单次 read 请求的最大字节数，用于验证读取粒度有界。"""

    def __init__(self, data: bytes):
        self._io = io.BytesIO(data)
        self.max_read = 0

    def read(self, n: int = -1) -> bytes:
        self.max_read = max(self.max_read, n)
        return self._io.read(n)


def test_frame_chunk_sizes_cover_frame_exactly():
    sizes = frame_chunk_sizes(4, 5, chunk_rows=2)
    assert sizes == [24, 24, 12]
    assert sum(sizes) == 4 * 5 * 3


def test_frame_chunk_sizes_single_chunk_when_rows_cover_frame():
    assert frame_chunk_sizes(2, 3, chunk_rows=100) == [2 * 3 * 3]


def test_frame_chunk_sizes_rejects_invalid_args():
    with pytest.raises(ValueError):
        frame_chunk_sizes(0, 5)
    with pytest.raises(ValueError):
        frame_chunk_sizes(4, 5, chunk_rows=0)


def test_identity_passthrough_is_byte_exact():
    """无参数（恒等 LUT）路径：输出与输入逐字节一致。"""
    data = _random_frames(W, H, 3)
    out = io.BytesIO()
    frames = map_stream(io.BytesIO(data), out, W, H, IDENTITY_LUT, chunk_rows=2)
    assert frames == 3
    assert out.getvalue() == data


def test_parallel_path_matches_whole_translate():
    """分块并行结果 == 对整段数据一次性 translate（顺序与映射都正确）。"""
    data = _random_frames(W, H, 4)
    lut = build_lut(brightness=20, contrast=30)
    out = io.BytesIO()
    frames = map_stream(io.BytesIO(data), out, W, H, lut, workers=2, chunk_rows=2)
    assert frames == 4
    assert out.getvalue() == data.translate(lut)


def test_parallel_path_with_uneven_last_chunk():
    """高度不能整除 chunk_rows 时，尾部小块也正确处理。"""
    data = _random_frames(W, H, 2)  # H=6, chunk_rows=4 → 块 [4行, 2行]
    lut = build_lut(brightness=-15, contrast=60)
    out = io.BytesIO()
    frames = map_stream(io.BytesIO(data), out, W, H, lut, workers=2, chunk_rows=4)
    assert frames == 2
    assert out.getvalue() == data.translate(lut)


def test_truncated_stream_raises_on_both_paths():
    half_frame = _random_frames(W, H, 1)[:-5]
    with pytest.raises(StreamTruncatedError):
        map_stream(io.BytesIO(half_frame), io.BytesIO(), W, H, IDENTITY_LUT, chunk_rows=2)
    with pytest.raises(StreamTruncatedError):
        map_stream(
            io.BytesIO(half_frame), io.BytesIO(), W, H,
            build_lut(10, 0), workers=2, chunk_rows=2,
        )


def test_read_granularity_is_bounded_by_chunk():
    """单次读取不超过一个行带：内存不随分辨率/时长增长的关键性质。"""
    chunk_rows = 3
    data = _random_frames(16, 10, 2)
    spy = _ReadSpy(data)
    map_stream(spy, io.BytesIO(), 16, 10, IDENTITY_LUT, chunk_rows=chunk_rows)
    assert spy.max_read <= chunk_rows * 16 * 3


def test_on_frame_callback_counts_frames():
    data = _random_frames(W, H, 3)
    seen: list[int] = []
    map_stream(io.BytesIO(data), io.BytesIO(), W, H, IDENTITY_LUT,
               chunk_rows=2, on_frame=seen.append)
    assert seen == [1, 2, 3]
