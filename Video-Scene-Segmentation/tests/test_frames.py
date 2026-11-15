"""frames.py 中精确读帧逻辑的纯测试（用内存管道，不启动 ffmpeg）。"""

from __future__ import annotations

import io

import pytest

from scene_seg_tool.errors import FFmpegRunError
from scene_seg_tool.frames import read_exact_frame


def test_read_exact_frame_handles_short_reads():
    # 模拟管道级短读：一次只给 3 字节，仍须拼出完整帧
    pipe = io.BytesIO(b"abcdefghij")
    frame = read_exact_frame(pipe, 5)
    assert frame == b"abcde"
    frame = read_exact_frame(pipe, 5)
    assert frame == b"fghij"


def test_read_exact_frame_clean_eof_returns_none():
    pipe = io.BytesIO(b"abc")
    assert read_exact_frame(pipe, 3) == b"abc"
    assert read_exact_frame(pipe, 3) is None


def test_read_exact_frame_partial_tail_raises():
    pipe = io.BytesIO(b"abcde")  # 5 字节流按 3 字节成帧，残留 2 字节
    assert read_exact_frame(pipe, 3) == b"abc"
    with pytest.raises(FFmpegRunError):
        read_exact_frame(pipe, 3)


def test_read_exact_frame_empty_stream_immediately_none():
    pipe = io.BytesIO(b"")
    assert read_exact_frame(pipe, 16) is None
