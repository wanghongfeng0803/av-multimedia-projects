"""帧内行带分块与多进程并行映射。

设计要点：

- **分块按帧内行带切分**：每块是同一帧内连续的若干行 rgb24 像素，
  块边界永不跨帧，因此解码 / 编码两侧的字节流无需任何重排，
  写出顺序与读入顺序天然一致。
- **映射表只构建一次**：LUT 由调用方构建后通过进程池
  ``initializer`` 分发给各 worker，worker 内只做一次
  ``bytes.translate``，不再重复建表。
- **内存不随分辨率暴涨**：在途块数有界（``max_inflight``），
  内存上界 ≈ ``max_inflight × chunk_rows × width × 3`` 字节，
  与帧率、时长、总帧数无关，也不随画面高度增长。
- 恒等 LUT 走零开销直通路径（不启动进程池），结果与并行路径
  逐字节一致，这是无参数调用的性能兜底。
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO, Callable

from .errors import StreamTruncatedError
from .lut import IDENTITY_LUT

#: 默认每块 128 行：1080p 下每块约 0.7 MiB，8K 下约 2.8 MiB，均在可接受范围。
DEFAULT_CHUNK_ROWS = 128

# worker 进程内的全局 LUT，由进程池 initializer 设置一次。
_WORKER_LUT: bytes | None = None


def _init_worker(lut: bytes) -> None:
    """进程池 initializer：每个 worker 启动时接收同一张 LUT。"""
    global _WORKER_LUT
    _WORKER_LUT = lut


def _translate_chunk(data: bytes) -> bytes:
    """worker 入口：对一块像素应用 LUT（必须是模块级函数才能被 pickle）。"""
    return data.translate(_WORKER_LUT)


def frame_chunk_sizes(width: int, height: int, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> list[int]:
    """返回一帧 rgb24 数据的分块字节大小序列（块边界不跨帧）。

    除最后一块可能不足 ``chunk_rows`` 行外，每块都是
    ``chunk_rows × width × 3`` 字节；序列总和恒等于一帧的字节数。
    """
    if width < 1 or height < 1:
        raise ValueError("width/height 必须为正整数")
    if chunk_rows < 1:
        raise ValueError("chunk_rows 必须为正整数")
    row_bytes = width * 3
    sizes: list[int] = []
    remaining = height
    while remaining > 0:
        rows = min(chunk_rows, remaining)
        sizes.append(rows * row_bytes)
        remaining -= rows
    return sizes


def _read_exact(stream: BinaryIO, n: int) -> bytes | None:
    """精确读取 n 字节；流在干净的边界结束返回 None，中途截断抛异常。"""
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            if not buf:
                return None
            raise StreamTruncatedError(
                f"裸视频流在块中途截断（期望 {n} 字节，实得 {len(buf)}），"
                "解码器可能已异常退出。"
            )
        buf.extend(part)
    return bytes(buf)


def map_stream(
    stream_in: BinaryIO,
    stream_out: BinaryIO,
    width: int,
    height: int,
    lut: bytes,
    *,
    workers: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_inflight: int | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> int:
    """把 ``stream_in`` 的 rgb24 裸流经 LUT 映射后写入 ``stream_out``。

    :param lut: 由 :func:`color_grading.lut.build_lut` 构建的 256 字节映射表，
        本函数不再重新构建。
    :param workers: 进程池大小，默认 ``os.cpu_count()``。
    :param chunk_rows: 每块行数，决定单次读写的粒度。
    :param max_inflight: 在途块数上限（默认 ``workers × 2``），
        是内存占用与分辨率的解耦阀门。
    :param on_frame: 每读完一帧回调一次，参数为已读帧数（用于进度显示）。
    :return: 处理的帧数。
    """
    sizes = frame_chunk_sizes(width, height, chunk_rows)
    if bytes(lut) == IDENTITY_LUT:
        # 恒等快速路径：无参数调用零进程开销，输出与输入逐字节一致。
        return _copy_stream(stream_in, stream_out, sizes, on_frame)

    if workers is None:
        workers = os.cpu_count() or 4
    if workers < 1:
        raise ValueError("workers 必须为正整数")
    if max_inflight is None:
        max_inflight = workers * 2
    if max_inflight < 1:
        raise ValueError("max_inflight 必须为正整数")

    lut = bytes(lut)
    frames = 0
    pos = 0  # 当前帧内下一块的序号
    inflight: deque = deque()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(lut,)
    ) as pool:
        eof = False
        while not eof or inflight:
            # 补满在途窗口：读取与翻译重叠，解码 / 映射 / 编码三线并行。
            while not eof and len(inflight) < max_inflight:
                data = _read_exact(stream_in, sizes[pos])
                if data is None:
                    if pos != 0:
                        raise StreamTruncatedError("裸视频流在帧边界前结束。")
                    eof = True
                    break
                inflight.append(pool.submit(_translate_chunk, data))
                pos += 1
                if pos == len(sizes):
                    pos = 0
                    frames += 1
                    if on_frame is not None:
                        on_frame(frames)
            if inflight:
                # 按提交顺序写出，保证帧内块序与字节流一致。
                stream_out.write(inflight.popleft().result())
    return frames


def _copy_stream(
    stream_in: BinaryIO,
    stream_out: BinaryIO,
    sizes: list[int],
    on_frame: Callable[[int], None] | None,
) -> int:
    """恒等映射的直通路径：同样的分块读取粒度，但不进进程池。"""
    frames = 0
    pos = 0
    while True:
        data = _read_exact(stream_in, sizes[pos])
        if data is None:
            if pos != 0:
                raise StreamTruncatedError("裸视频流在帧边界前结束。")
            return frames
        stream_out.write(data)
        pos += 1
        if pos == len(sizes):
            pos = 0
            frames += 1
            if on_frame is not None:
                on_frame(frames)
