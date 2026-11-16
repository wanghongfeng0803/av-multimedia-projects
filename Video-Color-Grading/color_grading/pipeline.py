"""解码 → 分块映射 → 编码 的 ffmpeg 管道编排。

数据流：

    输入文件 ──ffmpeg 解码──> rgb24 裸流 ──分块并行映射──> rgb24 裸流 ──ffmpeg 编码──> 输出文件
      │                                                                        ▲
      └────────────── 音轨不经过像素管道，-c:a copy 原样直通 ────────────────────┘

设计要点：

- **保留音轨**：音轨不进像素管道，编码器以原文件为第二输入，
  ``-map 1:a? -c:a copy`` 无重编码直通，没有音画同步漂移；
  无音轨的输入也能正常工作（``?`` 后缀使映射可选）。
- **映射表只算一次**：LUT 在 :func:`adjust` 内构建一次，
  之后全部帧、全部块复用。
- **stderr 排空**：两条 ffmpeg 子进程的 stderr 都由独立线程持续读取，
  避免管道缓冲区写满后父子进程互等死锁。
- **内存有界**：任意时刻只有少量行带在途（见
  :func:`color_grading.chunks.map_stream`），整段视频不会载入内存。
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .binaries import find_ffmpeg
from .chunks import DEFAULT_CHUNK_ROWS, map_stream
from .errors import (
    FFmpegRunError,
    InputNotFoundError,
    OutputExistsError,
)
from .lut import build_lut, clamp_params
from .probe import probe_video

# stderr 只保留尾部若干行用于错误报告，避免长视频日志占内存。
_STDERR_TAIL_LINES = 50


@dataclass
class GradeResult:
    """一次调色执行的汇总。"""

    input_path: str
    output_path: str
    brightness: float
    contrast: float
    width: int
    height: int
    fps: float
    frames: int
    has_audio: bool
    elapsed_seconds: float
    messages: list[str] = field(default_factory=list)


def _drain_stderr(stream, sink: deque) -> None:
    """持续读空子进程 stderr，防止管道写满死锁；只保留尾部若干行。"""
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", "replace"))
    finally:
        stream.close()


def _tail(sink: deque) -> str:
    return "".join(sink).strip()


def adjust(
    input_path: str,
    output_path: str,
    brightness: float = 0.0,
    contrast: float = 0.0,
    *,
    workers: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
    on_frame: Callable[[int], None] | None = None,
) -> GradeResult:
    """调整视频亮度 / 对比度并保留音轨。

    默认参数（``brightness=0, contrast=0``）为恒等映射，输出像素
    与输入逐字节一致（裸流级别），这是旧无参数调用的兼容基线。

    :param brightness: 亮度，[-100, 100]，超范围自动钳制并提示
    :param contrast: 对比度，[-100, 100]，超范围自动钳制并提示
    :param workers: 并行进程数，默认 CPU 核数
    :param chunk_rows: 分块行数（内存粒度）
    :param crf: x264 恒定质量因子
    :param preset: x264 速度/压缩率预设
    :param overwrite: 允许覆盖已存在的输出文件
    :param on_frame: 每处理一帧的回调（已处理帧数）
    :raises InputNotFoundError: 输入不存在
    :raises OutputExistsError: 输出已存在且未允许覆盖
    :raises InvalidInputError: 输入不是可处理的视频
    :raises FFmpegRunError: ffmpeg 执行失败
    """
    if not os.path.isfile(input_path):
        raise InputNotFoundError(f"输入文件不存在: {input_path}")
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(f"输出文件已存在（未指定覆盖）: {output_path}")

    # 超范围钳制并收集提示（同时也会触发 UserWarning）。
    brightness, contrast, messages = clamp_params(brightness, contrast)
    info = probe_video(input_path)

    # 整个流程唯一一次建表；之后所有帧 / 块 / worker 复用。
    lut = build_lut(brightness, contrast)

    vf_args: list[str] = []
    if info.width % 2 or info.height % 2:
        # yuv420p 要求偶数尺寸；向下取偶并明确告知调用方。
        vf_args = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        messages.append("输入宽/高为奇数，yuv420p 要求偶数尺寸，已向下缩放为偶数。")

    ffmpeg = find_ffmpeg()
    decoder = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y" if overwrite else "-n",
            # 第一路输入：处理后的 rgb24 裸流（stdin）
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{info.width}x{info.height}",
            "-r", info.fps_fraction,
            "-i", "-",
            # 第二路输入：原文件，只为直通音轨
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            *vf_args,
            "-c:a", "copy",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    dec_err: deque = deque(maxlen=_STDERR_TAIL_LINES)
    enc_err: deque = deque(maxlen=_STDERR_TAIL_LINES)
    drainers = [
        threading.Thread(target=_drain_stderr, args=(decoder.stderr, dec_err), daemon=True),
        threading.Thread(target=_drain_stderr, args=(encoder.stderr, enc_err), daemon=True),
    ]
    for t in drainers:
        t.start()

    started = time.monotonic()
    frames = 0
    broken_pipe = False
    try:
        try:
            frames = map_stream(
                decoder.stdout, encoder.stdin,
                info.width, info.height, lut,
                workers=workers, chunk_rows=chunk_rows, on_frame=on_frame,
            )
        except (BrokenPipeError, ConnectionResetError):
            # 编码器提前退出（如参数不被支持），其 stderr 有真正原因。
            broken_pipe = True
    finally:
        try:
            encoder.stdin.close()
        except OSError:
            pass
        if broken_pipe and decoder.poll() is None:
            # 不再消费解码输出，主动终止避免它阻塞在管道写上。
            decoder.terminate()
        enc_rc = encoder.wait()
        dec_rc = decoder.wait()
        for t in drainers:
            t.join(timeout=5)
    elapsed = time.monotonic() - started

    if broken_pipe:
        raise FFmpegRunError("ffmpeg 编码器提前退出，像素管道中断。", stderr=_tail(enc_err))
    if dec_rc != 0:
        raise FFmpegRunError(f"ffmpeg 解码失败（退出码 {dec_rc}）。", stderr=_tail(dec_err))
    if enc_rc != 0:
        raise FFmpegRunError(f"ffmpeg 编码失败（退出码 {enc_rc}）。", stderr=_tail(enc_err))

    return GradeResult(
        input_path=input_path,
        output_path=output_path,
        brightness=brightness,
        contrast=contrast,
        width=info.width,
        height=info.height,
        fps=info.fps,
        frames=frames,
        has_audio=info.has_audio,
        elapsed_seconds=elapsed,
        messages=messages,
    )
