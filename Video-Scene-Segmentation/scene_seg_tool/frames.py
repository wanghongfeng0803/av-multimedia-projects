"""ffmpeg 灰度原始流的单遍流式解码。

核心约束：**整段视频只解码一次，逐帧产出，不回放、不重复读取**。
ffmpeg 被要求输出 ``rawvideo`` + ``gray`` 像素格式（8-bit 单通道），
其 stdout 就是连续的 ``宽×高`` 字节帧，无需任何容器解析或图像库参与。

工程细节
--------
* stderr 由独立守护线程持续排空。stderr 管道容量很小（Windows 上约 4KB），
  若主线程只读取 stdout，ffmpeg 写满 stderr 后会阻塞，进而导致两个管道
  互相等待的死锁；
* 每帧按精确字节数循环读满，网络/管道级的短读不影响帧边界；
* 流末尾若残留不足一帧的字节，判定为损坏输入并显式报错，而不是悄悄
  用残缺帧参与统计；
* 生成器提前关闭（消费方异常或主动 break）时，finally 中终止子进程，
  确保不遗留 ffmpeg 进程与管道句柄。
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Iterator

from .binaries import find_ffmpeg
from .errors import FFmpegRunError

__all__ = ["stream_gray_frames", "read_exact_frame", "MAX_STDERR_BYTES"]

#: stderr 环形缓冲保留的最大字节数（只保留末尾，足够定位错误）
MAX_STDERR_BYTES = 64 * 1024

# Windows 下隐藏子进程控制台窗口；其它平台为空字典
_NO_WINDOW_KWARGS: dict[str, object] = (
    {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if os.name == "nt"
    else {}
)


def _drain_stderr(pipe: subprocess.PIPE, sink: bytearray) -> None:
    """持续排空子进程 stderr，只保留末尾 ``MAX_STDERR_BYTES`` 字节。"""
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            sink.extend(chunk)
            if len(sink) > MAX_STDERR_BYTES:
                del sink[: len(sink) - MAX_STDERR_BYTES]
    finally:
        pipe.close()


def read_exact_frame(pipe, frame_size: int) -> bytes | None:
    """从管道精确读取一帧的 ``frame_size`` 字节。

    :returns: 完整帧字节；流干净结束（EOF 且无残留）时返回 ``None``。
    :raises FFmpegRunError: EOF 处残留不足一帧的字节（损坏/截断的输入）。
    """
    chunks: list[bytes] = []
    remaining = frame_size
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining == frame_size:
        return None  # 干净 EOF
    if remaining > 0:
        got = frame_size - remaining
        raise FFmpegRunError(
            f"解码流在帧中途结束：期望 {frame_size} 字节，仅得到 {got} 字节，"
            "输入视频可能已损坏或被截断"
        )
    return b"".join(chunks)


def stream_gray_frames(
    path: str,
    width: int,
    height: int,
    *,
    ffmpeg: str | None = None,
    extra_args: list[str] | None = None,
) -> Iterator[bytes]:
    """以生成器形式逐帧产出 gray 原始字节，整段视频单遍解码。

    :param path: 输入视频路径。
    :param width: 探测得到的帧宽。
    :param height: 探测得到的帧高。
    :param extra_args: 追加到 ffmpeg 命令行的额外参数（插入在输出格式
                       参数之前），供测试与高级用途使用。
    :raises FFmpegRunError: ffmpeg 非零退出，或帧数据不完整。

    命令行选择 ``-map 0:v:0`` 固定首条视频流、``-pix_fmt gray`` 让 ffmpeg
    完成色度到灰度的转换（BT.609 加权由 ffmpeg 内部实现，本项目不重复
    造轮子；项目自研范围是直方图统计与差异度量，而非色彩科学），
    ``-vsync`` 保持默认以尊重源时间戳。
    """
    executable = ffmpeg or find_ffmpeg()
    frame_size = width * height
    cmd = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-i", path,
        "-map", "0:v:0",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_NO_WINDOW_KWARGS,
    )
    stderr_sink = bytearray()
    stderr_thread = threading.Thread(
        target=_drain_stderr,
        args=(process.stderr, stderr_sink),
        name="ffmpeg-stderr-drainer",
        daemon=True,
    )
    stderr_thread.start()

    try:
        while True:
            frame = read_exact_frame(process.stdout, frame_size)
            if frame is None:
                break
            yield frame
    except GeneratorExit:
        # 消费方提前终止：保证子进程被回收，不留下僵尸 ffmpeg
        raise
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        # 等待 ffmpeg 自然退出；消费方提前 break 时它通常仍在写管道，
        # 先给一个短窗口，再强制终止
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        stderr_thread.join(timeout=2.0)

    if process.returncode != 0:
        raise FFmpegRunError(
            f"ffmpeg 解码失败（退出码 {process.returncode}）: {path}",
            stderr=stderr_sink.decode("utf-8", errors="replace"),
            cmd=cmd,
        )
