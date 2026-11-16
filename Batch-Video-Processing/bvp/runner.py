"""ffmpeg 子进程运行器：流式进度解析 + 低内存占用。

两个必须守住的工程点（详见项目记忆中的 stderr 管道死锁教训）：

1. **stderr 直接落盘**：ffmpeg 统计日志随视频时长线性增长，全量读进内存
   会让长视频占用大量 RAM；这里边产生边写日志文件，失败时只把末尾片段
   带回异常。
2. **两条管道各自独立消费**：stdout（``-progress pipe:1`` 进度流）与
   stderr（日志流）用独立线程读取，任一管道缓冲区写满都会阻塞子进程，
   只读其中一条就会形成管道死锁。
"""

from __future__ import annotations

import subprocess
import threading
from typing import Callable

from .errors import FFmpegRunError

__all__ = ["run_ffmpeg", "build_progress_args", "ProgressCallback"]

# 进度回调：参数为 0.0-1.0 的完成比例（单调不减）
ProgressCallback = Callable[[float], None]

_STDERR_TAIL_CHARS = 2000
_CHUNK_SIZE = 8192


def run_ffmpeg(
    cmd: list[str],
    *,
    log_path: str | None = None,
    total_duration: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[int, str]:
    """执行 ffmpeg 命令并流式解析进度。

    :param cmd: 完整命令（调用方应包含 ``-progress pipe:1 -nostats``）
    :param log_path: stderr 落盘路径；None 时仅在内存保留尾部
    :param total_duration: 输入总时长（秒），用于把 out_time 换算成完成比例；
        未知时不上报比例，仅消费进度流
    :param progress_callback: 进度回调，接收 0.0-1.0 的完成比例
    :returns: ``(returncode, stderr_tail)``
    :raises FFmpegRunError: 进程无法启动
    """
    tail_keeper = _TailKeeper()

    log_handle = None
    if log_path:
        log_handle = open(
            log_path, "w", encoding="utf-8", errors="replace"
        )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        if log_handle:
            log_handle.close()
        raise FFmpegRunError(
            f"无法启动 ffmpeg 进程: {exc}", stderr=str(exc), cmd=cmd
        ) from exc

    stderr_tail = ""

    def _drain_stderr() -> None:
        nonlocal stderr_tail
        assert proc.stderr is not None
        while True:
            chunk = proc.stderr.read(_CHUNK_SIZE)
            if chunk == "":
                break
            if log_handle is not None:
                log_handle.write(chunk)
                log_handle.flush()
            tail_keeper.feed(chunk)
        stderr_tail = tail_keeper.text()

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        _consume_progress(proc, total_duration, progress_callback)
    finally:
        stderr_thread.join()
        if log_handle is not None:
            log_handle.close()

    returncode = proc.wait()
    if not stderr_tail:
        stderr_tail = tail_keeper.text()
    return returncode, stderr_tail


class _TailKeeper:
    """只保留写入内容的尾部，避免长任务的 stderr 无限占用内存。"""

    def __init__(self, limit: int = _STDERR_TAIL_CHARS * 2):
        self._limit = limit
        self._buffer = ""

    def feed(self, chunk: str) -> None:
        self._buffer += chunk
        if len(self._buffer) > self._limit:
            self._buffer = self._buffer[-self._limit:]

    def text(self) -> str:
        return self._buffer


def _consume_progress(
    proc: subprocess.Popen,
    total_duration: float | None,
    callback: ProgressCallback | None,
) -> None:
    """逐行读取 ffmpeg 的 key=value 进度流并上报完成比例。"""
    assert proc.stdout is not None
    last_emitted = 0.0
    current_out_us: int | None = None

    for line in proc.stdout:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "out_time_us" and value.isdigit():
            current_out_us = int(value)
        elif key == "out_time_ms" and current_out_us is None and value.isdigit():
            # 老版本 ffmpeg 的 out_time_ms 实际也是微秒
            current_out_us = int(value)
        elif key == "progress":
            ratio = _ratio_from_progress(
                current_out_us, total_duration, value == "end"
            )
            current_out_us = None
            if ratio is None:
                continue
            ratio = min(max(ratio, 0.0), 1.0)
            if ratio > last_emitted:
                last_emitted = ratio
                _safe_callback(callback, ratio)

    if callback is not None:
        # 正常结束时确保比例收敛到 100%
        if 1.0 > last_emitted:
            _safe_callback(callback, 1.0)


def _ratio_from_progress(
    out_time_us: int | None, total_duration: float | None, is_end: bool
) -> float | None:
    if is_end:
        return 1.0
    if not total_duration or total_duration <= 0 or out_time_us is None:
        return None
    return (out_time_us / 1_000_000) / total_duration


def _safe_callback(callback: ProgressCallback | None, ratio: float) -> None:
    """用户回调抛异常不应杀死转码进程。"""
    if callback is None:
        return
    try:
        callback(ratio)
    except Exception:
        pass


def build_progress_args() -> list[str]:
    """ffmpeg 开启机器可读进度、关闭人类可读统计的标准参数。"""
    return ["-progress", "pipe:1", "-nostats"]
