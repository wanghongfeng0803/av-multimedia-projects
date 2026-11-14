"""ffmpeg 子进程运行器：流式进度解析 + 低内存占用。

与直接 ``subprocess.run`` 捕获全部输出相比，本模块针对长任务/大文件做了两件事：

1. **stderr 直接落盘**：ffmpeg 的统计日志随视频时长线性增长，全量读进内存
   会让几个 GB 的转码占用大量 RAM。这里边产生边写入日志文件，失败时只把
   末尾片段带回异常，供调用方展示。
2. **机器可读进度流**：通过 ``-progress pipe:1`` 让 ffmpeg 把
   ``out_time_us`` / ``progress`` 等键值输出到 stdout，逐行解析后通过回调
   上报 0.0-1.0 的完成比例，无需正则解析人类可读的 stderr。

stdout（进度流）与 stderr（日志流）分别用独立线程读取，避免任一管道
缓冲区写满后子进程阻塞造成死锁。
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Callable

from .errors import FFmpegRunError

# 进度回调：参数为 0.0-1.0 的完成比例（单调不减）
ProgressCallback = Callable[[float], None]

# 带回异常的 stderr 末尾最大字符数
_STDERR_TAIL_CHARS = 2000
# 读取管道时的块大小
_CHUNK_SIZE = 8192


def run_ffmpeg(
    cmd: list[str],
    *,
    log_path: str | None = None,
    total_duration: float | None = None,
    progress_callback: ProgressCallback | None = None,
    last_ratio: float = 0.0,
    ratio_span: float = 1.0,
) -> tuple[int, str]:
    """执行 ffmpeg 命令并流式解析进度。

    :param cmd: 完整命令（调用方应包含 ``-progress pipe:1 -nostats``）
    :param log_path: stderr 落盘路径；为 None 时写入临时缓冲（仅保留尾部）
    :param total_duration: 输入总时长（秒），用于把 out_time 换算成完成比例；
        未知时不上报比例，仅消费进度流
    :param progress_callback: 进度回调，收到本任务区间内的全局比例
    :param last_ratio: 分段场景中本段之前已完成的全局比例
    :param ratio_span: 分段场景中本段占全局比例的跨度
    :returns: ``(returncode, stderr_tail)``
    :raises FFmpegRunError: 进程无法启动或被信号终止
    """
    stderr_chunks: list[str] = []
    tail_keeper = _TailKeeper(stderr_chunks)

    log_handle = None
    if log_path:
        # 行缓冲 + utf-8，保证外部也能 tail 实时查看进度
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")

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

    # 当前线程消费 stdout 进度流（键值对，块小、频率低）
    try:
        _consume_progress(
            proc, total_duration, progress_callback, last_ratio, ratio_span
        )
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

    def __init__(self, sink: list[str], limit: int = _STDERR_TAIL_CHARS * 2):
        self._sink = sink
        self._limit = limit
        self._buffer = ""

    def feed(self, chunk: str) -> None:
        self._buffer += chunk
        if len(self._buffer) > self._limit:
            self._buffer = self._buffer[-self._limit:]
        if not self._sink:
            self._sink.append(self._buffer)
        else:
            self._sink[0] = self._buffer

    def text(self) -> str:
        return self._buffer


def _consume_progress(
    proc: subprocess.Popen,
    total_duration: float | None,
    callback: ProgressCallback | None,
    last_ratio: float,
    ratio_span: float,
) -> None:
    """逐行读取 ffmpeg 的 key=value 进度流并上报完成比例。"""
    assert proc.stdout is not None
    last_emitted = last_ratio
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
            global_ratio = last_ratio + ratio * ratio_span
            global_ratio = min(max(global_ratio, 0.0), 1.0)
            if global_ratio > last_emitted:
                last_emitted = global_ratio
                if callback is not None:
                    _safe_callback(callback, global_ratio)

    if callback is not None and total_duration:
        # 正常结束时确保比例收敛到本段 100%
        end_ratio = min(last_ratio + ratio_span, 1.0)
        if end_ratio > last_emitted:
            _safe_callback(callback, end_ratio)


def _ratio_from_progress(
    out_time_us: int | None, total_duration: float | None, is_end: bool
) -> float | None:
    if not total_duration or total_duration <= 0:
        return 1.0 if is_end else None
    if out_time_us is None:
        return 1.0 if is_end else None
    ratio = (out_time_us / 1_000_000) / total_duration
    if is_end:
        ratio = 1.0
    return min(max(ratio, 0.0), 1.0)


def _safe_callback(callback: ProgressCallback, ratio: float) -> None:
    """用户回调抛异常不应杀死转码进程，静默忽略即可。"""
    try:
        callback(ratio)
    except Exception:
        pass


def build_progress_args() -> list[str]:
    """ffmpeg 开启机器可读进度、关闭人类可读统计的标准参数。"""
    return ["-progress", "pipe:1", "-nostats"]


def ensure_log_dir(log_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(log_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
