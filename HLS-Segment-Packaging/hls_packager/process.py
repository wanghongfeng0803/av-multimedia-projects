"""子进程执行工具。

统一封装 subprocess.run 的编码与超时处理，让 probe / ts_writer
只关心命令本身。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def stderr_text(self, max_chars: int = 4000) -> str:
        """返回 stderr 末尾的文本（ffmpeg 日志的关键信息通常在最后）。"""
        text = self.stderr.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            text = "..." + text[-max_chars:]
        return text


def run_capture(argv: list[str], timeout: float | None = None) -> CommandResult:
    """执行命令并捕获全部输出；不抛 CalledProcessError，由调用方判断。"""
    completed = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout or b"",
        stderr=completed.stderr or b"",
    )
