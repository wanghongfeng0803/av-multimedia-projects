"""工具内统一的异常类型。

CLI 根据异常类型决定退出码；作为库使用时调用方可以按类型捕获。
"""

from __future__ import annotations


class ComposeError(Exception):
    """所有视频合成相关错误的基类。"""

    exit_code = 1


class BinaryNotFoundError(ComposeError):
    """系统中找不到 ffmpeg / ffprobe 可执行文件。"""

    exit_code = 2


class InputNotFoundError(ComposeError):
    """输入文件不存在或不是普通文件。"""

    exit_code = 3


class InvalidInputError(ComposeError):
    """输入存在但无法作为可合成的媒体处理（非媒体、无视频流、时长未知等）。"""

    exit_code = 4


class OutputExistsError(ComposeError):
    """输出文件已存在且未允许覆盖。"""

    exit_code = 5


class FFmpegRunError(ComposeError):
    """ffmpeg / ffprobe 执行失败。"""

    exit_code = 6

    def __init__(self, message: str, stderr: str = "", cmd: list[str] | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.cmd = cmd or []
