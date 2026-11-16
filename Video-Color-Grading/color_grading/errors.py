"""工具内统一的异常类型。

CLI 根据异常类型决定退出码；作为库使用时调用方可以按类型捕获。
"""

from __future__ import annotations


class ColorGradingError(Exception):
    """所有调色相关错误的基类。"""

    exit_code = 1


class BinaryNotFoundError(ColorGradingError):
    """系统中找不到 ffmpeg / ffprobe 可执行文件。"""

    exit_code = 2


class InputNotFoundError(ColorGradingError):
    """输入文件不存在或不是普通文件。"""

    exit_code = 3


class InvalidInputError(ColorGradingError):
    """输入存在但无法作为视频处理（非媒体、无视频流、缺尺寸信息等）。"""

    exit_code = 4


class OutputExistsError(ColorGradingError):
    """输出文件已存在且未允许覆盖。"""

    exit_code = 5


class FFmpegRunError(ColorGradingError):
    """ffmpeg / ffprobe 执行失败。"""

    exit_code = 6

    def __init__(self, message: str, stderr: str = "", cmd: list[str] | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.cmd = cmd or []


class StreamTruncatedError(ColorGradingError):
    """解码出的裸视频流在帧中途结束（通常意味着解码器异常退出）。"""

    exit_code = 7
