"""工具内统一的异常类型。

CLI 根据异常类型决定退出码；作为库使用时调用方可以按类型捕获。
三层架构（键控 / alpha 后处理 / 输出编码）抛出的错误都派生自
:class:`ChromaKeyError`，调用方只接基类即可兜住全部已知失败。
"""

from __future__ import annotations

__all__ = [
    "ChromaKeyError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
    "StreamTruncatedError",
    "SampleError",
    "OutputFormatError",
]


class ChromaKeyError(Exception):
    """所有色度键相关错误的基类。"""

    exit_code = 1


class BinaryNotFoundError(ChromaKeyError):
    """系统中找不到 ffmpeg / ffprobe 可执行文件。"""

    exit_code = 2


class InputNotFoundError(ChromaKeyError):
    """输入文件不存在或不是普通文件。"""

    exit_code = 3


class InvalidInputError(ChromaKeyError):
    """输入存在但无法作为视频处理（非媒体、无视频流、缺尺寸信息等）。"""

    exit_code = 4


class OutputExistsError(ChromaKeyError):
    """输出文件已存在且未允许覆盖。"""

    exit_code = 5


class FFmpegRunError(ChromaKeyError):
    """ffmpeg / ffprobe 执行失败。"""

    exit_code = 6

    def __init__(self, message: str, stderr: str = "", cmd: list[str] | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.cmd = cmd or []


class StreamTruncatedError(ChromaKeyError):
    """解码出的裸视频流在帧中途结束（通常意味着解码器异常退出）。"""

    exit_code = 7


class SampleError(ChromaKeyError):
    """键色采样失败：采样区域为空、全部落在帧外、帧数不足等。"""

    exit_code = 8


class OutputFormatError(ChromaKeyError):
    """输出格式无法承载所请求的内容（典型：mp4 无法写 alpha 通道）。"""

    exit_code = 9
