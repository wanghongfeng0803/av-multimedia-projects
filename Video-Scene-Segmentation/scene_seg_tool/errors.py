"""场景分割工具内统一的异常类型。

CLI 根据异常类型决定退出码；作为库使用时调用方可以按类型捕获。
"""

from __future__ import annotations


class SceneSegError(Exception):
    """所有场景分割相关错误的基类。"""

    exit_code = 1


class BinaryNotFoundError(SceneSegError):
    """系统中找不到 ffmpeg / ffprobe 可执行文件。"""

    exit_code = 2


class InputNotFoundError(SceneSegError):
    """输入文件不存在或不是普通文件。"""

    exit_code = 3


class InvalidInputError(SceneSegError):
    """输入存在但无法作为可解码的视频处理（非媒体、无视频流等）。"""

    exit_code = 4


class FFmpegRunError(SceneSegError):
    """ffmpeg / ffprobe 执行失败或输出了不完整的帧数据。"""

    exit_code = 6

    def __init__(self, message: str, stderr: str = "", cmd: list[str] | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.cmd = cmd or []
