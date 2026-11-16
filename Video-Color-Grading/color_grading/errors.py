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


class LUTParseError(ColorGradingError):
    """.cube 文件无法解析（格式损坏、关键字缺失/重复、数据不足等）。

    所有错误都携带可定位信息：``file_path`` 指明是哪个文件，
    ``line_number`` 指明触发问题的行（``None`` 表示问题属于整个
    文件而非某一行，如完全没有 ``LUT_3D_SIZE``）。错误消息本身
    已包含行号，CLI 可直接展示。
    """

    exit_code = 8

    def __init__(self, message: str, file_path: str = "", line_number: int | None = None):
        located = message
        if file_path:
            located = f"{file_path}: {located}"
        if line_number is not None:
            located = f"第 {line_number} 行: {located}"
        super().__init__(located)
        self.file_path = file_path
        self.line_number = line_number


class LUTDomainError(ColorGradingError):
    """.cube 的域（DOMAIN_*）或尺寸与本工具约定不匹配，拒绝硬套。

    本工具只接受未显式声明非默认域、即归一化 [0, 1] 的 3D LUT；
    任何偏移 / 缩放 / 负数域都无法在 rgb24 字节域上直接套用，
    一律报错而不是静默钳制或重映射。
    """

    exit_code = 9
