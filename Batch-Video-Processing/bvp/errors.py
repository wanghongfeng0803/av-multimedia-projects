"""单文件核心的异常类型体系。

每个异常都通过构造参数绑定一个 :class:`~bvp.contract.ErrorCategory`，
该绑定是「单文件核心 → 批量层」错误分类契约的核心侧落点：批量层只通过
:func:`bvp.contract.classify_error` 读取 ``category``，不解析消息文本。

设计约束：

- 所有核心可预期错误都继承 :class:`TranscodeError`，其 ``category`` 在
  类定义处固定，调用方无法在 raise 时临时篡改，保证分类稳定；
- 异常消息（``str(exc)``）只供人读，可以随实现自由改写，不承担契约职责；
- CLI 可依据分类决定退出码，库调用方按异常类型或分类捕获即可。
"""

from __future__ import annotations

from .contract import ErrorCategory

__all__ = [
    "TranscodeError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
    "EncoderUnavailableError",
]


class TranscodeError(Exception):
    """所有单文件核心可预期错误的基类。

    :cvar category: 该错误对应的稳定分类，子类必须覆盖；
        基类默认 :data:`~ErrorCategory.TRANSCODE_FAILED` 仅作兜底。
    :cvar exit_code: CLI 退出码建议值。
    """

    category: ErrorCategory = ErrorCategory.TRANSCODE_FAILED
    exit_code = 1


class BinaryNotFoundError(TranscodeError):
    """系统中找不到 ffmpeg / ffprobe 可执行文件。"""

    category = ErrorCategory.BINARY_MISSING
    exit_code = 2


class InputNotFoundError(TranscodeError):
    """输入文件不存在、不是普通文件或无法访问（权限/IO 错误）。"""

    category = ErrorCategory.INPUT_NOT_FOUND
    exit_code = 3

    def __init__(self, message: str, *, os_error: OSError | None = None):
        super().__init__(message)
        # 保留原始 OSError 便于排查，但批量层不读取它，只认 category
        self.os_error = os_error


class InvalidInputError(TranscodeError):
    """输入存在但无法作为可转码视频处理（非媒体/无法识别/无视频流/空文件）。

    该异常是「跳过 vs 失败」边界的核心侧表达：它属于
    :data:`~bvp.contract.SKIP_CATEGORIES`，批量层据此记为跳过而非失败。
    """

    category = ErrorCategory.NOT_A_VIDEO
    exit_code = 4


class OutputExistsError(TranscodeError):
    """输出文件已存在且未允许覆盖。"""

    category = ErrorCategory.OUTPUT_EXISTS
    exit_code = 5


class FFmpegRunError(TranscodeError):
    """ffmpeg / ffprobe 执行失败，或执行后产物复核不过。"""

    category = ErrorCategory.TRANSCODE_FAILED
    exit_code = 6

    def __init__(self, message: str, stderr: str = "",
                 cmd: list[str] | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.cmd = cmd or []


class EncoderUnavailableError(TranscodeError):
    """本机 ffmpeg 不支持目标编码所需的任何编码器。"""

    category = ErrorCategory.ENCODER_UNAVAILABLE
    exit_code = 7

    def __init__(self, message: str, codec: str = "",
                 candidates: list[str] | None = None):
        super().__init__(message)
        self.codec = codec
        self.candidates = candidates or []
