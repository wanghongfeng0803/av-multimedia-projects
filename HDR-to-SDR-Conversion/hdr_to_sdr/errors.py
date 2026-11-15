"""异常体系：HDR→SDR 转换各阶段统一抛出的错误类型。"""

from __future__ import annotations

__all__ = [
    "HDRConversionError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "UnsupportedHDRTypeError",
    "FFmpegRunError",
]


class HDRConversionError(Exception):
    """本项目所有自定义异常的基类。"""


class BinaryNotFoundError(HDRConversionError):
    """PATH 与环境变量中都找不到 ffmpeg / ffprobe。"""


class InputNotFoundError(HDRConversionError):
    """输入文件路径不存在或不是普通文件。"""


class InvalidInputError(HDRConversionError):
    """输入无法被 ffprobe 解析，或缺少必需的视频流。"""


class UnsupportedHDRTypeError(HDRConversionError):
    """无法从元数据**确证**输入是 PQ 还是 HLG。

    设计原则（需求明确要求「不能靠猜」）：transfer 特征缺失或为 SDR 时，
    必须显式报错退出；只有调用方用 ``assume`` 显式指定时才允许继续。
    """


class FFmpegRunError(HDRConversionError):
    """ffmpeg 子进程以非零状态退出，携带末尾 stderr 便于定位。"""

    def __init__(self, cmd: list[str], returncode: int, stderr_tail: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        super().__init__(
            f"ffmpeg 执行失败（exit={returncode}）: {' '.join(cmd)}\n"
            f"stderr 末尾:\n{stderr_tail}"
        )
