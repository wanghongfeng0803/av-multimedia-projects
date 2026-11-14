"""自定义异常体系。

所有可预期的错误都继承 :class:`PackagerError`，CLI 层据此区分
"输入/环境问题"（退出码 2）与"未知内部错误"（退出码 3）。
"""

from __future__ import annotations


class PackagerError(Exception):
    """HLS 打包过程中的可预期错误基类。"""


class BinaryNotFoundError(PackagerError):
    """系统中找不到 ffmpeg / ffprobe 可执行文件。"""


class ProbeError(PackagerError):
    """ffprobe 无法解析输入文件（文件不存在、损坏或格式不支持）。"""


class UnsupportedInputError(PackagerError):
    """输入缺少可用的音视频轨道，或编码无法直接写入 MPEG-TS。"""


class SegmenterError(PackagerError):
    """ffmpeg 切片子进程执行失败。"""

    def __init__(self, message: str, stderr_tail: str = "", returncode: int | None = None):
        super().__init__(message)
        #: ffmpeg 输出的最后一段日志，便于定位失败原因
        self.stderr_tail = stderr_tail
        self.returncode = returncode


class PlaylistError(PackagerError):
    """生成 M3U8 播放列表时参数不合法。"""
