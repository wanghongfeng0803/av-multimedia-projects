"""MP4 -> H.264 视频转码工具（基础版本）。

对外主要入口：
- :func:`transcode_tool.probe.probe_media`：探测输入媒体信息
- :func:`transcode_tool.transcoder.transcode`：执行转码
"""

from .errors import (
    TranscodeError,
    BinaryNotFoundError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    FFmpegRunError,
)
from .models import MediaInfo, StreamInfo, TranscodeResult
from .probe import probe_media
from .transcoder import transcode

__all__ = [
    "probe_media",
    "transcode",
    "MediaInfo",
    "StreamInfo",
    "TranscodeResult",
    "TranscodeError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
]
