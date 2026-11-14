"""HLS MPEG-TS 切片打包工具。

公开 API 一览：

* :func:`package`            一键打包（探测 -> GOP 规划 -> 切片 -> M3U8）；
* :func:`probe_media`        只做 ffprobe 探测；
* :func:`plan_segments`      纯函数的 GOP 对齐分片规划；
* :func:`render_playlist`    渲染 M3U8 文本；
* :class:`PackagerError`     全部可预期错误的基类。
"""

from __future__ import annotations

from .errors import (
    BinaryNotFoundError,
    PackagerError,
    PlaylistError,
    ProbeError,
    SegmenterError,
    UnsupportedInputError,
)
from .m3u8 import PlaylistConfig, compute_target_duration, render_playlist
from .models import MediaInfo, PackagingResult, SegmentInfo, StreamInfo
from .packager import package
from .planner import plan_segments
from .probe import probe_media

__all__ = [
    "package",
    "probe_media",
    "plan_segments",
    "render_playlist",
    "compute_target_duration",
    "PlaylistConfig",
    "MediaInfo",
    "StreamInfo",
    "SegmentInfo",
    "PackagingResult",
    "PackagerError",
    "BinaryNotFoundError",
    "ProbeError",
    "UnsupportedInputError",
    "SegmenterError",
    "PlaylistError",
]

__version__ = "1.0.0"
