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
    WatermarkError,
)
from .m3u8 import PlaylistConfig, compute_target_duration, render_playlist
from .models import MediaInfo, PackagingResult, SegmentInfo, StreamInfo
from .packager import package
from .planner import plan_segments
from .probe import probe_media
from .watermark import (
    WatermarkConfig,
    build_watermarked_segment_command,
    decode_utf8,
    escape_filter_path,
    render_text_png,
)

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
    "WatermarkError",
    "WatermarkConfig",
    "render_text_png",
    "decode_utf8",
    "escape_filter_path",
    "build_watermarked_segment_command",
]

__version__ = "1.0.0"
