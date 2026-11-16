"""视频亮度 / 对比度调色工具。

对外主要入口：
- :func:`color_grading.pipeline.adjust`：调整亮度/对比度并保留音轨
- :func:`color_grading.lut.build_lut` / :func:`clamp_params`：映射表构建与参数钳制
- :func:`color_grading.chunks.map_stream`：rgb24 裸流的分块并行映射
- :func:`color_grading.probe.probe_video`：探测视频元信息
"""

from .errors import (
    ColorGradingError,
    BinaryNotFoundError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    FFmpegRunError,
    StreamTruncatedError,
)
from .lut import (
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
    CONTRAST_MIN,
    CONTRAST_MAX,
    IDENTITY_LUT,
    build_lut,
    clamp_params,
    is_identity,
)
from .chunks import DEFAULT_CHUNK_ROWS, frame_chunk_sizes, map_stream
from .probe import VideoInfo, probe_video
from .pipeline import GradeResult, adjust

__all__ = [
    # 核心 API
    "adjust",
    "GradeResult",
    "probe_video",
    "VideoInfo",
    # 映射表
    "build_lut",
    "clamp_params",
    "is_identity",
    "IDENTITY_LUT",
    "BRIGHTNESS_MIN",
    "BRIGHTNESS_MAX",
    "CONTRAST_MIN",
    "CONTRAST_MAX",
    # 分块并行
    "map_stream",
    "frame_chunk_sizes",
    "DEFAULT_CHUNK_ROWS",
    # 异常
    "ColorGradingError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
    "StreamTruncatedError",
]
