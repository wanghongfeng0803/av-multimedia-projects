"""视频亮度 / 对比度 / 色相 / 饱和度调色工具。

全工具像素统一为 **RGB** 字节序（ffmpeg ``rgb24``），无 BGR 分支。
处理顺序固定：亮度/对比度 LUT → 色相旋转 → 饱和度缩放。

对外主要入口：
- :func:`color_grading.pipeline.adjust`：调色并保留音轨
- :func:`color_grading.lut.build_lut` / :func:`clamp_params`：亮度/对比度映射表与钳制
- :func:`color_grading.colorspace.adjust_pixel` / :func:`clamp_color_params`：
  色相/饱和度的逐像素处理与钳制
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
from .colorspace import (
    HUE_MIN,
    HUE_MAX,
    SATURATION_MIN,
    SATURATION_MAX,
    adjust_pixel,
    clamp_color_params,
    map_chunk,
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
    # 亮度 / 对比度映射表
    "build_lut",
    "clamp_params",
    "is_identity",
    "IDENTITY_LUT",
    "BRIGHTNESS_MIN",
    "BRIGHTNESS_MAX",
    "CONTRAST_MIN",
    "CONTRAST_MAX",
    # 色相 / 饱和度
    "adjust_pixel",
    "map_chunk",
    "clamp_color_params",
    "HUE_MIN",
    "HUE_MAX",
    "SATURATION_MIN",
    "SATURATION_MAX",
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
