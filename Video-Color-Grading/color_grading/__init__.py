"""视频亮度 / 对比度 / 色相 / 饱和度调色工具，支持 3D LUT 节点。

全工具像素统一为 **RGB** 字节序（ffmpeg ``rgb24``），无 BGR 分支。
传统四项的处理顺序固定：亮度/对比度 LUT → 色相旋转 → 饱和度缩放；
3D LUT 是可按位置（first/last）插入链路的独立节点，多节点自由组合见
:mod:`color_grading.nodes` 与 :func:`color_grading.pipeline.adjust_nodes`。

对外主要入口：
- :func:`color_grading.pipeline.adjust`：调色并保留音轨（含可选 3D LUT）
- :func:`color_grading.pipeline.adjust_nodes`：用显式有序节点链调色
- :func:`color_grading.lut.build_lut` / :func:`clamp_params`：亮度/对比度映射表与钳制
- :func:`color_grading.colorspace.adjust_pixel` / :func:`clamp_color_params`：
  色相/饱和度的逐像素处理与钳制
- :func:`color_grading.lut3d.load_cube` / :class:`~color_grading.lut3d.CubeLUT`：
  .cube 3D LUT 的容错解析与三线性插值
- :mod:`color_grading.nodes`：调色节点抽象与有序节点链
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
    LUTParseError,
    LUTDomainError,
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
from .lut3d import (
    STRENGTH_MIN,
    STRENGTH_MAX,
    MAX_GRID_SIZE,
    MIN_GRID_SIZE,
    CubeLUT,
    parse_cube,
    load_cube,
    apply_chunk as apply_lut3d_chunk,
    clamp_strength,
)
from .nodes import (
    GradeNode,
    BrightnessContrastNode,
    HueSaturationNode,
    LUT3DNode,
    apply_chain,
    is_identity_chain,
)
from .chunks import DEFAULT_CHUNK_ROWS, frame_chunk_sizes, map_stream
from .probe import VideoInfo, probe_video
from .pipeline import (
    LUT3D_POSITIONS,
    GradeResult,
    adjust,
    adjust_nodes,
    build_legacy_nodes,
)

__all__ = [
    # 核心 API
    "adjust",
    "adjust_nodes",
    "build_legacy_nodes",
    "GradeResult",
    "LUT3D_POSITIONS",
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
    # 3D LUT
    "CubeLUT",
    "parse_cube",
    "load_cube",
    "apply_lut3d_chunk",
    "clamp_strength",
    "STRENGTH_MIN",
    "STRENGTH_MAX",
    "MAX_GRID_SIZE",
    "MIN_GRID_SIZE",
    # 节点链
    "GradeNode",
    "BrightnessContrastNode",
    "HueSaturationNode",
    "LUT3DNode",
    "apply_chain",
    "is_identity_chain",
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
    "LUTParseError",
    "LUTDomainError",
]
