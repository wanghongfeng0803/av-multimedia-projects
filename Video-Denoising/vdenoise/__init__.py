"""vdenoise：视频降噪（零三方依赖），均值、中值、时域三种模式。

流水线：ffmpeg 解码 → 滤波（空间分带 / 时域帧缓存）→ ffmpeg 编码，
音轨 ``-c:a copy`` 直通，像素格式自适应（RGB 工作格式自动切换
libx264rgb）。

- 均值模式（默认）：可分离盒式滤波（分带 + 接缝加权合并）；
- 中值模式：Huang 直方图加速中值滤波，镜像边界，专治椒盐噪声；
- 时域模式：全局位移估计对齐后再做时间窗混合，运动区掩码防拖影，
  缓存帧数 / 片长 / 内存预算三者互推、超预算提前拒绝。
"""

from .filter import horizontal_pass_row, vertical_pass, box_blur_plane
from .bands import plan_bands, box_blur_plane_banded, merge_rows
from .median import (
    border_mask,
    dump_border_frame,
    median_plane,
    median_plane_banded,
    mirror_index,
    mirror_indices,
    plan_median_bands,
)
from .temporal import (
    DEFAULT_MOTION_THRESHOLD,
    MAX_CACHE_FRAMES,
    FrameCache,
    derive_band_height,
    derive_cache_frames,
    estimate_shift,
    plan_temporal,
    temporal_blend_plane_banded,
    temporal_memory_bytes,
)
from .memory import available_memory, derive_budget
from .pixfmt import (
    SUPPORTED_FORMATS,
    plane_layout,
    filter_dims,
    frame_size,
    working_format,
    encoder_for,
    pix_fmt_matches,
    split_planes,
    join_planes,
)
from .pipeline import probe, denoise_video, PipelineError, FILTER_MODES

__all__ = [
    # filter（均值）
    "horizontal_pass_row",
    "vertical_pass",
    "box_blur_plane",
    # bands（均值分带）
    "plan_bands",
    "box_blur_plane_banded",
    "merge_rows",
    # median（中值）
    "median_plane",
    "median_plane_banded",
    "plan_median_bands",
    "mirror_index",
    "mirror_indices",
    "border_mask",
    "dump_border_frame",
    # temporal（时域）
    "FrameCache",
    "plan_temporal",
    "temporal_memory_bytes",
    "derive_cache_frames",
    "derive_band_height",
    "estimate_shift",
    "temporal_blend_plane_banded",
    "MAX_CACHE_FRAMES",
    "DEFAULT_MOTION_THRESHOLD",
    # memory
    "available_memory",
    "derive_budget",
    # pixfmt
    "SUPPORTED_FORMATS",
    "plane_layout",
    "filter_dims",
    "frame_size",
    "working_format",
    "encoder_for",
    "pix_fmt_matches",
    "split_planes",
    "join_planes",
    # pipeline
    "probe",
    "denoise_video",
    "PipelineError",
    "FILTER_MODES",
]
