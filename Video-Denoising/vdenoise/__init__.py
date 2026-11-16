"""vdenoise：均值滤波视频降噪（零三方依赖）。

流水线：ffmpeg 解码 → 可分离盒式滤波（分带 + 接缝加权合并）→
ffmpeg 编码，音轨 ``-c:a copy`` 直通，像素格式自适应
（RGB 工作格式自动切换 libx264rgb）。
"""

from .filter import horizontal_pass_row, vertical_pass, box_blur_plane
from .bands import plan_bands, box_blur_plane_banded, merge_rows
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
from .pipeline import probe, denoise_video, PipelineError

__all__ = [
    # filter
    "horizontal_pass_row",
    "vertical_pass",
    "box_blur_plane",
    # bands
    "plan_bands",
    "box_blur_plane_banded",
    "merge_rows",
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
]
