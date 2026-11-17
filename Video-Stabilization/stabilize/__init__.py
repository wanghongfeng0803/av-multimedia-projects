"""stabilize：视频防抖（零三方依赖），估计 → 平滑 → 补偿 → 裁剪 分阶段。

全局运动只建模「平移 + 绕中心旋转」两类，分别估计、分别输出：

- :mod:`stabilize.motion`：缩小灰度帧上的块匹配平移与环形签名
  循环互相关旋转，交替精修，场景切换显式置无效；
- :mod:`stabilize.smoothing`：无效帧插值、刚体轨迹累积、滑动
  平均/高斯平滑，并直接推导四边裁剪 inset；
- :mod:`stabilize.compensate`：sendcmd 驱动 ffmpeg rotate+crop
  完成像素补偿，裁剪作为独立子步骤复用同一份 smooth JSON。

两个稳定 JSON 契约（版本化，后续阶段只依赖契约）：

- ``stabilize-motion`` v1：逐帧增量运动（估计阶段产物）；
- ``stabilize-smooth`` v1：轨迹、补偿量与裁剪参数（平滑阶段产物）。

原视频在任何阶段都只读不写，每一步的输出都是独立文件，可单独
重跑、可人工检查参数后再执行补偿。
"""

from .fft_mod import (
    circular_correlation,
    fft,
    ifft,
    next_pow2,
    peak_lag,
)
from .motion import (
    ANGLE_BINS,
    EST_WIDTH,
    RING_COUNT,
    SCENE_CUT_RATIO,
    bilinear_sample,
    block_match,
    downsample,
    estimate_pair,
    estimate_rotation,
    is_scene_cut,
    parabolic_refine,
    ring_signature,
)
from .smoothing import (
    CROP_SAFETY_PX,
    DEFAULT_RADIUS,
    MAX_CROP_RATIO,
    accumulate_trajectory,
    clamp_compensation,
    compute_crop,
    dynamic_insets,
    gaussian_weights,
    interpolate_invalid,
    moving_average,
    rotation_inset,
    smooth_motion,
    smooth_trajectory,
)
from .pipeline import (
    MOTION_FORMAT,
    SMOOTH_FORMAT,
    PipelineError,
    dump_json,
    estimate_motion,
    fps_to_float,
    load_json,
    probe,
)
from .compensate import (
    BORDER_MODES,
    DEFAULT_CRF,
    build_sendcmd,
    build_sendcmd_dynamic,
    build_sendcmd_mirror,
    compensate_video,
    crop_video,
    mirror_margin,
)

__all__ = [
    # fft
    "fft", "ifft", "next_pow2", "circular_correlation", "peak_lag",
    # motion
    "EST_WIDTH", "RING_COUNT", "ANGLE_BINS", "SCENE_CUT_RATIO",
    "downsample", "block_match", "parabolic_refine", "bilinear_sample",
    "ring_signature", "estimate_rotation", "estimate_pair", "is_scene_cut",
    # smoothing
    "DEFAULT_RADIUS", "MAX_CROP_RATIO", "CROP_SAFETY_PX",
    "interpolate_invalid", "accumulate_trajectory", "moving_average",
    "gaussian_weights", "smooth_trajectory", "rotation_inset",
    "compute_crop", "clamp_compensation", "dynamic_insets", "smooth_motion",
    # pipeline
    "MOTION_FORMAT", "SMOOTH_FORMAT", "PipelineError",
    "probe", "fps_to_float", "estimate_motion", "load_json", "dump_json",
    # compensate
    "DEFAULT_CRF", "BORDER_MODES",
    "build_sendcmd", "build_sendcmd_dynamic", "build_sendcmd_mirror",
    "mirror_margin", "compensate_video", "crop_video",
]
