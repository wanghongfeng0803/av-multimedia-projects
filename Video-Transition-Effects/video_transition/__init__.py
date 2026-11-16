"""video_transition：从零实现的两段素材转场工具包（交叉淡化 + 四方向滑动 + 擦除）。

只依赖 Python 标准库与系统 ffmpeg/ffprobe。核心承诺：

* 成片总帧数恒为 ``Na + Nb - No``，重叠区之外一帧不丢；
* 视频帧与音频采样共用同一张帧映射表与同一条进度曲线，音视频严格同帧；
* 全部转场帧统一经 :class:`FrameReader` 的 RGB24 管道读取，淡化、滑动
  与擦除渲染复用同一套确定性整数帧混合器；
* 滑动两帧满幅互补覆盖、擦除掩码分区无缝铺满，边界构造上不露黑边；
* 解码帧数、写出帧数三方核对，不符即报错，不静默丢帧。
"""

from .audio import (
    S16_CHANNELS,
    assemble_audio,
    choose_sample_rate,
    frame_sample_bounds,
    overlap_weight_fraction,
    silence_pcm,
)
from .blend import blend_rgb24_frame, blend_s16_samples, div_round_half_away
from .easing import (
    EASINGS,
    EASING_EASE_IN,
    EASING_EASE_IN_OUT,
    EASING_EASE_OUT,
    EASING_LINEAR,
    eased_fraction,
    eased_shift,
)
from .errors import PipelineError, PlanError, ProbeError, TransitionError
from .pipeline import FrameWriter, TransitionResult, cross_fade
from .plan import (
    KIND_HEAD,
    KIND_OVERLAP,
    KIND_TAIL,
    TRANSITIONS,
    TRANSITION_FADE,
    TRANSITION_SLIDE,
    TRANSITION_WIPE,
    FrameJob,
    Segment,
    TransitionPlan,
    build_plan,
    fade_weight_b,
    frame_jobs,
    transition_gain_fraction,
)
from .probe import ClipMeta, ffmpeg_bin, ffprobe_bin, parse_rate, probe_clip
from .reader import FrameReader, load_pcm_bytes, read_pcm_track, read_rgb24_frames
from .slide import (
    DIRECTIONS,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    HORIZONTAL_DIRECTIONS,
    VERTICAL_DIRECTIONS,
    composite_slide_frame,
    slide_regions,
)
from .wipe import (
    WipeSoftWidthWarning,
    clamp_soft_width,
    composite_wipe_frame,
    wipe_edge,
    wipe_segments,
)

__all__ = [
    # 异常
    "TransitionError", "ProbeError", "PlanError", "PipelineError",
    # 探针
    "ClipMeta", "probe_clip", "parse_rate", "ffmpeg_bin", "ffprobe_bin",
    # 规划
    "Segment", "FrameJob", "TransitionPlan", "build_plan",
    "fade_weight_b", "frame_jobs", "transition_gain_fraction",
    "KIND_HEAD", "KIND_OVERLAP", "KIND_TAIL",
    "TRANSITION_FADE", "TRANSITION_SLIDE", "TRANSITION_WIPE", "TRANSITIONS",
    # 速度曲线
    "EASING_LINEAR", "EASING_EASE_IN", "EASING_EASE_OUT", "EASING_EASE_IN_OUT",
    "EASINGS", "eased_fraction", "eased_shift",
    # 滑动
    "DIRECTION_RIGHT", "DIRECTION_LEFT", "DIRECTION_DOWN", "DIRECTION_UP",
    "DIRECTIONS", "HORIZONTAL_DIRECTIONS", "VERTICAL_DIRECTIONS",
    "slide_regions", "composite_slide_frame",
    # 擦除
    "WipeSoftWidthWarning", "clamp_soft_width",
    "wipe_edge", "wipe_segments", "composite_wipe_frame",
    # 混合原语
    "blend_rgb24_frame", "blend_s16_samples", "div_round_half_away",
    # 音频
    "S16_CHANNELS", "choose_sample_rate", "frame_sample_bounds",
    "overlap_weight_fraction", "silence_pcm", "assemble_audio",
    # 帧读取器
    "FrameReader", "read_rgb24_frames", "read_pcm_track", "load_pcm_bytes",
    # 主管道
    "FrameWriter", "TransitionResult", "cross_fade",
]

__version__ = "1.2.0"
