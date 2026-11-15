"""video_transition：从零实现的两段素材交叉淡化转场工具包。

只依赖 Python 标准库与系统 ffmpeg/ffprobe。核心承诺：

* 成片总帧数恒为 ``Na + Nb - No``，重叠区之外一帧不丢；
* 视频帧与音频采样共用同一张帧映射表与淡化权重，音视频严格同帧淡化；
* 全部转场帧统一经 :class:`FrameReader` 的 RGB24 管道读取；
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
from .errors import PipelineError, PlanError, ProbeError, TransitionError
from .pipeline import FrameWriter, TransitionResult, cross_fade
from .plan import (
    KIND_HEAD,
    KIND_OVERLAP,
    KIND_TAIL,
    FrameJob,
    Segment,
    TransitionPlan,
    build_plan,
    fade_weight_b,
    frame_jobs,
)
from .probe import ClipMeta, ffmpeg_bin, ffprobe_bin, parse_rate, probe_clip
from .reader import FrameReader, load_pcm_bytes, read_pcm_track, read_rgb24_frames

__all__ = [
    # 异常
    "TransitionError", "ProbeError", "PlanError", "PipelineError",
    # 探针
    "ClipMeta", "probe_clip", "parse_rate", "ffmpeg_bin", "ffprobe_bin",
    # 规划
    "Segment", "FrameJob", "TransitionPlan", "build_plan",
    "fade_weight_b", "frame_jobs",
    "KIND_HEAD", "KIND_OVERLAP", "KIND_TAIL",
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

__version__ = "1.0.0"
