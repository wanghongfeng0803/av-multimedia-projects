"""双路视频左右并排合成工具。

对外主要入口：
- :func:`compose_tool.probe.probe_media`：探测输入媒体信息
- :func:`compose_tool.plan.plan_composition`：计算确定性对齐规划（纯逻辑）
- :func:`compose_tool.filtergraph.build_filter_complex`：由规划生成 filter_complex
- :func:`compose_tool.composer.compose`：执行合成（H.264 + AAC MP4）
"""

from .errors import (
    ComposeError,
    BinaryNotFoundError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    FFmpegRunError,
)
from .models import MediaInfo, StreamInfo, ComposeResult
from .probe import probe_media
from .plan import (
    ComposePlan,
    VideoSidePlan,
    plan_composition,
    parse_fps,
    parse_sar,
    even_ceil,
    round_even,
    scaled_width_for_height,
    DURATION_PAD,
    DURATION_TRIM,
    DURATION_MODES,
    LAYOUT_AUTO,
    LAYOUT_FIT,
    LAYOUTS,
    AUDIO_AUTO,
    AUDIO_MIX,
    AUDIO_LEFT,
    AUDIO_RIGHT,
    AUDIO_NONE,
    AUDIO_MODES,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    DEFAULT_FALLBACK_FPS,
)
from .filtergraph import build_filter_complex
from .composer import (
    compose,
    build_command,
    default_output_path,
    DEFAULT_CRF,
    DEFAULT_PRESET,
    AUDIO_BITRATE,
    VIDEO_TIMESCALE,
)

__all__ = [
    # 核心 API
    "probe_media",
    "plan_composition",
    "build_filter_complex",
    "compose",
    "build_command",
    "default_output_path",
    # 数据模型
    "MediaInfo",
    "StreamInfo",
    "ComposeResult",
    "ComposePlan",
    "VideoSidePlan",
    # 规划常量与纯函数
    "parse_fps",
    "parse_sar",
    "even_ceil",
    "round_even",
    "scaled_width_for_height",
    "DURATION_PAD",
    "DURATION_TRIM",
    "DURATION_MODES",
    "LAYOUT_AUTO",
    "LAYOUT_FIT",
    "LAYOUTS",
    "AUDIO_AUTO",
    "AUDIO_MIX",
    "AUDIO_LEFT",
    "AUDIO_RIGHT",
    "AUDIO_NONE",
    "AUDIO_MODES",
    "TARGET_SAMPLE_RATE",
    "TARGET_CHANNELS",
    "DEFAULT_FALLBACK_FPS",
    "DEFAULT_CRF",
    "DEFAULT_PRESET",
    "AUDIO_BITRATE",
    "VIDEO_TIMESCALE",
    # 异常
    "ComposeError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
]
