"""多编码视频转码工具。

对外主要入口：
- :func:`transcode_tool.probe.probe_media`：探测输入媒体信息
- :func:`transcode_tool.transcoder.transcode`：执行转码（H.264/H.265/VP9/AV1/VP8）
- :func:`transcode_tool.codecs.resolve_codec` / :func:`list_codecs`：编码注册表
- :func:`transcode_tool.containers.resolve_container`：容器注册表
"""

from .errors import (
    TranscodeError,
    BinaryNotFoundError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    FFmpegRunError,
    EncoderUnavailableError,
)
from .models import MediaInfo, StreamInfo, TranscodeResult
from .probe import probe_media
from .transcoder import transcode
from .codecs import (
    CodecSpec,
    EncoderSpec,
    list_codecs,
    resolve_codec,
    build_video_encoder_args,
)
from .containers import (
    ContainerSpec,
    ContainerError,
    list_containers,
    resolve_container,
)
from .bitrate import (
    BitratePlan,
    RateMode,
    parse_bitrate,
    format_bitrate,
    plan_bitrate,
    build_bitrate_encoder_args,
)
from .scale import ResolutionPlan, parse_resolution, build_scale_filter, even_floor

__all__ = [
    # 核心 API
    "probe_media",
    "transcode",
    # 数据模型
    "MediaInfo",
    "StreamInfo",
    "TranscodeResult",
    "CodecSpec",
    "EncoderSpec",
    "ContainerSpec",
    "BitratePlan",
    "ResolutionPlan",
    # 注册表查询
    "list_codecs",
    "resolve_codec",
    "build_video_encoder_args",
    "list_containers",
    "resolve_container",
    "ContainerError",
    # 码率控制
    "RateMode",
    "parse_bitrate",
    "format_bitrate",
    "plan_bitrate",
    "build_bitrate_encoder_args",
    # 分辨率规划
    "parse_resolution",
    "build_scale_filter",
    "even_floor",
    # 异常
    "TranscodeError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
    "EncoderUnavailableError",
]
