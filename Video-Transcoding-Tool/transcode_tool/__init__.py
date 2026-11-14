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
    # 注册表查询
    "list_codecs",
    "resolve_codec",
    "build_video_encoder_args",
    "list_containers",
    "resolve_container",
    "ContainerError",
    # 异常
    "TranscodeError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
    "EncoderUnavailableError",
]
