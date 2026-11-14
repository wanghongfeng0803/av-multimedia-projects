"""容器（封装格式）注册表：容器与音视频编码的兼容矩阵及音频策略。

容器决定三件事：
1. 哪些视频编码可以装进去（不兼容时在 ffmpeg 启动前就明确报错）；
2. 音频流何时可无损直通（``-c:a copy``），何时必须重编码以及编成什么；
3. 输出文件的扩展名与默认容器选择。

本模块同样不执行外部进程，所有决策都是纯函数，便于单测。
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from .codecs import CodecSpec


class ContainerError(ValueError):
    """容器与编码/音频不兼容，或请求了未知容器。"""


@dataclass(frozen=True)
class ContainerSpec:
    """一种容器的兼容矩阵与默认行为。"""

    canonical: str
    aliases: tuple[str, ...]
    extension: str                     # 输出文件扩展名（含点）
    # 可被该容器承载的视频编码规范名
    supported_video: frozenset[str]
    # 可直接 -c:a copy 的音频编码
    passthrough_audio: frozenset[str]
    # 需要重编码时的首选音频编码器及其产物编码名
    audio_encoder: str
    audio_codec_name: str
    # 该音频编码器在 ffmpeg 中的可选替代（按优先级），用于能力探测后的降级
    audio_encoder_fallbacks: tuple[str, ...] = ()
    # 是否需要 moov 前置（仅 MP4 系有 faststart 语义）
    faststart: bool = False
    label: str = ""


_REGISTRY: dict[str, ContainerSpec] = {}


def _register(spec: ContainerSpec) -> ContainerSpec:
    _REGISTRY[spec.canonical] = spec
    return spec


MP4 = _register(ContainerSpec(
    canonical="mp4",
    aliases=("m4v", "mov"),
    extension=".mp4",
    supported_video=frozenset({"h264", "h265", "av1"}),
    passthrough_audio=frozenset({"aac", "mp3", "ac3", "eac3", "alac"}),
    audio_encoder="aac",
    audio_codec_name="aac",
    faststart=True,
    label="MPEG-4 (MP4)",
))

WEBM = _register(ContainerSpec(
    canonical="webm",
    aliases=("webm-video",),
    extension=".webm",
    supported_video=frozenset({"vp9", "vp8", "av1"}),
    # WebM 只允许 Vorbis/Opus 音频
    passthrough_audio=frozenset({"opus", "vorbis"}),
    audio_encoder="libopus",
    audio_codec_name="opus",
    audio_encoder_fallbacks=("libvorbis",),
    label="WebM",
))

MKV = _register(ContainerSpec(
    canonical="mkv",
    aliases=("matroska", "webm-matroska"),
    extension=".mkv",
    # Matroska 几乎可以承载本工具支持的全部编码
    supported_video=frozenset({"h264", "h265", "vp9", "vp8", "av1"}),
    passthrough_audio=frozenset({
        "aac", "mp3", "ac3", "eac3", "alac", "opus", "vorbis",
        "flac", "dts", "truehd",
    }),
    audio_encoder="libopus",
    audio_codec_name="opus",
    audio_encoder_fallbacks=("aac", "libvorbis"),
    label="Matroska (MKV)",
))


def list_containers() -> tuple[ContainerSpec, ...]:
    return tuple(_REGISTRY.values())


def _normalize(name: str) -> str:
    return name.strip().lower().lstrip(".")


def resolve_container(name: str) -> ContainerSpec:
    """按规范名/别名/扩展名解析容器。"""
    key = _normalize(name)
    for spec in _REGISTRY.values():
        if key == spec.canonical or key in (_normalize(a) for a in spec.aliases) \
                or key == spec.extension.lstrip("."):
            return spec
    known = ", ".join(s.canonical for s in _REGISTRY.values())
    raise ContainerError(f"未知容器 '{name}'。支持: {known}")


def container_from_extension(path: str) -> ContainerSpec | None:
    """从输出路径扩展名推断容器；无法识别时返回 None。"""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if not ext:
        return None
    for spec in _REGISTRY.values():
        if ext == spec.extension.lstrip(".") or \
                ext in (_normalize(a) for a in spec.aliases):
            return spec
    return None


def ensure_video_compatible(container: ContainerSpec, codec: CodecSpec) -> None:
    """视频编码与容器不兼容时在启动 ffmpeg 前快速失败。"""
    if codec.canonical not in container.supported_video:
        supported = sorted(container.supported_video)
        raise ContainerError(
            f"容器 {container.canonical} 不能承载 {codec.label}（{codec.canonical}）。"
            f" {container.canonical} 支持的视频编码: {', '.join(supported)}。"
            f"可改用容器: {', '.join(codec.preferred_containers)}。"
        )


def default_container_for(codec: CodecSpec) -> ContainerSpec:
    """选择编码的首选容器。"""
    return resolve_container(codec.preferred_containers[0])


def normalize_output_extension(path: str, container: ContainerSpec) -> str:
    """把输出路径的扩展名替换为容器对应的扩展名（无扩展名则追加）。"""
    root = os.path.splitext(path)[0]
    return root + container.extension


def audio_strategy(
    container: ContainerSpec, source_audio_codec: str | None
) -> tuple[str, str | None]:
    """决定音频处理方式。

    :returns: ``(mode, encoder)``
        - ``("copy", None)``：源音频可被容器直接承载，无损直通；
        - ``("reencode", "<ffmpeg 编码器>")``：重编码为容器兼容格式；
        - ``("drop", None)``：无音频流，输出纯视频。
    """
    if source_audio_codec is None:
        return "drop", None
    if source_audio_codec in container.passthrough_audio:
        return "copy", None
    return "reencode", container.audio_encoder
