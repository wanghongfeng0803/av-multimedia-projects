"""分片容器规格：把容器名映射到 ffmpeg segment muxer 参数与播放列表语义。

本项目的播放列表由自己渲染（不用 ffmpeg hls muxer），因此每种容器
需要回答四个问题：

1. ``-segment_format`` 传什么 muxer 名；
2. 分片文件用什么扩展名（同时决定 ``{ext}`` 占位符的取值）；
3. 播放列表 ``EXT-X-VERSION`` 至少写几（RFC 8216 按容器/特性定版本）；
4. 哪些编码可以**流拷贝**进该容器（白名单在
   :func:`hls_packager.ts_writer.validate_codecs` 中按规格过滤）。

支持的容器
----------
``ts``（默认）
    MPEG-2 Transport Stream，RFC 8216 原生分片格式，兼容性最好，
    每片自带 PAT/PMT、可独立解码；版本 3。
``fmp4``
    自包含碎片化 MP4（CMAF 风格）：``empty_moov`` 让每个分片都带
    ftyp + 空 moov + 若干 fragment，**无需独立 init 段与 EXT-X-MAP**
    即可独立起播，契合本项目"每片可独立解码"的既有承诺；版本 7。
``mp4``
    普通 ISO-BMFF 分片，每片独立 ftyp/moov/mdat，可被通用播放器
    直接打开；不属于 RFC 8216 规定的 HLS 分片格式，播放列表仅作
    通用分片清单使用。
``mkv``
    Matroska 分片，编码承载面最宽（含 FLAC/Opus/Vorbis 等），
    同样是通用归档分片而非标准 HLS 分片。

容器只影响**封装**层：GOP 对齐规划、时间戳连续性、逐片时长复测、
AES 加密等逻辑与容器无关，全部复用既有流水线。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PackagerError

#: 默认容器（不指定时与历史版本完全一致）
DEFAULT_CONTAINER = "ts"

# MPEG-TS 可承载的常见编码（与历史白名单保持一致）
_TS_VIDEO = frozenset(("h264", "hevc", "mpeg2video", "mpeg1video", "mp4"))
_TS_AUDIO = frozenset(("aac", "mp3", "ac3", "eac3", "mp2", "opus"))

# ISO-BMFF 系（fmp4 / mp4）：MPEG-4 Part 2 的 ffprobe 名为 mpeg4
_MP4_VIDEO = frozenset(("h264", "hevc", "mpeg4"))
_MP4_AUDIO = frozenset(("aac", "mp3", "ac3", "eac3", "opus"))

# Matroska 承载面最宽：TS 全家桶 + MP4 系 + 无损音频
_MKV_VIDEO = _TS_VIDEO | frozenset(("mpeg4", "av1"))
_MKV_AUDIO = _TS_AUDIO | frozenset(("flac", "vorbis", "pcm_s16le"))


@dataclass(frozen=True)
class ContainerSpec:
    """一种分片容器的完整规格（不可变）。"""

    #: 规格名（CLI / API 入参使用的短名）
    name: str
    #: ffmpeg ``-segment_format`` 取值（内部 muxer 名）
    segment_format: str
    #: 分片文件扩展名（不含点），同时是命名模板 {ext} 的取值
    extension: str
    #: 播放列表声明的 EXT-X-VERSION
    hls_version: int
    #: 是否为 RFC 8216 规定的标准 HLS 分片容器
    is_hls_standard: bool
    #: 允许流拷贝的视频编码白名单
    video_codecs: frozenset[str]
    #: 允许流拷贝的音频编码白名单（不在表内的音频轨直接跳过）
    audio_codecs: frozenset[str]
    #: 追加给 ffmpeg 的输出级 muxer 参数（按 argv 顺序平铺）
    muxer_options: tuple[str, ...] = ()


#: 全部受支持容器的规格表（顺序即 CLI help 的展示顺序）
CONTAINER_SPECS: dict[str, ContainerSpec] = {
    "ts": ContainerSpec(
        name="ts",
        segment_format="mpegts",
        extension="ts",
        hls_version=3,
        is_hls_standard=True,
        video_codecs=_TS_VIDEO,
        audio_codecs=_TS_AUDIO,
    ),
    "fmp4": ContainerSpec(
        name="fmp4",
        segment_format="mp4",
        extension="m4s",
        hls_version=7,
        is_hls_standard=True,
        video_codecs=_MP4_VIDEO,
        audio_codecs=_MP4_AUDIO,
        # empty_moov：每片自包含、无需独立 init 段；
        # frag_keyframe：fragment 切在关键帧上；default_base_moof：
        # tfdt/base-is-moof 时间基，CMAF 播放器普遍要求
        muxer_options=(
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        ),
    ),
    "mp4": ContainerSpec(
        name="mp4",
        segment_format="mp4",
        extension="mp4",
        hls_version=7,
        is_hls_standard=False,
        video_codecs=_MP4_VIDEO,
        audio_codecs=_MP4_AUDIO,
    ),
    "mkv": ContainerSpec(
        name="mkv",
        segment_format="matroska",
        extension="mkv",
        hls_version=3,
        is_hls_standard=False,
        video_codecs=_MKV_VIDEO,
        audio_codecs=_MKV_AUDIO,
    ),
}

#: 受支持容器短名元组（对外暴露，供 CLI choices 与文档复用）
SUPPORTED_CONTAINERS = tuple(CONTAINER_SPECS)


def get_container(name: str) -> ContainerSpec:
    """按短名取容器规格；未知名抛出带全部合法值的中文错误。"""
    spec = CONTAINER_SPECS.get(name)
    if spec is None:
        raise PackagerError(
            f"不支持的分片容器 {name!r}；合法值：{', '.join(SUPPORTED_CONTAINERS)}"
        )
    return spec
