"""打包编排层：把探测、规划、切片、播放列表、元数据串成一条流水线。

流水线
------
1. ``probe_media``        拿到容器/流信息与关键帧时间序列；
2. ``plan_segments``      GOP 对齐生成分片规划；
3. ``validate_codecs``    确认编码可写入 MPEG-TS；
4. ``write_all_segments`` 一次 ffmpeg segment muxer 调用切出全部
   分片（全局连续时间戳），再用 ffprobe 逐片复测实际时长；
5. 用**实测时长**重建输出时间轴（每片 start_time = 前序实测时长之和），
   保证 EXTINF 之和与分片真实时长严格一致；
6. 计算 TARGETDURATION、渲染 M3U8（VOD 或滑动窗口）；
7. 落盘 ``segments.json``，记录每片的源内位置、输出时间轴与实测时长。

时间戳连续性在切片层闭环：segment muxer 保留全局时间戳，相邻分片的
PTS/DTS 首尾相接；而播放列表侧用实测时长逐片累计，避免规划值与
帧级实际片长的漂移累积到 EXTINF 上。
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from .errors import PackagerError
from .encryption.config import DEFAULT_KEY_FILENAME, EncryptionConfig
from .encryption.key_info import KeyDescriptor
from .encryption.segment_io import encrypt_segment_file, write_key_file
from .containers import DEFAULT_CONTAINER, get_container
from .m3u8 import PlaylistConfig, compute_target_duration, write_playlist
from .models import MediaInfo, PackagingResult, SegmentInfo
from .naming import NamingConfig
from .planner import (
    ALIGN_MODES,
    ALIGN_NEAREST,
    plan_segments,
)
from .probe import probe_media
from .ts_writer import WATERMARK_PNG_FILENAME, validate_codecs, write_all_segments
from .watermark.config import WatermarkConfig

#: 元数据文件名
METADATA_FILENAME = "segments.json"
#: 播放列表文件名
PLAYLIST_FILENAME = "playlist.m3u8"


def _rebuild_timeline(
    planned: list[SegmentInfo], actual_durations: list[float]
) -> list[SegmentInfo]:
    """用实测时长重建输出时间轴：start_time 从 0 起逐片累计。"""
    rebuilt: list[SegmentInfo] = []
    timeline_cursor = 0.0
    for seg, actual in zip(planned, actual_durations):
        rebuilt.append(
            replace(
                seg,
                start_time=round(timeline_cursor, 6),
                duration=round(actual, 6),
            )
        )
        timeline_cursor += actual
    return rebuilt


def _write_metadata(
    output_dir: str,
    media: MediaInfo,
    segments: list[SegmentInfo],
    target_duration: int,
    media_sequence: int,
    window_size: int | None,
    container_name: str,
    name_template: str,
    align_mode: str,
    hls_version: int,
    encryption: EncryptionConfig | None = None,
) -> str:
    """落盘机器可读的分片元数据。"""
    hls_block: dict = {
        "version": 3,
        "target_duration": target_duration,
        "media_sequence": media_sequence,
        "window_size": window_size,
        "segment_count": len(segments),
        "total_duration": round(sum(s.duration for s in segments), 6),
        # 本次打包的容器/命名/对齐参数，便于下游复现与核对
        "container": container_name,
        "hls_version": hls_version,
        "name_template": name_template,
        "align_mode": align_mode,
    }
    if encryption is not None:
        # 记录加密方法、密钥 URI 与 IV 语义，便于下游核对播放列表；
        # 密钥本身同时落盘为独立 .key 文件，这里只留其十六进制指纹
        hls_block["encryption"] = {
            "method": "AES-128",
            "key_uri": encryption.key_uri,
            "key_file": DEFAULT_KEY_FILENAME,
            "key_hex": encryption.key_hex,
            "iv_mode": "explicit" if encryption.iv is not None else "sequence",
            "iv_hex": encryption.iv_hex,
            "padding": "PKCS#7",
        }
    payload = {
        "source": {
            "path": media.path,
            "format": media.format_name,
            "duration": round(media.duration, 6),
            "video_codec": media.video_streams[0].codec_name
            if media.video_streams
            else None,
            "audio_codecs": [s.codec_name for s in media.audio_streams],
            "keyframe_count": len(media.keyframe_times),
        },
        "hls": hls_block,
        "segments": [
            {
                "sequence_number": seg.sequence_number,
                "filename": seg.filename,
                "start_time": seg.start_time,
                "duration": seg.duration,
                "source_start": seg.source_start,
            }
            for seg in segments
        ],
    }
    path = os.path.join(output_dir, METADATA_FILENAME)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def package(
    input_path: str,
    output_dir: str,
    target_duration: float = 6.0,
    *,
    window_size: int | None = None,
    media_sequence: int = 0,
    filename_prefix: str = "seg",
    overwrite: bool = False,
    watermark: WatermarkConfig | None = None,
    encryption: EncryptionConfig | None = None,
    name_template: str | None = None,
    container: str = DEFAULT_CONTAINER,
    align_mode: str = ALIGN_NEAREST,
) -> PackagingResult:
    """把一个媒体文件打包为 HLS 分片 + M3U8，可选 AES-128 加密。

    :param input_path: 源文件路径；
    :param output_dir: 输出目录（不存在则创建）；
    :param target_duration: 目标分片时长（秒），实际切点对齐 GOP；
    :param window_size: 直播滑动窗口大小；``None`` 表示 VOD 全量列表；
    :param media_sequence: 首片的 EXT-X-MEDIA-SEQUENCE 序号；
    :param filename_prefix: 分片文件名前缀（命名模板中 ``{name}`` 的值）；
    :param overwrite: 输出目录非空时是否允许写入（会覆盖同名产物）；
    :param watermark: 文字水印配置；``None`` 时全程流拷贝、行为与
        旧版本完全一致；传入时视频轨重编码叠加水印，音频仍流拷贝。
    :param encryption: AES-128 加密配置；``None`` 时不加密、行为与
        旧版本完全一致；传入时切片复测通过后逐片做 AES-128-CBC 加密，
        密钥写入输出目录、播放列表增加 EXT-X-KEY 标签。
    :param name_template: 分片命名模板（如
        ``"{name}_{sequence:05d}.{ext}"``）；``None`` 时用默认模板
        ``{name}_{sequence:04d}.{ext}``，TS 容器下与历史文件名一致。
    :param container: 分片容器短名：``ts``（默认）/ ``fmp4`` / ``mp4``
        / ``mkv``，仅影响封装层，GOP 对齐与时间戳逻辑不变。
    :param align_mode: 切点对齐模式：``nearest``（默认，距理想切点
        最近关键帧，同距取靠前）/ ``before``（不超过目标时长）/
        ``after``（不短于目标时长）；纯音频输入下该参数无意义。
    :raises PackagerError: 任意可预期的输入/环境/切片/加密错误。
    """
    if target_duration <= 0:
        raise PackagerError(f"目标分片时长必须为正数，实际为 {target_duration!r}")
    if media_sequence < 0:
        raise PackagerError("MEDIA-SEQUENCE 起始序号不能为负数")
    if align_mode not in ALIGN_MODES:
        raise PackagerError(
            f"对齐模式必须为 {ALIGN_MODES} 之一，实际为 {align_mode!r}"
        )
    if watermark is not None:
        watermark.validate()
    if encryption is not None:
        encryption.validate()

    # 容器规格与命名模板在任何 IO 之前确定，参数非法尽早报错
    spec = get_container(container)
    if name_template is None:
        naming = NamingConfig(
            name_prefix=filename_prefix, extension=spec.extension
        )
    else:
        naming = NamingConfig(
            template=name_template,
            name_prefix=filename_prefix,
            extension=spec.extension,
        )

    os.makedirs(output_dir, exist_ok=True)
    if not overwrite and os.listdir(output_dir):
        raise PackagerError(
            f"输出目录非空：{output_dir}（传入 overwrite=True 可覆盖同名产物）"
        )

    # 1) 探测
    media = probe_media(input_path)

    # 2) GOP 对齐规划（纯音频 / 无关键帧信息时退化为等长规划）
    planned = plan_segments(
        duration=media.duration,
        target_duration=target_duration,
        keyframe_times=media.keyframe_times,
        sequence_start=media_sequence,
        filename_prefix=filename_prefix,
        align_mode=align_mode,
        name_template=naming.template,
        extension=spec.extension,
    )
    if not planned:
        raise PackagerError("规划结果为空：源文件时长过短或内容不可切分")

    # 规划与命名模板双重确认一致，并在落盘前拦截重名
    expected_names = naming.render_many(len(planned), media_sequence)
    if [seg.filename for seg in planned] != expected_names:
        raise PackagerError("命名模板渲染结果与分片规划不一致")  # pragma: no cover

    # 3) 按目标容器的编码白名单生成 -map 列表
    maps = validate_codecs(media, spec)

    # 4) 一次性切出全部分片，并逐片复测实际时长
    #    （watermark=None 走流拷贝旧路径；否则自行栅格化文字后重编码叠加）
    written_segments = write_all_segments(
        media=media,
        planned=planned,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        maps=maps,
        watermark=watermark,
        container=spec,
        naming=naming,
    )
    actual_durations = [w.actual_duration for w in written_segments]

    # 5) 实测时长重建连续时间轴
    segments = _rebuild_timeline(planned, actual_durations)

    # 5.5) 可选 AES-128-CBC 加密：复测全部通过后才加密，保证写入磁盘的
    #      时长数据来自明文分片；逐片原子替换，密钥与 EXT-X-KEY 同步产出
    key_descriptor = None
    if encryption is not None:
        key_path = write_key_file(encryption, output_dir)
        for seg in segments:
            encrypt_segment_file(
                os.path.join(output_dir, seg.filename),
                encryption,
                seg.sequence_number,
            )
        key_descriptor = KeyDescriptor.from_encryption_config(encryption)

    # 6) 播放列表（TARGETDURATION 按实测最大片长向上取整；
    #    版本随容器：TS=3，fMP4/MP4=7）
    target = compute_target_duration(segments)
    config = PlaylistConfig(
        target_duration=target,
        window_size=window_size,
        key=key_descriptor,
        version=spec.hls_version,
    )
    playlist_path = write_playlist(
        segments,
        os.path.join(output_dir, PLAYLIST_FILENAME),
        config,
    )

    # 7) 元数据
    metadata_path = _write_metadata(
        output_dir, media, segments, target, media_sequence, window_size,
        spec.name, naming.template, align_mode, spec.hls_version,
        encryption,
    )

    extra = {
        "metadata_path": metadata_path,
        "container": spec.name,
        "name_template": naming.template,
        "align_mode": align_mode,
    }
    if watermark is not None:
        extra["watermark_png"] = os.path.abspath(
            os.path.join(output_dir, WATERMARK_PNG_FILENAME)
        )
    if encryption is not None:
        extra["encryption"] = {
            "method": "AES-128",
            "key_file": os.path.abspath(key_path),
            "key_uri": encryption.key_uri,
            "key_hex": encryption.key_hex,
            "iv_mode": "explicit" if encryption.iv is not None else "sequence",
            "iv_hex": encryption.iv_hex,
        }

    return PackagingResult(
        output_dir=os.path.abspath(output_dir),
        playlist_path=os.path.abspath(playlist_path),
        segments=tuple(segments),
        media=media,
        target_duration=target,
        media_sequence=media_sequence,
        window_size=window_size,
        extra=extra,
    )
