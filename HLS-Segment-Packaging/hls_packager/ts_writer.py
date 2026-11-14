"""MPEG-TS 分片写入器（基于 ffmpeg segment muxer 的一次成型切割）。

为什么不用"每片一个 ffmpeg + -ss/-t"
------------------------------------
逐进程手控切割在 B 帧重排与 AAC priming（编码器延迟采样）下会过切：
``-t`` 按"输出包起点"计时，音频 priming 会让容器多吞下一个 GOP 的
关键帧，导致分片变长、首帧之外混入额外 IDR。ffmpeg 的 segment muxer
在解复用器内部按关键帧边界切分，能正确处理重排与 priming，切点精确。

本模块仍然完全掌握分片策略——切点 ``segment_times`` 由
:mod:`hls_packager.planner` 规划（只含关键帧时刻），segment muxer
只负责"在给定时刻做流拷贝切割"，M3U8 也由本项目自己生成，
不使用 ffmpeg 的 hls muxer / 自带播放列表。

时间戳与可独立解码性
--------------------
* 不启用 ``-reset_timestamps``：各分片保留**全局连续**时间戳，
  相邻分片 PTS/DTS 首尾相接，不回退、不跳变；
* 切点全部是 IDR，且每个 TS 分片都由 muxer 写入独立的 PAT/PMT，
  因此任意一片都能被播放器独立解析、从首帧正常起播；
* 全程 ``-c copy`` 流拷贝，不重编码，开销接近纯 IO。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .binaries import ffprobe_path, ffmpeg_path
from .errors import SegmenterError, UnsupportedInputError
from .models import MediaInfo, SegmentInfo
from .process import run_capture

# MPEG-TS 支持的编码白名单（ffmpeg mpegts muxer 可直接承载的常见编码）
_TS_VIDEO_CODECS = {"h264", "hevc", "mpeg2video", "mpeg1video", "mp4"}
_TS_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "mp2", "opus"}

# 子进程超时系数：按整文件时长放宽，最低 60 秒，兼容慢速磁盘
_TIMEOUT_BASE = 60.0
_TIMEOUT_PER_SECOND = 10.0
# Windows CreateProcess 命令行上限约 32767 字符，预留安全余量
_ARGV_SOFT_LIMIT = 30000


@dataclass(frozen=True)
class WrittenSegment:
    """切片落盘后的实测信息。"""

    info: SegmentInfo
    actual_duration: float
    size_bytes: int


def validate_codecs(media: MediaInfo) -> list[str]:
    """检查待打包的流能否写进 MPEG-TS，返回 ffmpeg -map 参数列表。

    策略：视频是主轨道，编码不支持时**直接报错**（避免静默产出只有
    声音的"视频"）；音频是次要轨道，不支持的音频跳过即可；
    字幕 / 数据流 MPEG-TS 承载能力有限且 HLS TS 场景极少使用，直接忽略。
    """
    maps: list[str] = []
    skipped_audio: list[str] = []

    videos = media.video_streams
    if videos:
        video = videos[0]
        if video.codec_name not in _TS_VIDEO_CODECS:
            raise UnsupportedInputError(
                f"视频编码 {video.codec_name} 无法直接写入 MPEG-TS"
                "（需要先转码，本工具只做流拷贝重封装）"
            )
        maps.append("0:v:0?")

    for audio in media.audio_streams:
        if audio.codec_name in _TS_AUDIO_CODECS:
            maps.append(f"0:{audio.index}?")
        else:
            skipped_audio.append(audio.codec_name)

    if not maps:
        detail = (
            f"（已跳过不支持的音频编码：{'、'.join(skipped_audio)}）"
            if skipped_audio else ""
        )
        raise UnsupportedInputError(f"没有可写入 MPEG-TS 的音视频流{detail}")
    return maps


def build_segmentation_command(
    media: MediaInfo,
    planned: list[SegmentInfo],
    output_dir: str,
    filename_prefix: str,
    maps: list[str],
) -> list[str]:
    """构造一次性切片的 ffmpeg segment muxer 命令（独立成函数便于单测）。

    切点取规划中除首片起点（恒为 0）外的全部 ``source_start``；
    输出文件名序号用 ``-segment_start_number`` 对齐 MEDIA-SEQUENCE。
    """
    if not planned:
        raise SegmenterError("分片规划为空，无法构造切片命令")

    # segment_times 是"绝对源时间轴"上的切分时刻；首片从 0 开始无需列出
    cut_times = ",".join(f"{seg.source_start:.6f}" for seg in planned[1:])
    start_number = planned[0].sequence_number
    output_pattern = os.path.join(output_dir, f"{filename_prefix}_%04d.ts")

    argv = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-y",
        "-i", media.path,
        "-map", maps[0],
    ]
    for extra_map in maps[1:]:
        argv.extend(["-map", extra_map])

    argv.extend(
        [
            "-c", "copy",
            "-f", "segment",
            "-segment_format", "mpegts",
            # 起始序号对齐 EXT-X-MEDIA-SEQUENCE，使文件名与规划一致
            "-segment_start_number", str(start_number),
        ]
    )
    # 只有存在中间切点时才需要 segment_times（单分片文件不需要）
    if cut_times:
        argv.extend(["-segment_times", cut_times])
    argv.append(output_pattern)

    if sum(len(part) for part in argv) > _ARGV_SOFT_LIMIT:
        raise SegmenterError(
            "分片数量过多导致命令行超长（切点列表超过系统限制），"
            "请增大 --segment-duration 以减少分片数量"
        )
    return argv


def _probe_duration(path: str) -> float:
    """用 ffprobe 读取分片的容器级实际时长（秒）。"""
    argv = [
        ffprobe_path(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-print_format", "json",
        path,
    ]
    result = run_capture(argv, timeout=30)
    if not result.ok:
        raise SegmenterError(
            f"无法探测分片时长 {path}：{result.stderr_text()}"
        )
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SegmenterError(f"分片时长探测结果非法：{path}") from exc


def write_all_segments(
    media: MediaInfo,
    planned: list[SegmentInfo],
    output_dir: str,
    filename_prefix: str,
    maps: list[str],
) -> list[WrittenSegment]:
    """一次性切出全部分片，并逐片复测实际时长与文件大小。

    依赖 segment muxer 的输出序号与规划序号一一对应（文件名均可由
    :attr:`SegmentInfo.filename` 直接定位），逐片校验存在性与时长。
    """
    argv = build_segmentation_command(
        media, planned, output_dir, filename_prefix, maps
    )
    # 处理整文件，超时按源时长估算
    timeout = _TIMEOUT_BASE + media.duration * _TIMEOUT_PER_SECOND

    result = run_capture(argv, timeout=timeout)
    if not result.ok:
        raise SegmenterError(
            f"ffmpeg 切片失败（退出码 {result.returncode}）",
            stderr_tail=result.stderr_text(),
            returncode=result.returncode,
        )

    written: list[WrittenSegment] = []
    for seg in planned:
        path = os.path.join(output_dir, seg.filename)
        if not os.path.isfile(path):
            raise SegmenterError(
                f"规划中的分片 {seg.filename} 未生成，"
                "通常是输入缺少关键帧或切点不在关键帧上"
            )
        size_bytes = os.path.getsize(path)
        if size_bytes == 0:
            raise SegmenterError(f"分片 {seg.filename} 输出为空文件")

        actual_duration = _probe_duration(path)
        if actual_duration <= 0:
            raise SegmenterError(
                f"分片 {seg.filename} 实测时长异常：{actual_duration}"
            )
        written.append(
            WrittenSegment(
                info=seg,
                actual_duration=actual_duration,
                size_bytes=size_bytes,
            )
        )
    return written
