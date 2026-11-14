"""基于 ffprobe 的输入探测。

职责：
1. 解析容器与每条流的元数据（编码、分辨率、帧率、采样率等）；
2. 提取视频流**全部关键帧的时间戳**，供规划器做 GOP 对齐切点；
3. 对无法打包的输入（无音视频轨道）给出明确错误。

关键帧提取策略（两级）：
- 优先 ``-show_packets``，按包标志 ``K``（keyframe）收集 pts_time。
  包级查询只解复用不解码，速度远快于逐帧解码；
- 部分容器（如某些裸流 / TS）的包标志不可靠或缺失时，回退
  ``-show_frames``（``pict_type=I``），此时 ffprobe 必须解码，较慢，
  仅在包级结果明显不可用时启用。
"""

from __future__ import annotations

import json
import os

from .binaries import ffprobe_path
from .errors import ProbeError, UnsupportedInputError
from .models import MediaInfo, StreamInfo
from .process import run_capture

# 时间戳去重容差（秒）：同一关键帧在 pts/dts 路径下可能出现极小浮点差
_TIME_EPS = 1e-6


def _ffprobe_json(argv: list[str]) -> dict:
    """运行 ffprobe 并解析其 JSON 输出。"""
    result = run_capture(argv, timeout=120)
    if not result.ok:
        raise ProbeError(
            f"ffprobe 执行失败（退出码 {result.returncode}）：\n"
            f"{result.stderr_text()}"
        )
    try:
        return json.loads(result.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe 输出不是合法 JSON：{exc}") from exc


def _parse_rate(value: str | None) -> float | None:
    """把 ffprobe 的 r_framerate/avg_frame_rate（形如 '30000/1001'）转 float。"""
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_f = float(denominator)
        if denominator_f == 0:
            return None
        return float(numerator) / denominator_f
    return float(value)


def _parse_int(value) -> int | None:
    if value in (None, "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value) -> float | None:
    if value in (None, "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_streams(raw_streams: list[dict]) -> tuple[StreamInfo, ...]:
    streams: list[StreamInfo] = []
    for raw in raw_streams:
        tags = raw.get("tags") or {}
        streams.append(
            StreamInfo(
                index=int(raw.get("index", 0)),
                codec_type=raw.get("codec_type", "data"),
                codec_name=raw.get("codec_name", "unknown"),
                codec_tag=raw.get("codec_tag_string", ""),
                language=tags.get("language", "und"),
                width=_parse_int(raw.get("width")),
                height=_parse_int(raw.get("height")),
                frame_rate=_parse_rate(raw.get("avg_frame_rate"))
                or _parse_rate(raw.get("r_frame_rate")),
                sample_rate=_parse_int(raw.get("sample_rate")),
                channels=_parse_int(raw.get("channels")),
            )
        )
    return tuple(streams)


def _probe_format(path: str) -> tuple[dict, list[dict]]:
    argv = [
        ffprobe_path(),
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-print_format", "json",
        path,
    ]
    data = _ffprobe_json(argv)
    if "format" not in data:
        raise ProbeError("ffprobe 未返回 format 段，文件可能已损坏或不是媒体文件")
    return data["format"], data.get("streams", [])


def _keyframe_times_from_packets(path: str, stream_index: int) -> list[float]:
    """包级关键帧查询：flags 含 'K' 的包即为关键包。"""
    argv = [
        ffprobe_path(),
        "-v", "error",
        # 裸整数 = 按全局流索引选择（"0:0" 这类文件:索引写法不是合法选择器）
        "-select_streams", str(stream_index),
        "-show_packets",
        "-show_entries", "packet=pts_time,flags",
        "-print_format", "json",
        path,
    ]
    data = _ffprobe_json(argv)
    times: list[float] = []
    for packet in data.get("packets", []):
        flags = packet.get("flags", "")
        pts_text = packet.get("pts_time")
        if "K" in flags and pts_text not in (None, "N/A"):
            times.append(float(pts_text))
    return times


def _keyframe_times_from_frames(path: str, stream_index: int) -> list[float]:
    """帧级回退查询：pict_type == 'I'（需要解码，较慢）。"""
    argv = [
        ffprobe_path(),
        "-v", "error",
        "-select_streams", str(stream_index),
        "-show_frames",
        "-show_entries", "frame=pts_time,pict_type",
        "-print_format", "json",
        path,
    ]
    data = _ffprobe_json(argv)
    times: list[float] = []
    for frame in data.get("frames", []):
        if frame.get("pict_type") != "I":
            continue
        pts_text = frame.get("pts_time")
        if pts_text not in (None, "N/A"):
            times.append(float(pts_text))
    return times


def _dedupe_sorted(times: list[float]) -> tuple[float, ...]:
    """升序排序并按时间戳容差去重。"""
    result: list[float] = []
    for t in sorted(times):
        if not result or t - result[-1] > _TIME_EPS:
            result.append(t)
    return tuple(result)


def extract_keyframe_times(path: str, video_stream_index: int) -> tuple[float, ...]:
    """提取关键帧时间序列；包级结果不可信时自动回退帧级查询。

    "不可信"的判定：完全没有关键帧，但文件本身是可解码视频——
    正常视频至少存在一个 IDR，此时多半是容器没有标记包标志。
    """
    packet_times = _keyframe_times_from_packets(path, video_stream_index)
    if packet_times:
        return _dedupe_sorted(packet_times)

    frame_times = _keyframe_times_from_frames(path, video_stream_index)
    return _dedupe_sorted(frame_times)


def probe_media(path: str) -> MediaInfo:
    """探测输入文件，返回 :class:`MediaInfo`。

    :raises ProbeError: 文件不存在、无法解复用或 ffprobe 失败；
    :raises UnsupportedInputError: 没有任何音频/视频轨道。
    """
    if not os.path.isfile(path):
        raise ProbeError(f"输入文件不存在：{path}")

    fmt, raw_streams = _probe_format(path)
    streams = _parse_streams(raw_streams)

    av_streams = [s for s in streams if s.is_video or s.is_audio]
    if not av_streams:
        raise UnsupportedInputError("输入文件不包含任何音频或视频轨道，无法打包 HLS")

    duration_f = _parse_float(fmt.get("duration")) or 0.0

    keyframe_times: tuple[float, ...] = ()
    video_streams = [s for s in streams if s.is_video]
    if video_streams:
        keyframe_times = extract_keyframe_times(path, video_streams[0].index)

    return MediaInfo(
        path=os.path.abspath(path),
        format_name=fmt.get("format_name", "unknown"),
        duration=duration_f,
        bit_rate=_parse_int(fmt.get("bit_rate")),
        streams=streams,
        keyframe_times=keyframe_times,
    )
