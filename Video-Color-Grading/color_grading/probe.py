"""基于 ffprobe 的视频流探测（只取调色所需的最小信息集）。"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

from .binaries import find_ffprobe
from .errors import FFmpegRunError, InputNotFoundError, InvalidInputError


@dataclass
class VideoInfo:
    """调色管道需要的视频元信息。"""

    width: int
    height: int
    fps: float
    fps_fraction: str  # 原样传给 ffmpeg -r，避免浮点换算造成时基漂移
    pix_fmt: str | None
    has_audio: bool
    frame_count: int | None
    duration_seconds: float | None


def _fraction_to_float(text: str) -> float | None:
    try:
        num, _, den = text.partition("/")
        num_v, den_v = int(num), int(den)
        if den_v == 0:
            return None
        return num_v / den_v
    except (ValueError, AttributeError):
        return None


def _valid_fraction(text: object) -> bool:
    return isinstance(text, str) and _fraction_to_float(text) not in (None, 0.0)


def probe_video(file_path: str) -> VideoInfo:
    """探测视频文件，返回 :class:`VideoInfo`。

    :raises InputNotFoundError: 文件不存在
    :raises InvalidInputError: 无法解析、不含视频流或缺少尺寸信息
    :raises FFmpegRunError: ffprobe 自身异常退出
    """
    if not os.path.isfile(file_path):
        raise InputNotFoundError(f"输入文件不存在: {file_path}")

    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        file_path,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise InvalidInputError(
            "ffprobe 无法解析该文件，它可能不是媒体文件或已严重损坏。"
            f" 最后错误: {(proc.stderr or '').strip()[-300:]}"
        )
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegRunError("ffprobe 输出无法解析为 JSON。", stderr=proc.stderr, cmd=cmd) from exc

    streams = parsed.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise InvalidInputError("输入文件不含视频流，无法进行调色处理。")

    try:
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError("视频流缺少有效的宽/高信息。") from exc

    # 帧率优先取 avg_frame_rate（恒定帧率下最准），其次 r_frame_rate，最后兜底 25。
    fps_fraction = "25"
    for key in ("avg_frame_rate", "r_frame_rate"):
        if _valid_fraction(video.get(key)):
            fps_fraction = video[key]
            break
    fps = _fraction_to_float(fps_fraction) or 25.0

    frame_count = None
    try:
        frame_count = int(video.get("nb_frames"))
    except (TypeError, ValueError):
        pass

    duration = None
    try:
        duration = float((parsed.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        pass

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        fps_fraction=fps_fraction,
        pix_fmt=video.get("pix_fmt"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        frame_count=frame_count,
        duration_seconds=duration,
    )
