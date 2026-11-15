"""用 ffprobe 探测输入视频的尺寸、帧率与帧数。

只读取元数据，不解码任何帧；真正的解码在 :mod:`scene_seg_tool.frames`
中以单遍流式完成。帧率解析遵循「平均帧率优先、基础帧率兜底、常量收尾」
的顺序，兼顾 VFR（可变帧率）视频与元数据缺失的原始流。
"""

from __future__ import annotations

import json
import os
import subprocess

from .binaries import find_ffprobe
from .errors import FFmpegRunError, InputNotFoundError, InvalidInputError

__all__ = ["VideoProbe", "probe_video", "parse_frame_rate", "DEFAULT_FPS"]

#: ffprobe 两个帧率字段都缺失/非法时使用的兜底帧率（帧/秒）
DEFAULT_FPS = 25.0


class VideoProbe:
    """视频流探测结果。"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        fps_source: str,
        nb_frames: int | None,
        duration_seconds: float | None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        #: 帧率来源：avg_frame_rate / r_frame_rate / fallback
        self.fps_source = fps_source
        #: 流头声明的帧数；元数据缺失（如裸流）时为 None
        self.nb_frames: int | None = nb_frames
        #: 容器级时长（秒）；缺失时为 None
        self.duration_seconds: float | None = duration_seconds

    def __repr__(self) -> str:  # pragma: no cover - 调试表示
        return (
            f"VideoProbe(width={self.width}, height={self.height}, "
            f"fps={self.fps!r}, fps_source={self.fps_source!r}, "
            f"nb_frames={self.nb_frames}, duration_seconds={self.duration_seconds})"
        )


def parse_frame_rate(text: str | None) -> float:
    """解析 ffprobe 的 ``num/den`` 帧率字符串。

    :returns: 浮点帧率；``None``、空串、``0/0`` 或分母为 0 时返回 0.0，
              表示该字段不可用，由调用方尝试下一个来源。
    """
    if not text or text == "0/0":
        return 0.0
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            denominator = float(den)
            if denominator == 0.0:
                return 0.0
            return float(num) / denominator
        return float(text)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str, ffprobe: str | None = None) -> VideoProbe:
    """探测视频文件的首条视频流信息。

    :raises InputNotFoundError: 路径不存在或不是普通文件。
    :raises InvalidInputError: ffprobe 能运行但找不到任何视频流。
    :raises FFmpegRunError: ffprobe 进程执行失败或输出无法解析。
    """
    if not os.path.isfile(path):
        raise InputNotFoundError(f"输入文件不存在: {path}")

    executable = ffprobe or find_ffprobe()
    cmd = [
        executable,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-print_format", "json",
        path,
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
        raise FFmpegRunError(
            f"ffprobe 探测失败（退出码 {proc.returncode}）: {path}",
            stderr=proc.stderr,
            cmd=cmd,
        )

    try:
        meta = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegRunError(
            f"ffprobe 输出无法解析为 JSON: {path}", stderr=proc.stderr, cmd=cmd
        ) from exc

    streams = meta.get("streams") or []
    if not streams:
        raise InvalidInputError(f"输入中不存在视频流: {path}")
    stream = streams[0]

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"视频流缺少有效的宽高信息: {path}"
        ) from exc
    if width <= 0 or height <= 0:
        raise InvalidInputError(f"视频流宽高非法: {width}x{height}")

    # VFR 视频 avg_frame_rate 反映真实平均节奏，优先采用；
    # 不可用时退回 r_frame_rate（流的时基帧率）
    avg_fps = parse_frame_rate(stream.get("avg_frame_rate"))
    r_fps = parse_frame_rate(stream.get("r_frame_rate"))
    if avg_fps > 0.0:
        fps, fps_source = avg_fps, "avg_frame_rate"
    elif r_fps > 0.0:
        fps, fps_source = r_fps, "r_frame_rate"
    else:
        fps, fps_source = DEFAULT_FPS, "fallback"

    nb_frames: int | None = None
    raw_nb = stream.get("nb_frames")
    if raw_nb not in (None, "N/A"):
        try:
            nb_frames = int(raw_nb)
        except (TypeError, ValueError):
            nb_frames = None

    duration_seconds = _parse_optional_float(stream.get("duration"))
    if duration_seconds is None:
        duration_seconds = _parse_optional_float(
            (meta.get("format") or {}).get("duration")
        )

    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        fps_source=fps_source,
        nb_frames=nb_frames,
        duration_seconds=duration_seconds,
    )


def _parse_optional_float(value: object) -> float | None:
    """解析 ffprobe 可能给出的 N/A / 缺失 / 非法浮点字段。"""
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
