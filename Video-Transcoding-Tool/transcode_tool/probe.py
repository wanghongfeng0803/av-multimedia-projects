"""基于 ffprobe 的输入媒体探测。"""

from __future__ import annotations

import json
import os
import subprocess

from .binaries import find_ffprobe
from .errors import FFmpegRunError, InvalidInputError
from .models import MediaInfo, StreamInfo


def _to_float(value: object) -> float | None:
    try:
        if value in (None, "N/A"):
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        if value in (None, "N/A"):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_streams(raw_streams: list[dict]) -> list[StreamInfo]:
    streams: list[StreamInfo] = []
    for raw in raw_streams:
        streams.append(
            StreamInfo(
                index=_to_int(raw.get("index")) or 0,
                codec_type=raw.get("codec_type", "data"),
                codec_name=raw.get("codec_name"),
                width=_to_int(raw.get("width")),
                height=_to_int(raw.get("height")),
                pix_fmt=raw.get("pix_fmt"),
                sample_rate=_to_int(raw.get("sample_rate")),
                channels=_to_int(raw.get("channels")),
                duration_seconds=_to_float(raw.get("duration")),
            )
        )
    return streams


def probe_media(file_path: str) -> MediaInfo:
    """探测媒体文件，返回 :class:`MediaInfo`。

    对损坏但仍可部分读取的文件尽量给出结果，并把问题记录到
    ``MediaInfo.warnings``；对完全无法解析或不含任何流的文件抛出
    :class:`InvalidInputError`。

    :raises InputNotFoundError: 文件不存在（由上层转码流程提前拦截）
    :raises InvalidInputError: ffprobe 无法解析或文件中没有任何流
    :raises FFmpegRunError: ffprobe 自身异常退出
    """
    if not os.path.isfile(file_path):
        # 延迟导入避免循环依赖
        from .errors import InputNotFoundError

        raise InputNotFoundError(f"输入文件不存在: {file_path}")

    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
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

    stderr_text = proc.stderr or ""
    warnings: list[str] = []

    if proc.returncode != 0:
        # 截断/损坏文件时 ffprobe 常返回非零但仍向 stdout 输出部分 JSON，
        # 先尝试解析；解析不出来才判定为无效输入。
        parsed = _try_parse_json(proc.stdout)
        if parsed is None or not parsed.get("streams"):
            raise InvalidInputError(
                "ffprobe 无法解析该文件，它可能不是媒体文件或已严重损坏。"
                f" 最后错误: {_tail(stderr_text)}"
            )
        warnings.append(f"输入文件可能已损坏，探测返回非零状态: {_tail(stderr_text)}")
    elif stderr_text.strip():
        # returncode == 0 但有 error 级日志（如部分索引缺失）
        warnings.append(f"探测警告: {_tail(stderr_text)}")
    else:
        parsed = _try_parse_json(proc.stdout)
        if parsed is None:
            raise FFmpegRunError(
                "ffprobe 输出无法解析为 JSON。", stderr=stderr_text, cmd=cmd
            )

    raw_streams = parsed.get("streams") or []
    if not raw_streams:
        raise InvalidInputError("文件中没有任何媒体流，无法转码。")

    raw_format = parsed.get("format") or {}
    info = MediaInfo(
        file_path=os.path.abspath(file_path),
        format_name=raw_format.get("format_name"),
        duration_seconds=_to_float(raw_format.get("duration")),
        streams=_parse_streams(raw_streams),
        warnings=warnings,
    )

    if not info.video_streams:
        raise InvalidInputError(
            "输入文件不含视频流（可能是纯音频或其它非视频文件），无法进行视频转码。"
        )

    return info


def _try_parse_json(text: str | None) -> dict | None:
    if not text or not text.strip():
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _tail(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return "…" + text[-max_chars:]
