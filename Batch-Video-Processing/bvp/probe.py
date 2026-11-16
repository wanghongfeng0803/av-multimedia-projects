"""基于 ffprobe 的输入媒体探测。

探测结论直接服务于错误分类契约：

- 文件不存在 / 无法访问 → :class:`~bvp.errors.InputNotFoundError`
  （``INPUT_NOT_FOUND``，批量层记为失败）；
- 文件存在但 ffprobe 无法解析、没有任何流、或不含视频流 →
  :class:`~bvp.errors.InvalidInputError`（``NOT_A_VIDEO``，批量层记为跳过）；
- ffprobe 自身异常退出且无任何可用输出 → 同样按无法识别处理。

对截断/损坏但仍能部分读取的文件，尽量返回 :class:`MediaInfo` 并把问题
记入 ``warnings``，而不是一刀切拒绝。
"""

from __future__ import annotations

import json
import os
import subprocess

from .binaries import find_ffprobe
from .errors import InvalidInputError
from .models import MediaInfo, StreamInfo

__all__ = ["probe_media"]


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

    :raises InputNotFoundError: 文件不存在或不是普通文件
    :raises InvalidInputError: ffprobe 无法解析、无任何流或不含视频流
    """
    # 延迟导入避免与 errors 模块的契约加载顺序产生不必要耦合
    from .errors import InputNotFoundError

    if not os.path.isfile(file_path):
        raise InputNotFoundError(f"输入文件不存在或不是普通文件: {file_path}")

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
        # 截断/损坏文件常返回非零但仍输出部分 JSON，先尝试解析；
        # 解析不出来才判定为无法识别（NOT_A_VIDEO → 跳过）
        parsed = _try_parse_json(proc.stdout)
        if parsed is None or not parsed.get("streams"):
            raise InvalidInputError(
                "ffprobe 无法解析该文件，它可能不是视频文件或已严重损坏。"
                f" 最后错误: {_tail(stderr_text)}"
            )
        warnings.append(
            f"输入文件可能已损坏，探测返回非零状态: {_tail(stderr_text)}"
        )
    else:
        parsed = _try_parse_json(proc.stdout)
        if parsed is None:
            raise InvalidInputError(
                "ffprobe 输出无法解析为 JSON，无法识别该文件。"
            )
        if stderr_text.strip():
            warnings.append(f"探测警告: {_tail(stderr_text)}")

    raw_streams = parsed.get("streams") or []
    if not raw_streams:
        raise InvalidInputError("文件中没有任何媒体流，无法识别为视频。")

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
            "输入文件不含视频流（可能是纯音频或其它非视频文件）。"
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
