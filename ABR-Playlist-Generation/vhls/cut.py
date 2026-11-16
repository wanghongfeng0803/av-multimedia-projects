"""切片切割：按统一时间轴计划逐片调 ffmpeg 重编码，并实测每片时长。

切割严格服从 :class:`vhls.timeline.TimelinePlan`，本模块不自行决定任何
切点——这是「列表写入与切片切割共用同一套时间轴」在切割侧的落地。

两个关键取舍：

- **重编码而非流复制（``-c copy``）**：流复制只能在源关键帧处切开，
  切点会被吸附到最近的 IDR，实际分片边界与计划不符，EXTINF 只能跟着
  漂。重编码配合 ``-ss/-t`` 可在任意帧精确切割，并对每片强制
  ``-force_key_frames 0``，保证每个分片都以 IDR 起始、可独立解码。
- **切完独立 ffprobe 实测**：编码器在毫秒级可能有半帧出入，播放列表
  必须写「文件的真实时长」而不是计划值。实测结果回传给 playlist，
  若实测与计划偏差超过容差则直接失败，不让对不上的列表落盘。

时间参数用计划的毫秒换算回秒（3 位小数，与内部单位同分辨率），不经过
任何浮点累加。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .probe import ProbeError, SourceInfo, check_tools, run_capture
from .timeline import (
    MS_PER_SEC,
    SegmentSpec,
    TimelinePlan,
    ms_to_seconds,
)

__all__ = [
    "CutError",
    "SegmentResult",
    "cut_segments",
    "probe_segment_duration_ms",
    "DEFAULT_DURATION_TOLERANCE_MS",
]

#: 实测分片时长（视频轨口径）与计划值的默认容差（毫秒）。
#: 重编码帧精确切割下视频轨偏差实测为 0；留 20ms 兜底时间戳封装对齐，
#: 超过即判定时间轴被破坏，拒绝写播放列表。注意 MPEG-TS 的 format
#: duration 含音频帧填充（可长出约一个 AAC 帧 ~23ms），本项目实测取
#: 视频流时长而非 format duration，故容差无需为容器填充放大。
DEFAULT_DURATION_TOLERANCE_MS = 20


class CutError(RuntimeError):
    """ffmpeg 切割失败、产物缺失或实测时长与计划不一致。"""


@dataclass(frozen=True)
class SegmentResult:
    """单片切割结果：计划规格 + 落盘路径 + ffprobe 实测时长。

    :ivar spec: 对应的计划分片（序列号/帧范围/计划时长）。
    :ivar path: 分片落盘路径。
    :ivar actual_duration_ms: ffprobe 实测时长（毫秒）。
    """

    spec: SegmentSpec
    path: Path
    actual_duration_ms: int


def _seconds_text(ms: int) -> str:
    """毫秒 → ffmpeg 时间参数字符串（3 位小数，无浮点累加）。"""
    return f"{ms_to_seconds(ms):.3f}"


def probe_segment_duration_ms(path: Path) -> int:
    """用 ffprobe 实测分片时长，返回整数毫秒（半值向上取整）。

    口径上**优先取视频流时长**而非 format duration：MPEG-TS 复用器会在
    容器尾部做 PCR/音频帧对齐填充，实测 format duration 比视频轨长出
    一个 AAC 帧（约 23ms），那段时间没有任何画面，不是真实播放时长。
    HLS 播放器按视频轨时间线呈现，EXTINF 以视频时长为准才能与帧边界
    计划严格对应。视频流时长缺失时才回退到 format duration。
    不以计划值参与计算，保证是独立观测值。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=duration,codec_type",
        "-of", "json", str(path),
    ]
    data = json.loads(run_capture(cmd).stdout.decode("utf-8"))

    raw = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and \
                stream.get("duration") not in (None, "N/A"):
            raw = float(stream["duration"])
            break
    if raw is None:
        fmt = data.get("format")
        if fmt and fmt.get("duration") not in (None, "N/A"):
            raw = float(fmt["duration"])
    if raw is None:
        raise CutError(f"ffprobe 无法读取分片时长：{path}")
    return int(raw * MS_PER_SEC + 0.5)


def _build_ffmpeg_cmd(source: SourceInfo, spec: SegmentSpec,
                      out_path: Path) -> list:
    """构造单片重编码命令（帧精确切割 + 片首强制 IDR）。"""
    start = _seconds_text(spec.start_ms)
    length = _seconds_text(spec.planned_duration_ms)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", start,                  # 先 seek 到片起点（重编码下帧精确）
        "-i", str(source.path),
        "-t", length,                  # 单片时长，由计划毫秒换算，不累积
        "-map", "0:v:0",
        "-force_key_frames", "0",      # 片首强制 IDR，每片可独立起播
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
    ]
    if source.has_audio:
        # AAC 帧约 23ms 粒度，-t 在音频侧按帧对齐，偏差由实测容差兜底。
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-f", "mpegts", str(out_path)]
    return cmd


def cut_segments(
    source: SourceInfo,
    plan: TimelinePlan,
    output_dir: str | Path,
    tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS,
) -> tuple:
    """按计划逐片切割并实测时长，返回与计划一一对应的切割结果。

    任一片切割失败、产物为空或实测时长超出容差即抛 :class:`CutError`，
    已产出的分片保留在输出目录（便于排查），但上层不应写播放列表。

    :param tolerance_ms: 实测与计划时长的最大允许偏差（毫秒）。
    :return: :class:`SegmentResult` 元组，顺序与 ``plan.segments`` 一致。
    """
    check_tools()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tolerance_ms < 0:
        raise CutError(f"时长容差不能为负：{tolerance_ms}")

    results: list = []
    for spec in plan.segments:
        out_path = output_dir / spec.filename
        cmd = _build_ffmpeg_cmd(source, spec, out_path)
        try:
            run_capture(cmd)
        except ProbeError as exc:
            raise CutError(
                f"分片 {spec.sequence}（{spec.filename}）切割失败：{exc}"
            ) from exc

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise CutError(
                f"分片 {spec.sequence} 产物缺失或为空：{out_path}"
            )

        actual_ms = probe_segment_duration_ms(out_path)
        drift = actual_ms - spec.planned_duration_ms
        if abs(drift) > tolerance_ms:
            raise CutError(
                f"分片 {spec.sequence} 实测时长 {actual_ms}ms 与计划 "
                f"{spec.planned_duration_ms}ms 偏差 {drift:+d}ms，"
                f"超过容差 ±{tolerance_ms}ms，时间轴一致性被破坏"
            )
        results.append(SegmentResult(
            spec=spec, path=out_path, actual_duration_ms=actual_ms,
        ))
    return tuple(results)
