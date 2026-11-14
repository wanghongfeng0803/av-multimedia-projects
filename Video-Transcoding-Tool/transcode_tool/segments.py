"""大文件分段转码：规划、拼接与临时资源管理。

超大视频一次性转码意味着单次 ffmpeg 运行时间极长、失败后必须从头再来。
本模块按**时长**把任务切成多个独立的小段分别编码，再用 ffmpeg 的
``concat`` 分离器（``-f concat -c copy``）无损拼接——拼接阶段不重新编码，
不损失画质，速度也与拷贝文件相当。

分段边界与进度都基于 ffprobe 报告的时长计算，全部为纯数值逻辑；
真正的 ffmpeg 调用由编排层（transcoder）通过回调注入，本模块只负责
“分几段、每段多长、怎么拼、临时文件放哪”。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile

from .binaries import find_ffmpeg
from .errors import FFmpegRunError

# 分段数硬上限：防止异常时长/过小段长导致创建海量进程与临时文件
MAX_SEGMENTS = 1000
# 允许的最小段长（秒），过短的分段没有意义且关键帧开销占比过高
MIN_SEGMENT_SECONDS = 1.0


@dataclass(frozen=True)
class Segment:
    """单个分段的时间范围。"""

    index: int
    start: float          # 起点（秒）
    duration: float       # 时长（秒）

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class SegmentPlan:
    """一次分段任务的完整规划。"""

    segments: tuple[Segment, ...]
    warnings: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.segments)

    def span_ratio(self, segment: Segment, total_duration: float) -> float:
        """该段占全局进度的比例。"""
        if total_duration <= 0:
            return 1.0 / self.count
        return segment.duration / total_duration


def plan_segments(
    total_duration: float | None,
    *,
    segment_duration: float | None = None,
    segment_count: int | None = None,
) -> SegmentPlan:
    """根据时长与用户偏好生成分段规划。

    :param total_duration: ffprobe 探测到的总时长（秒）；未知/为 0 时
        回退为单段（由调用方按整体转码处理）
    :param segment_duration: 期望的单段时长（秒），与 count 二选一
    :param segment_count: 期望的段数，与 duration 二选一；duration 优先
    :raises ValueError: 参数本身非法（非正数、段数超上限等）
    """
    warnings: list[str] = []

    if segment_duration is not None and segment_duration <= 0:
        raise ValueError("segment_duration 必须是正数（秒）。")
    if segment_count is not None:
        if isinstance(segment_count, bool) or not isinstance(segment_count, int):
            raise ValueError("segment_count 必须是正整数。")
        if segment_count <= 0:
            raise ValueError("segment_count 必须是正整数。")
        if segment_count > MAX_SEGMENTS:
            raise ValueError(
                f"segment_count 不能超过 {MAX_SEGMENTS}，收到 {segment_count}。"
            )

    if not total_duration or total_duration <= 0:
        warnings.append("无法获取输入时长，分段策略退化为单段整体转码。")
        return SegmentPlan((Segment(0, 0.0, 0.0),), tuple(warnings))

    if segment_duration is None and segment_count is None:
        # 未启用分段：单段覆盖整个文件
        return SegmentPlan((Segment(0, 0.0, total_duration),), ())

    if segment_duration is not None:
        target = max(segment_duration, MIN_SEGMENT_SECONDS)
        count = _ceil_div(total_duration, target)
        if count > MAX_SEGMENTS:
            # 段长太小导致段数爆炸：自动放大段长收敛到上限，并明确告知
            target = total_duration / MAX_SEGMENTS
            count = MAX_SEGMENTS
            warnings.append(
                f"按段长 {segment_duration:g}s 计算的段数超过上限 "
                f"{MAX_SEGMENTS}，已自动调整为每段约 {target:g}s。"
            )
    else:
        count = min(segment_count, MAX_SEGMENTS)  # type: ignore[arg-type]
        target = total_duration / count

    # 均匀切分，最后一段吸收舍入余量，保证各段首尾相接且不超出总时长
    segments: list[Segment] = []
    boundaries = [round(total_duration * i / count, 3) for i in range(count + 1)]
    boundaries[-1] = total_duration
    for idx in range(count):
        start = boundaries[idx]
        duration = round(boundaries[idx + 1] - start, 3)
        if duration <= 0:
            continue
        segments.append(Segment(idx, start, duration))

    if not segments:
        return SegmentPlan((Segment(0, 0.0, total_duration),), tuple(warnings))
    return SegmentPlan(tuple(segments), tuple(warnings))


def _ceil_div(value: float, divisor: float) -> int:
    """浮点向上取整除法，避免浮点误差少算一段。"""
    quotient = value / divisor
    rounded = int(quotient)
    if quotient > rounded + 1e-9:
        rounded += 1
    return max(rounded, 1)


def make_temp_workdir(prefix: str = "vtt_seg_") -> str:
    """创建分段工作临时目录（调用方负责最终清理）。"""
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_workdir(workdir: str) -> None:
    """尽力清理临时目录；清理失败不应掩盖主流程结果。"""
    if not workdir or not os.path.isdir(workdir):
        return
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except OSError:
        pass


def segment_output_path(workdir: str, segment: Segment, extension: str) -> str:
    extension = extension if extension.startswith(".") else "." + extension
    return os.path.join(workdir, f"segment_{segment.index:04d}{extension}")


def write_concat_list(workdir: str, segment_paths: list[str]) -> str:
    """生成 ffmpeg concat 分离器的列表文件。

    列表内统一写**相对文件名**（与列表文件同目录），从根上规避 Windows
    反斜杠、盘符冒号、空格等转义问题；单引号文件名按 concat 语法转义。
    """
    list_path = os.path.join(workdir, "concat_list.txt")
    lines: list[str] = []
    for path in segment_paths:
        name = os.path.basename(path)
        escaped = name.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    with open(list_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return list_path


def concat_segments(
    list_path: str,
    output_path: str,
    *,
    ffmpeg_path: str | None = None,
    log_path: str | None = None,
) -> None:
    """用 concat 分离器无损拼接分段（流拷贝，不重新编码）。

    :raises FFmpegRunError: 拼接失败（通常是某分段损坏或容器不支持流拷贝）
    """
    ffmpeg = ffmpeg_path or find_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
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
        if log_path:
            try:
                with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
                    fh.write(proc.stderr or "")
            except OSError:
                pass
        raise FFmpegRunError(
            f"分段拼接失败（退出码 {proc.returncode}）: "
            f"{(proc.stderr or '')[-500:]}",
            stderr=proc.stderr,
            cmd=cmd,
        )
