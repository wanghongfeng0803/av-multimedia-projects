"""管线编排：探测 → 时间轴规划 → 逐片切割 → 播放列表原子写入。

四个阶段严格单向，时间轴对象（:class:`vhls.timeline.TimelinePlan`）在
规划阶段一次性生成，切割与写列表都只读它，任何阶段都不另起一套切点
计算，这就是「列表写入与切片切割共用同一套时间轴」的编排保证。

默认目标分片时长取 6 秒：Apple HLS Authoring 推荐 6s，与 24/25/30fps
均为整数帧边界，常规 VOD 场景下分片数与首屏开销折中良好；可通过
``target_seconds`` 显式覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cut import (
    DEFAULT_DURATION_TOLERANCE_MS,
    CutError,
    cut_segments,
)
from .playlist import PlaylistError, PlaylistResult, write_playlist
from .probe import ProbeError, SourceInfo, probe_source
from .timeline import (
    TimelineError,
    TimelinePlan,
    build_timeline,
)

__all__ = [
    "DEFAULT_TARGET_SECONDS",
    "PackError",
    "PackResult",
    "pack_hls",
]

#: 默认目标分片时长（秒），依据见模块 docstring。
DEFAULT_TARGET_SECONDS = 6.0
#: 默认分片文件名前缀。
DEFAULT_SEGMENT_PREFIX = "seg"


class PackError(RuntimeError):
    """打包管线失败（包装下层 ProbeError/TimelineError/CutError 等）。"""


@dataclass(frozen=True)
class PackResult:
    """一次 HLS 打包的完整可观测结果，供 CLI 与测试断言。

    :ivar playlist: 播放列表产出摘要。
    :ivar plan: 统一时间轴计划。
    :ivar results: 逐片切割结果（实测时长）。
    :ivar source: 源探测信息。
    :ivar output_dir: 分片与列表所在目录。
    """

    playlist: PlaylistResult
    plan: TimelinePlan
    results: tuple
    source: SourceInfo
    output_dir: Path

    @property
    def short_segments(self) -> tuple:
        """所有被标记为短尾的分片（常规 VOD 最多末片一个）。"""
        return tuple(r for r in self.results if r.spec.is_short)


def pack_hls(
    input_path: str | Path,
    output_dir: str | Path,
    playlist_name: str = "index.m3u8",
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    segment_prefix: str = DEFAULT_SEGMENT_PREFIX,
    tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS,
    source: SourceInfo | None = None,
) -> PackResult:
    """执行完整的单码率 HLS 打包。

    :param input_path: 源媒体路径（含视频流；音频可选）。
    :param output_dir: 产物目录，分片与 m3u8 都写在这里，自动创建。
    :param playlist_name: 播放列表文件名（默认 ``index.m3u8``）。
    :param target_seconds: 目标分片时长（秒），默认 6。
    :param segment_prefix: 分片文件名前缀（默认 ``seg``）。
    :param tolerance_ms: 分片实测时长与计划值的容差（毫秒）。
    :param source: 已探测的源信息；None 时内部调用 ffprobe。给出时
        其 ``path`` 仍须指向待切割的源文件。
    :raises PackError: 任一阶段失败；播放列表仅在全部校验通过后才落盘。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not segment_prefix:
        raise PackError("分片文件名前缀不能为空")
    if not playlist_name:
        raise PackError("播放列表文件名不能为空")

    # ---- 阶段一：探测（时间戳以 ffprobe 为准）----
    try:
        if source is None:
            source = probe_source(input_path)
        elif source.path != input_path:
            raise PackError(
                f"传入的源信息路径 {source.path} 与输入 {input_path} 不一致"
            )
    except ProbeError as exc:
        raise PackError(f"探测失败：{exc}") from exc

    # ---- 阶段二：统一时间轴（整数毫秒，切点只落帧边缘）----
    try:
        plan = build_timeline(
            source, target_seconds, segment_prefix=segment_prefix,
        )
    except TimelineError as exc:
        raise PackError(f"时间轴规划失败：{exc}") from exc

    # ---- 阶段三：按计划逐片切割 + 独立实测时长 ----
    try:
        results = cut_segments(source, plan, output_dir, tolerance_ms)
    except CutError as exc:
        raise PackError(f"切片切割失败：{exc}") from exc

    # ---- 阶段四：结构校验通过后原子写播放列表 ----
    playlist_path = output_dir / playlist_name
    try:
        playlist = write_playlist(plan, results, playlist_path)
    except (PlaylistError, OSError) as exc:
        raise PackError(f"播放列表写入失败：{exc}") from exc

    return PackResult(
        playlist=playlist, plan=plan, results=results,
        source=source, output_dir=output_dir,
    )
