"""M3U8 播放列表：结构校验后原子写入（纯 Python，不依赖 ffmpeg）。

播放列表的每一项都由「同一份时间轴计划 + 分片实测时长」共同决定，
本模块不重新计算任何切点：

- EXTINF 写 :class:`vhls.cut.SegmentResult` 的**实测毫秒时长**（3 位
  小数浮点舍入），保证声明值与磁盘上的分片严格对应；
- EXT-X-TARGETDURATION 取各片实测时长上限的整数秒向上取整，确保
  「TARGETDURATION 与各分片时长上限对得上」，不靠猜测；
- 媒体序列号从 0 起、按计划顺序连续；
- 末片不足目标时长时由计划的 ``is_short`` 标出（结果对象与 CLI 提示），
  其 EXTINF 仍写真实时长，不做补齐。

写盘采用「临时文件 + os.replace」原子替换：进程中途失败不会留下半截
m3u8，播放器永远看不到写了一半的列表。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .timeline import (
    MS_PER_SEC,
    TimelinePlan,
    ceil_div,
    format_extinf,
)

__all__ = [
    "PlaylistError",
    "PlaylistResult",
    "validate_against_plan",
    "render_playlist",
    "write_playlist",
]

#: 播放列表协议版本（3 起支持浮点 EXTINF，兼容所有主流播放器）。
PLAYLIST_VERSION = 3


class PlaylistError(RuntimeError):
    """播放列表校验失败（序列号不连续、时长超限、分片缺失等）。"""


@dataclass(frozen=True)
class PlaylistResult:
    """播放列表产出摘要，供 CLI 与测试断言。

    :ivar path: m3u8 文件路径。
    :ivar target_duration: 写入 EXT-X-TARGETDURATION 的整数秒值。
    :ivar segment_count: 分片总数。
    :ivar media_sequence: 首个分片的媒体序列号（恒为 0）。
    :ivar last_segment_short: 末片是否为不足目标时长的短尾片。
    :ivar last_segment_ms: 末片实测时长（毫秒）。
    :ivar total_ms: 各片实测时长之和（毫秒）。
    """

    path: Path
    target_duration: int
    segment_count: int
    media_sequence: int
    last_segment_short: bool
    last_segment_ms: int
    total_ms: int


def validate_against_plan(plan: TimelinePlan,
                          results: tuple) -> None:
    """写盘前结构校验，任何一条不满足都拒绝产出播放列表。

    校验项对应需求中的四条硬约束：

    1. 切割结果与计划分片数量相同、序列号连续且一一对应；
    2. 每个分片文件真实存在且非空；
    3. 每片实测时长 <= TARGETDURATION（ceil 后的整数秒）；
    4. 实测时长与计划时长的偏差已在切割阶段按容差拦截，这里复核序列号
       与文件名，防止调用方传入错配的结果集合。
    """
    if len(results) != plan.segment_count:
        raise PlaylistError(
            f"切割结果数 {len(results)} 与计划分片数 "
            f"{plan.segment_count} 不一致"
        )

    expected_seq = [s.sequence for s in plan.segments]
    actual_seq = [r.spec.sequence for r in results]
    if actual_seq != expected_seq:
        raise PlaylistError(
            f"分片序列号不连续或错配：期望 {expected_seq}，实际 {actual_seq}"
        )

    target = ceil_div(
        max(r.actual_duration_ms for r in results), MS_PER_SEC
    )
    # 以计划为基准逐项交叉比对，不能用结果自带的 spec 自证。
    for plan_spec, result in zip(plan.segments, results):
        if result.spec.sequence != plan_spec.sequence:
            raise PlaylistError(
                f"位置 {plan_spec.sequence} 的分片序列号错配："
                f"计划 {plan_spec.sequence}，实际 {result.spec.sequence}"
            )
        if plan_spec.filename != result.path.name:
            raise PlaylistError(
                f"分片文件名错配：计划 {plan_spec.filename}，"
                f"实际 {result.path.name}"
            )
        if not result.path.is_file() or result.path.stat().st_size == 0:
            raise PlaylistError(f"分片文件缺失或为空：{result.path}")
        secs = result.actual_duration_ms / MS_PER_SEC
        if secs > target:
            raise PlaylistError(
                f"分片 {plan_spec.sequence} 实测 {secs:.3f}s 超过 "
                f"EXT-X-TARGETDURATION {target}s"
            )
        if plan_spec.start_ms >= plan_spec.end_ms:
            raise PlaylistError(
                f"分片 {plan_spec.sequence} 时间区间非正："
                f"[{plan_spec.start_ms}, {plan_spec.end_ms})ms"
            )


def render_playlist(plan: TimelinePlan, results: tuple) -> str:
    """渲染 m3u8 文本（``\\n`` 行尾，末尾带换行）。

    TARGETDURATION 以**实测**时长最大值 ceil 得到，与各片 EXTINF 上限
    严格对得上；EXTINF 逐个写实测值，序列号由计划保证连续。
    """
    target = ceil_div(
        max(r.actual_duration_ms for r in results), MS_PER_SEC
    )
    lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{PLAYLIST_VERSION}",
        f"#EXT-X-TARGETDURATION:{target}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-INDEPENDENT-SEGMENTS",
    ]
    for result in results:
        lines.append(f"#EXTINF:{format_extinf(result.actual_duration_ms)},")
        lines.append(result.spec.filename)
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def write_playlist(plan: TimelinePlan, results: tuple,
                   playlist_path: str | Path) -> PlaylistResult:
    """校验 → 渲染 → 原子写入播放列表，返回产出摘要。

    :raises PlaylistError: 结构校验未通过（此时不会有任何文件改动）。
    """
    playlist_path = Path(playlist_path)
    validate_against_plan(plan, results)
    text = render_playlist(plan, results)

    playlist_path.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件保证 os.replace 是同卷原子操作，失败不留半截列表。
    fd, tmp_name = tempfile.mkstemp(
        prefix=playlist_path.name + ".", suffix=".tmp",
        dir=str(playlist_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, playlist_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    last = results[-1]
    return PlaylistResult(
        path=playlist_path,
        target_duration=ceil_div(
            max(r.actual_duration_ms for r in results), MS_PER_SEC
        ),
        segment_count=len(results),
        media_sequence=0,
        last_segment_short=last.spec.is_short,
        last_segment_ms=last.actual_duration_ms,
        total_ms=sum(r.actual_duration_ms for r in results),
    )
