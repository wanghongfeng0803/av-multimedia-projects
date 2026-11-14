"""GOP 对齐的分片规划器（纯函数、无 IO，便于单测）。

输入：源文件时长、目标分片时长、关键帧时间序列。
输出：每个分片的源内起点与计划时长。

核心规则
--------
1. **切点只落在关键帧上**。视频分片必须从可独立解码的 IDR 帧开始，
   否则播放器跳到该分片时会花屏 / 黑屏，直到下一个 IDR 才能恢复。
2. 每个分片从当前边界 ``current`` 出发，理想切点为
   ``current + target_duration``，按对齐模式在关键帧中选点：

   * ``nearest``（默认）：距理想切点最近的关键帧，距离相同取靠前
     的一个（保证分片不会无谓变长）；
   * ``before``：只取理想切点**之前（含重合）**最近的关键帧，分片
     时长不会超过目标值（末段除外）；前面没有候选时退而取之后
     最近的关键帧，避免切出零长度分片；
   * ``after``：只取理想切点**之后（含重合）**最近的关键帧，分片
     时长不会短于目标值（末段除外），适合希望分片尽量"装满"的
     归档场景。

3. 若理想切点已经越过源文件末尾，则不再切，最后一个分片自然收尾。
4. 纯音频输入（或关键帧信息缺失）没有 GOP 概念，按目标时长做等长
   规划，尾部过短时并入前一片——对齐模式对其无意义（切点不受
   关键帧约束）。
5. 大 GOP 场景（关键帧间隔 > 目标时长）下分片会比目标时长长——
   这是 GOP 对齐的必然代价，``EXT-X-TARGETDURATION`` 会按实际最大
   片长向上取整，符合 RFC 8216 的要求。

时长精度
--------
所有时间戳以 float 秒在内部传递，比较统一走 :data:`_EPS`（1ms）
容差，落盘前 round 到 6 位小数（微秒级，高于 MPEG-TS 90kHz 时钟
的 ~11.1us 单 tick 精度需求）；分片计划时长之和与源时长严格相等，
实际片长以切后 ffprobe 复测为准（见 :mod:`hls_packager.packager`）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SegmentInfo
from .naming import DEFAULT_NAME_TEMPLATE, NamingConfig

# 时间判零容差：小于该值的分片视为空片，不产出（1ms，覆盖容器时间戳抖动）
_EPS = 1e-3
# 尾部短片并入前一片的阈值系数（相对目标时长）
_TAIL_MERGE_RATIO = 0.5

#: 切点对齐模式
ALIGN_NEAREST = "nearest"
ALIGN_BEFORE = "before"
ALIGN_AFTER = "after"
ALIGN_MODES = (ALIGN_NEAREST, ALIGN_BEFORE, ALIGN_AFTER)


@dataclass(frozen=True)
class SegmentPlan:
    """规划阶段的分片描述（尚未实际切片）。"""

    source_start: float       # 在源文件中的起点（秒）
    planned_duration: float   # 计划时长（秒），实际时长以切后探测为准


def _clamp_keyframes(
    keyframe_times: tuple[float, ...], duration: float
) -> list[float]:
    """过滤并规整关键帧：落在 [0, duration] 内、升序、去重、夹紧边界。"""
    cleaned: list[float] = []
    for t in keyframe_times:
        if t < -_EPS or t > duration + _EPS:
            continue
        t = max(0.0, min(float(duration), t))
        if not cleaned or t - cleaned[-1] > _EPS:
            cleaned.append(t)
    return cleaned


def _pick_nearest(
    keyframes: list[float], current: float, ideal: float, duration: float
) -> float | None:
    """默认模式：距理想切点最近的关键帧，同距取靠前（历史行为）。"""
    best: float | None = None
    best_diff = float("inf")
    for kf in keyframes:
        if kf <= current + _EPS:
            continue
        if kf >= duration - _EPS:
            break
        diff = abs(kf - ideal)
        if diff < best_diff:
            best_diff = diff
            best = kf
    return best


def _pick_before(
    keyframes: list[float], current: float, ideal: float, duration: float
) -> float | None:
    """只取理想切点之前（含重合）最近的关键帧。

    前面没有候选时（首个 GOP 就比目标时长长）退而取之后最近的
    关键帧，否则会切不出边界、整片被拉成一个超长分片。
    """
    before: float | None = None
    after: float | None = None
    for kf in keyframes:
        if kf <= current + _EPS:
            continue
        if kf >= duration - _EPS:
            break
        if kf <= ideal + _EPS:
            # 升序遍历，最后一个满足条件的即最靠近 ideal 的前侧候选
            before = kf
        else:
            after = kf  # 第一个超过 ideal 的即最近的后侧候选
            break
    return before if before is not None else after


def _pick_after(
    keyframes: list[float], current: float, ideal: float, duration: float
) -> float | None:
    """只取理想切点之后（含重合）最近的关键帧。"""
    for kf in keyframes:
        if kf <= current + _EPS:
            continue
        if kf >= duration - _EPS:
            break
        if kf >= ideal - _EPS:
            return kf
    return None


_PICKERS = {
    ALIGN_NEAREST: _pick_nearest,
    ALIGN_BEFORE: _pick_before,
    ALIGN_AFTER: _pick_after,
}


def _video_boundaries(
    duration: float,
    target_duration: float,
    keyframes: list[float],
    align_mode: str,
) -> list[float]:
    """GOP 对齐算法：返回包含 0 在内的全部分片起点。"""
    picker = _PICKERS[align_mode]
    boundaries = [0.0]
    current = 0.0

    while current < duration - _EPS:
        ideal = current + target_duration
        # 理想切点已到/越过末尾：剩余内容全部作为最后一片
        if ideal >= duration - _EPS:
            break

        best = picker(keyframes, current, ideal, duration)
        if best is None:
            # 后面没有可用关键帧，剩余内容并入当前分片
            break
        if best - current < _EPS:
            break

        boundaries.append(best)
        current = best

    return boundaries


def _uniform_boundaries(duration: float, target_duration: float) -> list[float]:
    """等长规划（纯音频 / 无关键帧信息时的回退路径）。"""
    if duration <= _EPS or target_duration <= 0:
        return [0.0] if duration > _EPS else []

    boundaries = [0.0]
    next_start = target_duration
    while next_start < duration - _EPS:
        boundaries.append(next_start)
        next_start += target_duration

    # 尾部过短则并入前一片：撤销最后一个边界
    tail = duration - boundaries[-1]
    if len(boundaries) > 1 and tail < target_duration * _TAIL_MERGE_RATIO:
        boundaries.pop()
    return boundaries


def _resolve_naming(
    filename_prefix: str,
    name_template: str | None,
    extension: str,
) -> NamingConfig:
    """把旧的 prefix 参数与新的模板参数归一为 :class:`NamingConfig`。

    显式模板优先；未给模板时用默认模板 + 前缀，扩展名随容器变化，
    保证旧调用（只传 prefix）在 TS 容器下产出与历史完全一致的文件名。
    """
    if name_template is None:
        return NamingConfig(
            template=DEFAULT_NAME_TEMPLATE,
            name_prefix=filename_prefix,
            extension=extension,
        )
    return NamingConfig(
        template=name_template,
        name_prefix=filename_prefix,
        extension=extension,
    )


def plan_segments(
    duration: float,
    target_duration: float,
    keyframe_times: tuple[float, ...] = (),
    *,
    sequence_start: int = 0,
    filename_prefix: str = "seg",
    align_mode: str = ALIGN_NEAREST,
    name_template: str | None = None,
    extension: str = "ts",
) -> list[SegmentInfo]:
    """生成分片规划。

    :param duration: 源文件总时长（秒）；
    :param target_duration: 目标分片时长（秒）；
    :param keyframe_times: 视频关键帧时间序列（升序，秒），为空走等长规划；
    :param sequence_start: 首个分片的 EXT-X-MEDIA-SEQUENCE 序号；
    :param filename_prefix: 分片文件名前缀（``{name}`` 占位符的取值）；
    :param align_mode: 切点对齐模式，``nearest`` / ``before`` / ``after``，
        仅对有关键帧的视频输入生效；
    :param name_template: 命名模板；``None`` 时使用默认模板
        ``{name}_{sequence:04d}.{ext}``，TS 容器下与历史文件名一致；
    :param extension: 分片扩展名（``{ext}`` 占位符的取值，由容器规格提供）。
    :returns: 按时间排序的 :class:`SegmentInfo` 列表（时长为计划值，
        实际切片后由打包层用 ffprobe 复测值替换）。
    :raises ValueError: 参数非法时。
    """
    if duration <= 0:
        raise ValueError(f"源文件时长必须为正数，实际为 {duration!r}")
    if target_duration <= 0:
        raise ValueError(
            f"目标分片时长必须为正数，实际为 {target_duration!r}"
        )
    if sequence_start < 0:
        raise ValueError("起始序号不能为负数")
    if align_mode not in ALIGN_MODES:
        raise ValueError(
            f"对齐模式必须为 {ALIGN_MODES} 之一，实际为 {align_mode!r}"
        )

    naming = _resolve_naming(filename_prefix, name_template, extension)

    keyframes = _clamp_keyframes(keyframe_times, duration)
    if keyframes:
        boundaries = _video_boundaries(
            duration, target_duration, keyframes, align_mode
        )
    else:
        # 纯音频 / 无关键帧：切点不受 GOP 约束，对齐模式无意义
        boundaries = _uniform_boundaries(duration, target_duration)

    if not boundaries:
        return []

    # 边界补齐末尾时间戳，逐片计算计划时长
    cut_points = boundaries + [float(duration)]
    segments: list[SegmentInfo] = []
    for i, start in enumerate(boundaries):
        seg_duration = cut_points[i + 1] - start
        if seg_duration < _EPS:
            continue
        sequence = sequence_start + i
        segments.append(
            SegmentInfo(
                sequence_number=sequence,
                start_time=round(start, 6),
                duration=round(seg_duration, 6),
                filename=naming.render(index=i, sequence=sequence),
                source_start=round(start, 6),
            )
        )
    return segments
