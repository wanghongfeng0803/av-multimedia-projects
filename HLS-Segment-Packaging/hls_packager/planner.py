"""GOP 对齐的分片规划器（纯函数、无 IO，便于单测）。

输入：源文件时长、目标分片时长、关键帧时间序列。
输出：每个分片的源内起点与计划时长。

核心规则
--------
1. **切点只落在关键帧上**。视频分片必须从可独立解码的 IDR 帧开始，
   否则播放器跳到该分片时会花屏 / 黑屏，直到下一个 IDR 才能恢复。
2. 每个分片从当前边界 ``current`` 出发，理想切点为
   ``current + target_duration``，在其后的关键帧中选择距理想切点
   最近的一个；距离相同取靠前的关键帧（保证分片不会无谓变长）。
3. 若理想切点已经越过源文件末尾，则不再切，最后一个分片自然收尾。
4. 纯音频输入（或关键帧信息缺失）没有 GOP 概念，按目标时长做等长
   规划，尾部过短时并入前一片。
5. 大 GOP 场景（关键帧间隔 > 目标时长）下分片会比目标时长长——
   这是 GOP 对齐的必然代价，``EXT-X-TARGETDURATION`` 会按实际最大
   片长向上取整，符合 RFC 8216 的要求。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SegmentInfo

# 时间判零容差：小于该值的分片视为空片，不产出
_EPS = 1e-3
# 尾部短片并入前一片的阈值系数（相对目标时长）
_TAIL_MERGE_RATIO = 0.5


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


def _video_boundaries(
    duration: float, target_duration: float, keyframes: list[float]
) -> list[float]:
    """GOP 对齐算法：返回包含 0 在内的全部分片起点。"""
    boundaries = [0.0]
    current = 0.0

    while current < duration - _EPS:
        ideal = current + target_duration
        # 理想切点已到/越过末尾：剩余内容全部作为最后一片
        if ideal >= duration - _EPS:
            break

        # 在当前边界之后寻找距理想切点最近的关键帧（同距取靠前）
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


def plan_segments(
    duration: float,
    target_duration: float,
    keyframe_times: tuple[float, ...] = (),
    *,
    sequence_start: int = 0,
    filename_prefix: str = "seg",
) -> list[SegmentInfo]:
    """生成分片规划。

    :param duration: 源文件总时长（秒）；
    :param target_duration: 目标分片时长（秒）；
    :param keyframe_times: 视频关键帧时间序列（升序，秒），为空走等长规划；
    :param sequence_start: 首个分片的 EXT-X-MEDIA-SEQUENCE 序号；
    :param filename_prefix: 分片文件名前缀。
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

    keyframes = _clamp_keyframes(keyframe_times, duration)
    if keyframes:
        boundaries = _video_boundaries(duration, target_duration, keyframes)
    else:
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
                filename=f"{filename_prefix}_{sequence:04d}.ts",
                source_start=round(start, 6),
            )
        )
    return segments
