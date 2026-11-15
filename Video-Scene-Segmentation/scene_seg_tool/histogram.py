"""灰度直方图统计与相邻帧差异度量（纯标准库实现）。

本模块是整个工具的算法核心，不依赖 opencv / numpy / Pillow 等任何图像库，
只使用 Python 标准库。设计要点：

1. **灰度级数固定 256**。上游 ffmpeg 以 ``gray``（8-bit 单通道）像素格式
   解码，帧数据即一串 0..255 的字节，:class:`bytes` 自带 ``bytes.count``
   与 :class:`collections.Counter` 两条 C 级统计路径，无需逐像素 Python 循环。
2. **差异度量用整数 L1（Bhattacharyya 意义下的直方图交的补）**：
   ``sum_k |h_a[k] - h_b[k]|``，全部为整数加法，不存在浮点舍入，天然可复现。
   归一化放到检测末尾统一做一次浮点除法，同机跨次、跨平台（只要字节序不影响
   字节流解析，gray 原始流本身与字节序无关）结果完全一致。
3. **统计复杂度 O(帧数 × 灰度级数)**。每帧统计 O(像素数)（C 级扫描），
   每边界的 L1 距离固定 256 次减法累加；流式处理只保留 3 条直方图，
   与视频长度无关的 O(256) 直方图内存。
"""

from __future__ import annotations

from collections import Counter

__all__ = [
    "GRAY_LEVELS",
    "build_histogram",
    "histogram_l1_distance",
    "normalize_distance",
    "median",
    "median_absolute_deviation",
    "robust_threshold",
    "robust_threshold_with_source",
    "suppress_flash_boundaries",
    "enforce_min_scene_length",
    "THRESHOLD_ROBUST",
    "THRESHOLD_GAP_MIDPOINT",
    "THRESHOLD_FALLBACK",
]

#: 8-bit 灰度的灰度级数，也是直方图数组的固定长度
GRAY_LEVELS = 256

#: 自适应阈值缺省参数：阈值 = 中位数 + k × 1.4826 × MAD
DEFAULT_MAD_K = 9.0
#: 中位数/MAD 退化（整体静止，MAD=0）时使用的兜底归一化阈值
FALLBACK_THRESHOLD = 0.35
#: 闪光判定的两侧距离比上限（另一侧不超过候选距离的该比例）
FLASH_RATIO = 0.25

#: 阈值来源标签：MAD 稳健统计
THRESHOLD_ROBUST = "robust"
#: 阈值来源标签：MAD 退化为 0 时取常态与异常之间的空隙中点
THRESHOLD_GAP_MIDPOINT = "gap_midpoint"
#: 阈值来源标签：序列为空或全相同（含长时间全静止），使用兜底阈值
THRESHOLD_FALLBACK = "fallback"


def build_histogram(frame: bytes) -> list[int]:
    """统计一帧灰度图像的 256 级直方图。

    :param frame: ``gray`` 像素格式的原始帧字节，每个字节即一个像素的灰度值。
    :returns: 长度 256 的列表，``h[v]`` 是灰度值 ``v`` 出现的像素数，
              ``sum(h) == len(frame)``。

    实现选择：:class:`collections.Counter` 在 CPython 中由 C 级循环完成
    元素计数（``_count_elements``），随后只做至多 256 次 Python 级装箱；
    比 ``bytes.count`` 扫 256 遍和纯 Python 逐像素循环都快（基准数据见
    README「实现与性能」一节）。空帧返回全零直方图，由调用方决定是否拒绝。
    """
    counts = Counter(frame)
    hist = [0] * GRAY_LEVELS
    for value, count in counts.items():
        # gray 像素格式保证字节必然落在 0..255，无需再做范围检查
        hist[value] = count
    return hist


def histogram_l1_distance(hist_a: list[int], hist_b: list[int]) -> int:
    """两条直方图的整数 L1 距离。

    对同尺寸帧恒有 ``0 <= d <= 2 × 像素数``：两帧完全相同时为 0，
    灰度分布完全不重叠时取上界。整数运算无舍入误差，是可复现性的基础。
    """
    distance = 0
    for a, b in zip(hist_a, hist_b):
        distance += abs(a - b)
    return distance


def normalize_distance(distance: int, pixel_count: int) -> float:
    """整数距离归一化到 ``[0.0, 1.0]``。

    :param distance: :func:`histogram_l1_distance` 的整数结果。
    :param pixel_count: 单帧像素数（宽 × 高）。
    :returns: ``distance / (2 × pixel_count)``；``pixel_count`` 为 0
              （空帧）时返回 0.0，避免除零。

    归一化只在输出与判定阶段做一次，中间统计全程保持整数。
    """
    if pixel_count <= 0:
        return 0.0
    return distance / (2 * pixel_count)


def median(values: list[float]) -> float:
    """中位数（确定性实现，不依赖 statistics 模块的隐式排序策略）。"""
    if not values:
        raise ValueError("median() 需要至少一个元素")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def median_absolute_deviation(values: list[float], med: float) -> float:
    """中位绝对偏差（MAD）：``median(|x - med|)``。

    MAD 对硬切产生的少量极端值不敏感，是长时间静止视频中把阈值抬到
    噪声之上、同时又不会被单个大值带偏的关键统计量。
    """
    return median([abs(v - med) for v in values])


def robust_threshold_with_source(
    scores: list[float],
    mad_k: float = DEFAULT_MAD_K,
    fallback: float = FALLBACK_THRESHOLD,
) -> tuple[float, str]:
    """由完整差异序列计算自适应判定阈值，并返回阈值来源标签。

    规则（全部基于确定性排序统计，无随机量、无迭代收敛）：

    1. 空序列或序列元素完全相同（含全 0，即长时间无变化）：
       返回 ``(fallback, "fallback")``；
    2. ``MAD > 0``：返回 ``(median + mad_k × 1.4826 × MAD, "robust")``。
       1.4826 是 MAD 对标正态分布标准差的一致性常数，``mad_k`` 可理解为
       「偏离常态多少个标准差才算切换」；
    3. MAD 为 0 但序列非全相同（离散的少数取值、无连续噪声带）：
       取「严格大于中位数的最小取值」与中位数的中点，使阈值落在常态与
       异常之间的空隙里，硬切仍可被检出而常态不会误报，来源记
       ``"gap_midpoint"``。

    阈值不做 ``min(1.0)`` 截断：超过 1.0 时天然没有任何边界能通过判定，
    等价于「本视频无硬切」，语义更直接。
    """
    if not scores:
        return fallback, THRESHOLD_FALLBACK
    med = median(scores)
    mad = median_absolute_deviation(scores, med)
    if mad > 0.0:
        return med + mad_k * 1.4826 * mad, THRESHOLD_ROBUST
    above = sorted(v for v in scores if v > med)
    if not above:
        # 所有得分完全相同（含全 0 的长静止视频）
        return fallback, THRESHOLD_FALLBACK
    return (med + above[0]) / 2.0, THRESHOLD_GAP_MIDPOINT


def robust_threshold(
    scores: list[float],
    mad_k: float = DEFAULT_MAD_K,
    fallback: float = FALLBACK_THRESHOLD,
) -> float:
    """:func:`robust_threshold_with_source` 的只取阈值版本。"""
    return robust_threshold_with_source(scores, mad_k=mad_k, fallback=fallback)[0]


def suppress_flash_boundaries(
    distances: list[int],
    returns: list[int],
    threshold: float,
    pixel_count: int,
    flash_ratio: float = FLASH_RATIO,
) -> set[int]:
    """识别应被闪光抑制的边界下标。

    亮度/曝光突变（闪光灯、白闪、黑闪）与真实场景切换在单条 L1 距离上
    无法区分，但单帧闪光在差异序列上留下**成对**的尖峰：

    * 闪光帧 ``t`` 的两条边界 ``t``（A→F）与 ``t+1``（F→A）距离都很大；
    * 跳过闪光帧后两侧画面几乎相同，即三帧回返距离
      ``|h[t-1] - h[t+1]|`` 很小。

    真实硬切 A→B 后若紧接着又切 B→C，回退距离 ``|h[A] - h[C]|`` 通常
    依然很大，不会被误抑制。因此本函数以**闪光帧**为单位判定，
    一次性抑制其前后两条边界（首尾帧不可能成为闪光帧，因为无法构成
    三元组，天然不会被处理）。

    判定（``1 <= t <= N-2``，distances 下标从 0 开始）：

    * 两条边界归一化距离都达到阈值；
    * 回返距离小于两条边界距离最大值的 *flash_ratio* 倍。

    多帧闪光（持续 2 帧及以上，如慢淡入淡出）不存在单帧三元组回返关系，
    不在本函数处理范围内，README 中明确说明这一能力边界。

    :param distances: ``distances[b-1]`` 为边界 b 的整数 L1 距离，长度 N-1。
    :param returns: 与 distances 等长，``returns[b-1]`` 为三帧回返距离，
                    无法构成三元组的位置为 -1。
    :returns: 被抑制的边界编号集合（1..N-1）。
    """
    suppressed: set[int] = set()
    if pixel_count <= 0:
        return suppressed
    span = 2.0 * pixel_count
    # 闪光帧 t：其进入边界下标 t-1、离开边界下标 t，要求两条都在序列内
    for flash_frame in range(1, len(distances)):
        entry_idx = flash_frame - 1
        exit_idx = flash_frame
        d_entry = distances[entry_idx]
        d_exit = distances[exit_idx]
        if d_entry / span < threshold or d_exit / span < threshold:
            continue
        back = returns[entry_idx]
        if back < 0:
            continue
        # 操作数均为确定的整数值与固定常量，IEEE 754 基本运算结果唯一，
        # 不引入任何随机性，跨次运行判定一致
        if back < flash_ratio * max(d_entry, d_exit):
            suppressed.add(flash_frame)       # 进入边界
            suppressed.add(flash_frame + 1)   # 离开边界
    return suppressed


def enforce_min_scene_length(
    candidates: list[int],
    distances: list[int],
    min_scene_frames: int,
) -> list[int]:
    """最小场景长度的确定性非极大值抑制（NMS）。

    两个被保留切点之间至少相隔 *min_scene_frames* 帧。冲突时按
    「整数距离更大者优先、距离相同则边界下标更小者优先」的全序规则处理，
    保证任何输入下结果唯一、可复现。

    :param candidates: 候选边界编号（1..N-1），调用方应已完成阈值判定
                       与闪光抑制。
    :param distances: 完整整数距离序列（按距离排序冲突用）。
    :param min_scene_frames: 最小场景长度（帧），<=1 表示不做抑制。
    """
    if min_scene_frames <= 1 or len(candidates) <= 1:
        return list(candidates)
    # 距离降序、下标升序：全序，无并列歧义
    ranked = sorted(candidates, key=lambda b: (-distances[b - 1], b))
    kept: list[int] = []
    for boundary in ranked:
        if all(abs(boundary - other) >= min_scene_frames for other in kept):
            kept.append(boundary)
    return sorted(kept)
