"""histogram.py 的纯逻辑测试（无 ffmpeg 依赖）。"""

from __future__ import annotations

import pytest

from scene_seg_tool.histogram import (
    GRAY_LEVELS,
    build_histogram,
    enforce_min_scene_length,
    histogram_l1_distance,
    median,
    median_absolute_deviation,
    normalize_distance,
    robust_threshold,
    robust_threshold_with_source,
    suppress_flash_boundaries,
    THRESHOLD_FALLBACK,
    THRESHOLD_GAP_MIDPOINT,
    THRESHOLD_ROBUST,
)


# ---------------------------------------------------------------- 直方图统计

def test_build_histogram_counts_every_level():
    frame = bytes([0, 1, 2, 2, 255, 255, 0])
    hist = build_histogram(frame)
    assert len(hist) == GRAY_LEVELS
    assert hist[0] == 2
    assert hist[1] == 1
    assert hist[2] == 2
    assert hist[255] == 2
    assert sum(hist) == len(frame)


def test_build_histogram_empty_frame_is_all_zero():
    hist = build_histogram(b"")
    assert hist == [0] * GRAY_LEVELS
    assert sum(hist) == 0


def test_histogram_supports_all_256_levels():
    frame = bytes(range(GRAY_LEVELS))
    hist = build_histogram(frame)
    assert hist == [1] * GRAY_LEVELS


# ---------------------------------------------------------------- L1 距离

def test_l1_distance_identical_histograms_is_zero():
    hist = build_histogram(bytes([10, 20, 30, 20]))
    assert histogram_l1_distance(hist, hist) == 0


def test_l1_distance_disjoint_histograms_hits_upper_bound():
    pixels = 100
    a = [0] * GRAY_LEVELS
    b = [0] * GRAY_LEVELS
    a[0] = pixels       # 全部像素落在灰度 0
    b[255] = pixels     # 全部像素落在灰度 255，两分布完全不重叠
    assert histogram_l1_distance(a, b) == 2 * pixels


def test_l1_distance_symmetric_and_integer():
    a = build_histogram(bytes([0] * 4 + [100] * 6))
    b = build_histogram(bytes([0] * 7 + [100] * 3))
    d_ab = histogram_l1_distance(a, b)
    d_ba = histogram_l1_distance(b, a)
    assert d_ab == d_ba
    assert isinstance(d_ab, int)
    # 3 个像素从灰度 100 迁移到 0，每个贡献 2 倍差值计数
    assert d_ab == 6


def test_normalize_distance_range_and_zero_division():
    pixels = 50
    assert normalize_distance(0, pixels) == 0.0
    assert normalize_distance(2 * pixels, pixels) == 1.0
    assert normalize_distance(pixels, pixels) == 0.5
    # 空帧不抛除零异常
    assert normalize_distance(10, 0) == 0.0


# ---------------------------------------------------------------- 排序统计

def test_median_odd_and_even():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


def test_mad_zero_on_constant_sequence():
    assert median_absolute_deviation([5.0, 5.0, 5.0], 5.0) == 0.0


def test_mad_resists_single_outlier():
    # 99 个常态值 + 1 个极端切点，MAD 仍为 0，阈值不会被异常值带偏
    values = [0.01] * 99 + [0.9]
    med = median(values)
    assert med == pytest.approx(0.01)
    assert median_absolute_deviation(values, med) == 0.0


# ---------------------------------------------------------------- 自适应阈值

def test_threshold_empty_sequence_uses_fallback():
    value, source = robust_threshold_with_source([])
    assert value == pytest.approx(0.35)
    assert source == THRESHOLD_FALLBACK


def test_threshold_all_zero_long_static_uses_fallback():
    # 长时间无变化：全部差异为 0，MAD 退化，走兜底阈值
    value, source = robust_threshold_with_source([0.0] * 100)
    assert source == THRESHOLD_FALLBACK
    assert value == pytest.approx(0.35)


def test_threshold_gap_midpoint_when_mad_degenerate():
    # 离散取值、无连续噪声带：阈值落在常态与异常的空隙中点
    scores = [0.1] * 20 + [0.8] * 2
    value, source = robust_threshold_with_source(scores, fallback=0.35)
    assert source == THRESHOLD_GAP_MIDPOINT
    assert value == pytest.approx((0.1 + 0.8) / 2.0)
    assert 0.1 < value < 0.8


def test_threshold_robust_branch_with_noise_band():
    # 连续噪声带使 MAD > 0，走稳健统计分支
    scores = [0.05, 0.06, 0.07, 0.05, 0.08, 0.06, 0.9]
    value, source = robust_threshold_with_source(scores, mad_k=9.0)
    assert source == THRESHOLD_ROBUST
    med = median(scores)
    mad = median_absolute_deviation(scores, med)
    assert value == pytest.approx(med + 9.0 * 1.4826 * mad)
    # 极端切点必须落在阈值之上才能被检出
    assert 0.9 >= value


def test_threshold_is_deterministic_across_input_orderings():
    scores = [0.05, 0.9, 0.06, 0.05, 0.8, 0.07, 0.06]
    first = robust_threshold(scores)
    for _ in range(5):
        shuffled = list(reversed(scores))
        assert robust_threshold(shuffled) == first


# ---------------------------------------------------------------- 闪光抑制

def test_flash_suppression_removes_pair_of_spikes():
    # 帧序列：A A ... A, F(闪光), A ...；两条边界都很大，回返距离为 0
    distances = [0, 0, 1000, 1000, 0, 0]
    returns = [-1, 0, 0, 0, 0, -1]
    suppressed = suppress_flash_boundaries(
        distances, returns, threshold=0.3, pixel_count=1000
    )
    # 闪光帧下标 3 -> 抑制边界 3 与 4
    assert suppressed == {3, 4}


def test_flash_suppression_keeps_genuine_cut_with_large_return():
    # A→B→C 两次真实切换：回退距离 |h[A]-h[C]| 同样很大，不应抑制
    distances = [0, 900, 900, 0]
    returns = [-1, 800, -1, -1]
    suppressed = suppress_flash_boundaries(
        distances, returns, threshold=0.3, pixel_count=1000
    )
    assert suppressed == set()


def test_flash_suppression_ignores_boundaries_below_threshold():
    distances = [10, 10, 10]
    returns = [-1, 0, -1]
    suppressed = suppress_flash_boundaries(
        distances, returns, threshold=0.9, pixel_count=1000
    )
    assert suppressed == set()


def test_flash_suppression_deterministic():
    distances = [0, 1000, 1000, 0, 0, 1000, 1000, 0]
    returns = [-1, 0, 0, 0, 0, 0, 0, -1]
    first = suppress_flash_boundaries(distances, returns, 0.3, 1000)
    for _ in range(3):
        assert suppress_flash_boundaries(distances, returns, 0.3, 1000) == first


# ---------------------------------------------------------------- 最小场景 NMS

def test_nms_keeps_stronger_when_boundaries_too_close():
    # 候选边界 5（距离 100）与 8（距离 200）相隔仅 3 帧，保留更强的 8
    distances = [0] * 10
    distances[4] = 100
    distances[7] = 200
    kept = enforce_min_scene_length([5, 8], distances, min_scene_frames=5)
    assert kept == [8]


def test_nms_tie_breaks_by_smaller_index():
    distances = [0] * 10
    distances[4] = 100
    distances[6] = 100
    kept = enforce_min_scene_length([5, 7], distances, min_scene_frames=5)
    assert kept == [5]


def test_nms_disabled_when_min_length_one():
    distances = [10, 20]
    assert enforce_min_scene_length([1, 2], distances, 1) == [1, 2]


def test_nms_far_apart_both_kept_and_sorted():
    distances = [0] * 30
    distances[2] = 200
    distances[20] = 100
    kept = enforce_min_scene_length([3, 21], distances, min_scene_frames=10)
    assert kept == [3, 21]
