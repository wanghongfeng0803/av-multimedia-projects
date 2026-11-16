"""中值滤波单元测试（纯逻辑，不依赖 ffmpeg）。

对照基准用标准库 statistics.median_low 暴力直算（含镜像填充），
Huang 直方图实现与分带实现都要求与之零容差逐像素相等。
"""

import random
import statistics
import time

import pytest

from vdenoise import (
    border_mask,
    dump_border_frame,
    median_plane,
    median_plane_banded,
    mirror_index,
    mirror_indices,
    plan_median_bands,
)


def brute_median(data, width, height, radius):
    """暴力直算镜像填充窗口的中值（statistics.median_low 低中值）。"""
    out = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            vals = [
                data[mirror_index(y + dy, height) * width
                     + mirror_index(x + dx, width)]
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
            ]
            out[y * width + x] = statistics.median_low(vals)
    return bytes(out)


# ---------- 镜像填充 ----------

def test_mirror_index_reflect_101_no_edge_duplication():
    # 源 [0,1,2,3]：p=-1 -> 1（不是 0），p=4 -> 2（不是 3）
    assert mirror_index(-1, 4) == 1
    assert mirror_index(-2, 4) == 2
    assert mirror_index(4, 4) == 2
    assert mirror_index(5, 4) == 1
    assert mirror_index(6, 4) == 0


def test_mirror_index_inside_identity():
    n = 6
    assert [mirror_index(p, n) for p in range(n)] == list(range(n))


def test_mirror_index_degenerate_n_one():
    assert mirror_index(0, 1) == 0
    assert mirror_index(-7, 1) == 0
    assert mirror_index(7, 1) == 0


def test_mirror_index_radius_larger_than_axis():
    # 半径远超轴长也要落在 [0, n) 内
    for n in (2, 3, 5):
        for p in range(-20, 21):
            assert 0 <= mirror_index(p, n) < n


def test_mirror_indices_full_sequence():
    # n=4, r=2：前 2 + 源 4 + 后 2
    assert mirror_indices(4, 2) == [2, 1, 0, 1, 2, 3, 2, 1]
    assert mirror_indices(5, 1) == [1, 0, 1, 2, 3, 4, 3]


# ---------- 整帧中值 vs 暴力直算（零容差） ----------

@pytest.mark.parametrize("w,h,r", [
    (1, 1, 1),
    (3, 3, 1),
    (5, 4, 1),
    (7, 5, 2),
    (6, 6, 3),
    (4, 9, 2),
    (10, 3, 1),
    (8, 8, 4),
    (2, 2, 5),     # 半径远超帧尺寸
    (1, 7, 2),     # 单列
    (7, 1, 2),     # 单行
    (13, 11, 3),
])
def test_median_plane_matches_brute_force(w, h, r):
    rng = random.Random(w * 977 + h * 13 + r)
    data = bytes(rng.randrange(256) for _ in range(w * h))
    assert median_plane(data, w, h, r) == brute_median(data, w, h, r)


def test_median_radius_zero_identity():
    rng = random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(30))
    assert median_plane(data, 6, 5, 0) == data


def test_median_uniform_unchanged():
    data = bytes([128] * 10 * 8)
    assert median_plane(data, 10, 8, 3) == data


def test_median_1x1():
    assert median_plane(bytes([200]), 1, 1, 2) == bytes([200])


def test_median_known_3x3_window():
    # 5x5 中心 (2,2) 的 3x3 窗口完整落在帧内，无镜像重复：
    # 窗口值 1..8 + 100，排序后低中值为 5
    data = bytearray([0] * 25)
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 100]
    k = 0
    for y in range(1, 4):
        for x in range(1, 4):
            data[y * 5 + x] = vals[k]
            k += 1
    out = median_plane(bytes(data), 5, 5, 1)
    assert out[2 * 5 + 2] == 5


# ---------- 椒盐噪声特性 ----------

def test_salt_and_pepper_impulses_removed_on_constant_background():
    w = h = 9
    data = bytearray([100] * w * h)
    # 冲激距边缘 >= 2，3x3 窗口内其余 8 像素全是背景 100
    data[4 * w + 4] = 255
    data[3 * w + 5] = 0
    data[5 * w + 3] = 255
    out = median_plane(bytes(data), w, h, 1)
    assert set(out) == {100}


def test_salt_and_pepper_extreme_values_do_not_bleed():
    # 单个白冲激：中值输出处处 <= 背景邻域，冲激位置不被保留
    w = h = 7
    data = bytearray([40] * w * h)
    data[3 * w + 3] = 255
    out = median_plane(bytes(data), w, h, 1)
    assert out[3 * w + 3] == 40
    assert max(out) == 40


# ---------- 分带规划 ----------

def test_plan_median_single_band_when_budget_ample():
    assert plan_median_bands(10, 4, 2, 10 ** 9) == [(0, 10)]


def test_plan_median_deducts_2r_context_rows():
    # per_row = 2*1*2 = 4；budget 4*16 -> afford 16；r=3 -> core = 16-6 = 10
    bands = plan_median_bands(20, 2, 3, 4 * 16)
    assert bands == [(0, 10), (10, 20)]


def test_plan_median_tiny_budget_core_at_least_one():
    bands = plan_median_bands(5, 1000, 7, 1)
    assert bands == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_plan_median_radius_zero_single_band():
    assert plan_median_bands(9, 4, 0, 1) == [(0, 9)]


def test_plan_median_empty_height():
    assert plan_median_bands(0, 4, 2, 100) == []


def test_plan_median_bands_cover_exactly_once():
    bands = plan_median_bands(37, 8, 2, 100)
    assert bands[0][0] == 0
    assert bands[-1][1] == 37
    for (s, e), (ns, _) in zip(bands, bands[1:]):
        assert e == ns and e > s


# ---------- 分带 == 整帧（零容差） ----------

@pytest.mark.parametrize("w,h,r,budget", [
    (16, 12, 1, 10 ** 9),    # 单带
    (16, 12, 1, 160),        # 多带
    (13, 9, 2, 120),         # 奇数尺寸多带
    (8, 8, 3, 10 ** 9),      # 半径接近帧尺寸
    (8, 8, 5, 64),           # 半径超过半帧 + 多带
    (5, 7, 2, 1),            # 极小预算，核心行高=1
    (1, 6, 2, 10 ** 9),      # 单列
    (6, 1, 2, 10 ** 9),      # 单行
    (24, 20, 4, 300),        # 较大半径多带
])
def test_banded_median_equals_whole_frame_zero_tolerance(w, h, r, budget):
    rng = random.Random(w * 1000 + h * 10 + r)
    data = bytes(rng.randrange(256) for _ in range(w * h))
    whole = median_plane(data, w, h, r)
    bands = plan_median_bands(h, w, r, budget)
    banded = median_plane_banded(data, w, h, r, bands)
    assert banded == whole
    assert banded == brute_median(data, w, h, r)


def test_banded_median_radius_zero_identity():
    rng = random.Random(9)
    data = bytes(rng.randrange(256) for _ in range(30))
    assert median_plane_banded(data, 6, 5, 0, [(0, 5)]) == data


# ---------- 性能：窗口放大耗时不成倍恶化 ----------

def test_runtime_scales_sublinearly_with_window_area():
    # 直方图加速：窗口面积 x9（r=1 -> r=4），耗时比应远小于面积比。
    # 取多次最佳值规避调度抖动，阈值 4 倍（实测约 1.1 倍；暴力排序约 9 倍）。
    rng = random.Random(7)
    w, h = 192, 128
    data = bytes(rng.randrange(256) for _ in range(w * h))

    def best(radius, reps=3):
        best_t = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            median_plane(data, w, h, radius)
            best_t = min(best_t, time.perf_counter() - t0)
        return best_t

    ratio = best(4) / best(1)
    assert ratio < 4.0, f"r=4/r=1 耗时比 {ratio:.2f}，疑似退化为 O(r^2)"


# ---------- 边界复核辅助 ----------

def test_border_mask_geometry():
    mask = border_mask(5, 6, 1)
    grid = [list(mask[y * 5:(y + 1) * 5]) for y in range(6)]
    assert grid[0] == [1] * 5 and grid[5] == [1] * 5      # 上下边带
    assert grid[2] == [1, 1, 0, 1, 1]                     # 中部仅左右各 r+1 列
    assert grid[3] == [1, 1, 0, 1, 1]


def test_border_mask_covers_whole_plane_when_radius_large():
    assert border_mask(3, 3, 2) == bytes([1] * 9)


def test_border_mask_zero_radius_is_one_pixel_rim():
    mask = border_mask(4, 4, 0)
    grid = [list(mask[y * 4:(y + 1) * 4]) for y in range(4)]
    assert grid[0] == [1, 1, 1, 1]
    assert grid[3] == [1, 1, 1, 1]
    assert grid[1] == [1, 0, 0, 1]
    assert grid[2] == [1, 0, 0, 1]


def test_dump_border_frame_zeroes_interior_only(tmp_path):
    planes = [bytes([200] * 30), bytes([90] * 30)]
    path = tmp_path / "f.raw"
    dump_border_frame(path, planes, [(5, 6), (5, 6)], 1)
    raw = path.read_bytes()
    assert len(raw) == 60
    assert list(raw[0:5]) == [200] * 5                 # 顶行边界保留
    assert list(raw[10:15]) == [200, 200, 0, 200, 200]  # 中部内点置 0
    assert list(raw[40:45]) == [90, 90, 0, 90, 90]      # 第二平面同样处理
