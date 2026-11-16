"""分带规划与接缝加权合并测试（纯逻辑，不依赖 ffmpeg）。"""

import random

import pytest

from vdenoise import (
    box_blur_plane,
    box_blur_plane_banded,
    merge_rows,
    plan_bands,
)


# ---------- plan_bands ----------

def test_plan_single_band_when_budget_ample():
    assert plan_bands(10, 4, 2, 10 ** 9) == [(0, 10)]


def test_plan_splits_by_budget():
    # per_row = 4*1*2 = 8；budget 56 -> afford 7 行；core = 7 - 4*1 = 3
    bands = plan_bands(10, 4, 1, 56)
    assert bands == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_plan_deducts_4r_context_rows():
    # 4r 行上下文扣预算：per_row = 2*1*2 = 4；budget 4*20 -> afford 20
    # r=3 -> core = 20 - 12 = 8
    bands = plan_bands(20, 2, 3, 4 * 20)
    assert bands == [(0, 8), (8, 16), (16, 20)]


def test_plan_tiny_budget_core_at_least_one():
    bands = plan_bands(5, 1000, 7, 1)
    assert bands == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_plan_radius_zero_single_band():
    assert plan_bands(9, 4, 0, 1) == [(0, 9)]


def test_plan_empty_height():
    assert plan_bands(0, 4, 2, 100) == []


def test_plan_bands_cover_exactly_once():
    bands = plan_bands(37, 8, 2, 100)
    assert bands[0][0] == 0
    assert bands[-1][1] == 37
    for (s, e), (ns, _) in zip(bands, bands[1:]):
        assert e == ns
        assert e > s


# ---------- merge_rows ----------

def test_merge_rows_identity_when_equal():
    row = bytes([10, 200, 30, 255, 0])
    for t in range(5):
        assert merge_rows(row, row, 5, t) == row


def test_merge_rows_endpoints():
    old = bytes([0, 0, 0])
    new = bytes([90, 90, 90])
    assert merge_rows(old, new, 4, 0) == old   # t=0 全取 old
    assert merge_rows(old, new, 4, 3) == new   # t=L-1 全取 new


def test_merge_rows_linear_ramp():
    old = bytes([0])
    new = bytes([90])
    # L=4, den=3：t=1 -> 30，t=2 -> 60
    assert merge_rows(old, new, 4, 1) == bytes([30])
    assert merge_rows(old, new, 4, 2) == bytes([60])


def test_merge_rows_length_one_takes_new():
    assert merge_rows(bytes([1]), bytes([2]), 1, 0) == bytes([2])


# ---------- 分带 == 整帧（零容差） ----------

@pytest.mark.parametrize("w,h,r,budget", [
    (16, 12, 1, 10 ** 9),     # 单带
    (16, 12, 1, 200),         # 多带
    (13, 9, 2, 120),          # 奇数尺寸多带
    (8, 8, 3, 10 ** 9),       # 半径接近帧尺寸
    (8, 8, 5, 64),            # 半径超过半帧 + 多带
    (5, 7, 2, 1),             # 极小预算，核心行高=1
    (1, 6, 2, 10 ** 9),       # 单列
    (6, 1, 2, 10 ** 9),       # 单行
    (24, 20, 4, 300),         # 较大半径多带
])
def test_banded_equals_whole_frame_zero_tolerance(w, h, r, budget):
    rng = random.Random(w * 1000 + h * 10 + r)
    data = bytes(rng.randrange(256) for _ in range(w * h))
    whole = box_blur_plane(data, w, h, r)
    bands = plan_bands(h, w, r, budget)
    banded = box_blur_plane_banded(data, w, h, r, bands)
    assert banded == whole  # 逐像素相等，不容差


def test_banded_radius_zero_identity():
    rng = random.Random(9)
    data = bytes(rng.randrange(256) for _ in range(30))
    assert box_blur_plane_banded(data, 6, 5, 0, [(0, 5)]) == data


def test_banded_output_rows_all_filled():
    rng = random.Random(13)
    w, h, r = 10, 11, 2
    data = bytes(rng.randrange(256) for _ in range(w * h))
    bands = plan_bands(h, w, r, 40)
    out = box_blur_plane_banded(data, w, h, r, bands)
    assert isinstance(out, bytes)
    assert len(out) == w * h
