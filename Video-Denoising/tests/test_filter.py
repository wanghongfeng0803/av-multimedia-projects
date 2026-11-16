"""可分离盒式滤波单元测试（纯逻辑，不依赖 ffmpeg）。"""

import random

from vdenoise import horizontal_pass_row, vertical_pass, box_blur_plane


def direct_mean(data, width, height, radius, x, y):
    """暴力直算 (2r+1)^2 截断窗口均值（浮点），用于对照。"""
    lo_y, hi_y = max(0, y - radius), min(height - 1, y + radius)
    lo_x, hi_x = max(0, x - radius), min(width - 1, x + radius)
    vals = [data[j * width + i]
            for j in range(lo_y, hi_y + 1)
            for i in range(lo_x, hi_x + 1)]
    return sum(vals) / len(vals)


# ---------- 水平遍 ----------

def test_horizontal_uniform_row():
    assert horizontal_pass_row(bytes([7] * 9), 2) == [7] * 9


def test_horizontal_known_values():
    row = bytes(range(9))
    out = horizontal_pass_row(row, 1)
    # x=0: 窗口 [0,1] -> (0+1+1)//2 = 1；x=1: [0,1,2] -> 1
    # x=8: 窗口 [7,8] -> (15+1)//2 = 8
    assert out[0] == 1
    assert out[1] == 1
    assert out[4] == 4
    assert out[8] == 8


def test_horizontal_boundary_no_overflow():
    # 半径远大于行宽：不越界，窗口截断为整行
    row = bytes([10, 20, 30])
    out = horizontal_pass_row(row, 100)
    assert out == [20, 20, 20]


def test_horizontal_radius_zero_identity():
    row = bytes([3, 1, 4, 1, 5, 9, 2, 6])
    assert horizontal_pass_row(row, 0) == list(row)


def test_horizontal_single_pixel():
    assert horizontal_pass_row(bytes([42]), 3) == [42]


# ---------- 垂直遍 ----------

def test_vertical_uniform_columns():
    rows = [[5] * 4 for _ in range(6)]
    out = vertical_pass(rows, 2)
    assert out == [bytes([5] * 4)] * 6


def test_vertical_known_values():
    rows = [[0] * 3, [10] * 3, [20] * 3]
    out = vertical_pass(rows, 1)
    # y=0: mean(0,10)=5；y=1: mean(0,10,20)=10；y=2: mean(10,20)=15
    assert [list(r) for r in out] == [[5] * 3, [10] * 3, [15] * 3]


def test_vertical_sliding_matches_recompute():
    # 滑动运行和必须与每行独立重算一致
    rng = random.Random(7)
    rows = [[rng.randrange(256) for _ in range(11)] for _ in range(13)]
    out = vertical_pass(rows, 3)
    for y in range(13):
        lo, hi = max(0, y - 3), min(12, y + 3)
        cnt = hi - lo + 1
        expect = bytes(
            (sum(rows[j][x] for j in range(lo, hi + 1)) + cnt // 2) // cnt
            for x in range(11)
        )
        assert out[y] == expect


def test_vertical_output_slice():
    rng = random.Random(3)
    rows = [[rng.randrange(256) for _ in range(7)] for _ in range(9)]
    full = vertical_pass(rows, 2)
    part = vertical_pass(rows, 2, 2, 5)
    assert part == full[2:5]


def test_vertical_radius_zero_identity():
    rng = random.Random(11)
    rows = [[rng.randrange(256) for _ in range(5)] for _ in range(4)]
    out = vertical_pass(rows, 0)
    assert [list(r) for r in out] == rows


# ---------- 整帧 ----------

def test_box_blur_radius_zero_identity():
    rng = random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(6 * 5))
    assert box_blur_plane(data, 6, 5, 0) == data


def test_box_blur_uniform_unchanged():
    data = bytes([128] * 10 * 8)
    assert box_blur_plane(data, 10, 8, 3) == data


def test_box_blur_output_type_and_length():
    data = bytes(range(12))
    out = box_blur_plane(data, 4, 3, 1)
    assert isinstance(out, bytes)
    assert len(out) == len(data)


def test_box_blur_vs_direct_mean_within_one():
    # 两遍各自舍入：与全局均值直算最多差 ±1
    rng = random.Random(42)
    w, h, r = 9, 7, 2
    data = bytes(rng.randrange(256) for _ in range(w * h))
    out = box_blur_plane(data, w, h, r)
    for y in range(h):
        for x in range(w):
            assert abs(out[y * w + x] - direct_mean(data, w, h, r, x, y)) <= 1


def test_box_blur_radius_larger_than_frame():
    rng = random.Random(5)
    w, h = 4, 3
    data = bytes(rng.randrange(256) for _ in range(w * h))
    out = box_blur_plane(data, w, h, 10)
    # 半径覆盖整帧：每个像素都是全帧均值（两遍舍入后仍应一致）
    assert len(set(out)) == 1
    mean = sum(data) / len(data)
    assert abs(out[0] - mean) <= 1


def test_box_blur_1x1():
    assert box_blur_plane(bytes([200]), 1, 1, 2) == bytes([200])


def test_box_blur_single_row():
    out = box_blur_plane(bytes([0, 10, 20]), 3, 1, 1)
    assert list(out) == [5, 10, 15]


def test_box_blur_impulse_energy_spreads():
    # 中心冲激经均值滤波后向四周扩散，中心值降低
    w = h = 7
    data = bytearray(w * h)
    data[3 * w + 3] = 255
    out = box_blur_plane(bytes(data), w, h, 1)
    assert out[3 * w + 3] < 255
    assert out[3 * w + 2] > 0  # 邻居被波及
    assert out[0] == 0         # 远处不受影响
