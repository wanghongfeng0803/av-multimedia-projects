"""slide.py 纯逻辑测试：区域全覆盖、四方向取源正确、合成逐像素可锁定。

滑动不经过 ffmpeg，全部在 RGB24 字节缓冲上完成，因此可以做
逐字节断言（区别于集成测试里 H.264 往返的容差核对）。
"""

from __future__ import annotations

import pytest

from video_transition.errors import PlanError
from video_transition.slide import (
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    DIRECTIONS,
    composite_slide_frame,
    slide_regions,
)

W, H = 4, 3
FRAME_SIZE = W * H * 3


def _frame(fill: int) -> bytearray:
    return bytearray([fill & 0xFF] * FRAME_SIZE)


def _pixel(frame: bytearray, x: int, y: int) -> tuple[int, int, int]:
    i = (y * W + x) * 3
    return frame[i], frame[i + 1], frame[i + 2]


class TestSlideRegions:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_regions_partition_axis_without_gap_or_overlap(self, direction):
        axis = W if direction in ("right", "left") else H
        for shift in range(axis + 1):
            r0, r1 = slide_regions(direction, shift, W, H)
            (start0, len0, src0), (start1, len1, src1) = r0, r1
            # 首尾相接、长度之和恰为轴长 -> 无黑边、无重复覆盖
            assert start0 == 0
            assert start0 + len0 == start1
            assert start1 + len1 == axis
            assert {src0, src1} == {"A", "B"}

    def test_horizontal_regions_are_column_ranges(self):
        r0, r1 = slide_regions(DIRECTION_RIGHT, shift=3, width=W, height=H)
        # right：B 占左 [0,3)，A 占右 [3,4)
        assert r0 == (0, 3, "B")
        assert r1 == (3, W - 3, "A")
        r0, r1 = slide_regions(DIRECTION_LEFT, shift=3, width=W, height=H)
        assert r0 == (0, W - 3, "A")
        assert r1 == (W - 3, 3, "B")

    def test_vertical_regions_are_row_ranges(self):
        r0, r1 = slide_regions(DIRECTION_DOWN, shift=2, width=W, height=H)
        # down：B 占上 [0,2)，A 占下 [2,3)
        assert r0 == (0, 2, "B")
        assert r1 == (2, H - 2, "A")
        r0, r1 = slide_regions(DIRECTION_UP, shift=2, width=W, height=H)
        assert r0 == (0, H - 2, "A")
        assert r1 == (H - 2, 2, "B")
    def test_unknown_direction_rejected(self):
        with pytest.raises(PlanError):
            slide_regions("north", 1, W, H)

    def test_shift_out_of_axis_rejected(self):
        with pytest.raises(PlanError):
            slide_regions(DIRECTION_RIGHT, W + 1, W, H)


class TestCompositeSlideFrame:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_zero_shift_is_byte_identical_to_a(self, direction):
        a, b = _frame(0x11), _frame(0x22)
        dst = bytearray(FRAME_SIZE)
        composite_slide_frame(dst, a, b, W, H, direction, 0)
        assert bytes(dst) == bytes(a)

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_full_shift_is_byte_identical_to_b(self, direction):
        axis = W if direction in ("right", "left") else H
        a, b = _frame(0x11), _frame(0x22)
        dst = bytearray(FRAME_SIZE)
        composite_slide_frame(dst, a, b, W, H, direction, axis)
        assert bytes(dst) == bytes(b)

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_every_pixel_covered_no_black_at_any_shift(self, direction):
        axis = W if direction in ("right", "left") else H
        a, b = _frame(0x10), _frame(0x20)
        for shift in range(axis + 1):
            dst = bytearray(FRAME_SIZE)
            composite_slide_frame(dst, a, b, W, H, direction, shift)
            # 不存在零值像素（两源都非零），即没有未覆盖的黑边
            assert min(dst) > 0
            # 每个像素只可能来自 A 或 B
            assert set(dst) <= {0x10, 0x20}

    def test_right_horizontal_source_layout(self):
        a, b = _frame(0x10), _frame(0x20)
        dst = bytearray(FRAME_SIZE)
        composite_slide_frame(dst, a, b, W, H, DIRECTION_RIGHT, shift=3)
        # 左 3 列来自 B，右 1 列来自 A
        assert _pixel(dst, 0, 0) == (0x20,) * 3
        assert _pixel(dst, 2, 0) == (0x20,) * 3
        assert _pixel(dst, 3, 0) == (0x10,) * 3
        assert _pixel(dst, 3, H - 1) == (0x10,) * 3

    def test_left_horizontal_source_layout(self):
        a, b = _frame(0x10), _frame(0x20)
        dst = bytearray(FRAME_SIZE)
        composite_slide_frame(dst, a, b, W, H, DIRECTION_LEFT, shift=3)
        assert _pixel(dst, 0, 0) == (0x10,) * 3
        assert _pixel(dst, 1, 0) == (0x20,) * 3
        assert _pixel(dst, 3, H - 1) == (0x20,) * 3

    def test_down_vertical_source_layout(self):
        a, b = _frame(0x10), _frame(0x20)
        dst = bytearray(FRAME_SIZE)
        composite_slide_frame(dst, a, b, W, H, DIRECTION_DOWN, shift=2)
        # 上 2 行来自 B，底 1 行来自 A
        assert _pixel(dst, 0, 0) == (0x20,) * 3
        assert _pixel(dst, 0, 1) == (0x20,) * 3
        assert _pixel(dst, W - 1, H - 1) == (0x10,) * 3

    def test_up_vertical_source_layout(self):
        a, b = _frame(0x10), _frame(0x20)
        dst = bytearray(FRAME_SIZE)
        composite_slide_frame(dst, a, b, W, H, DIRECTION_UP, shift=2)
        assert _pixel(dst, 0, 0) == (0x10,) * 3
        assert _pixel(dst, 0, 1) == (0x20,) * 3
        assert _pixel(dst, W - 1, H - 1) == (0x20,) * 3

    def test_deterministic_twice_same_output(self):
        a, b = _frame(0x10), _frame(0x20)
        d1, d2 = bytearray(FRAME_SIZE), bytearray(FRAME_SIZE)
        composite_slide_frame(d1, a, b, W, H, DIRECTION_RIGHT, 2)
        composite_slide_frame(d2, a, b, W, H, DIRECTION_RIGHT, 2)
        assert bytes(d1) == bytes(d2)

    def test_size_mismatch_rejected(self):
        with pytest.raises(PlanError):
            composite_slide_frame(
                bytearray(3), _frame(0), _frame(0), W, H, DIRECTION_RIGHT, 1
            )
