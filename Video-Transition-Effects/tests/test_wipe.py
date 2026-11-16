"""wipe.py 纯逻辑测试：掩码几何、软边渐变、钳制告警、合成逐像素可锁定。

擦除不经过 ffmpeg，全部在 RGB24 字节缓冲上完成，因此可以做逐字节
断言（区别于集成测试里 H.264 往返的容差核对）。
"""

from __future__ import annotations

import warnings

import pytest

from video_transition.errors import PlanError
from video_transition.slide import (
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    DIRECTIONS,
    HORIZONTAL_DIRECTIONS,
    slide_regions,
)
from video_transition.wipe import (
    WipeSoftWidthWarning,
    clamp_soft_width,
    composite_wipe_frame,
    wipe_edge,
    wipe_segments,
)

W, H = 4, 3
FRAME_SIZE = W * H * 3


def _frame(fill: int) -> bytearray:
    return bytearray([fill & 0xFF] * FRAME_SIZE)


def _pixel(frame: bytearray, x: int, y: int) -> tuple[int, int, int]:
    i = (y * W + x) * 3
    return frame[i], frame[i + 1], frame[i + 2]


class TestClampSoftWidth:
    def test_within_axis_returned_unchanged(self):
        assert clamp_soft_width(0, 10) == 0
        assert clamp_soft_width(5, 10) == 5
        assert clamp_soft_width(10, 10) == 10

    def test_overflow_clamped_and_warns(self):
        with pytest.warns(WipeSoftWidthWarning) as record:
            got = clamp_soft_width(99, 10)
        assert got == 10
        # 告警文本必须同时包含原始值与钳制值，调用方才能定位配置问题
        msg = str(record[0].message)
        assert "99" in msg and "10" in msg

    def test_overflow_does_not_warn_twice_on_repeated_valid_call(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert clamp_soft_width(10, 10) == 10

    @pytest.mark.parametrize("bad", [-1, -100])
    def test_negative_rejected(self, bad):
        with pytest.raises(PlanError):
            clamp_soft_width(bad, 10)

    @pytest.mark.parametrize("bad", [1.5, True, False, "8", None])
    def test_non_int_rejected(self, bad):
        with pytest.raises(PlanError):
            clamp_soft_width(bad, 10)

    def test_non_positive_axis_rejected(self):
        with pytest.raises(PlanError):
            clamp_soft_width(1, 0)


class TestWipeEdge:
    def test_linear_anchors_on_extended_travel(self):
        L, soft, no = 10, 2, 5
        edges = [wipe_edge(k, no, L, soft, "linear") for k in range(no)]
        # round(k*(L+soft)/(No-1)) - soft：[-2, 1, 4, 7, 10]
        assert edges == [-2, 1, 4, 7, 10]

    def test_anchors_are_exact_for_any_soft_width(self):
        L, no = 12, 7
        for soft in (0, 1, 5, 12):
            edges = [wipe_edge(k, no, L, soft, "ease_in_out") for k in range(no)]
            assert edges[0] == -soft
            assert edges[-1] == L
            assert edges == sorted(edges)  # 单调不减、掩码不回退

    def test_hard_edge_travel_is_plain_axis(self):
        # soft=0 时扩展行程退化为 [0, L]，与滑动位移同一条曲线
        # （半数远离零取整：round(k*10/4) = 0,3,5,8,10）
        L, no = 10, 5
        edges = [wipe_edge(k, no, L, 0, "linear") for k in range(no)]
        assert edges == [0, 3, 5, 8, 10]

    def test_single_frame_overlap_lands_midpoint(self):
        # No=1 进度取 1/2：边界落在扩展行程中点
        edge = wipe_edge(0, 1, 10, 2, "linear")
        assert edge == round((10 + 2) / 2) - 2

    def test_bad_soft_width_rejected(self):
        with pytest.raises(PlanError):
            wipe_edge(0, 5, 10, 11, "linear")


class TestWipeSegments:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_segments_partition_axis_every_frame(self, direction):
        axis = W if direction in HORIZONTAL_DIRECTIONS else H
        soft = 2
        no = 6
        for k in range(no):
            edge = wipe_edge(k, no, axis, soft, "linear")
            segs = wipe_segments(direction, edge, soft, W, H)
            # 长度之和恒为轴长、坐标无缝相接、按序排列
            assert sum(length for _, length, _, _ in segs) == axis
            cursor = 0
            for start, length, num, den in segs:
                assert start == cursor
                assert length > 0
                assert 0 <= num <= den and den > 0
                cursor = start + length
            assert cursor == axis

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_first_frame_pure_a_last_frame_pure_b(self, direction):
        axis = W if direction in HORIZONTAL_DIRECTIONS else H
        soft = 2
        no = 8
        first = wipe_segments(
            direction, wipe_edge(0, no, axis, soft, "linear"), soft, W, H
        )
        last = wipe_segments(
            direction, wipe_edge(no - 1, no, axis, soft, "linear"), soft, W, H
        )
        assert first == ((0, axis, 0, 1),)
        assert last == ((0, axis, 1, 1),)

    def test_forward_gradient_weights_increase_linearly(self):
        # right：edge=2、soft=3，带内列 u=2,3,4 权重 1/3、2/3、3/3
        segs = wipe_segments(DIRECTION_RIGHT, edge=2, soft_width=3, width=10, height=2)
        band = [(s, n, d) for s, _, n, d in segs if d == 3]
        assert band == [(2, 1, 3), (3, 2, 3), (4, 3, 3)]
        # 纯 B 在左 [0,2)、纯 A 在右 [5,10)
        assert segs[0] == (0, 2, 1, 1)
        assert segs[-1] == (5, 5, 0, 1)

    def test_reverse_gradient_weights_increase_toward_high_coordinate(self):
        # left：edge=2、soft=3，B 在物理高坐标侧。归一坐标
        # u'=axis-1-u：u'=0..2 对应物理 u=9,8,7... 实际带内为物理
        # u=5,6,7（u'=4,3,2），权重随 u' 递增、随物理 u 递减：
        # u=5->3/3（最 B）、u=6->2/3、u=7->1/3（最 A）
        segs = wipe_segments(DIRECTION_LEFT, edge=2, soft_width=3, width=10, height=2)
        by_start = {s: (n, d) for s, _, n, d in segs}
        assert by_start[5] == (3, 3)
        assert by_start[6] == (2, 3)
        assert by_start[7] == (1, 3)
        # 纯 A 在最左 [0,5)、纯 B 在最右 [8,10)
        assert segs[0] == (0, 5, 0, 1)
        assert segs[-1] == (8, 2, 1, 1)
        # 权重沿物理坐标单调不增（越靠右越 B 的反向语义）
        band_weights = [n / d for _, _, n, d in segs if d == 3]
        assert band_weights == sorted(band_weights, reverse=True)

    def test_band_partially_off_canvas_still_full_coverage(self):
        # edge 为负：渐变带左半在画布外，画布最左列权重应小于 1 但非 0
        segs = wipe_segments(DIRECTION_RIGHT, edge=-1, soft_width=3, width=6, height=1)
        assert sum(length for _, length, _, _ in segs) == 6
        first_num, first_den = segs[0][2], segs[0][3]
        assert 0 < first_num < first_den

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_hard_wipe_matches_slide_regions(self, direction):
        axis = W if direction in HORIZONTAL_DIRECTIONS else H
        for shift in range(axis + 1):
            slide = tuple(t for t in slide_regions(direction, shift, W, H) if t[1])
            wipe = wipe_segments(direction, shift, 0, W, H)
            got = tuple(
                (start, length, "B" if num == 1 else "A")
                for start, length, num, _den in wipe
            )
            assert got == slide

    def test_unknown_direction_rejected(self):
        with pytest.raises(PlanError):
            wipe_segments("north", 0, 0, W, H)

    def test_edge_out_of_extended_travel_rejected(self):
        with pytest.raises(PlanError):
            wipe_segments(DIRECTION_RIGHT, 5, 2, W, H)  # 上界是 W=4
        with pytest.raises(PlanError):
            wipe_segments(DIRECTION_RIGHT, -3, 2, W, H)  # 下界是 -soft=-2


class TestCompositeWipeFrame:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_start_anchor_byte_identical_to_a(self, direction):
        axis = W if direction in HORIZONTAL_DIRECTIONS else H
        soft = 2
        edge = wipe_edge(0, 8, axis, soft, "linear")
        a, b = _frame(0x11), _frame(0x22)
        dst = bytearray(FRAME_SIZE)
        composite_wipe_frame(dst, a, b, W, H, direction, edge, soft)
        assert bytes(dst) == bytes(a)

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_end_anchor_byte_identical_to_b(self, direction):
        axis = W if direction in HORIZONTAL_DIRECTIONS else H
        soft = 2
        edge = wipe_edge(7, 8, axis, soft, "linear")
        a, b = _frame(0x11), _frame(0x22)
        dst = bytearray(FRAME_SIZE)
        composite_wipe_frame(dst, a, b, W, H, direction, edge, soft)
        assert bytes(dst) == bytes(b)

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_no_uncovered_pixel_at_any_frame(self, direction):
        axis = W if direction in HORIZONTAL_DIRECTIONS else H
        soft = axis  # 最大合法软边：渐变可铺满全轴
        a, b = _frame(0x10), _frame(0x20)
        no = 10
        for k in range(no):
            edge = wipe_edge(k, no, axis, soft, "ease_in_out")
            dst = bytearray(FRAME_SIZE)
            composite_wipe_frame(dst, a, b, W, H, direction, edge, soft)
            # 两源都非零：任何未覆盖黑边都会出现 0；每个像素只能来自
            # A、B 或两者的凸组合，故取值必落在 [0x10, 0x20]
            assert min(dst) >= 0x10
            assert max(dst) <= 0x20

    def test_horizontal_midband_is_integer_blend(self):
        # width=4、right、edge=1、soft=2：列 0 纯 B、列 1 权重 1/2、
        # 列 2 权重 2/2=纯 B、列 3 纯 A
        a, b = _frame(0x10), _frame(0x20)
        dst = bytearray(FRAME_SIZE)
        composite_wipe_frame(dst, a, b, W, H, DIRECTION_RIGHT, 1, 2)
        half = (0x10 + 0x20 + 1) // 2  # 混合器半数远离零
        assert _pixel(dst, 0, 0) == (0x20,) * 3
        assert _pixel(dst, 1, 0) == (half,) * 3
        assert _pixel(dst, 2, 0) == (0x20,) * 3
        assert _pixel(dst, 3, 0) == (0x10,) * 3
        # 同一列权重对全部行生效
        assert _pixel(dst, 1, H - 1) == (half,) * 3

    def test_vertical_midband_blends_whole_rows(self):
        # height=3、down、edge=1、soft=1（硬渐变，每段一行）：
        # 行 0 纯 B、行 1 权重 1/1=纯 B、行 2 纯 A
        a, b = _frame(0x10), _frame(0x20)
        dst = bytearray(FRAME_SIZE)
        composite_wipe_frame(dst, a, b, W, H, DIRECTION_DOWN, 1, 1)
        assert _pixel(dst, 0, 0) == (0x20,) * 3
        assert _pixel(dst, 0, H - 1) == (0x10,) * 3

    def test_hard_wipe_equals_hard_slide_byte_for_byte(self):
        from video_transition.slide import composite_slide_frame

        a, b = _frame(0x30), _frame(0x60)
        for direction in DIRECTIONS:
            axis = W if direction in HORIZONTAL_DIRECTIONS else H
            for shift in range(axis + 1):
                d_slide = bytearray(FRAME_SIZE)
                d_wipe = bytearray(FRAME_SIZE)
                composite_slide_frame(d_slide, a, b, W, H, direction, shift)
                composite_wipe_frame(d_wipe, a, b, W, H, direction, shift, 0)
                assert bytes(d_slide) == bytes(d_wipe), (direction, shift)

    def test_deterministic_twice_same_output(self):
        a, b = _frame(0x10), _frame(0x20)
        d1, d2 = bytearray(FRAME_SIZE), bytearray(FRAME_SIZE)
        composite_wipe_frame(d1, a, b, W, H, DIRECTION_LEFT, 0, 2)
        composite_wipe_frame(d2, a, b, W, H, DIRECTION_LEFT, 0, 2)
        assert bytes(d1) == bytes(d2)

    def test_size_mismatch_rejected(self):
        with pytest.raises(PlanError):
            composite_wipe_frame(
                bytearray(3), _frame(0), _frame(0), W, H, DIRECTION_RIGHT, 0, 0
            )
