"""easing.py 纯逻辑测试：锚点 0/1、单调性、整数位移与边界夹取。"""

from __future__ import annotations

import pytest

from video_transition.easing import (
    EASING_EASE_IN,
    EASING_EASE_IN_OUT,
    EASING_EASE_OUT,
    EASING_LINEAR,
    EASINGS,
    eased_fraction,
    eased_shift,
)
from video_transition.errors import PlanError


class TestEasedFraction:
    @pytest.mark.parametrize("easing", EASINGS)
    def test_anchors_are_exact_zero_and_one(self, easing):
        no = 11
        first_num, first_den = eased_fraction(0, no, easing)
        last_num, last_den = eased_fraction(no - 1, no, easing)
        assert first_num == 0  # 首帧进度精确为 0
        assert last_num == last_den  # 末帧进度精确为 1（允许未约分形式）

    @pytest.mark.parametrize("easing", EASINGS)
    def test_monotonic_non_decreasing(self, easing):
        no = 21
        values = [num / den for num, den in
                  (eased_fraction(k, no, easing) for k in range(no))]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_linear_is_k_over_steps(self):
        assert eased_fraction(0, 5) == (0, 4)
        assert eased_fraction(2, 5) == (2, 4)
        assert eased_fraction(4, 5) == (4, 4)

    def test_ease_in_squares_progress(self):
        # t=0.5（k=2, steps=4）：t²=0.25 -> 4/16
        assert eased_fraction(2, 5, EASING_EASE_IN) == (4, 16)

    def test_ease_out_formula(self):
        # k=2, steps=4：2*2*4-4 = 12 / 16 = 0.75
        assert eased_fraction(2, 5, EASING_EASE_OUT) == (12, 16)

    def test_ease_in_out_midpoint_is_half(self):
        # steps=4，中点 k=2 落在后半分支起点：1 - 2*(2/4)² = 0.5
        num, den = eased_fraction(2, 5, EASING_EASE_IN_OUT)
        assert num * 2 == den

    def test_ease_in_out_piecewise_values(self):
        # steps=4，统一分母 32：
        # k=1 前半段 4·1²=4/32；k=3 后半段 32-2·1=30/32
        assert eased_fraction(1, 5, EASING_EASE_IN_OUT) == (4, 32)
        assert eased_fraction(3, 5, EASING_EASE_IN_OUT) == (30, 32)

    def test_single_frame_overlap_is_half(self):
        for easing in EASINGS:
            assert eased_fraction(0, 1, easing) == (1, 2)

    def test_unknown_easing_rejected(self):
        with pytest.raises(PlanError):
            eased_fraction(0, 5, "bounce")

    @pytest.mark.parametrize("bad_k,no", [(5, 5), (-1, 5), (0, 0), (1, -3)])
    def test_out_of_range_rejected(self, bad_k, no):
        with pytest.raises(PlanError):
            eased_fraction(bad_k, no)


class TestEasedShift:
    @pytest.mark.parametrize("easing", EASINGS)
    def test_shift_anchors_cover_full_axis(self, easing):
        width, no = 320, 9
        assert eased_shift(0, no, width, easing) == 0
        assert eased_shift(no - 1, no, width, easing) == width

    @pytest.mark.parametrize("easing", EASINGS)
    def test_shift_monotonic_and_in_bounds(self, easing):
        height, no = 240, 17
        shifts = [eased_shift(k, no, height, easing) for k in range(no)]
        assert shifts == sorted(shifts)
        assert all(0 <= s <= height for s in shifts)

    def test_linear_shift_matches_round_half_away(self):
        # width=100, no=5：位移 = round(k*100/4) -> 0,25,50,75,100
        assert [eased_shift(k, 5, 100, EASING_LINEAR) for k in range(5)] == [
            0, 25, 50, 75, 100,
        ]

    def test_half_pixel_rounds_away_from_zero(self):
        # width=3, no=3：k=1 -> round(1.5)=2（半数远离零）
        assert eased_shift(1, 3, 3, EASING_LINEAR) == 2

    def test_single_frame_shift_is_half_axis(self):
        # 单帧进度 1/2，半数远离零取整：101/2 = 50.5 -> 51
        assert eased_shift(0, 1, 101, EASING_LINEAR) == 51

    def test_non_positive_axis_rejected(self):
        with pytest.raises(PlanError):
            eased_shift(0, 5, 0)
