"""blend.py 混合原语测试：锚点逐字节相等、整数四舍五入、s16 负值对称。"""

from __future__ import annotations

import array

import pytest

from video_transition.blend import (
    blend_rgb24_frame,
    blend_s16_samples,
    div_round_half_away,
)
from video_transition.errors import PlanError


class TestDivRound:
    @pytest.mark.parametrize(
        "num,den,expected",
        [
            (0, 4, 0), (7, 4, 2), (9, 4, 2), (10, 4, 3),
            (-7, 4, -2), (-9, 4, -2), (-10, 4, -3),
            (1, 2, 1), (-1, 2, -1),
        ],
    )
    def test_half_away_from_zero(self, num, den, expected):
        assert div_round_half_away(num, den) == expected

    def test_den_must_be_positive(self):
        with pytest.raises(PlanError):
            div_round_half_away(1, 0)


class TestRgbBlend:
    def test_endpoints_are_byte_identical_to_sources(self):
        a = bytearray([10, 200, 0, 255, 128, 64])
        b = bytearray([200, 10, 255, 0, 128, 192])
        # 权重 0 -> 完全等于 A；权重 1（num==den）-> 完全等于 B
        out0, out1 = bytearray(len(a)), bytearray(len(a))
        blend_rgb24_frame(out0, a, b, 0, 9)
        blend_rgb24_frame(out1, a, b, 9, 9)
        assert bytes(out0) == bytes(a)
        assert bytes(out1) == bytes(b)

    def test_midpoint_is_average_with_half_rounding(self):
        a = bytearray([0, 10, 11, 255])
        b = bytearray([10, 11, 20, 255])
        out = bytearray(4)
        blend_rgb24_frame(out, a, b, 1, 2)
        # (a+b+1)//2
        assert list(out) == [5, 11, 16, 255]

    def test_deterministic_integer_result(self):
        a = bytearray(range(256))
        b = bytearray(range(255, -1, -1))
        out1, out2 = bytearray(256), bytearray(256)
        blend_rgb24_frame(out1, a, b, 3, 7)
        blend_rgb24_frame(out2, a, b, 3, 7)
        assert bytes(out1) == bytes(out2)
        # 逐字节核对公式
        for i in range(256):
            assert out1[i] == (a[i] * 4 + b[i] * 3 + 3) // 7

    def test_size_mismatch_rejected(self):
        with pytest.raises(PlanError):
            blend_rgb24_frame(bytearray(3), b"\x00" * 3, b"\x00" * 2, 1, 2)

    def test_weight_out_of_range_rejected(self):
        a = b = bytearray(3)
        with pytest.raises(PlanError):
            blend_rgb24_frame(bytearray(3), a, b, 5, 2)


class TestS16Blend:
    def _pcm(self, values: list[int]) -> bytearray:
        arr = array.array("h", values)
        return bytearray(arr.tobytes())

    def test_endpoints_identical(self):
        a = self._pcm([-1000, 0, 1000, 32767])
        b = self._pcm([2000, -2000, 500, -32768])
        out0, out1 = bytearray(len(a)), bytearray(len(a))
        blend_s16_samples(out0, a, b, 0, 9)
        blend_s16_samples(out1, a, b, 9, 9)
        assert bytes(out0) == bytes(a)
        assert bytes(out1) == bytes(b)

    def test_negative_values_symmetric_rounding(self):
        a = self._pcm([-9])
        b = self._pcm([-11])
        out = bytearray(len(a))
        blend_s16_samples(out, a, b, 1, 2)
        got = array.array("h")
        got.frombytes(bytes(out))
        # (-9 + -11) / 2 = -10 精确
        assert got[0] == -10

    def test_midpoint_matches_audio_and_video_weight_semantics(self):
        # 与 RGB 同一套 num/den 权重：画面与声音淡化同源
        a = self._pcm([1000, -1000])
        b = self._pcm([2000, -2000])
        out = bytearray(len(a))
        blend_s16_samples(out, a, b, 1, 2)
        got = array.array("h")
        got.frombytes(bytes(out))
        assert list(got) == [1500, -1500]

    def test_length_mismatch_rejected(self):
        with pytest.raises(PlanError):
            blend_s16_samples(
                bytearray(4), self._pcm([1, 2]), self._pcm([1]), 1, 2
            )
