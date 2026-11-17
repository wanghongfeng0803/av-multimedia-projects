"""缩放测试：奇数对齐、面积平均抗锯齿、双线性、统一选路与索引最近邻。"""

import pytest

from vgif.scale import (
    _axis_spans,
    area_resize,
    bilinear_resize,
    nearest_resize_indexed,
    resize_rgb,
    round_odd,
)


class TestRoundOdd:
    def test_even_rounds_down_to_odd(self):
        # 偶数与相邻两奇数等距，统一向下取奇，规则确定。
        assert round_odd(10) == 9
        assert round_odd(2) == 1

    def test_odd_unchanged(self):
        assert round_odd(1) == 1
        assert round_odd(3) == 3
        assert round_odd(481) == 481

    def test_minimum_clamp(self):
        assert round_odd(0) == 1
        assert round_odd(2, minimum=3) == 3   # 向下会跌破下限，改向上
        assert round_odd(5, minimum=3) == 5

    def test_even_minimum_normalized_to_odd(self):
        assert round_odd(1, minimum=4) == 5

    def test_below_one_minimum_treated_as_one(self):
        assert round_odd(0, minimum=0) == 1


class TestBilinear:
    def test_identity_byte_exact(self):
        frame = bytes((i * 7) % 256 for i in range(4 * 5 * 3))
        assert bilinear_resize(frame, 4, 5, 4, 5) == frame

    def test_output_dimensions(self):
        frame = bytes(4 * 6 * 3)
        out = bilinear_resize(frame, 4, 6, 8, 3)
        assert len(out) == 8 * 3 * 3

    def test_solid_color_preserved_under_upscale(self):
        r, g, b = 33, 77, 200
        frame = bytes([r, g, b]) * (6 * 6)
        out = bilinear_resize(frame, 6, 6, 12, 12)
        for i in range(0, len(out), 3):
            assert (out[i], out[i + 1], out[i + 2]) == (r, g, b)

    def test_rejects_bad_frame_length(self):
        with pytest.raises(ValueError):
            bilinear_resize(bytes(10), 4, 4, 2, 2)

    def test_rejects_nonpositive_target(self):
        frame = bytes(2 * 2 * 3)
        with pytest.raises(ValueError):
            bilinear_resize(frame, 2, 2, 0, 2)

    def test_corner_pixels_anchored(self):
        # 2x2 放大到 4x4：四角应取源四角值。
        frame = bytes([
            0, 0, 0,     100, 100, 100,     # noqa: E241
            200, 200, 200, 255, 255, 255,
        ])
        out = bilinear_resize(frame, 2, 2, 4, 4)
        assert tuple(out[0:3]) == (0, 0, 0)
        assert tuple(out[9:12]) == (100, 100, 100)
        assert tuple(out[36:39]) == (200, 200, 200)
        assert tuple(out[45:48]) == (255, 255, 255)

    def test_deterministic(self):
        frame = bytes((i * 11) % 256 for i in range(8 * 8 * 3))
        a = bilinear_resize(frame, 8, 8, 5, 5)
        b = bilinear_resize(frame, 8, 8, 5, 5)
        assert a == b


class TestAxisSpans:
    def test_weights_partition_axis_without_gap_or_overlap(self):
        cells, wsums = _axis_spans(9, 3)
        covered = sum(w for cell in cells for _, w in cell)
        # 各单元权重求和恰为整轴长度（定点），无缺口无重叠。
        assert covered == 9 * (1 << 16)
        assert len(cells) == 3 and all(ws > 0 for ws in wsums)

    def test_identity_spans_single_full_cell(self):
        cells, wsums = _axis_spans(4, 4)
        for d, (cell, wsum) in enumerate(zip(cells, wsums)):
            assert cell == [(d, 1 << 16)]
            assert wsum == 1 << 16


class TestAreaResize:
    def test_identity_byte_exact(self):
        frame = bytes((i * 7) % 256 for i in range(6 * 5 * 3))
        assert area_resize(frame, 6, 5, 6, 5) == frame

    def test_output_dimensions(self):
        frame = bytes(8 * 6 * 3)
        out = area_resize(frame, 8, 6, 3, 2)
        assert len(out) == 3 * 2 * 3

    def test_solid_color_preserved_any_ratio(self):
        # 归一化按实际权重和进行，任意缩放比（含放大、非整数比）纯色不变。
        r, g, b = 33, 77, 200
        for sw, sh, dw, dh in [
            (10, 8, 4, 3), (9, 9, 3, 3), (10, 10, 20, 16),
            (7, 5, 3, 9), (5, 5, 5, 5),
        ]:
            frame = bytes([r, g, b]) * (sw * sh)
            out = area_resize(frame, sw, sh, dw, dh)
            assert out == bytes([r, g, b]) * (dw * dh), (sw, sh, dw, dh)

    def test_checkerboard_downscale_is_midgray(self):
        # 4x4 黑白棋盘缩到 2x2：每个输出像素覆盖 2x2 源块，面积平均
        # 得中灰 128，高频纹理被低通而非硬采样。
        cb = bytearray()
        for y in range(4):
            for x in range(4):
                v = 255 if (x + y) % 2 == 0 else 0
                cb += bytes([v, v, v])
        out = area_resize(bytes(cb), 4, 4, 2, 2)
        assert len(out) == 2 * 2 * 3
        for i in range(0, len(out), 3):
            assert tuple(out[i:i + 3]) == (128, 128, 128)

    def test_antialiasing_vs_bilinear_on_fine_stripes(self):
        # 9px 明暗细条（周期 2）缩到 3px（3:1）：面积平均把每个输出
        # 单元内的能量都保留为中间调；双线性只碰 2 个邻居，硬采样出
        # 0/255 极端值——这正是锯齿/摩尔纹的量化特征。
        strip = bytearray()
        for x in range(9):
            v = 255 if x % 2 == 0 else 0
            strip += bytes([v, v, v])
        area = area_resize(bytes(strip), 9, 1, 3, 1)
        bil = bilinear_resize(bytes(strip), 9, 1, 3, 1)
        area_vals = [area[i * 3] for i in range(3)]
        bil_vals = [bil[i * 3] for i in range(3)]
        assert all(0 < v < 255 for v in area_vals)
        assert 0 in bil_vals or 255 in bil_vals
        # 面积平均保留的平均能量与源均值（约 142）同量级。
        assert abs(sum(area_vals) / 3 - 142) < 20

    def test_downscale_bounds_within_byte_range(self):
        frame = bytes((i * 13) % 256 for i in range(15 * 11 * 3))
        out = area_resize(frame, 15, 11, 4, 3)
        assert max(out) <= 255 and min(out) >= 0

    def test_rejects_bad_frame_length(self):
        with pytest.raises(ValueError):
            area_resize(bytes(10), 4, 4, 2, 2)

    def test_rejects_nonpositive_target(self):
        frame = bytes(2 * 2 * 3)
        with pytest.raises(ValueError):
            area_resize(frame, 2, 2, 2, 0)

    def test_deterministic(self):
        frame = bytes((i * 11) % 256 for i in range(12 * 12 * 3))
        assert area_resize(frame, 12, 12, 5, 5) == area_resize(
            frame, 12, 12, 5, 5
        )


class TestResizeRgb:
    def test_identity_returns_input_bytes(self):
        frame = bytes((i * 7) % 256 for i in range(4 * 4 * 3))
        assert resize_rgb(frame, 4, 4, 4, 4) == frame

    def test_downscale_uses_area(self):
        frame = bytes((i * 13) % 256 for i in range(8 * 8 * 3))
        assert resize_rgb(frame, 8, 8, 4, 4) == area_resize(
            frame, 8, 8, 4, 4
        )

    def test_upscale_uses_bilinear(self):
        frame = bytes((i * 13) % 256 for i in range(4 * 4 * 3))
        assert resize_rgb(frame, 4, 4, 8, 8) == bilinear_resize(
            frame, 4, 4, 8, 8
        )

    def test_mixed_axes_cascade_dimensions(self):
        # 宽缩小（面积）、高放大（双线性）：级联后尺寸正确、字节合法。
        frame = bytes((i * 17) % 256 for i in range(8 * 6 * 3))
        out = resize_rgb(frame, 8, 6, 4, 12)
        assert len(out) == 4 * 12 * 3
        assert max(out) <= 255

    def test_mixed_axes_solid_color_preserved(self):
        r, g, b = 10, 128, 240
        frame = bytes([r, g, b]) * (8 * 6)
        out = resize_rgb(frame, 8, 6, 4, 12)
        assert out == bytes([r, g, b]) * (4 * 12)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            resize_rgb(bytes(10), 4, 4, 2, 2)
        with pytest.raises(ValueError):
            resize_rgb(bytes(4 * 4 * 3), 4, 4, 0, 2)


class TestNearestIndexed:
    def test_identity(self):
        idx = bytes([0, 1, 2, 3])
        assert nearest_resize_indexed(idx, 2, 2, 2, 2) == idx

    def test_upscale_keeps_valid_indices(self):
        idx = bytes([0, 5, 10, 200])
        out = nearest_resize_indexed(idx, 2, 2, 6, 6)
        assert len(out) == 36
        assert set(out) <= set(idx)

    def test_downscale_picks_source_cells(self):
        idx = bytes(range(16))
        out = nearest_resize_indexed(idx, 4, 4, 2, 2)
        assert len(out) == 4
        assert set(out) <= set(range(16))

    def test_rejects_bad_length(self):
        with pytest.raises(ValueError):
            nearest_resize_indexed(bytes(3), 2, 2, 4, 4)
