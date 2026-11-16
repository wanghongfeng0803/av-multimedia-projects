"""缩放测试：恒等基线、角点对齐、索引域最近邻。"""

import pytest

from vgif.scale import bilinear_resize, nearest_resize_indexed


class TestBilinear:
    def test_identity_byte_exact(self):
        frame = bytes((i * 7) % 256 for i in range(4 * 5 * 3))
        assert bilinear_resize(frame, 4, 5, 4, 5) == frame

    def test_output_dimensions(self):
        frame = bytes(4 * 6 * 3)
        out = bilinear_resize(frame, 4, 6, 8, 3)
        assert len(out) == 8 * 3 * 3

    def test_solid_color_preserved_under_scale(self):
        r, g, b = 33, 77, 200
        frame = bytes([r, g, b]) * (6 * 6)
        out = bilinear_resize(frame, 6, 6, 3, 3)
        # 纯色缩放后仍是同一颜色（插值权重和为 1）。
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
