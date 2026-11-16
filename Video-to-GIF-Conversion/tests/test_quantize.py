"""中位切分量化与最近邻映射测试。"""

import pytest

from vgif.quantize import (
    MAX_GIF_COLORS,
    QuantizeError,
    QuantizedFrame,
    map_frame_to_index,
    median_cut_palette,
    next_pow2_colors,
    quantize_frames,
)


class TestNextPow2:
    @pytest.mark.parametrize(
        "n,expected",
        [(1, 2), (2, 2), (3, 4), (100, 128), (255, 256), (256, 256)],
    )
    def test_pow2(self, n, expected):
        assert next_pow2_colors(n) == expected

    def test_out_of_range(self):
        with pytest.raises(QuantizeError):
            next_pow2_colors(0)
        with pytest.raises(QuantizeError):
            next_pow2_colors(257)


class TestMedianCut:
    def test_single_color(self):
        px = [(10, 20, 30)] * 50
        pal = median_cut_palette(px, 256)
        assert pal == ((10, 20, 30),)

    def test_few_unique_colors_returned_as_is(self):
        px = [(0, 0, 0), (255, 255, 255), (255, 0, 0)]
        pal = median_cut_palette(px, 256)
        assert set(pal) == set(px)

    def test_respects_max_colors(self):
        px = [(i, 255 - i, (i * 7) % 256) for i in range(256)]
        pal = median_cut_palette(px, 16)
        assert len(pal) <= 16
        assert len(pal) >= 1

    def test_palette_entries_in_range(self):
        px = [((i * 13) % 256, (i * 29) % 256, (i * 53) % 256)
              for i in range(500)]
        pal = median_cut_palette(px, 32)
        for r, g, b in pal:
            assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_empty_rejected(self):
        with pytest.raises(QuantizeError):
            median_cut_palette([], 16)

    def test_deterministic(self):
        px = [((i * 13) % 256, (i * 29) % 256, (i * 53) % 256)
              for i in range(500)]
        assert median_cut_palette(px, 32) == median_cut_palette(px, 32)


class TestMapFrame:
    def test_exact_palette_colors_map_to_self(self):
        palette = ((0, 0, 0), (255, 255, 255), (255, 0, 0))
        frame = bytes([0, 0, 0, 255, 255, 255, 255, 0, 0])
        idx = map_frame_to_index(frame, palette)
        assert idx == bytes([0, 1, 2])

    def test_nearest_color_basic(self):
        palette = ((0, 0, 0), (255, 255, 255))
        # 近黑归 0，近白归 1。
        frame = bytes([10, 10, 10, 240, 240, 240])
        idx = map_frame_to_index(frame, palette)
        assert idx == bytes([0, 1])

    def test_cache_reuse_across_frames(self):
        palette = ((0, 0, 0), (255, 255, 255))
        cache = {}
        f1 = bytes([0, 0, 0, 255, 255, 255])
        f2 = bytes([255, 255, 255, 0, 0, 0])
        i1 = map_frame_to_index(f1, palette, cache)
        i2 = map_frame_to_index(f2, palette, cache)
        assert i1 == bytes([0, 1]) and i2 == bytes([1, 0])
        assert (0, 0, 0) in cache and (255, 255, 255) in cache


class TestQuantizeFrames:
    def test_solid_frame_single_palette_entry(self):
        w = h = 4
        frame = bytes([12, 120, 200]) * (w * h)
        palette, qframes = quantize_frames([frame], w, h, 256)
        assert len(palette) == 1
        assert qframes[0].indices == bytes([0]) * (w * h)
        assert isinstance(qframes[0], QuantizedFrame)

    def test_shared_palette_across_frames(self):
        w = h = 2
        f1 = bytes([0, 0, 0]) * 4
        f2 = bytes([255, 255, 255]) * 4
        palette, qframes = quantize_frames([f1, f2], w, h, 256)
        # 两帧引用同一张全局表，索引在表内合法。
        for q in qframes:
            assert max(q.indices) < len(palette)
        assert len(palette) == 2

    def test_size_mismatch_rejected(self):
        with pytest.raises(QuantizeError):
            quantize_frames([bytes(3)], 4, 4, 16)

    def test_empty_rejected(self):
        with pytest.raises(QuantizeError):
            quantize_frames([], 2, 2, 16)

    def test_sample_cap_does_not_change_output_length(self):
        w = h = 8
        frame = bytes(range(256))[: w * h * 3]
        frame = frame * (w * h * 3 // len(frame) + 1)
        frame = frame[: w * h * 3]
        palette, qframes = quantize_frames(
            [frame], w, h, 16, sample_cap=10
        )
        assert len(qframes[0].indices) == w * h
        assert len(palette) <= MAX_GIF_COLORS
