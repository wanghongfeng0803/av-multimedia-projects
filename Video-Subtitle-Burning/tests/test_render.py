"""字体光栅化与像素混合测试。"""

from __future__ import annotations

import pytest

from subburn.fonts import BitmapFont, load_font
from subburn.render import OverlayRenderer, Style, blend_rgba_tile


class TestBitmapFont:
    def test_glyph_metrics(self):
        f = BitmapFont(scale=2)
        w, h = f.text_size("AB")
        # 每字 advance=(5+1)*2=12，两字去掉末尾一个 scale：24-2=22；墨高 7*2=14
        assert (w, h) == (22, 14)
        assert f.line_height == 16

    def test_empty_text(self):
        f = BitmapFont(1)
        assert f.text_size("") == (0, f.line_height)
        assert f.rasterize_line("", (255, 255, 255)) == bytearray()

    def test_raster_is_rgba_with_transparent_background(self):
        f = BitmapFont(1)
        buf = f.rasterize_line("A", (255, 255, 255))
        # 必存在墨迹像素（alpha=255）与背景像素（alpha=0）
        alphas = {buf[i + 3] for i in range(0, len(buf), 4)}
        assert 255 in alphas and 0 in alphas

    def test_black_text_preserved(self):
        # 关键回归：黑色文字不能因为"按颜色判墨迹"而丢失
        f = BitmapFont(1)
        buf = f.rasterize_line("A", (0, 0, 0))
        ink = [buf[i:i + 4] for i in range(0, len(buf), 4) if buf[i + 3] == 255]
        assert ink and all(bytes(p[:3]) == b"\x00\x00\x00" for p in ink)

    def test_unknown_glyph_renders_box_not_crash(self):
        f = BitmapFont(1)
        buf = f.rasterize_line("中", (255, 255, 255))  # 位图字体无中文
        assert any(buf[i + 3] == 255 for i in range(0, len(buf), 4))

    def test_bad_scale(self):
        with pytest.raises(ValueError):
            BitmapFont(0)

    def test_load_fallback_without_pillow(self):
        font = load_font(allow_pillow=False)
        assert isinstance(font, BitmapFont)


class TestBlend:
    def test_opaque_overwrite(self):
        frame = bytearray([10, 20, 30] * 2)
        tile = bytearray([255, 0, 0, 255, 0, 255, 0, 255])  # 2x1
        blend_rgba_tile(frame, 2, 1, tile, 2, 1, 0, 0)
        assert frame[:3] == bytearray([255, 0, 0])
        assert frame[3:6] == bytearray([0, 255, 0])

    def test_semi_transparent_integer_math(self):
        frame = bytearray([100, 100, 100])
        tile = bytearray([255, 0, 0, 128])  # 1x1
        blend_rgba_tile(frame, 1, 1, tile, 1, 1, 0, 0)
        # (255*128 + 100*127) >> 8 = 177；(0*128+100*127)>>8 = 49
        assert frame == bytearray([177, 49, 49])

    def test_full_transparent_noop(self):
        frame = bytearray([9, 8, 7])
        tile = bytearray([255, 255, 255, 0])
        blend_rgba_tile(frame, 1, 1, tile, 1, 1, 0, 0)
        assert frame == bytearray([9, 8, 7])

    def test_clip_completely_outside(self):
        frame = bytearray([0] * 12)  # 2x2
        tile = bytearray([255, 0, 0, 255, 0, 255, 0, 255])
        blend_rgba_tile(frame, 2, 2, tile, 2, 1, 9, 9)
        assert frame == bytearray(12)

    def test_clip_partial_overlap(self):
        frame = bytearray([0] * 12)  # 2x2
        tile = bytearray([1, 1, 1, 255, 2, 2, 2, 255,
                          3, 3, 3, 255, 4, 4, 4, 255])  # 2x2
        # 瓦片左上角 (-1, 0)：左列在画面外，右列两个像素贴到画面 x=0 列
        blend_rgba_tile(frame, 2, 2, tile, 2, 2, -1, 0)
        assert frame[0:3] == bytearray([2, 2, 2])
        assert frame[6:9] == bytearray([4, 4, 4])
        assert frame[3:6] == bytearray([0, 0, 0])  # x=1 列未被触碰


class TestOverlayRenderer:
    def setup_method(self):
        self.renderer = OverlayRenderer(BitmapFont(2), Style(box=True))

    def test_tile_size_and_alpha(self):
        tile, w, h = self.renderer._get_tile("AB")
        assert len(tile) == w * h * 4
        # 底盒开启：瓦片四角应为半透明黑（box_alpha=128）
        assert tile[3] == Style().box_alpha

    def test_cache_returns_same_object(self):
        a, _, _ = self.renderer._get_tile("same")
        b, _, _ = self.renderer._get_tile("same")
        assert a is b

    def test_cache_eviction(self):
        r = OverlayRenderer(BitmapFont(1), Style(box=False), cache_size=2)
        r._get_tile("a")
        r._get_tile("b")
        r._get_tile("c")  # 超出容量，"a" 被淘汰
        assert "a" not in r._cache and "c" in r._cache

    def test_render_frame_modifies_pixels(self):
        frame = bytearray([30] * 640 * 360 * 3)
        self.renderer.render_frame(frame, 640, 360, ["HELLO"])
        assert any(v != 30 for v in frame)

    def test_render_empty_texts_is_noop(self):
        frame = bytearray([30] * 100)
        self.renderer.render_frame(frame, 10, 10, [])
        assert frame == bytearray([30] * 100)

    def test_small_frame_keeps_subtitle_visible(self):
        # 回归：默认 margin 大于画面高度时，字幕曾被整体推出画面
        frame = bytearray([20] * 32 * 32 * 3)
        self.renderer.render_frame(frame, 32, 32, ["TEST"])
        assert any(v != 20 for v in frame)

    def test_multiline_tile_taller_than_single(self):
        one, _, h1 = self.renderer._get_tile("A")
        two, _, h2 = self.renderer._get_tile("A\nB")
        assert h2 > h1

    def test_stacked_cues_both_render(self):
        frame = bytearray([20] * 320 * 240 * 3)
        self.renderer.render_frame(frame, 320, 240, ["AAA", "BBB"])
        # 改写过的像素应当分散在两个不同的 y 区段
        changed_rows = {
            (i // 3) // 320 for i in range(0, len(frame), 3) if frame[i] != 20
        }
        assert max(changed_rows) - min(changed_rows) > 10
