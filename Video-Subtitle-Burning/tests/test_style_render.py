"""新样式系统在渲染层的行为测试：透明度、对齐、边距越界、缓存隔离与端到端。

全部以确定性的 :class:`BitmapFont` 驱动，不依赖系统 TTF；端到端用例沿用
conftest 的 ffmpeg 夹具，无 ffmpeg 时自动 skip。
"""

from __future__ import annotations

import pytest

from subburn.fonts import BitmapFont, FontRegistry
from subburn.render import OverlayRenderer, Style
from subburn.style import (
    Color,
    FontSpec,
    HAlign,
    Margins,
    Position,
    SubtitleStyle,
    VAlign,
)
from subburn import burn_subtitles, parse_srt

from .conftest import make_test_video

W, H = 320, 240


def _frame(fill: int = 20) -> bytearray:
    return bytearray([fill] * W * H * 3)


def ink_bbox(frame: bytearray, background: int = 20):
    """返回被改写像素的 (min_x, max_x, min_y, max_y)。"""
    xs, ys = [], []
    for y in range(H):
        for x in range(W):
            i = (y * W + x) * 3
            if any(abs(frame[i + c] - background) > 0 for c in range(3)):
                xs.append(x)
                ys.append(y)
    assert xs, "帧上没有任何字幕像素"
    return min(xs), max(xs), min(ys), max(ys)


def _render(style: SubtitleStyle, texts, background: int = 20):
    renderer = OverlayRenderer(BitmapFont(2), style)
    frame = _frame(background)
    renderer.render_frame(frame, W, H, texts)
    return frame, renderer


class TestTransparency:
    def test_semi_transparent_fill_uses_source_over(self):
        # 黑底 + 半透明白字（无描边/阴影/底盒），结果可由混合公式精确预测
        st = SubtitleStyle(
            color=Color(255, 255, 255, 128),
            outline_width=0, shadow_offset=0, box=False,
        )
        frame, _ = _render(st, ["A"], background=0)
        values = {frame[i] for i in range(0, len(frame), 3) if frame[i] != 0}
        assert values == {(255 * 128) >> 8}  # 127

    def test_full_transparent_text_is_invisible(self):
        st = SubtitleStyle(
            color=Color(255, 255, 255, 0),
            outline_width=0, shadow_offset=0, box=False,
        )
        frame, _ = _render(st, ["A"])
        assert frame == _frame()

    def test_box_alpha_comes_from_box_color(self):
        st = SubtitleStyle(box=True, box_color=Color(0, 0, 0, 64),
                           shadow_offset=0)
        renderer = OverlayRenderer(BitmapFont(1), st)
        tile, tw, _ = renderer._get_tile("A")
        assert tile[3] == 64  # 左上角即底盒像素

    def test_outline_alpha_appears_in_tile(self):
        st = SubtitleStyle(outline_width=1,
                           outline_color=Color(0, 0, 0, 100),
                           shadow_offset=0, box=False)
        renderer = OverlayRenderer(BitmapFont(1), st)
        tile, _, _ = renderer._get_tile("A")
        alphas = {tile[i + 3] for i in range(0, len(tile), 4)}
        assert 100 in alphas  # 描边-only 像素保留描边透明度

    def test_shadow_color_alpha_respected(self):
        st = SubtitleStyle(shadow_offset=2,
                           shadow_color=Color(0, 0, 0, 140),
                           outline_width=0, box=False)
        renderer = OverlayRenderer(BitmapFont(1), st)
        tile, _, _ = renderer._get_tile("A")
        alphas = {tile[i + 3] for i in range(0, len(tile), 4)}
        assert 140 in alphas


class TestHorizontalAlignment:
    def _style(self, halign):
        return SubtitleStyle(
            position=Position(halign=halign, margins=Margins.uniform(10)),
            outline_width=0, shadow_offset=0, box=False,
        )

    def test_left_center_right_anchor_x(self):
        frame_l, _ = _render(self._style(HAlign.LEFT), ["HELLO"])
        frame_c, _ = _render(self._style(HAlign.CENTER), ["HELLO"])
        frame_r, _ = _render(self._style(HAlign.RIGHT), ["HELLO"])
        x_min_l = ink_bbox(frame_l)[0]
        x_max_r = ink_bbox(frame_r)[1]
        cx = (ink_bbox(frame_c)[0] + ink_bbox(frame_c)[1]) / 2
        assert x_min_l == 10
        # 右对齐：瓦片右边缘距画面右边 10px（瓦片含描边外扩，这里描边已关）
        assert x_max_r == W - 10 - 1
        assert 130 < cx < 190  # 居中：画面中心 160 附近

    def test_multiline_respects_line_alignment(self):
        # 短行在左对齐时贴左、右对齐时贴右；居中为旧行为
        left = SubtitleStyle(
            position=Position(halign=HAlign.LEFT,
                              margins=Margins.uniform(0)),
            outline_width=0, shadow_offset=0)
        right = SubtitleStyle(
            position=Position(halign=HAlign.RIGHT,
                              margins=Margins.uniform(0)),
            outline_width=0, shadow_offset=0)
        frame_l, _ = _render(left, ["AAAA\nA"])
        frame_r, _ = _render(right, ["AAAA\nA"])

        def short_line_left(frame):
            # 短行是瓦片的第二行，落在画面下半部；取该行带最左的改写列
            cols = []
            for y in range(H // 2, H):
                row = [x for x in range(W)
                       if any(frame[(y * W + x) * 3 + c] != 20 for c in range(3))]
                if row:
                    cols.append(min(row))
            return min(cols) if cols else None

        assert short_line_left(frame_r) > short_line_left(frame_l)


class TestVerticalAlignment:
    def _style(self, valign, margin=10):
        return SubtitleStyle(
            position=Position(valign=valign,
                              margins=Margins(margin, margin, margin, margin)),
            outline_width=0, shadow_offset=0, box=False,
        )

    def test_top_middle_bottom_anchor_y(self):
        frame_t, _ = _render(self._style(VAlign.TOP), ["HELLO"])
        frame_m, _ = _render(self._style(VAlign.MIDDLE), ["HELLO"])
        frame_b, _ = _render(self._style(VAlign.BOTTOM), ["HELLO"])
        assert ink_bbox(frame_t)[2] == 10
        cy = (ink_bbox(frame_m)[2] + ink_bbox(frame_m)[3]) / 2
        assert 95 < cy < 145  # 画面中心 120 附近
        # 单瓦片高 14：底边锚定 y = 240 - 10 - 14 = 216
        assert ink_bbox(frame_b)[2] == H - 10 - 14

    def test_top_valign_stacks_subsequent_cues_downward(self):
        st = self._style(VAlign.TOP)
        frame, _ = _render(st, ["AAA", "BBB"])
        _, _, y0, y1 = ink_bbox(frame)
        assert y0 == 10 and y1 > 40  # 两条字幕占据不同行带

class TestMarginOverflow:
    def test_huge_margins_keep_subtitle_visible(self):
        st = SubtitleStyle(
            position=Position(margins=Margins(500, 500, 500, 500)),
            outline_width=0, shadow_offset=0)
        frame, _ = _render(st, ["TEST"])
        assert frame != _frame()
        x0, _, y0, _ = ink_bbox(frame)
        assert x0 >= 0 and y0 >= 0  # 没有负坐标，画面内必有内容

    def test_opposing_margins_collapse_then_render(self):
        st = SubtitleStyle(
            position=Position(
                halign=HAlign.LEFT, valign=VAlign.TOP,
                margins=Margins(left=300, right=300, top=0, bottom=0)),
            outline_width=0, shadow_offset=0)
        frame, _ = _render(st, ["X"])
        assert ink_bbox(frame)[0] == 0  # 边距归零后贴左

    def test_taller_than_frame_clamps_inside(self):
        st = SubtitleStyle(
            position=Position(margins=Margins.uniform(4)),
            outline_width=0, shadow_offset=0, line_gap=2)
        many = "\n".join("LINE" for _ in range(40))
        frame, _ = _render(st, [many])
        assert ink_bbox(frame)[2] >= 0


class TestCacheAndCompatibility:
    def test_different_styles_get_different_tiles(self):
        thin = OverlayRenderer(BitmapFont(1), SubtitleStyle(outline_width=0))
        thick = OverlayRenderer(BitmapFont(1), SubtitleStyle(outline_width=3))
        t1, _, h1 = thin._get_tile("A")
        t2, _, h2 = thick._get_tile("A")
        assert h2 > h1 and t1 is not t2

    def test_same_style_cache_key_stable_across_renderers(self):
        # 字体身份相同、样式相同 -> 缓存键相同（可安全共享瓦片）
        font = BitmapFont(1)
        st = SubtitleStyle()
        r1 = OverlayRenderer(font, st)
        r2 = OverlayRenderer(font, st)
        assert r1._cache_key("A") == r2._cache_key("A")

    def test_different_font_instance_changes_cache_key(self):
        st = SubtitleStyle()
        r1 = OverlayRenderer(BitmapFont(1), st)
        r2 = OverlayRenderer(BitmapFont(1), st)
        assert r1._cache_key("A") != r2._cache_key("A")

    def test_whitespace_only_texts_is_noop(self):
        frame = _frame()
        OverlayRenderer(BitmapFont(1), SubtitleStyle()).render_frame(
            frame, W, H, ["   ", "\t"])
        assert frame == _frame()

    def test_legacy_and_new_default_style_equivalent_tiles(self):
        legacy = OverlayRenderer(BitmapFont(2), Style(box=True))
        modern = OverlayRenderer(
            BitmapFont(2), SubtitleStyle.from_legacy(Style(box=True)))
        t1, w1, h1 = legacy._get_tile("AB\nC")
        t2, w2, h2 = modern._get_tile("AB\nC")
        assert (w1, h1) == (w2, h2)
        assert bytes(t1) == bytes(t2)


@pytest.mark.usefixtures("ffmpeg")
class TestStyleIntegration:
    def test_burn_with_new_style_and_bitmap_font(self, tmp_path):
        video = tmp_path / "v.mp4"
        out = tmp_path / "o.mp4"
        make_test_video(video)
        cues, _ = parse_srt(
            "1\n00:00:00,100 --> 00:00:01,000\nSTYLED\n")
        style = SubtitleStyle(
            color=Color(255, 200, 0, 220),
            position=Position(halign=HAlign.LEFT,
                              margins=Margins(12, 12, 12, 12)),
            box=True)
        result = burn_subtitles(str(video), str(out), cues,
                                font=BitmapFont(2), style=style,
                                overwrite=True)
        assert result.frames_written == 50 and out.exists()

    def test_style_drives_font_resolution_when_font_omitted(self, tmp_path):
        video = tmp_path / "v.mp4"
        out = tmp_path / "o.mp4"
        make_test_video(video)
        cues, _ = parse_srt(
            "1\n00:00:00,100 --> 00:00:01,000\nX\n")
        registry = FontRegistry()
        registry._scan()
        if not registry._by_family:
            pytest.skip("系统无可用 TTF 字体")
        style = SubtitleStyle(font=FontSpec(size=24))
        result = burn_subtitles(str(video), str(out), cues, style=style,
                                overwrite=True)
        assert result.frames_written == 50
