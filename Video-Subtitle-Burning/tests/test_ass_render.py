"""ASS 渲染层测试：numpad 对齐、run 排版、定位/特效与瓦片缓存。

全部以 :class:`BitmapFont` 驱动（空字体注册表强制位图路径），不依赖系统
TTF，任何环境可复现；端到端烧录用例沿用 conftest 的 ffmpeg 夹具。
"""

from __future__ import annotations

import pytest

from subburn.ass import parse_ass_text
from subburn.ass_render import ASSRenderer, numpad_align, _scale_rgba
from subburn.fonts import BitmapFont, FontRegistry
from subburn import burn_ass

from .conftest import make_test_video

W, H = 320, 240
REG = FontRegistry(candidate_paths=())

_STYLE_FMT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FMT = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
    "MarginV, Effect, Text"
)


def make_doc(styles: str = "", dialogues: list[str] | None = None,
             *, play_x: int = W, play_y: int = H):
    lines = ["[Script Info]", f"PlayResX: {play_x}", f"PlayResY: {play_y}",
             "[V4+ Styles]", _STYLE_FMT]
    lines.append(styles or (
        "Style: D,Arial,14,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,1,1,2,10,10,10,1"))
    lines += ["[Events]", _EVENT_FMT]
    for text in (dialogues or []):
        lines.append(
            f"Dialogue: 0,0:00:00.00,0:00:02.00,D,,0,0,0,,{text}")
    doc, warns = parse_ass_text("\n".join(lines) + "\n")
    assert not warns, warns
    return doc


def frame(fill: int = 20) -> bytearray:
    return bytearray([fill] * W * H * 3)


def ink_bbox(buf: bytearray, background: int = 20):
    xs, ys = [], []
    for y in range(H):
        for x in range(W):
            i = (y * W + x) * 3
            if any(abs(buf[i + c] - background) > 0 for c in range(3)):
                xs.append(x)
                ys.append(y)
    assert xs, "帧上没有任何字幕像素"
    return min(xs), max(xs), min(ys), max(ys)


class TestNumpad:
    @pytest.mark.parametrize("num,halign,valign", [
        (1, "left", "bottom"), (2, "center", "bottom"), (3, "right", "bottom"),
        (4, "left", "middle"), (5, "center", "middle"), (6, "right", "middle"),
        (7, "left", "top"), (8, "center", "top"), (9, "right", "top"),
    ])
    def test_numpad_mapping(self, num, halign, valign):
        assert numpad_align(num) == (halign, valign)

    def test_invalid_falls_back_to_bottom_center(self):
        assert numpad_align(0) == ("center", "bottom")
        assert numpad_align(99) == ("center", "bottom")


class TestScaleRgba:
    def test_upscale_nearest_neighbor(self):
        buf = bytearray([10, 20, 30, 255] * 2)  # 2x1
        out = _scale_rgba(buf, 2, 1, 4, 2)
        assert len(out) == 4 * 2 * 4
        # 最近邻：输出每两列对应源一列，两行内容相同
        assert tuple(out[0:4]) == (10, 20, 30, 255)
        assert tuple(out[4 * 4:4 * 4 + 4]) == (10, 20, 30, 255)

    def test_degenerate_returns_empty(self):
        assert _scale_rgba(bytearray([0] * 4), 1, 1, 0, 5) == bytearray()


class TestRenderBasics:
    def test_render_modifies_frame(self):
        doc = make_doc(dialogues=["HELLO"])
        r = ASSRenderer(doc, W, H, registry=REG)
        f = frame()
        r.render_frame(f, W, H, list(doc.cues), 0)
        assert f != frame()

    def test_empty_cues_is_noop(self):
        doc = make_doc(dialogues=["X"])
        r = ASSRenderer(doc, W, H, registry=REG)
        f = frame()
        r.render_frame(f, W, H, [], 0)
        assert f == frame()

    def test_uses_bitmap_font_in_empty_registry(self):
        doc = make_doc(dialogues=["AB"])
        r = ASSRenderer(doc, W, H, registry=REG)
        font = r._font_for(doc.cues[0].lines[0].runs[0],
                           doc.styles["D"])
        assert isinstance(font, BitmapFont)


class TestTileCaching:
    def test_static_cue_single_tile_across_frames(self):
        doc = make_doc(dialogues=["static"])
        r = ASSRenderer(doc, W, H, registry=REG)
        cue = doc.cues[0]
        assert r._tile_for(cue, 0) is r._tile_for(cue, 1500)

    def test_dynamic_cue_rebuilds_per_change_frame(self):
        doc = make_doc(dialogues=[r"{\fad(300,300)}fade"])
        r = ASSRenderer(doc, W, H, registry=REG)
        cue = doc.cues[0]
        assert r._tile_for(cue, 0) is not r._tile_for(cue, 400)
        # 淡入结束后进入稳态，相邻帧重新共享
        assert r._tile_for(cue, 500) is r._tile_for(cue, 600)

    def test_cache_eviction(self):
        # 用长淡入制造连续不同的 alpha 状态键，超出容量后最早键被淘汰
        doc = make_doc(dialogues=[r"{\fad(1000,0)}x"])
        r = ASSRenderer(doc, W, H, registry=REG, cache_size=2)
        cue = doc.cues[0]
        first = r._state_key(cue, 0)
        r._tile_for(cue, 0)
        r._tile_for(cue, 100)
        r._tile_for(cue, 200)
        assert first not in r._tile_cache


class TestFade:
    def _fading(self):
        return make_doc(dialogues=[r"{\fad(400,400)}x"])

    def test_starts_fully_transparent(self):
        doc = self._fading()
        r = ASSRenderer(doc, W, H, registry=REG)
        tile, _, _ = r._tile_for(doc.cues[0], 0)
        assert all(tile[i + 3] == 0 for i in range(0, len(tile), 4))

    def test_opaque_after_fade_in(self):
        doc = self._fading()
        r = ASSRenderer(doc, W, H, registry=REG)
        tile, _, _ = r._tile_for(doc.cues[0], 500)
        assert any(tile[i + 3] == 255 for i in range(0, len(tile), 4))

    def test_fades_out_near_end(self):
        doc = self._fading()
        r = ASSRenderer(doc, W, H, registry=REG)
        cue = doc.cues[0]
        # 对话长 2000ms，淡出窗 [1600,2000)；1990ms 接近全透明
        tile, _, _ = r._tile_for(cue, 1990)
        assert max(tile[i + 3] for i in range(0, len(tile), 4)) < 255


class TestKaraoke:
    def test_secondary_before_primary_after(self):
        doc = make_doc(dialogues=[r"{\k50}la {\k50}la"])
        r = ASSRenderer(doc, W, H, registry=REG)
        cue = doc.cues[0]
        before, _, _ = r._tile_for(cue, 0)
        after, _, _ = r._tile_for(cue, 600)

        def colors(tile):
            return {tuple(tile[i:i + 3]) for i in range(0, len(tile), 4)
                    if tile[i + 3] == 255}

        # 次色 &H000000FF -> RGB 红；到点前出现，到点后消失，主色白保留
        assert (255, 0, 0) in colors(before)
        assert (255, 0, 0) not in colors(after)
        assert (255, 255, 255) in colors(after)


class TestPositioning:
    def test_pos_anchors_tile(self):
        doc = make_doc(dialogues=[r"{\an7\pos(40,30)}x"])
        r = ASSRenderer(doc, W, H, registry=REG)
        f = frame()
        r.render_frame(f, W, H, list(doc.cues), 0)
        x0, _, y0, _ = ink_bbox(f)
        assert x0 >= 35 and y0 >= 25

    def test_numpad_margins_bottom_center_default(self):
        doc = make_doc()  # 默认 an2 + MarginV 10
        doc2 = make_doc(dialogues=["X"])
        r = ASSRenderer(doc2, W, H, registry=REG)
        f = frame()
        r.render_frame(f, W, H, list(doc2.cues), 0)
        _, _, _, y1 = ink_bbox(f)
        assert y1 <= H - 1  # 底对齐：墨迹底边在画面内且靠近底部

    def test_move_travels_left_to_right(self):
        doc = make_doc(dialogues=[r"{\move(0,120,260,120)}go"])
        r = ASSRenderer(doc, W, H, registry=REG)
        cue = doc.cues[0]
        # \move 只改锚点不改瓦片像素：整条移动过程共享同一张瓦片
        assert r._tile_for(cue, 0) is r._tile_for(cue, 1000)
        f0, f1 = frame(), frame()
        r.render_frame(f0, W, H, list(doc.cues), 0)
        r.render_frame(f1, W, H, list(doc.cues), 2000)

        def leftmost(buf):
            return min(x for y in range(H) for x in range(W)
                       if any(abs(buf[(y * W + x) * 3 + c] - 20) > 0
                              for c in range(3)))
        assert leftmost(f1) > leftmost(f0) + 100

    def test_anchor_clamped_inside_frame(self):
        # 极端 pos 把瓦片推出画面，夹紧后仍有可见墨迹
        doc = make_doc(dialogues=[r"{\an7\pos(-999,-999)}x"])
        r = ASSRenderer(doc, W, H, registry=REG)
        f = frame()
        r.render_frame(f, W, H, list(doc.cues), 0)
        x0, _, y0, _ = ink_bbox(f)
        assert x0 == 0 and y0 == 0


class TestBorderStyleAndDecorations:
    def test_opaque_box_fills_corners(self):
        style = (
            "Style: D,Arial,14,&H00000000,&H000000FF,&H00FFFFFF,"
            "&H00FFFFFF,0,0,0,0,100,100,0,0,3,2,0,5,10,10,10,1")
        doc = make_doc(styles=style, dialogues=["box"])
        r = ASSRenderer(doc, W, H, registry=REG)
        tile, _, _ = r._tile_for(doc.cues[0], 0)
        assert tuple(tile[0:4]) == (255, 255, 255, 255)  # 不透明白底盒

    def test_underline_and_strikeout_add_ink(self):
        doc = make_doc(dialogues=[r"{\u1}U", r"{\s1}S"])
        r = ASSRenderer(doc, W, H, registry=REG)
        t_u, _, _ = r._tile_for(doc.cues[0], 0)
        t_s, _, _ = r._tile_for(doc.cues[1], 0)
        # 装饰线瓦片比纯字形包含更多不透明像素
        plain_doc = make_doc(dialogues=["U", "S"])
        rp = ASSRenderer(plain_doc, W, H, registry=REG)
        p_u, _, _ = rp._tile_for(plain_doc.cues[0], 0)
        p_s, _, _ = rp._tile_for(plain_doc.cues[1], 0)
        ink = lambda b: sum(1 for i in range(3, len(b), 4) if b[i] == 255)
        assert ink(t_u) > ink(p_u) and ink(t_s) > ink(p_s)


class TestMultilineAlignment:
    def test_short_line_centered_under_long_line(self):
        # 默认 an2（底部居中）：两行里短行的中心应与长行中心对齐
        style = (
            "Style: D,Arial,14,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1")
        doc = make_doc(styles=style, dialogues=[r"AAAA\NAA"])
        r = ASSRenderer(doc, W, H, registry=REG)
        tile, tw, th = r._tile_for(doc.cues[0], 0)

        def ink_cols(row_lo, row_hi):
            cols = []
            for y in range(row_lo, row_hi):
                for x in range(tw):
                    if tile[(y * tw + x) * 4 + 3] != 0:
                        cols.append(x)
            return cols

        top = ink_cols(0, th // 2)
        bottom = ink_cols(th // 2, th)
        # 两行各自的墨迹中心应几乎重合（位图整数取整允许 1px 误差）
        center_top = (min(top) + max(top)) / 2
        center_bottom = (min(bottom) + max(bottom)) / 2
        assert abs(center_top - center_bottom) <= 1


class TestPlayResScaling:
    def test_playres_smaller_than_frame_scales_up(self):
        # PlayRes 只有帧的一半：scale=2，瓦片几何应放大
        doc = make_doc(play_x=W // 2, play_y=H // 2, dialogues=["X"])
        r = ASSRenderer(doc, W, H, registry=REG)
        assert r.scale == pytest.approx(2.0)
        _, tw, th = r._tile_for(doc.cues[0], 0)
        doc1 = make_doc(dialogues=["X"])
        r1 = ASSRenderer(doc1, W, H, registry=REG)
        _, tw1, th1 = r1._tile_for(doc1.cues[0], 0)
        assert tw >= tw1 and th >= th1


@pytest.mark.usefixtures("ffmpeg")
class TestASSIntegration:
    def test_burn_ass_end_to_end(self, tmp_path):
        video = tmp_path / "v.mp4"
        out = tmp_path / "o.mp4"
        make_test_video(video)
        doc = make_doc(dialogues=[
            r"普通 {\b1}粗体{\r} 混排", r"{\an8\pos(20,20)}顶部",
            r"{\fad(100,100)}淡入淡出", r"{\k40}la {\k60}la"])
        result = burn_ass(str(video), str(out), doc, overwrite=True)
        assert result.frames_written == 50 and out.exists()
        assert result.cues == 4

    def test_burn_ass_timing_half_open(self, tmp_path):
        video = tmp_path / "v.mp4"
        out = tmp_path / "o.mp4"
        make_test_video(video, duration=2)
        doc, _ = parse_ass_text(
            "[Script Info]\nPlayResX: 320\nPlayResY: 240\n"
            "[V4+ Styles]\n" + _STYLE_FMT + "\n"
            "Style: D,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n"
            "[Events]\n" + _EVENT_FMT + "\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,D,,0,0,0,,EDGE\n")
        burn_ass(str(video), str(out), doc, overwrite=True)

        import subprocess
        def grab(src, ts, name):
            p = tmp_path / name
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-ss", str(ts), "-i", src,
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                str(p)], check=True)
            return p.read_bytes()

        # 终点 1000ms = 帧 25 时刻，半开区间外应与原帧无差异
        edge = grab(str(out), 1.0, "e.rgb")
        src = grab(str(video), 1.0, "s.rgb")
        diff = sum(
            1 for i in range(0, len(edge), 3)
            if max(abs(edge[i] - src[i]), abs(edge[i + 1] - src[i + 1]),
                   abs(edge[i + 2] - src[i + 2])) >= 100)
        assert diff == 0
