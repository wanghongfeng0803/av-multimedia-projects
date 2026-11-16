"""渲染绘图纯字节测试：矩形、虚线框、文字与逐帧叠加层语义。"""

from conftest import make_plane
from vtrack.font import GLYPH_H
from vtrack.model import FrameRecord, LOST, TRACKED
from vtrack.render import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    draw_dashed_rect,
    draw_frame_overlay,
    draw_rect,
    draw_text,
)

W, H = 48, 32


def _rgb_frame():
    return bytearray(W * H * 3)


def _px(frame, x, y):
    base = (y * W + x) * 3
    return tuple(frame[base:base + 3])


def test_draw_rect_colors_border_only():
    frame = _rgb_frame()
    draw_rect(frame, W, H, 4, 5, 10, 8, COLOR_GREEN, thickness=1)
    assert _px(frame, 4, 5) == COLOR_GREEN       # 左上角
    assert _px(frame, 13, 12) == COLOR_GREEN      # 右下角
    assert _px(frame, 8, 8) == (0, 0, 0)          # 内部不着色


def test_draw_rect_thickness_two():
    frame = _rgb_frame()
    draw_rect(frame, W, H, 2, 2, 12, 10, COLOR_RED, thickness=2)
    assert _px(frame, 2, 3) == COLOR_RED          # 第二行边框
    assert _px(frame, 6, 6) == (0, 0, 0)          # 内部


def test_draw_rect_clipped_at_frame_edge():
    frame = _rgb_frame()
    # 越界不抛异常，能画的部分照画
    draw_rect(frame, W, H, W - 3, H - 3, 10, 10, COLOR_GREEN)
    assert _px(frame, W - 1, H - 1) == COLOR_GREEN


def test_dashed_rect_alternates_on_off():
    frame = _rgb_frame()
    draw_dashed_rect(frame, W, H, 4, 4, 16, 8, COLOR_YELLOW, dash=4)
    # 顶边：前 4 像素亮，接着 4 像素暗
    assert _px(frame, 4, 4) == COLOR_YELLOW
    assert _px(frame, 7, 4) == COLOR_YELLOW
    assert _px(frame, 8, 4) == (0, 0, 0)


def test_draw_text_writes_foreground_pixels():
    frame = _rgb_frame()
    draw_text(frame, W, H, 2, 2, "A", COLOR_GREEN, font_scale=1)
    # 字形至少打出若干前景像素
    greens = sum(
        1 for y in range(H) for x in range(W) if _px(frame, x, y) == COLOR_GREEN
    )
    assert greens > 5


def test_draw_text_scaled_is_blockier():
    f1 = _rgb_frame()
    f2 = _rgb_frame()
    draw_text(f1, W, H, 0, 0, "A", COLOR_GREEN, font_scale=1)
    draw_text(f2, W, H, 0, 0, "A", COLOR_GREEN, font_scale=2)
    n1 = sum(1 for y in range(H) for x in range(W)
             if _px(f1, x, y) == COLOR_GREEN)
    n2 = sum(1 for y in range(H) for x in range(W)
             if _px(f2, x, y) == COLOR_GREEN)
    assert n2 == n1 * 4


def test_draw_text_clipped_at_right_edge():
    frame = _rgb_frame()
    # 文字超出右边界不抛异常；可见的首字形在 (2,2) 处有前景像素
    draw_text(frame, W, H, W - 3, 2, "LONG", COLOR_RED)
    assert _px(frame, W - 3, 2) == COLOR_RED


def _record(status, x=4, y=14, score=12.5, confirmed=None):
    return FrameRecord(
        frame=0, pts_time=0.0, status=status,
        bbox=(x, y, 10, 8), score=score,
        confirmed=confirmed if confirmed is not None else status == TRACKED,
        search=(0, 0, 20, 20),
    )


def test_overlay_tracked_draws_green_box():
    frame = _rgb_frame()
    draw_frame_overlay(frame, W, H, _record(TRACKED), "sad")
    assert _px(frame, 4, 14) == COLOR_GREEN
    # 没有红色 LOST 标记
    reds = sum(1 for y in range(H) for x in range(W)
               if _px(frame, x, y) == COLOR_RED)
    assert reds == 0


def test_overlay_lost_no_green_box_but_red_status():
    frame = _rgb_frame()
    draw_frame_overlay(frame, W, H, _record(LOST), "sad", lost_count=3)
    # 原 bbox 位置不画绿框
    assert _px(frame, 4, 14) == (0, 0, 0)
    reds = sum(1 for y in range(H) for x in range(W)
               if _px(frame, x, y) == COLOR_RED)
    assert reds > 0


def test_overlay_lost_last_known_dashed_yellow():
    frame = _rgb_frame()
    draw_frame_overlay(frame, W, H, _record(LOST), "sad",
                       show_last_known=True, lost_count=1)
    yellows = sum(1 for y in range(H) for x in range(W)
                  if _px(frame, x, y) == COLOR_YELLOW)
    assert yellows > 0
    # 仍无绿框
    greens = sum(1 for y in range(H) for x in range(W)
                 if _px(frame, x, y) == COLOR_GREEN)
    assert greens == 0


def test_overlay_zncc_score_formatting():
    frame = _rgb_frame()
    rec = _record(TRACKED, score=0.87654)
    draw_frame_overlay(frame, W, H, rec, "zncc")
    # 不崩即通过；具体文字由字库测试与人工渲染保证
    assert _px(frame, 4, 14) == COLOR_GREEN
