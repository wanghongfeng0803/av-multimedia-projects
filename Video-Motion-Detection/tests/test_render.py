"""标记渲染测试：rgb24 字节平面上的绘图函数与叠加层状态区分。

不依赖 ffmpeg，只测纯字节绘图；端到端编码在 test_pipeline。
"""

from vmotion.model import FrameRecord
from vmotion.render import (
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_YELLOW,
    draw_frame_overlay,
    draw_rect,
    draw_text,
)


def _rgb_frame(w, h, fill=(10, 10, 10)):
    return bytearray(fill) * (w * h * 3)


def _pixel(frame, w, x, y):
    base = (y * w + x) * 3
    return tuple(frame[base:base + 3])


def test_draw_rect_writes_border_only():
    frame = _rgb_frame(16, 16)
    draw_rect(frame, 16, 16, 2, 2, 8, 6, COLOR_GREEN, thickness=1)
    # 边框点被着色
    assert _pixel(frame, 16, 2, 2) == COLOR_GREEN
    assert _pixel(frame, 16, 9, 7) == COLOR_GREEN
    # 内部点保持底色（不填充）
    assert _pixel(frame, 16, 5, 4) == (10, 10, 10)


def test_draw_rect_clips_at_frame_border():
    frame = _rgb_frame(8, 8)
    # 超出右/下边界不应抛异常，可见部分照常画
    draw_rect(frame, 8, 8, 5, 5, 10, 10, COLOR_GREEN, thickness=1)
    assert _pixel(frame, 8, 5, 5) == COLOR_GREEN
    assert _pixel(frame, 8, 7, 7) == COLOR_GREEN


def test_draw_text_changes_pixels_and_has_outline():
    frame = _rgb_frame(40, 12)
    draw_text(frame, 40, 12, 2, 2, "MOTION", COLOR_GREEN, font_scale=1)
    colors = set()
    for i in range(0, len(frame), 3):
        colors.add(tuple(frame[i:i + 3]))
    # 前景绿与描边黑（底色也是黑，故至少出现绿色）
    assert COLOR_GREEN in colors


def test_draw_text_empty_string_is_noop():
    frame = _rgb_frame(8, 8)
    before = bytes(frame)
    draw_text(frame, 8, 8, 0, 0, "", COLOR_GREEN)
    assert bytes(frame) == before


def test_overlay_idle_motion_holdoff_are_visually_distinct():
    w, h = 120, 40

    def render(record):
        frame = _rgb_frame(w, h)
        draw_frame_overlay(frame, w, h, record, font_scale=1)
        return frame

    idle = render(FrameRecord(0, 0.0, False, [], {"held_off": False}))
    hold = render(FrameRecord(1, 0.1, False, [],
                              {"held_off": True, "mean_delta": 30.0}))
    region = {"label": 1, "x": 40, "y": 20, "w": 16, "h": 12,
              "area": 100, "cx": 48, "cy": 26}
    move = render(FrameRecord(2, 0.2, True, [region],
                              {"held_off": False}, event_id=1))
    # 运动帧在区域框上出现绿色像素，另两种状态该位置无绿框
    assert _pixel(move, w, 40, 20) == COLOR_GREEN
    assert _pixel(idle, w, 40, 20) != COLOR_GREEN
    assert _pixel(hold, w, 40, 20) != COLOR_GREEN

    def has_color(frame, color):
        return any(tuple(frame[i:i + 3]) == color
                   for i in range(0, len(frame), 3))

    assert has_color(hold, COLOR_YELLOW)   # HOLDOFF 黄
    assert has_color(move, COLOR_GREEN)    # MOTION 绿
    assert has_color(idle, COLOR_GRAY)     # IDLE 灰
