"""动画叠加引擎测试：运动求值、旋转合成、蒙版逐帧重算与静态退化等价。"""

import pytest

from watermark_overlay.animation import Keyframe, MotionConfig
from watermark_overlay.config import WatermarkConfig
from watermark_overlay.engine import overlay_animated_frame, overlay_image
from watermark_overlay.errors import InvalidParameterError
from watermark_overlay.image import Image

from conftest import solid_image


def _motion(*keyframes, easing="linear", anchor_x=0.5, anchor_y=0.5):
    return MotionConfig(
        keyframes=list(keyframes),
        easing=easing,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )


# ---------------------------------------------------------------- 静态退化等价


def test_zero_motion_matches_static_overlay(frame_4x4, red_watermark):
    config = WatermarkConfig(anchor="top-left")
    static_out, static_info = overlay_image(frame_4x4, red_watermark, config)
    anim_out, anim_info = overlay_animated_frame(
        frame_4x4,
        red_watermark,
        config,
        _motion(Keyframe(0.0), Keyframe(1.0, x=0.0, y=0.0, angle=0.0)),
        frame_index=10,
        fps=30.0,
    )
    assert bytes(anim_out.pixels) == bytes(static_out.pixels)
    assert anim_info.origin == static_info.origin
    assert anim_info.blended_pixels == static_info.blended_pixels
    assert anim_info.angle == 0.0
    assert anim_info.motion_offset == (0.0, 0.0)


def test_none_motion_is_static_path(frame_4x4, red_watermark):
    config = WatermarkConfig()
    static_out, _ = overlay_image(frame_4x4, red_watermark, config)
    anim_out, info = overlay_animated_frame(
        frame_4x4, red_watermark, config, motion=None, frame_index=5, fps=25.0
    )
    assert bytes(anim_out.pixels) == bytes(static_out.pixels)
    assert info.angle == 0.0


def test_opacity_zero_with_motion_is_identity(frame_4x4, red_watermark):
    out, info = overlay_animated_frame(
        frame_4x4,
        red_watermark,
        WatermarkConfig(opacity=0.0),
        _motion(Keyframe(0.0, x=2.0, angle=45.0)),
    )
    assert not info.overlapped
    assert bytes(out.pixels) == bytes(frame_4x4.pixels)
    assert info.angle == 45.0


# ---------------------------------------------------------------- 位置轨迹


def test_position_keyframes_move_watermark():
    frame = solid_image(8, 8, (0, 0, 0, 255))
    wm = solid_image(2, 2, (255, 0, 0, 255))
    config = WatermarkConfig(anchor="top-left")
    motion = _motion(Keyframe(0.0, x=0.0, y=0.0), Keyframe(1.0, x=4.0, y=2.0))

    out0, info0 = overlay_animated_frame(
        frame, wm, config, motion, frame_index=0, fps=25.0
    )
    assert info0.origin == (0, 0)
    assert info0.blended_pixels == 4
    assert out0.get_pixel(0, 0) == (255, 0, 0, 255)

    # t = 12/25 = 0.48s → 线性插值 x=1.92, y=0.96（亚像素，画布原点取整）
    out1, info1 = overlay_animated_frame(
        frame, wm, config, motion, frame_index=12, fps=25.0
    )
    assert info1.origin == (1, 0)
    assert info1.blended_pixels > 0
    assert info1.motion_offset == (pytest.approx(1.92), pytest.approx(0.96))

    # 末关键帧之后保持终点状态
    _, info2 = overlay_animated_frame(
        frame, wm, config, motion, frame_index=100, fps=25.0
    )
    assert info2.origin == (4, 2)


def test_fractional_fps_frame_mapping():
    frame = solid_image(8, 8, (0, 0, 0, 255))
    wm = solid_image(2, 2, (255, 0, 0, 255))
    config = WatermarkConfig(anchor="top-left")
    motion = _motion(Keyframe(0.0, x=0.0), Keyframe(1.0, x=6.0))
    # 29.97 帧率：第 30 帧 ≈ 1.001s → 已越过末关键帧，clamp 到终点
    _, info = overlay_animated_frame(
        frame, wm, config, motion, frame_index=30, fps=29.97
    )
    assert info.origin == (6, 0)


def test_watermark_moved_fully_outside_frame():
    frame = solid_image(4, 4, (10, 20, 30, 255))
    wm = solid_image(2, 2, (255, 0, 0, 255))
    out, info = overlay_animated_frame(
        frame,
        wm,
        WatermarkConfig(anchor="top-left"),
        _motion(Keyframe(0.0, x=100.0, y=0.0)),
    )
    assert info.blended_pixels == 0
    assert not info.overlapped
    assert bytes(out.pixels) == bytes(frame.pixels)


# ---------------------------------------------------------------- 旋转


def test_rotation_90_in_engine():
    frame = solid_image(6, 6, (0, 0, 0, 255))
    wm = Image(2, 2)
    wm.set_pixel(0, 0, 255, 0, 0, 255)
    wm.set_pixel(1, 0, 0, 255, 0, 255)
    wm.set_pixel(0, 1, 0, 0, 255, 255)
    wm.set_pixel(1, 1, 255, 255, 255, 255)
    out, info = overlay_animated_frame(
        frame,
        wm,
        WatermarkConfig(anchor="top-left"),
        _motion(Keyframe(0.0, angle=90.0)),
    )
    assert info.angle == 90.0
    assert info.watermark_size == (2, 2)
    assert info.origin == (0, 0)
    # 与 transform 层一致：C A / D B
    assert out.get_pixel(0, 0) == (0, 0, 255, 255)
    assert out.get_pixel(1, 0) == (255, 0, 0, 255)


def test_rotation_45_expands_and_shifts_origin():
    frame = solid_image(16, 16, (0, 0, 0, 255))
    wm = solid_image(2, 2, (255, 0, 0, 255))
    out, info = overlay_animated_frame(
        frame,
        wm,
        WatermarkConfig(anchor="top-left"),
        _motion(Keyframe(0.0, angle=45.0)),
    )
    # 2x2 旋转 45° → 4x4 画布，原点 (-1, -1)，越界部分被裁剪
    assert info.watermark_size == (4, 4)
    assert info.origin == (-1, -1)
    assert 0 < info.blended_pixels < 16


def test_angle_normalization_equivalent_results():
    frame = solid_image(8, 8, (0, 0, 0, 255))
    wm = solid_image(2, 2, (255, 0, 0, 255))
    config = WatermarkConfig(anchor="top-left")
    out90, _ = overlay_animated_frame(
        frame, wm, config, _motion(Keyframe(0.0, angle=90.0))
    )
    out450, info = overlay_animated_frame(
        frame, wm, config, _motion(Keyframe(0.0, angle=450.0))
    )
    out_neg, _ = overlay_animated_frame(
        frame, wm, config, _motion(Keyframe(0.0, angle=-270.0))
    )
    assert info.angle == 90.0
    assert bytes(out450.pixels) == bytes(out90.pixels)
    assert bytes(out_neg.pixels) == bytes(out90.pixels)


def test_rotation_anchor_configurable():
    frame = solid_image(16, 16, (0, 0, 0, 255))
    wm = solid_image(4, 4, (255, 0, 0, 255))
    config = WatermarkConfig(anchor="center")
    center_pivot, info_c = overlay_animated_frame(
        frame, wm, config, _motion(Keyframe(0.0, angle=90.0))
    )
    corner_pivot, info_k = overlay_animated_frame(
        frame,
        wm,
        config,
        _motion(Keyframe(0.0, angle=90.0), anchor_x=0.0, anchor_y=0.0),
    )
    # 绕角点旋转 90° 后画布整体平移，原点与绕中心不同
    assert info_c.origin != info_k.origin


# ---------------------------------------------------------------- 自适应蒙版逐帧重算


def test_adaptive_mask_recomputed_at_each_position():
    # 左半黑右半白，边缘在 x=3/4 之间
    frame = Image(8, 4)
    for y in range(4):
        for x in range(8):
            v = 255 if x >= 4 else 0
            frame.set_pixel(x, y, v, v, v, 255)
    wm = solid_image(2, 4, (255, 0, 0, 255))
    config = WatermarkConfig(
        anchor="top-left",
        adaptive=True,
        mask_strength=0.8,
        detail_low=16,
        detail_high=96,
    )
    plain = WatermarkConfig(anchor="top-left")

    # 位置一：水印在 x=0（平坦区），自适应与固定透明度结果一致
    motion_flat = _motion(Keyframe(0.0, x=0.0))
    adaptive_flat, _ = overlay_animated_frame(frame, wm, config, motion_flat)
    fixed_flat, _ = overlay_animated_frame(frame, wm, plain, motion_flat)
    assert adaptive_flat.get_pixel(0, 0) == fixed_flat.get_pixel(0, 0)

    # 位置二：水印移到 x=3（压在边缘上），蒙版必须在新位置压暗红色
    motion_edge = _motion(Keyframe(0.0, x=3.0))
    adaptive_edge, info = overlay_animated_frame(frame, wm, config, motion_edge)
    fixed_edge, _ = overlay_animated_frame(frame, wm, plain, motion_edge)
    assert info.origin == (3, 0)
    assert adaptive_edge.get_pixel(3, 0)[0] < fixed_edge.get_pixel(3, 0)[0]


# ---------------------------------------------------------------- 输入校验


def test_invalid_frame_index_or_fps_rejected(frame_4x4, red_watermark):
    motion = _motion(Keyframe(0.0))
    with pytest.raises(InvalidParameterError) as exc:
        overlay_animated_frame(
            frame_4x4, red_watermark, WatermarkConfig(), motion, frame_index=-1
        )
    assert "frame_index" in str(exc.value)
    with pytest.raises(InvalidParameterError):
        overlay_animated_frame(
            frame_4x4, red_watermark, WatermarkConfig(), motion, fps=0
        )


def test_invalid_motion_rejected(frame_4x4, red_watermark):
    with pytest.raises(InvalidParameterError):
        overlay_animated_frame(
            frame_4x4, red_watermark, WatermarkConfig(), MotionConfig(keyframes=[])
        )
