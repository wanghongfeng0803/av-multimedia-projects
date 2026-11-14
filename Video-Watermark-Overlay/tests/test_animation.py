"""运动轨迹测试：关键帧插值、缓动、角度归一化与帧率映射。"""

import pytest

from watermark_overlay.animation import (
    EASINGS,
    Keyframe,
    MotionConfig,
    frame_to_time,
    interpolate_angle,
    normalize_angle,
)
from watermark_overlay.errors import InvalidParameterError


# ---------------------------------------------------------------- 角度归一化


def test_normalize_angle_basic():
    assert normalize_angle(0.0) == 0.0
    assert normalize_angle(90.0) == 90.0
    assert normalize_angle(359.5) == 359.5


def test_normalize_angle_negative_and_over_360():
    assert normalize_angle(-90.0) == 270.0
    assert normalize_angle(-360.0) == 0.0
    assert normalize_angle(360.0) == 0.0
    assert normalize_angle(450.0) == 90.0
    assert normalize_angle(720.0) == 0.0
    assert normalize_angle(-810.0) == 270.0


def test_interpolate_angle_takes_shortest_arc():
    # 350° -> 10° 应走 +20° 的短弧而不是 -340°
    assert interpolate_angle(350.0, 10.0, 0.5) == 0.0
    # 反向：10° -> 350° 走 -20°
    assert interpolate_angle(10.0, 350.0, 0.5) == 0.0
    # 端点不变
    assert interpolate_angle(30.0, 200.0, 0.0) == 30.0
    assert interpolate_angle(30.0, 200.0, 1.0) == 200.0


# ---------------------------------------------------------------- 帧率映射


def test_frame_to_time_integer_fps():
    assert frame_to_time(0, 25) == 0.0
    assert frame_to_time(25, 25) == 1.0
    assert frame_to_time(7, 30) == pytest.approx(7 / 30)


def test_frame_to_time_fractional_fps():
    # 29.97 等非整数帧率
    assert frame_to_time(30, 29.97) == pytest.approx(30 / 29.97)
    assert frame_to_time(1, 23.976) == pytest.approx(1 / 23.976)


@pytest.mark.parametrize("bad_fps", [0, -25, float("nan"), float("inf"), "25", True])
def test_frame_to_time_rejects_bad_fps(bad_fps):
    with pytest.raises(InvalidParameterError):
        frame_to_time(0, bad_fps)


@pytest.mark.parametrize("bad_index", [-1, 1.5, "3", True, None])
def test_frame_to_time_rejects_bad_index(bad_index):
    with pytest.raises(InvalidParameterError):
        frame_to_time(bad_index, 25)


# ---------------------------------------------------------------- 缓动函数


def test_easings_keep_endpoints():
    for name, fn in EASINGS.items():
        assert fn(0.0) == pytest.approx(0.0), name
        assert fn(1.0) == pytest.approx(1.0), name


def test_easing_shapes():
    assert EASINGS["linear"](0.5) == pytest.approx(0.5)
    assert EASINGS["ease-in"](0.5) == pytest.approx(0.125)
    assert EASINGS["ease-out"](0.5) == pytest.approx(0.875)
    assert EASINGS["ease-in-out"](0.5) == pytest.approx(0.5)
    # ease-in 前半段落后于线性，ease-out 超前
    assert EASINGS["ease-in"](0.25) < 0.25
    assert EASINGS["ease-out"](0.25) > 0.25


# ---------------------------------------------------------------- 轨迹求值


def test_evaluate_linear_interpolation():
    motion = MotionConfig(
        keyframes=[Keyframe(0.0, x=0.0, y=0.0), Keyframe(2.0, x=10.0, y=-4.0)]
    )
    motion.validate()
    state = motion.evaluate(1.0)
    assert state.x == pytest.approx(5.0)
    assert state.y == pytest.approx(-2.0)
    assert state.angle == 0.0


def test_evaluate_clamps_outside_keyframe_range():
    motion = MotionConfig(
        keyframes=[Keyframe(1.0, x=3.0), Keyframe(2.0, x=7.0, angle=90.0)]
    )
    motion.validate()
    assert motion.evaluate(0.0).x == 3.0
    assert motion.evaluate(0.5).x == 3.0
    last = motion.evaluate(99.0)
    assert last.x == 7.0
    assert last.angle == 90.0


def test_evaluate_applies_easing():
    motion = MotionConfig(
        keyframes=[Keyframe(0.0, x=0.0), Keyframe(1.0, x=1.0)],
        easing="ease-in",
    )
    motion.validate()
    assert motion.evaluate(0.5).x == pytest.approx(0.125)


def test_evaluate_interpolates_angle_on_shortest_arc():
    motion = MotionConfig(
        keyframes=[Keyframe(0.0, angle=350.0), Keyframe(1.0, angle=10.0)]
    )
    motion.validate()
    assert motion.evaluate(0.5).angle == pytest.approx(0.0)
    assert motion.evaluate(0.25).angle == pytest.approx(355.0)


def test_evaluate_normalizes_wild_angles():
    motion = MotionConfig(keyframes=[Keyframe(0.0, angle=-450.0)])
    motion.validate()
    # -450° ≡ 270°
    assert motion.evaluate(0.0).angle == 270.0


def test_evaluate_multi_segment():
    motion = MotionConfig(
        keyframes=[
            Keyframe(0.0, x=0.0),
            Keyframe(1.0, x=10.0),
            Keyframe(3.0, x=30.0),
        ]
    )
    motion.validate()
    assert motion.evaluate(0.5).x == pytest.approx(5.0)
    assert motion.evaluate(2.0).x == pytest.approx(20.0)


def test_evaluate_rejects_negative_time():
    motion = MotionConfig(keyframes=[Keyframe(0.0)])
    motion.validate()
    with pytest.raises(InvalidParameterError):
        motion.evaluate(-0.5)
    with pytest.raises(InvalidParameterError):
        motion.evaluate(float("nan"))


# ---------------------------------------------------------------- 配置校验


def test_validate_rejects_empty_keyframes():
    with pytest.raises(InvalidParameterError):
        MotionConfig(keyframes=[]).validate()


def test_validate_rejects_unsorted_keyframes():
    motion = MotionConfig(keyframes=[Keyframe(2.0), Keyframe(1.0)])
    with pytest.raises(InvalidParameterError):
        motion.validate()


def test_validate_rejects_duplicate_times():
    motion = MotionConfig(keyframes=[Keyframe(1.0), Keyframe(1.0)])
    with pytest.raises(InvalidParameterError):
        motion.validate()


def test_validate_rejects_unknown_easing():
    motion = MotionConfig(keyframes=[Keyframe(0.0)], easing="bounce")
    with pytest.raises(InvalidParameterError):
        motion.validate()


@pytest.mark.parametrize("anchor", [-0.1, 1.1, float("nan")])
def test_validate_rejects_out_of_range_anchor(anchor):
    motion = MotionConfig(keyframes=[Keyframe(0.0)], anchor_x=anchor)
    with pytest.raises(InvalidParameterError):
        motion.validate()


@pytest.mark.parametrize(
    "field_name", ["time", "x", "y", "angle"]
)
def test_validate_rejects_non_finite_fields(field_name):
    kwargs = {"time": 0.0, field_name: float("inf")}
    # time 不能为负无穷之外的非法值；inf 同样被拒绝
    motion = MotionConfig(keyframes=[Keyframe(**kwargs)])
    with pytest.raises(InvalidParameterError):
        motion.validate()


def test_validate_rejects_negative_time():
    motion = MotionConfig(keyframes=[Keyframe(-1.0)])
    with pytest.raises(InvalidParameterError):
        motion.validate()
