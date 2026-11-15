"""水印运动轨迹：多关键帧插值、缓动函数与帧率到时间的映射。

水印的位置偏移 (x, y) 与旋转角 angle 都是时间 t（秒）的连续函数，
由若干 :class:`Keyframe` 分段插值得到：

- 位置按分段线性插值（段内再施加缓动函数）；
- 角度沿**最短弧**插值（跨越 ±180° 时走短边），最终结果归一化到
  [0, 360)——负角度与超过 360° 的输入都会被正确折叠；
- t 落在首/末关键帧之外时，保持首/末关键帧的状态（clamp，不外推）。

任意帧率通过 :func:`frame_to_time` 把整数帧索引映射为秒
（``t = frame_index / fps``），29.97 等非整数帧率同样适用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import InvalidParameterError

# ---------------------------------------------------------------- 缓动函数


def _ease_linear(u: float) -> float:
    return u


def _ease_in(u: float) -> float:
    return u * u * u


def _ease_out(u: float) -> float:
    v = 1.0 - u
    return 1.0 - v * v * v


def _ease_in_out(u: float) -> float:
    if u < 0.5:
        return 4.0 * u * u * u
    v = -2.0 * u + 2.0
    return 1.0 - v * v * v / 2.0


#: 可选缓动函数名 -> 函数，输入/输出均为 [0, 1] 区间内的进度
EASINGS = {
    "linear": _ease_linear,
    "ease-in": _ease_in,
    "ease-out": _ease_out,
    "ease-in-out": _ease_in_out,
}


# ---------------------------------------------------------------- 角度工具


def normalize_angle(angle: float) -> float:
    """把任意角度折叠到 [0, 360) 区间（负角度、超 360° 均正确处理）。"""
    normalized = math.fmod(angle, 360.0)
    if normalized < 0.0:
        normalized += 360.0
    # fmod 可能把 360 的整数倍折成 -0.0，统一成正 0
    return normalized + 0.0 if normalized else 0.0


def interpolate_angle(a: float, b: float, u: float) -> float:
    """沿最短弧在 a、b 之间插值（u ∈ [0, 1]），结果归一化到 [0, 360)。"""
    delta = normalize_angle(b - a + 180.0) - 180.0
    return normalize_angle(a + delta * u)


# ---------------------------------------------------------------- 帧率映射


def frame_to_time(frame_index: int, fps: float) -> float:
    """把整数帧索引映射为时间（秒）：``t = frame_index / fps``。

    支持任意正帧率（含 29.97 等非整数）；非法输入抛出
    :class:`InvalidParameterError`。
    """
    if not isinstance(frame_index, int) or isinstance(frame_index, bool):
        raise InvalidParameterError("frame_index 必须为整数")
    if frame_index < 0:
        raise InvalidParameterError(f"frame_index 必须非负，收到 {frame_index}")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise InvalidParameterError("fps 必须为数值")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise InvalidParameterError(f"fps 必须为正有限数，收到 {fps}")
    return frame_index / fps


# ---------------------------------------------------------------- 关键帧轨迹


def _check_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidParameterError(f"{name} 必须为数值")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidParameterError(f"{name} 必须为有限数，收到 {value}")
    return value


@dataclass
class Keyframe:
    """一个关键帧：``time`` 秒时水印的偏移 (x, y) 与旋转角 ``angle``（度）。

    x/y 为相对静态定位结果的像素偏移（浮点，支持亚像素），
    angle 允许为负或超过 360°，求值时自动归一化。
    """

    time: float
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0


@dataclass
class MotionState:
    """某一时刻求值得到的运动状态。"""

    x: float
    y: float
    angle: float  # 已归一化到 [0, 360)


def motion_from_dict(data: object) -> "MotionConfig":
    """从 JSON 风格字典构建 :class:`MotionConfig` 并校验。

    期望格式：``{"keyframes": [{"time": 0, "x": 0, "y": 0, "angle": 0}, ...],
    "easing": "linear", "anchor_x": 0.5, "anchor_y": 0.5}``，
    格式非法统一抛出 :class:`InvalidParameterError`。
    """
    if not isinstance(data, dict):
        raise InvalidParameterError("运动配置必须是 JSON 对象")
    raw_keyframes = data.get("keyframes")
    if not isinstance(raw_keyframes, list):
        raise InvalidParameterError("运动配置缺少 keyframes 数组")

    keyframes = []
    for i, item in enumerate(raw_keyframes):
        if not isinstance(item, dict) or "time" not in item:
            raise InvalidParameterError(f"第 {i} 个关键帧必须是含 time 字段的对象")
        try:
            keyframes.append(
                Keyframe(
                    time=item["time"],
                    x=item.get("x", 0.0),
                    y=item.get("y", 0.0),
                    angle=item.get("angle", 0.0),
                )
            )
        except TypeError as exc:
            raise InvalidParameterError(f"第 {i} 个关键帧字段非法：{exc}") from exc

    motion = MotionConfig(
        keyframes=keyframes,
        easing=data.get("easing", "linear"),
        anchor_x=data.get("anchor_x", 0.5),
        anchor_y=data.get("anchor_y", 0.5),
    )
    motion.validate()
    return motion


@dataclass
class MotionConfig:
    """水印运动配置：按时间排序的关键帧序列 + 全局缓动 + 旋转锚点。

    ``anchor_x`` / ``anchor_y`` 为旋转锚点的相对坐标（0..1，
    相对缩放后的水印宽/高），默认 (0.5, 0.5) 即绕中心旋转。
    """

    keyframes: list[Keyframe] = field(default_factory=list)
    easing: str = "linear"
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def validate(self) -> None:
        """逐项校验，失败抛出 :class:`InvalidParameterError`。"""
        if not self.keyframes:
            raise InvalidParameterError("keyframes 不能为空，至少需要一个关键帧")
        if self.easing not in EASINGS:
            raise InvalidParameterError(
                f"未知缓动函数 {self.easing!r}，可选值：{', '.join(sorted(EASINGS))}"
            )
        for name, value in (("anchor_x", self.anchor_x), ("anchor_y", self.anchor_y)):
            value = _check_finite(name, value)
            if not 0.0 <= value <= 1.0:
                raise InvalidParameterError(f"{name} 必须在 [0, 1] 区间内，收到 {value}")

        prev_time = None
        for i, kf in enumerate(self.keyframes):
            if not isinstance(kf, Keyframe):
                raise InvalidParameterError(f"第 {i} 个关键帧不是 Keyframe 实例")
            t = _check_finite(f"keyframes[{i}].time", kf.time)
            if t < 0.0:
                raise InvalidParameterError(f"keyframes[{i}].time 必须非负，收到 {t}")
            if prev_time is not None and t <= prev_time:
                raise InvalidParameterError(
                    "关键帧时间必须严格递增："
                    f"keyframes[{i}].time={t} 不大于前一帧 {prev_time}"
                )
            prev_time = t
            _check_finite(f"keyframes[{i}].x", kf.x)
            _check_finite(f"keyframes[{i}].y", kf.y)
            _check_finite(f"keyframes[{i}].angle", kf.angle)

    def evaluate(self, t: float) -> MotionState:
        """求 t 时刻的运动状态；超出关键帧范围时保持端点状态。"""
        t = _check_finite("t", t)
        if t < 0.0:
            raise InvalidParameterError(f"t 必须非负，收到 {t}")
        # 引擎保证先 validate()，这里防御性兜底
        if not self.keyframes:
            raise InvalidParameterError("keyframes 不能为空，至少需要一个关键帧")

        first, last = self.keyframes[0], self.keyframes[-1]
        if t <= first.time:
            return MotionState(first.x, first.y, normalize_angle(first.angle))
        if t >= last.time:
            return MotionState(last.x, last.y, normalize_angle(last.angle))

        # 二分定位所在段：kfs[i].time <= t < kfs[i+1].time
        lo, hi = 0, len(self.keyframes) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.keyframes[mid].time <= t:
                lo = mid
            else:
                hi = mid
        k0, k1 = self.keyframes[lo], self.keyframes[hi]

        span = k1.time - k0.time
        u = (t - k0.time) / span if span > 0 else 0.0
        eased = EASINGS[self.easing](u)
        return MotionState(
            x=k0.x + (k1.x - k0.x) * eased,
            y=k0.y + (k1.y - k0.y) * eased,
            angle=interpolate_angle(k0.angle, k1.angle, eased),
        )
