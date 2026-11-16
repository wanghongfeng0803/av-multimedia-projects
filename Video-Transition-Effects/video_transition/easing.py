"""转场进度速度曲线：把重叠区帧号 k 映射为有理数进度与整数位移。

全部纯函数、无 I/O，输出两种确定性形式：

* :func:`eased_fraction` 返回 ``(num, den)`` 有理数进度，
  供音频增益等凸组合直接使用，绕开浮点；
* :func:`eased_shift` 返回沿运动轴的**整数像素位移**，
  供滑动转场按像素定位（见 :mod:`video_transition.slide`）。

进度语义与锚点
==============

``k`` 是重叠区帧号（0 起），``No`` 是重叠帧数，进度
``p`` 满足：

* 无论哪种曲线，``k=0`` 时 ``p=0``、``k=No-1`` 时 ``p=1``，
  即锚点精确落在 0/1，滑动首尾帧与纯 A/纯 B 逐像素连续，
  边界不会出现跳变帧或黑边；
* 所有曲线在 ``[0,1]`` 上单调不减，位移因此单调推进、不会回退；
* 单帧重叠（``No=1``）没有插值空间，进度取 1/2、位移取半个轴长。

四条曲线
========

``linear``（线性，默认，与旧淡化的等间隔语义一致）、``ease_in``
（慢入快出）、``ease_out``（快入慢出）、``ease_in_out``
（两端慢、中间快，分段二次，分母统一为 ``2*(No-1)^2``）。
"""

from __future__ import annotations

from .blend import div_round_half_away
from .errors import PlanError

__all__ = [
    "EASING_LINEAR",
    "EASING_EASE_IN",
    "EASING_EASE_OUT",
    "EASING_EASE_IN_OUT",
    "EASINGS",
    "eased_fraction",
    "eased_shift",
]

#: 线性：等间隔推进
EASING_LINEAR = "linear"
#: 慢入快出：p = t²
EASING_EASE_IN = "ease_in"
#: 快入慢出：p = 1-(1-t)²
EASING_EASE_OUT = "ease_out"
#: 两端慢中间快：分段二次
EASING_EASE_IN_OUT = "ease_in_out"

#: 全部合法曲线名（CLI 与 build_plan 共用同一份白名单）
EASINGS = (EASING_LINEAR, EASING_EASE_IN, EASING_EASE_OUT, EASING_EASE_IN_OUT)


def _check_index(k: int, overlap_frames: int) -> None:
    """帧号/重叠帧数合法性检查，与 fade_weight_b 的边界语义一致。"""
    if overlap_frames <= 0:
        raise PlanError(f"重叠帧数必须为正，收到 {overlap_frames}")
    if not 0 <= k < overlap_frames:
        raise PlanError(f"重叠区帧号越界: {k} / {overlap_frames}")


def eased_fraction(
    k: int, overlap_frames: int, easing: str = EASING_LINEAR
) -> tuple[int, int]:
    """重叠区第 k 帧的进度，返回 ``(分子, 分母)`` 有理数（值域 [0,1]）。

    单帧重叠返回 ``(1, 2)``；多帧时首尾锚点精确为 ``0/1`` 与 ``1/1``。
    非法曲线名或越界帧号抛 :class:`PlanError`。
    """
    _check_index(k, overlap_frames)
    if easing not in EASINGS:
        raise PlanError(f"未知速度曲线: {easing!r}，可选 {EASINGS}")
    if overlap_frames == 1:
        return 1, 2

    steps = overlap_frames - 1
    t_num, t_den = k, steps
    if easing == EASING_LINEAR:
        return t_num, t_den
    if easing == EASING_EASE_IN:
        # t² = k² / steps²
        return t_num * t_num, t_den * t_den
    if easing == EASING_EASE_OUT:
        # 1-(1-t)² = (2k·steps - k²) / steps²
        return 2 * k * steps - k * k, steps * steps
    # ease_in_out，以 2·steps² 为统一分母，用整数比较 2k 与 steps
    # 选段（不引入浮点）：
    #   前半 2t²   -> 2k²/steps² = 4k²/(2·steps²)
    #   后半 1-2(1-t)² -> (2·steps² - 2(steps-k)²)/(2·steps²)
    # 偶数 steps 的中点 2k==steps 归前半段：2·(1/2)² = 1/2，
    # 后半公式在该点同样为 1/2，曲线连续且关于中点中心对称。
    den = 2 * steps * steps
    if 2 * k <= steps:
        return 4 * k * k, den
    return den - 2 * (steps - k) * (steps - k), den


def eased_shift(
    k: int,
    overlap_frames: int,
    axis_length: int,
    easing: str = EASING_LINEAR,
) -> int:
    """第 k 帧沿运动轴的整数像素位移（``[0, axis_length]``，单调不减）。

    ``axis_length`` 即水平滑动的帧宽、垂直滑动的帧高。位移按
    :func:`eased_fraction` 的有理数进度做「半数远离零」取整，
    再夹回闭区间——锚点帧恒为 0 与 ``axis_length``，不会因取整
    越过边界而露出黑边。
    """
    if axis_length <= 0:
        raise PlanError(f"运动轴长度必须为正，收到 {axis_length}")
    num, den = eased_fraction(k, overlap_frames, easing)
    shift = div_round_half_away(num * axis_length, den)
    if shift < 0:
        return 0
    if shift > axis_length:
        return axis_length
    return shift
