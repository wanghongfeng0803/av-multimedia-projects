"""擦除转场（wipe）：固定画面 + 移动掩码，方向与软边宽度可配。

与滑动（:mod:`video_transition.slide`）的区别
=============================================

滑动让 A/B 两帧整体平移；擦除时两帧都**固定不动**，由一条沿运动轴
推进的掩码边界（edge）逐步「揭开」B、盖住 A。边界之外是纯 A/纯 B，
边界附近宽度 ``soft`` 的条带内按线性渐变逐像素混合。

掩码时间线
==========

掩码边界与转场时长共用同一条进度曲线——边界的推进进度直接取
:func:`video_transition.easing.eased_fraction` 的有理数进度（与滑动
位移、音频增益同源），再按扩展行程换算成整数边界位置（见
:func:`wipe_edge`）。画面揭到哪、声音增益就跟到哪，音视频严格同帧。

软边模型（无黑边 + 精确锚点）
=============================

设运动轴长为 ``L``、软边宽度为 ``soft``（沿轴像素数，0 表示硬边）。
若只让边界在 ``[0, L]`` 内移动，恒宽渐变带在终点仍压在画布内，
末帧就到不了纯 B——锚点被破坏。因此让边界走一条**扩展行程**::

    edge(k) = round(p(k) * (L + soft)) - soft，p 为 eased 进度

取值范围恰为 ``[-soft, L]``：

* ``k=0``（p=0）：``edge=-soft``，整条渐变带 ``[edge, edge+soft)``
  落在画布外，满幅纯 A，与 head 末帧逐字节连续；
* ``k=No-1``（p=1）：``edge=L``，渐变带整体移出另一端画布外，
  满幅纯 B，与 tail 首帧逐字节连续；
* 中间帧：渐变带夹在两个纯区之间，带内权重线性过渡。

任意一帧画布上每个像素都有确定的 A/B 凸组合权重，**构造上不存在
未覆盖像素**，不会画出黑边。``soft=0`` 时扩展行程退化为 ``[0, L]``、
擦除退化为硬边，分区与 :func:`video_transition.slide.slide_regions`
逐字节同构（同帧 A/B、同进度下硬擦除与硬滑动输出一致）。

软边宽度钳制
============

软边宽度只沿运动轴生效，合法上限即轴长（水平=宽、垂直=高）。
超过轴长的配置在规划阶段（:func:`video_transition.plan.build_plan`）
被钳到轴长并发出 :class:`WipeSoftWidthWarning`——不报错中断，但
调用方一定会在告警里看到原始值与钳制后的值；钳制后渐变铺满全轴、
依旧无黑边、锚点依旧精确。

与帧混合器的关系
================

擦除不引入第二条像素通路：输出沿运动轴切成「纯 B / 渐变 / 纯 A」
三个连续分区，每段调用 :func:`video_transition.blend.blend_rgb24_frame`，
纯区取端点权重 0/1、渐变带取 ``num/soft`` 的整数权重，淡化、滑动、
擦除因此共用同一套确定性整数混合原语。
"""

from __future__ import annotations

import warnings

from .blend import blend_rgb24_frame, div_round_half_away
from .easing import eased_fraction
from .errors import PlanError
from .slide import (
    DIRECTIONS,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    HORIZONTAL_DIRECTIONS,
)

__all__ = [
    "WipeSoftWidthWarning",
    "clamp_soft_width",
    "wipe_edge",
    "wipe_segments",
    "composite_wipe_frame",
]


class WipeSoftWidthWarning(UserWarning):
    """软边宽度超过运动轴长度被钳制时发出的告警（规划阶段，每计划一次）。"""


def clamp_soft_width(soft_width: int, axis_length: int) -> int:
    """把软边宽度钳进合法区间 ``[0, axis_length]``，超界即告警。

    负宽无物理意义、直接抛 :class:`PlanError`；超过轴长时钳到轴长
    （此时渐变可铺满全轴，仍是合法的无黑边擦除）并发出
    :class:`WipeSoftWidthWarning`，原始值与钳制值都写进告警文本。
    """
    if not isinstance(soft_width, int) or isinstance(soft_width, bool):
        raise PlanError(f"软边宽度必须为整数，收到 {soft_width!r}")
    if soft_width < 0:
        raise PlanError(f"软边宽度不能为负，收到 {soft_width}")
    if axis_length <= 0:
        raise PlanError(f"运动轴长度必须为正，收到 {axis_length}")
    if soft_width > axis_length:
        warnings.warn(
            f"擦除软边宽度 {soft_width}px 超过运动轴长度 {axis_length}px，"
            f"已钳制为 {axis_length}px（渐变铺满全轴，不会出现黑边）",
            WipeSoftWidthWarning,
            stacklevel=2,
        )
        return axis_length
    return soft_width


def wipe_edge(
    k: int,
    overlap_frames: int,
    axis_length: int,
    soft_width: int,
    easing: str,
) -> int:
    """重叠区第 k 帧的掩码边界位置（``[-soft_width, axis_length]``）。

    进度取 :func:`video_transition.easing.eased_fraction`（与滑动位移、
    音频增益同一条时间线），再在扩展行程 ``axis_length + soft_width``
    上做「半数远离零」取整并平移 ``-soft_width``：

    * ``k=0`` 恒为 ``-soft_width``（渐变带全在画布外，满幅纯 A）；
    * ``k=No-1`` 恒为 ``axis_length``（渐变带移出画布，满幅纯 B）；
    * 中间帧单调不减，掩码不会回退。
    """
    if axis_length <= 0:
        raise PlanError(f"运动轴长度必须为正，收到 {axis_length}")
    if not 0 <= soft_width <= axis_length:
        raise PlanError(
            f"软边宽度 {soft_width} 越界，合法区间 [0, {axis_length}]"
        )
    num, den = eased_fraction(k, overlap_frames, easing)
    travel = axis_length + soft_width
    edge = div_round_half_away(num * travel, den) - soft_width
    # 锚点钳制：取整误差不允许把边界推出扩展行程
    if edge < -soft_width:
        return -soft_width
    if edge > axis_length:
        return axis_length
    return edge


def wipe_segments(
    direction: str,
    edge: int,
    soft_width: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """给定边界与软边宽度，返回沿运动轴、按物理坐标升序的权重分区。

    返回 ``(起点, 长度, 权重分子, 权重分母)`` 元组序列，权重即该段
    每个像素的 B 混合权重 ``num/den``：

    * 水平方向：坐标是**列区间**，对全部行生效；
    * 垂直方向：坐标是**行区间**，对全部列生效；
    * 纯 B 段权重 ``1/1``、纯 A 段权重 ``0/1``；
    * 软边段（仅 ``soft_width > 0`` 且与画布相交时存在）被展开为
      长度恒为 1 的逐坐标段，权重分子沿推进方向从 1 递增到
      ``soft_width``，分母为 ``soft_width``。

    所有段长度之和恒等于轴长，每像素恰好属于一个段、权重恒在
    ``[0, 1]``——无空缺故无黑边。``edge`` 合法区间为
    ``[-soft_width, axis_length]``（扩展行程，见模块文档）。
    """
    if direction not in DIRECTIONS:
        raise PlanError(f"未知擦除方向: {direction!r}，可选 {DIRECTIONS}")
    axis = width if direction in HORIZONTAL_DIRECTIONS else height
    soft = soft_width
    if not 0 <= soft <= axis:
        raise PlanError(f"软边宽度 {soft} 越界，合法区间 [0, {axis}]")
    if not -soft <= edge <= axis:
        raise PlanError(
            f"掩码边界 {edge} 越界，合法区间 [-{soft}, {axis}]"
        )

    reverse = direction in (DIRECTION_LEFT, DIRECTION_UP)
    segments: list[tuple[int, int, int, int]] = []

    if not reverse:
        # right/down：B 从低坐标侧推入。渐变带 = [edge, edge+soft)，
        # 带左侧（坐标 < edge）纯 B、带右侧（坐标 >= edge+soft）纯 A。
        pure_b_len = max(0, edge)
        if pure_b_len:
            segments.append((0, pure_b_len, 1, 1))
        if soft:
            band_lo = max(0, edge)
            band_hi = min(axis, edge + soft)
            for u in range(band_lo, band_hi):
                segments.append((u, 1, u - edge + 1, soft))
        pure_a_start = max(0, edge + soft)
        if pure_a_start < axis:
            segments.append((pure_a_start, axis - pure_a_start, 0, 1))
    else:
        # left/up：B 从高坐标侧推入。沿物理坐标 u 从小到大是
        # 纯 A → 渐变 → 纯 B。用 u' = axis-1-u 把高坐标侧映射成
        # 与正向同构的归一坐标后，权重随 u 增大而增大。
        pure_a_len = max(0, axis - (edge + soft))
        if pure_a_len:
            segments.append((0, pure_a_len, 0, 1))
        if soft:
            band_lo = max(0, edge)
            band_hi = min(axis, edge + soft)
            for up in range(band_lo, band_hi):
                u = axis - 1 - up
                segments.append((u, 1, up - edge + 1, soft))
        pure_b_len = max(0, edge)
        if pure_b_len:
            segments.append((axis - pure_b_len, pure_b_len, 1, 1))
        segments.sort(key=lambda item: item[0])

    # 构造自检：分区必须无缝铺满整轴（测试之外的最后一道防线）
    covered = sum(length for _, length, _, _ in segments)
    if covered != axis or not segments:
        raise PlanError(
            f"内部错误：擦除分区覆盖 {covered} 像素，轴长 {axis}"
        )
    return tuple(segments)


def _row_view(buf, width: int, row: int):
    """取一帧 RGB24 缓冲第 row 行的 memoryview（行宽 = width*3 字节）。"""
    start = row * width * 3
    return memoryview(buf)[start:start + width * 3]


def composite_wipe_frame(
    dst: bytearray,
    frame_a: bytes | bytearray | memoryview,
    frame_b: bytes | bytearray | memoryview,
    width: int,
    height: int,
    direction: str,
    edge: int,
    soft_width: int,
) -> None:
    """把 A/B 两帧按擦除掩码合成进 ``dst``（固定画面、移动边界）。

    水平方向按列切片（每行都过一遍同样的列权重段），垂直方向按行
    切片；每段经 :func:`blend_rgb24_frame` 以段权重整数混合，不新增
    像素通路。``dst`` 与两源尺寸必须同为 ``width*height*3``。
    """
    frame_size = width * height * 3
    if len(dst) != frame_size or len(frame_a) != frame_size or len(frame_b) != frame_size:
        raise PlanError(
            f"帧尺寸不一致: dst={len(dst)}, a={len(frame_a)}, "
            f"b={len(frame_b)}, 期望 {frame_size}"
        )
    if width <= 0 or height <= 0:
        raise PlanError(f"宽高必须为正，收到 {width}x{height}")

    segments = wipe_segments(direction, edge, soft_width, width, height)
    horizontal = direction in HORIZONTAL_DIRECTIONS
    row_bytes = width * 3

    if horizontal:
        # 列区间对每一行都生效：外层行、内层权重段，逐段混合
        for row in range(height):
            dst_row = _row_view(dst, width, row)
            a_row = _row_view(frame_a, width, row)
            b_row = _row_view(frame_b, width, row)
            for start, length, wnum, wden in segments:
                seg = slice(start * 3, (start + length) * 3)
                blend_rgb24_frame(dst_row[seg], a_row[seg], b_row[seg], wnum, wden)
        return

    # 垂直方向：段坐标本身就是行区间，同一段内每行权重相同，
    # 外层段、内层行，整行送入混合器
    for start, length, wnum, wden in segments:
        for row in range(start, start + length):
            dst_row = _row_view(dst, width, row)
            a_row = _row_view(frame_a, width, row)
            b_row = _row_view(frame_b, width, row)
            blend_rgb24_frame(
                dst_row[0:row_bytes], a_row[0:row_bytes],
                b_row[0:row_bytes], wnum, wden,
            )
