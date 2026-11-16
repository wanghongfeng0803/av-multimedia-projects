"""四方向滑动转场：整数像素定位 + 边界全覆盖，渲染复用帧混合器。

几何模型
========

重叠区第 k 帧，A 与 B 沿同一运动轴**整体平移、互不缩放**，
位移 ``s`` 由 :func:`video_transition.easing.eased_shift` 给出
（``0 <= s <= 轴长``）。两帧都是满幅 ``width*height``，在任意
位移下铺满输出画布的两个互补区域，因此**任何一帧都不存在未覆盖
像素**——边界不露黑边是构造保证，不靠事后补色。

四个方向（名字描述 B 的运动方向）::

    right: B 从左向右推入   s=0 时满幅 B 在左外，s=轴长时 B 就位
    left : B 从右向左推入
    down : B 从上向下推入
    up   : B 从下向上推入

首尾锚点：``s=0`` 的输出逐像素等于 A（与 head 末帧连续），
``s=轴长`` 的输出逐像素等于 B（与 tail 首帧连续）。

与帧混合器的关系
================

滑动不引入第二条像素通路：输出按行切成 A/B 两个连续条带，
每个条带调用 :func:`video_transition.blend.blend_rgb24_frame`，
权重取端点 0/1（``0/1`` 取 A 源、``1/1`` 取 B 源）——淡化与滑动
因此共用同一套确定性整数混合原语，差异只在「取哪个源的哪块
内存」。所有切片都是 :class:`memoryview`，不复制像素。
"""

from __future__ import annotations

from .blend import blend_rgb24_frame
from .easing import eased_shift
from .errors import PlanError

__all__ = [
    "DIRECTION_RIGHT",
    "DIRECTION_LEFT",
    "DIRECTION_DOWN",
    "DIRECTION_UP",
    "DIRECTIONS",
    "HORIZONTAL_DIRECTIONS",
    "VERTICAL_DIRECTIONS",
    "slide_regions",
    "composite_slide_frame",
]

#: B 从左向右推入（A 向右移出）
DIRECTION_RIGHT = "right"
#: B 从右向左推入（A 向左移出）
DIRECTION_LEFT = "left"
#: B 从上向下推入（A 向下移出）
DIRECTION_DOWN = "down"
#: B 从下向上推入（A 向上移出）
DIRECTION_UP = "up"

#: 全部合法方向
DIRECTIONS = (
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_DOWN,
    DIRECTION_UP,
)
#: 水平方向（运动轴为宽度）
HORIZONTAL_DIRECTIONS = (DIRECTION_RIGHT, DIRECTION_LEFT)
#: 垂直方向（运动轴为高度）
VERTICAL_DIRECTIONS = (DIRECTION_DOWN, DIRECTION_UP)


def slide_regions(
    direction: str, shift: int, width: int, height: int
) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    """给定位移，返回输出画布上两个覆盖区域的几何描述。

    返回 ``((起点, 长度, 源), (起点, 长度, 源))``：

    * 水平方向：坐标是**列区间** ``[x0, x0+length)``，对全部行生效；
    * 垂直方向：坐标是**行区间** ``[y0, y0+length)``，对全部列生效；
    * ``源`` 为 ``"A"`` 或 ``"B"``，标注该区域取哪一路素材。

    两个区域长度之和恒等于轴长（水平为宽、垂直为高），即输出
    每个像素恰好被一个源覆盖，无重叠也无空缺。
    """
    if direction not in DIRECTIONS:
        raise PlanError(f"未知滑动方向: {direction!r}，可选 {DIRECTIONS}")
    axis = width if direction in HORIZONTAL_DIRECTIONS else height
    if not 0 <= shift <= axis:
        raise PlanError(f"位移 {shift} 越界，轴长 {axis}")

    # 统一物理模型：s=0 时 B 完全在画布外（输出纯 A），s=轴长时 B 就位
    # （输出纯 B），方向名描述 B 的推入方向。
    if direction == DIRECTION_RIGHT:
        # B 从左向右推入：左边缘由 -W 移到 0，占左 [0, s)，A 占右 [s, W)
        return (0, shift, "B"), (shift, axis - shift, "A")
    if direction == DIRECTION_LEFT:
        # B 从右向左推入：左边缘由 W 移到 0，A 占左 [0, W-s)，B 占右
        return (0, axis - shift, "A"), (axis - shift, shift, "B")
    if direction == DIRECTION_DOWN:
        # B 从上向下推入：占上 [0, s)，A 占下 [s, H)
        return (0, shift, "B"), (shift, axis - shift, "A")
    # DIRECTION_UP：B 从下向上推入，A 占上 [0, H-s)，B 占下
    return (0, axis - shift, "A"), (axis - shift, shift, "B")


def _row_view(buf, width: int, row: int):
    """取一帧 RGB24 缓冲第 row 行的 memoryview（行宽 = width*3 字节）。"""
    start = row * width * 3
    return memoryview(buf)[start:start + width * 3]


def composite_slide_frame(
    dst: bytearray,
    frame_a: bytes | bytearray | memoryview,
    frame_b: bytes | bytearray | memoryview,
    width: int,
    height: int,
    direction: str,
    shift: int,
) -> None:
    """把 A/B 两帧按 ``direction`` 与整数位移 ``shift`` 合成进 ``dst``。

    逐行处理：水平滑动时每行切成左右两段、垂直滑动时整行归属同一源；
    每段经 :func:`blend_rgb24_frame` 以端点权重拷贝（不新增像素通路）。
    ``dst`` 与两源尺寸必须同为 ``width*height*3``。
    """
    frame_size = width * height * 3
    if len(dst) != frame_size or len(frame_a) != frame_size or len(frame_b) != frame_size:
        raise PlanError(
            f"帧尺寸不一致: dst={len(dst)}, a={len(frame_a)}, "
            f"b={len(frame_b)}, 期望 {frame_size}"
        )
    if width <= 0 or height <= 0:
        raise PlanError(f"宽高必须为正，收到 {width}x{height}")
    region0, region1 = slide_regions(direction, shift, width, height)
    row_bytes = width * 3
    horizontal = direction in HORIZONTAL_DIRECTIONS

    if horizontal:
        # 区域坐标是列区间：每行都切成同样的左右两段。
        # 端点位移下其中一段长度为 0，跳过即可——另一段覆盖整行，
        # 两段长度之和恒等于宽度，不会留下未覆盖像素。
        for row in range(height):
            dst_row = _row_view(dst, width, row)
            a_row = _row_view(frame_a, width, row)
            b_row = _row_view(frame_b, width, row)
            for start, length, source in (region0, region1):
                if length == 0:
                    continue
                seg = slice(start * 3, (start + length) * 3)
                # 端点权重：取 A 用 0/1，取 B 用 1/1，混合器内零浮点
                wnum = 0 if source == "A" else 1
                blend_rgb24_frame(dst_row[seg], a_row[seg], b_row[seg], wnum, 1)
        return

    # 垂直方向：区域坐标是行区间，整行来自同一个源。
    # 边界行可能为 0（首帧 region0 满幅、末帧 region1 满幅），
    # 比较运算自然把全部行归入非空的那一段。
    boundary_row = region0[1]  # region0 覆盖 [0, boundary_row)
    for row in range(height):
        source = region0[2] if row < boundary_row else region1[2]
        dst_row = _row_view(dst, width, row)
        a_row = _row_view(frame_a, width, row)
        b_row = _row_view(frame_b, width, row)
        seg = slice(0, row_bytes)
        wnum = 0 if source == "A" else 1
        blend_rgb24_frame(dst_row[seg], a_row[seg], b_row[seg], wnum, 1)