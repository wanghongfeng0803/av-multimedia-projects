"""九宫格锚点定位与越界处理。

九宫格由 3 个水平位置（left/center/right）× 3 个垂直位置
（top/center/bottom）组合而成。锚点决定水印（或其对齐盒）的基准位置，
``offset_x / offset_y`` 再做像素级微调（x 向右为正、y 向下为正）。

定位阶段**不做裁剪**：水印可以部分或完全落在帧外，返回的原点允许为负；
真正的逐像素越界裁剪在 :func:`watermark_overlay.blend.apply_watermark` 中完成。
"""

from __future__ import annotations

from .errors import InvalidParameterError

ANCHOR_TOP_LEFT = "top-left"
ANCHOR_TOP_CENTER = "top-center"
ANCHOR_TOP_RIGHT = "top-right"
ANCHOR_CENTER_LEFT = "center-left"
ANCHOR_CENTER = "center"
ANCHOR_CENTER_RIGHT = "center-right"
ANCHOR_BOTTOM_LEFT = "bottom-left"
ANCHOR_BOTTOM_CENTER = "bottom-center"
ANCHOR_BOTTOM_RIGHT = "bottom-right"

ANCHORS = frozenset(
    {
        ANCHOR_TOP_LEFT,
        ANCHOR_TOP_CENTER,
        ANCHOR_TOP_RIGHT,
        ANCHOR_CENTER_LEFT,
        ANCHOR_CENTER,
        ANCHOR_CENTER_RIGHT,
        ANCHOR_BOTTOM_LEFT,
        ANCHOR_BOTTOM_CENTER,
        ANCHOR_BOTTOM_RIGHT,
    }
)


def place_watermark(
    frame_width: int,
    frame_height: int,
    watermark_width: int,
    watermark_height: int,
    anchor: str,
    offset_x: int = 0,
    offset_y: int = 0,
) -> tuple[int, int]:
    """计算水印左上角在帧坐标系中的原点（可为负，表示越界）。"""
    if anchor not in ANCHORS:
        raise InvalidParameterError(
            f"未知锚点 {anchor!r}，可选值：{', '.join(sorted(ANCHORS))}"
        )

    vertical, _, horizontal = anchor.partition("-")

    if horizontal == "left":
        x = 0
    elif horizontal == "right":
        x = frame_width - watermark_width
    else:  # center
        x = (frame_width - watermark_width) // 2

    if vertical == "top":
        y = 0
    elif vertical == "bottom":
        y = frame_height - watermark_height
    else:  # center
        y = (frame_height - watermark_height) // 2

    return x + offset_x, y + offset_y
