"""九宫格锚点定位与偏移测试。"""

import pytest

from watermark_overlay.errors import InvalidParameterError
from watermark_overlay.geometry import ANCHORS, place_watermark


def test_all_nine_anchors_on_exact_fit():
    # 水印与帧等大时，九个锚点的原点都应为 (0, 0)
    for anchor in ANCHORS:
        assert place_watermark(10, 10, 10, 10, anchor) == (0, 0)


def test_anchor_corners_and_edges():
    # 帧 10x8，水印 4x2
    cases = {
        "top-left": (0, 0),
        "top-center": (3, 0),
        "top-right": (6, 0),
        "center-left": (0, 3),
        "center": (3, 3),
        "center-right": (6, 3),
        "bottom-left": (0, 6),
        "bottom-center": (3, 6),
        "bottom-right": (6, 6),
    }
    for anchor, expected in cases.items():
        assert place_watermark(10, 8, 4, 2, anchor) == expected


def test_offsets_apply_after_anchor():
    assert place_watermark(10, 10, 2, 2, "top-left", 3, -1) == (3, -1)
    assert place_watermark(10, 10, 2, 2, "bottom-right", -1, 2) == (7, 10)


def test_oversized_watermark_origin_may_be_negative():
    # 水印比帧大：center/right/bottom 锚点产生负原点，交由混合层裁剪
    assert place_watermark(4, 4, 8, 8, "center") == (-2, -2)
    assert place_watermark(4, 4, 8, 8, "top-left") == (0, 0)
    assert place_watermark(4, 4, 8, 8, "bottom-right") == (-4, -4)


def test_unknown_anchor_rejected():
    with pytest.raises(InvalidParameterError):
        place_watermark(4, 4, 2, 2, "middle")
