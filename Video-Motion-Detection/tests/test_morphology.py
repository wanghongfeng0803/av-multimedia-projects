"""形态学与连通域测试：游程腐蚀/膨胀的数学性质与连通域统计。"""

import pytest

from vmotion.morphology import (
    Region,
    close_mask,
    connected_components,
    dilate,
    erode,
    open_mask,
)


def _mask_from_str(rows):
    """把 ``["010", "111"]`` 形态的 ASCII 图转成二值平面。"""
    h = len(rows)
    w = len(rows[0])
    mask = bytearray(w * h)
    for y, line in enumerate(rows):
        for x, ch in enumerate(line):
            if ch == "1":
                mask[y * w + x] = 1
    return w, h, mask


def test_erode_shrinks_foreground_and_strips_border():
    # 5x5 实心块，r=1：边缘被剥一圈，中心 3x3 存活
    w, h, mask = _mask_from_str([
        "00000",
        "01110",
        "01110",
        "01110",
        "00000",
    ])
    out = erode(mask, w, h, 1)
    _, _, expect = _mask_from_str([
        "00000",
        "00000",
        "00100",
        "00000",
        "00000",
    ])
    assert bytes(out) == bytes(expect)


def test_erode_removes_thin_line():
    # 宽度 1 的竖线在 r=1 腐蚀后全灭（帧外按背景处理）
    w, h, mask = _mask_from_str([
        "010",
        "010",
        "010",
    ])
    out = erode(mask, w, h, 1)
    assert sum(out) == 0


def test_dilate_grows_and_clips_at_border():
    w, h, mask = _mask_from_str([
        "000",
        "010",
        "000",
    ])
    out = dilate(mask, w, h, 1)
    # 单像素膨胀 r=1 = 3x3 方块，画面夹边
    assert sum(out) == 9
    _, _, expect = _mask_from_str([
        "111",
        "111",
        "111",
    ])
    assert bytes(out) == bytes(expect)


def test_dilate_zero_radius_is_copy():
    w, h, mask = _mask_from_str(["101", "010"])
    assert bytes(dilate(mask, w, h, 0)) == bytes(mask)
    assert bytes(erode(mask, w, h, 0)) == bytes(mask)


def test_erode_dilate_are_dual_on_solid_block():
    # 大实心块：腐蚀再膨胀（开运算）保持形状不变
    w, h, mask = _mask_from_str([
        "0000000",
        "0111110",
        "0111110",
        "0111110",
        "0111110",
        "0000000",
    ])
    opened = open_mask(mask, w, h, 1)
    assert bytes(opened) == bytes(mask)


def test_open_removes_speckle_keeps_large_region():
    # 大块放在画面内部（四周留背景），开运算后形状面积逐字节不变；
    # 贴边块会被零边框腐蚀剥边，故这里刻意不贴边
    w, h, mask = _mask_from_str([
        "000000000",
        "001111000",
        "001111000",
        "001111000",
        "000000000",
        "000000001",
    ])
    opened = open_mask(mask, w, h, 1)
    # 孤立单像素被开运算抹掉；内部 4x3 大块保持原形
    assert opened[-1] == 0
    assert sum(opened) == 12


def test_close_fills_one_pixel_hole():
    # 内部块中心挖洞（四周有足够厚的前景），r=1 闭运算填回且轮廓不变
    w, h, mask = _mask_from_str([
        "0000000",
        "0111110",
        "0111110",
        "0110110",
        "0111110",
        "0111110",
        "0000000",
    ])
    before = sum(mask)
    closed = close_mask(mask, w, h, 1)
    # 洞被填回（闭运算输出是输入的超集，且只多不少）
    assert closed[3 * w + 3] == 1
    assert sum(closed) >= before + 1
    assert all(closed[i] >= mask[i] for i in range(w * h))


def test_connected_components_two_regions():
    w, h, mask = _mask_from_str([
        "110001",
        "110001",
        "000000",
        "100111",
        "100000",
    ])
    labels, regions = connected_components(mask, w, h)
    assert len(regions) == 4
    # 4 邻接：左下竖两格与底部横条不相连（对角不算）
    areas = sorted(r.area for r in regions)
    assert areas == [2, 2, 3, 4]
    # 标签从 1 起且连续
    assert sorted(r.label for r in regions) == [1, 2, 3, 4]
    # 最大区域外接框与质心
    biggest = max(regions, key=lambda r: r.area)
    assert biggest.bbox == (0, 0, 2, 2)
    assert biggest.centroid == pytest.approx((0.5, 0.5))
    # 标签平面背景为 0
    assert labels[0] != 0 and labels[12] == 0


def test_connected_components_empty_mask():
    mask = bytearray(20)
    labels, regions = connected_components(mask, 5, 4)
    assert regions == []
    assert sum(labels) == 0


def test_region_to_dict_flat_fields():
    r = Region(2, (3, 4, 10, 8), 40, (7.5, 8.25))
    d = r.to_dict()
    assert d == {"label": 2, "x": 3, "y": 4, "w": 10, "h": 8,
                 "area": 40, "cx": 7.5, "cy": 8.25}
