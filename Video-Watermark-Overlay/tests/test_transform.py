"""旋转/亚像素重采样测试：几何正确性、预乘空间采样与边界情况。"""

import pytest

from watermark_overlay.errors import InvalidParameterError
from watermark_overlay.image import Image
from watermark_overlay.transform import rotated_bounds, warp_watermark

from conftest import solid_image


def _quad_image():
    """2x2 四色图：A红 B绿 / C蓝 D白（全不透明）。"""
    img = Image(2, 2)
    img.set_pixel(0, 0, 255, 0, 0, 255)    # A
    img.set_pixel(1, 0, 0, 255, 0, 255)    # B
    img.set_pixel(0, 1, 0, 0, 255, 255)    # C
    img.set_pixel(1, 1, 255, 255, 255, 255)  # D
    return img


# ---------------------------------------------------------------- 退化与归一化


def test_identity_returns_exact_copy():
    img = _quad_image()
    out, ox, oy = warp_watermark(img, 0.0, 1.0, 1.0)
    assert (ox, oy) == (0, 0)
    assert (out.width, out.height) == (2, 2)
    assert bytes(out.pixels) == bytes(img.pixels)


def test_angle_360_and_negative_normalize_to_identity():
    img = _quad_image()
    for angle in (360.0, -360.0, 720.0):
        out, ox, oy = warp_watermark(img, angle, 1.0, 1.0)
        assert (ox, oy) == (0, 0)
        assert bytes(out.pixels) == bytes(img.pixels)


def test_integer_translation_is_exact():
    img = _quad_image()
    out, ox, oy = warp_watermark(img, 0.0, 1.0, 1.0, dx=2.0, dy=1.0)
    assert (ox, oy) == (2, 1)
    assert (out.width, out.height) == (2, 2)
    assert bytes(out.pixels) == bytes(img.pixels)


def test_warp_does_not_mutate_input():
    img = _quad_image()
    before = bytes(img.pixels)
    warp_watermark(img, 33.0, 1.0, 1.0, dx=0.3, dy=-0.7)
    assert bytes(img.pixels) == before


# ---------------------------------------------------------------- 旋转几何


def test_rotate_90_clockwise_quadrants():
    img = _quad_image()
    out, ox, oy = warp_watermark(img, 90.0, 1.0, 1.0)
    assert (ox, oy) == (0, 0)
    assert (out.width, out.height) == (2, 2)
    # 正向映射 (dx,dy)->(-dy,dx)：C A / D B
    assert out.get_pixel(0, 0) == (0, 0, 255, 255)      # C
    assert out.get_pixel(1, 0) == (255, 0, 0, 255)      # A
    assert out.get_pixel(0, 1) == (255, 255, 255, 255)  # D
    assert out.get_pixel(1, 1) == (0, 255, 0, 255)      # B


def test_rotate_180():
    img = _quad_image()
    out, _, _ = warp_watermark(img, 180.0, 1.0, 1.0)
    assert out.get_pixel(0, 0) == (255, 255, 255, 255)  # D
    assert out.get_pixel(1, 0) == (0, 0, 255, 255)      # C
    assert out.get_pixel(0, 1) == (0, 255, 0, 255)      # B
    assert out.get_pixel(1, 1) == (255, 0, 0, 255)      # A


def test_rotate_minus_90_equals_270():
    img = _quad_image()
    out_neg, _, _ = warp_watermark(img, -90.0, 1.0, 1.0)
    out_pos, _, _ = warp_watermark(img, 270.0, 1.0, 1.0)
    assert bytes(out_neg.pixels) == bytes(out_pos.pixels)


def test_rotate_45_expands_canvas_with_transparent_corners():
    img = _quad_image()
    out, ox, oy = warp_watermark(img, 45.0, 1.0, 1.0)
    # 对角线 ±√2，包围盒约 [-0.41, 2.41] → 原点 (-1,-1)，4x4 画布
    assert (ox, oy) == (-1, -1)
    assert (out.width, out.height) == (4, 4)
    # 画布四角落在旋转后内容之外，必须全透明
    assert out.get_pixel(0, 0)[3] == 0
    assert out.get_pixel(3, 0)[3] == 0
    assert out.get_pixel(0, 3)[3] == 0
    assert out.get_pixel(3, 3)[3] == 0
    # 靠近中心的边缘像素只被部分覆盖，alpha 介于 0 与 255 之间
    assert 0 < out.get_pixel(1, 1)[3] < 255


def test_rotate_45_odd_size_has_opaque_center():
    # 奇数尺寸使几何中心落在像素中心：中心像素被菱形完全覆盖
    img = solid_image(5, 5, (255, 0, 0, 255))
    out, ox, oy = warp_watermark(img, 45.0, 2.5, 2.5)
    assert (ox, oy) == (-2, -2)
    assert (out.width, out.height) == (9, 9)
    assert out.get_pixel(4, 4)[3] == 255
    assert out.get_pixel(0, 0)[3] == 0
    assert out.get_pixel(8, 8)[3] == 0


def test_rotation_anchor_at_corner():
    # 绕左上角 (0,0) 旋转 90°：内容整体转到 x 轴负方向
    img = _quad_image()
    out, ox, oy = warp_watermark(img, 90.0, 0.0, 0.0)
    assert (ox, oy) == (-2, 0)
    assert (out.width, out.height) == (2, 2)
    # 锚点不动：源 (0,0) 的 A 映射到新画布中锚点所在像素
    assert out.get_pixel(1, 0) == (255, 0, 0, 255)


def test_rotated_bounds_matches_warp():
    img = _quad_image()
    for angle in (0.0, 30.0, 90.0, 135.0, 270.0):
        box = rotated_bounds(2, 2, angle, 1.0, 1.0, 0.5, -0.25)
        out, ox, oy = warp_watermark(img, angle, 1.0, 1.0, 0.5, -0.25)
        assert box == (ox, oy, out.width, out.height)


# ---------------------------------------------------------------- 亚像素平移


def test_subpixel_translation_redistributes_alpha():
    img = Image(2, 1)
    img.set_pixel(0, 0, 255, 0, 0, 255)
    img.set_pixel(1, 0, 0, 255, 0, 255)
    out, ox, oy = warp_watermark(img, 0.0, 1.0, 0.5, dx=0.5, dy=0.0)
    assert (ox, oy) == (0, 0)
    assert (out.width, out.height) == (3, 1)
    # 左边缘像素：一半红色一半透明
    r, g, b, a = out.get_pixel(0, 0)
    assert a == 128
    assert r > 250 and g == 0 and b == 0
    # 中间像素：红绿各半
    r, g, b, a = out.get_pixel(1, 0)
    assert a == 255
    assert 120 <= r <= 136 and 120 <= g <= 136
    # 右边缘像素：一半绿色一半透明
    r, g, b, a = out.get_pixel(2, 0)
    assert a == 128
    assert g > 250 and r == 0 and b == 0


# ---------------------------------------------------------------- 预乘 alpha 采样质量


def test_semitransparent_edge_has_no_black_fringe():
    # 半透明纯红旋转 45°：所有非透明像素必须保持纯红色相，
    # 若不在预乘空间采样，透明邻居的黑色会混入使边缘变暗
    img = solid_image(3, 3, (255, 0, 0, 128))
    out, _, _ = warp_watermark(img, 45.0, 1.5, 1.5)
    for y in range(out.height):
        for x in range(out.width):
            r, g, b, a = out.get_pixel(x, y)
            if a > 0:
                assert g == 0 and b == 0, f"({x},{y}) 出现黑边污染: {(r, g, b, a)}"
                assert r == 255


def test_fully_transparent_source_stays_transparent():
    img = solid_image(2, 2, (0, 0, 0, 0))
    out, _, _ = warp_watermark(img, 30.0, 1.0, 1.0, dx=0.4, dy=0.4)
    assert all(out.pixels[i] == 0 for i in range(len(out.pixels)))


# ---------------------------------------------------------------- 参数校验


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angle_deg": float("nan")},
        {"angle_deg": float("inf")},
        {"anchor_x": float("nan")},
        {"anchor_y": "center"},
        {"dx": float("inf")},
        {"dy": None},
    ],
)
def test_warp_rejects_non_finite_params(kwargs):
    img = _quad_image()
    args = {"angle_deg": 0.0, "anchor_x": 1.0, "anchor_y": 1.0}
    args.update(kwargs)
    with pytest.raises(InvalidParameterError):
        warp_watermark(img, **args)
