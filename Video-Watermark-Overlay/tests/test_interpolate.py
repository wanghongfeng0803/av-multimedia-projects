"""双线性插值缩放（预乘 alpha 空间）测试。"""

import pytest

from watermark_overlay.errors import InvalidParameterError
from watermark_overlay.image import Image
from watermark_overlay.interpolate import bilinear_resize, resolve_target_size


def make_gradient(width: int, height: int) -> Image:
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            img.set_pixel(x, y, x * 10 % 256, y * 10 % 256, (x + y) * 5 % 256, 255)
    return img


def test_resize_keeps_size_returns_equal_copy():
    img = make_gradient(3, 3)
    out = bilinear_resize(img, 3, 3)
    assert (out.width, out.height) == (3, 3)
    assert bytes(out.pixels) == bytes(img.pixels)
    assert out.pixels is not img.pixels  # 是拷贝而非同一缓冲


def test_resize_solid_color_stays_exact():
    img = Image(3, 3)
    for i in range(9):
        img.pixels[i * 4:i * 4 + 4] = bytes((37, 91, 173, 201))
    out = bilinear_resize(img, 7, 5)
    assert (out.width, out.height) == (7, 5)
    for i in range(7 * 5):
        j = i * 4
        assert tuple(out.pixels[j:j + 4]) == (37, 91, 173, 201)


def test_resize_upscale_2x_midpoint_values():
    # 1x2 -> 1x4，半像素中心映射的采样坐标为 [-0.25, 0.25, 0.75, 1.25]
    img = Image(1, 2)
    img.set_pixel(0, 0, 0, 0, 0, 0)
    img.set_pixel(0, 1, 200, 100, 50, 255)
    out = bilinear_resize(img, 1, 4)
    # alpha 严格按双线性权重（边缘 clamp）：不透明像素在 y=1，
    # 采样坐标 [-0.25, 0.25, 0.75, 1.25] 对应权重 [0.75, 0.25, 0.75, 1.0]
    alphas = [out.get_pixel(0, dy)[3] for dy in range(4)]
    assert alphas == [191, 64, 191, 255]
    # 预乘插值保证：只要 alpha>0，反预乘后的颜色贴近源不透明像素颜色
    # （8bit 预乘往返允许 ±1 量化误差）
    for dy in range(4):
        r, g, b, a = out.get_pixel(0, dy)
        if a > 0:
            assert abs(r - 200) <= 1
            assert g == 100 and b == 50


def test_resize_premultiplied_avoids_black_fringe():
    # 左列不透明红、右列全透明黑：半透明插值列不应出现黑边污染红色
    img = Image(2, 1)
    img.set_pixel(0, 0, 255, 0, 0, 255)
    img.set_pixel(1, 0, 0, 0, 0, 0)
    out = bilinear_resize(img, 4, 1)
    # fx = [-0.25, 0.25, 0.75, 1.25]，对红像素的权重经 clamp 后为 [0.25, 0.75, 0.25, 0.0]
    alphas = [out.get_pixel(x, 0)[3] for x in range(4)]
    assert alphas == [64, 191, 64, 0]
    # 非全透明插值像素的反预乘 R 接近 255（8bit 往返 ±1 量化误差）；
    # 若误用直接 alpha 插值，R 会低至 64，出现明显黑边
    for x in range(4):
        r, g, b, a = out.get_pixel(x, 0)
        if a > 0:
            assert r >= 254
            assert g <= 1 and b <= 1


def test_resize_downscale_2x_center_sample():
    # 2x2 四色 -> 1x1：采样中心位于四像素正中，结果是四者均值
    img = Image(2, 2)
    img.set_pixel(0, 0, 0, 0, 0, 255)
    img.set_pixel(1, 0, 100, 100, 100, 255)
    img.set_pixel(0, 1, 200, 200, 200, 255)
    img.set_pixel(1, 1, 255, 255, 255, 255)
    out = bilinear_resize(img, 1, 1)
    r, g, b, a = out.get_pixel(0, 0)
    assert (r, g, b) == (139, 139, 139)  # round((0+100+200+255)/4)=139
    assert a == 255


def test_resize_1px_wide_does_not_read_out_of_bounds():
    img = Image(1, 1)
    img.set_pixel(0, 0, 10, 20, 30, 40)
    out = bilinear_resize(img, 5, 5)
    for i in range(25):
        j = i * 4
        assert tuple(out.pixels[j:j + 4]) == (10, 20, 30, 40)


def test_resize_upscale_negative_sample_uses_floor():
    # 2 -> 3 上采样时首个目标像素的采样坐标为 -1/6：
    # floor(-1/6)=-1（clamp 到 0，权重为正）；若误用 int() 截断会得到负权重，
    # 把结果拉到源像素值域之外
    img = Image(2, 1)
    img.set_pixel(0, 0, 10, 0, 0, 255)
    img.set_pixel(1, 0, 20, 0, 0, 255)
    out = bilinear_resize(img, 3, 1)
    values = [out.get_pixel(x, 0)[0] for x in range(3)]
    # 正确结果（fx=-1/6，floor=-1 后 clamp，wx=5/6）：10*1/6 + 20*5/6 ≈ 18；
    # 误用 int() 截断会产生负权重，得到值域之外的 8
    assert values[0] == 18
    assert all(10 <= v <= 20 for v in values)
    # 末点严格映射到最后一个源像素
    assert values[2] == 20


def test_resize_invalid_dimensions():
    img = Image(2, 2)
    for bad_w, bad_h in ((0, 2), (2, 0), (-1, 2), (2, -3)):
        with pytest.raises(InvalidParameterError):
            bilinear_resize(img, bad_w, bad_h)
    with pytest.raises(TypeError):
        bilinear_resize(img, 2.0, 2)


def test_resolve_target_size_variants():
    img = Image(4, 2)  # 宽高比 2:1
    assert resolve_target_size(img, None, None) == (4, 2)
    assert resolve_target_size(img, 8, 4) == (8, 4)
    assert resolve_target_size(img, 8, None) == (8, 4)
    assert resolve_target_size(img, None, 4) == (8, 4)
    # 极小尺寸也至少为 1
    assert resolve_target_size(img, 1, None) == (1, 1)
