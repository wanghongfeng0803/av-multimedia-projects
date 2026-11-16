"""采样模块纯逻辑测试：区域解析、中位数统计；帧解码部分留给集成测试。"""

import pytest

from vkeyer.sampler import median_rgb, region_pixels
from vkeyer.errors import SampleError


def test_region_pixels_builtin_modes():
    # 全帧
    assert region_pixels(100, 80, "all") == (0, 0, 100, 80)
    # 顶部条带：宽铺满，高 10%
    x0, y0, x1, y1 = region_pixels(100, 80, "top")
    assert (x0, y0, x1) == (0, 0, 100)
    assert y1 == 8
    # 右上角
    x0, y0, x1, y1 = region_pixels(100, 80, "topright")
    assert x0 == 90 and x1 == 100 and y0 == 0 and y1 == 8
    # 左侧条带
    x0, y0, x1, y1 = region_pixels(100, 80, "left")
    assert (x0, x1, y0, y1) == (0, 10, 0, 80)


def test_region_pixels_rect_mode():
    box = region_pixels(100, 100, "rect", rect=[0.25, 0.25, 0.75, 0.75])
    assert box == (25, 25, 75, 75)


def test_region_pixels_rect_clamped_and_min_size():
    # 归一化坐标超界被钳到 [0,1]
    box = region_pixels(100, 100, "rect", rect=[-0.5, 0.0, 2.0, 1.0])
    assert box == (0, 0, 100, 100)
    # 极小区域仍保证至少 1 像素
    box = region_pixels(100, 100, "rect", rect=[0.0, 0.0, 0.001, 0.001])
    x0, y0, x1, y1 = box
    assert x1 > x0 and y1 > y0


def test_region_pixels_rejects_bad_rect():
    with pytest.raises(SampleError):
        region_pixels(100, 100, "rect", rect=[0.5, 0.5, 0.4, 0.6])
    with pytest.raises(SampleError):
        region_pixels(100, 100, "rect", rect=[0, 0, 1])
    with pytest.raises(SampleError):
        region_pixels(100, 100, "nowhere")


def test_median_rgb_basic_odd_count():
    pixels = [(0, 0, 0), (100, 100, 100), (200, 200, 200)]
    assert median_rgb(pixels) == (100, 100, 100)


def test_median_rgb_even_count_uses_low_median():
    # 偶数样本：median_low 取中间两值的低者（[10,20,30,40] → 20），
    # 与标准库 statistics.median_low 同语义
    import statistics
    raw = [10, 20, 30, 40]
    pixels = [(v, 0, 0) for v in raw]
    assert median_rgb(pixels) == (statistics.median_low(raw), 0, 0)


def test_median_rgb_per_channel_independent():
    pixels = [(255, 0, 10), (0, 255, 20), (128, 128, 30)]
    # 各通道独立取中位数
    assert median_rgb(pixels) == (128, 128, 20)


def test_median_rgb_robust_to_outlier():
    # 100 个幕布像素 + 1 个误入的前景亮点，中位数不被带偏
    screen = (0, 200, 0)
    pixels = [screen] * 100 + [(255, 255, 255)]
    assert median_rgb(pixels) == screen


def test_median_rgb_empty_rejected():
    with pytest.raises(SampleError):
        median_rgb([])
