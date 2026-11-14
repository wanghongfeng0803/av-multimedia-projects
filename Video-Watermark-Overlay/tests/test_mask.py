"""自适应蒙版：亮度、Sobel 边缘、局部亮度变化与 alpha 系数图测试。"""

from watermark_overlay.image import Image
from watermark_overlay.mask import build_alpha_map, build_detail_map, luma


def flat_frame(width: int, height: int, color=(0, 0, 0, 255)) -> Image:
    img = Image(width, height)
    for i in range(width * height):
        j = i * 4
        img.pixels[j:j + 4] = bytes(color)
    return img


def half_black_half_white(width: int = 6, height: int = 4) -> Image:
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            value = 255 if x >= width // 2 else 0
            img.set_pixel(x, y, value, value, value, 255)
    return img


def test_luma_rec601_weights():
    assert luma(0, 0, 0) == 0
    assert luma(255, 255, 255) == 255
    assert luma(255, 0, 0) == 76
    # 绿色亮度权重最高
    assert luma(0, 255, 0) > luma(255, 0, 0) > luma(0, 0, 255)


def test_detail_map_flat_image_is_zero():
    detail = build_detail_map(flat_frame(6, 4, (120, 60, 200, 255)))
    assert max(detail) == 0
    assert len(detail) == 24


def test_detail_map_detects_hard_edge():
    frame = half_black_half_white()
    detail = build_detail_map(frame)
    # 边界两列（x=2、x=3）局部亮度偏差最大（窗口 3:2 时 |偏差|*2=204）
    assert detail[1 * 6 + 2] >= 200
    assert detail[2 * 6 + 3] >= 200
    # 远离边界的平坦区域为 0
    assert detail[0] == 0
    assert detail[6] == 0


def test_alpha_map_uniform_when_strength_zero():
    frame = half_black_half_white()
    alpha_map = build_alpha_map(
        frame, 0, 0, 6, 4, base_alpha=180, strength=0.0,
        detail_low=24, detail_high=96,
    )
    # 矩形内统一系数，矩形外为 0
    assert set(alpha_map) == {180}


def test_alpha_map_darkens_edge_region():
    frame = half_black_half_white()
    alpha_map = build_alpha_map(
        frame, 0, 0, 6, 4, base_alpha=255, strength=0.5,
        detail_low=24, detail_high=96, radius=2,
    )
    # 平坦区域保持不透明
    assert alpha_map[6 + 0] == 255
    # 边界细节最强处：t=1，系数 = 255*(1-0.5) = 128
    assert alpha_map[6 + 2] == 128
    assert alpha_map[6 + 3] == 128


def test_alpha_map_respects_rectangle_and_clipping():
    frame = flat_frame(6, 4)
    # 原点带负值且水印超出帧边界
    alpha_map = build_alpha_map(
        frame, -2, -1, 4, 4, base_alpha=200, strength=0.0,
        detail_low=24, detail_high=96,
    )
    # 覆盖区域 x: 0..1, y: 0..2
    assert alpha_map[0] == 200
    assert alpha_map[1] == 200
    assert alpha_map[2] == 0  # x=2 在水印矩形外
    assert alpha_map[3 * 6] == 0  # y=3 在水印矩形外


def test_alpha_map_no_overlap_returns_all_zero():
    frame = flat_frame(4, 4)
    alpha_map = build_alpha_map(
        frame, 10, 10, 2, 2, base_alpha=255, strength=0.8,
        detail_low=24, detail_high=96,
    )
    assert max(alpha_map) == 0


def test_alpha_map_full_strength_makes_edge_invisible():
    frame = half_black_half_white()
    alpha_map = build_alpha_map(
        frame, 0, 0, 6, 4, base_alpha=255, strength=1.0,
        detail_low=24, detail_high=96,
    )
    # strength=1.0 时最强细节处系数归零
    assert alpha_map[6 + 2] == 0
    # 平坦区域不受影响
    assert alpha_map[6 + 0] == 255
