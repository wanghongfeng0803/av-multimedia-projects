"""预乘 alpha src-over 混合的逐像素与越界行为测试。"""

from watermark_overlay.blend import SCALE, apply_watermark, src_over_pixel
from watermark_overlay.image import Image


def test_src_over_fully_opaque_source_over_opaque_dst():
    # 不透明源直接覆盖目标颜色
    assert src_over_pixel((10, 20, 30, 255), (200, 100, 50, 255)) == (200, 100, 50, 255)


def test_src_over_fully_transparent_source_keeps_dst():
    dst = (10, 20, 30, 200)
    assert src_over_pixel(dst, (255, 255, 255, 0)) == dst


def test_src_over_transparent_destination_returns_source():
    # 目标全透明：结果等于源（反预乘误差为 0）
    assert src_over_pixel((0, 0, 0, 0), (80, 160, 240, 128)) == (80, 160, 240, 128)


def test_src_over_half_alpha_math():
    # 手算：alpha=128 约等于 0.502
    r, g, b, a = src_over_pixel((100, 150, 200, 255), (255, 0, 0, 128))
    assert a == 255
    # 预乘 R：128 + 100*127/255 ≈ 177.8 → 178
    assert r == 178
    assert g == 75
    assert b == 100


def test_src_over_semi_transparent_destination_alpha():
    # out_a = a_s + a_d - a_s*a_d/255
    _, _, _, a = src_over_pixel((0, 0, 0, 100), (0, 0, 0, 100))
    expected = 100 + 100 - round(100 * 100 / 255)
    assert a == expected


def test_apply_watermark_skips_zero_multiplier(frame_4x4, red_watermark):
    alpha_map = bytearray(16)  # 全部为 0
    before = bytes(frame_4x4.pixels)
    count = apply_watermark(frame_4x4, red_watermark, 0, 0, alpha_map)
    assert count == 0
    assert bytes(frame_4x4.pixels) == before


def test_apply_watermark_skips_transparent_watermark_pixels(frame_4x4):
    wm = Image(2, 2)  # 默认全透明
    alpha_map = bytearray([SCALE] * 16)
    before = bytes(frame_4x4.pixels)
    count = apply_watermark(frame_4x4, wm, 0, 0, alpha_map)
    assert count == 0
    assert bytes(frame_4x4.pixels) == before


def test_apply_watermark_negative_origin_clips(frame_4x4, red_watermark):
    # 水印左上角在 (-1, -1)：4 个像素中只有 (0,0) 一个落在帧内
    alpha_map = bytearray([SCALE] * 16)
    count = apply_watermark(frame_4x4, red_watermark, -1, -1, alpha_map)
    assert count == 1
    # 被混合的像素不再等于底色
    assert frame_4x4.get_pixel(0, 0) != (100, 150, 200, 255)
    # 相邻未覆盖像素保持不变
    assert frame_4x4.get_pixel(1, 1) == (100, 150, 200, 255)


def test_apply_watermark_completely_outside(frame_4x4, red_watermark):
    alpha_map = bytearray([SCALE] * 16)
    assert apply_watermark(frame_4x4, red_watermark, 10, 10, alpha_map) == 0
    assert apply_watermark(frame_4x4, red_watermark, -5, 0, alpha_map) == 0


def test_apply_watermark_larger_than_frame_clips():
    frame = Image(2, 2)
    for i in range(4):
        frame.pixels[i * 4 + 3] = 255
    wm = Image(4, 4)
    for i in range(16):
        wm.pixels[i * 4:i * 4 + 4] = bytes((255, 0, 0, 255))
    alpha_map = bytearray([SCALE] * 4)
    count = apply_watermark(frame, wm, -1, -1, alpha_map)
    # 帧的全部 4 个像素都应被混合
    assert count == 4
    for i in range(4):
        assert frame.get_pixel(i % 2, i // 2) == (255, 0, 0, 255)


def test_apply_watermark_multiplier_scales_alpha(frame_4x4, red_watermark):
    # multiplier=128 把水印 alpha 再减半
    full = bytearray([SCALE] * 16)
    half = bytearray([128] * 16)
    img_full = frame_4x4.copy()
    img_half = frame_4x4.copy()
    apply_watermark(img_full, red_watermark, 0, 0, full)
    apply_watermark(img_half, red_watermark, 0, 0, half)
    # 减半后的红通道贡献应明显更弱
    assert img_half.get_pixel(0, 0)[0] < img_full.get_pixel(0, 0)[0]
    # 直接与单像素公式对照（水印有效 alpha = round(128*128/255)=64）
    expected = src_over_pixel((100, 150, 200, 255), (255, 0, 0, 64))
    assert img_half.get_pixel(0, 0) == expected
