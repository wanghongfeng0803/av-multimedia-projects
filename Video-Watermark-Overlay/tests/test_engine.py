"""引擎端到端编排测试（缩放/定位/蒙版/混合的组合行为与边界输入）。"""

from watermark_overlay.config import WatermarkConfig
from watermark_overlay.engine import overlay_files, overlay_image
from watermark_overlay.image import Image
from watermark_overlay.interpolate import bilinear_resize


def test_overlay_basic(frame_4x4, red_watermark):
    out, info = overlay_image(frame_4x4, red_watermark, WatermarkConfig(anchor="top-left"))
    assert info.overlapped
    assert info.origin == (0, 0)
    assert info.watermark_size == (2, 2)
    assert info.blended_pixels == 4
    # 覆盖区被修改，未覆盖区保持底色
    assert out.get_pixel(0, 0) != (100, 150, 200, 255)
    assert out.get_pixel(3, 3) == (100, 150, 200, 255)


def test_overlay_does_not_mutate_inputs(frame_4x4, red_watermark):
    frame_before = bytes(frame_4x4.pixels)
    wm_before = bytes(red_watermark.pixels)
    overlay_image(frame_4x4, red_watermark, WatermarkConfig(anchor="top-left"))
    assert bytes(frame_4x4.pixels) == frame_before
    assert bytes(red_watermark.pixels) == wm_before


def test_overlay_opacity_zero_is_identity(frame_4x4, red_watermark):
    out, info = overlay_image(frame_4x4, red_watermark, WatermarkConfig(opacity=0.0))
    assert not info.overlapped
    assert bytes(out.pixels) == bytes(frame_4x4.pixels)


def test_overlay_completely_outside_frame(frame_4x4, red_watermark):
    config = WatermarkConfig(anchor="top-left", offset_x=10, offset_y=10)
    out, info = overlay_image(frame_4x4, red_watermark, config)
    assert info.blended_pixels == 0
    assert not info.overlapped
    assert bytes(out.pixels) == bytes(frame_4x4.pixels)


def test_overlay_negative_offset_partial_clipping(frame_4x4, red_watermark):
    config = WatermarkConfig(anchor="top-left", offset_x=-1, offset_y=-1)
    _, info = overlay_image(frame_4x4, red_watermark, config)
    assert info.origin == (-1, -1)
    assert info.blended_pixels == 1


def test_overlay_oversized_watermark_covers_frame():
    frame = Image(2, 2)
    for i in range(4):
        frame.pixels[i * 4:i * 4 + 4] = bytes((0, 0, 0, 255))
    wm = Image(6, 6)
    for i in range(36):
        wm.pixels[i * 4:i * 4 + 4] = bytes((0, 255, 0, 255))
    config = WatermarkConfig(anchor="center")
    out, info = overlay_image(frame, wm, config)
    assert info.origin == (-2, -2)
    assert info.blended_pixels == 4
    for i in range(4):
        assert out.get_pixel(i % 2, i // 2) == (0, 255, 0, 255)


def test_overlay_with_resize(frame_4x4, red_watermark):
    config = WatermarkConfig(anchor="top-left", target_width=4, target_height=4)
    out, info = overlay_image(frame_4x4, red_watermark, config)
    assert info.watermark_size == (4, 4)
    assert info.blended_pixels == 16
    # 与“先自行缩放再叠加”的结果一致
    scaled = bilinear_resize(red_watermark, 4, 4)
    ref, _ = overlay_image(frame_4x4, scaled, WatermarkConfig(anchor="top-left"))
    assert bytes(out.pixels) == bytes(ref.pixels)


def test_overlay_aspect_ratio_resize(frame_4x4):
    wm = Image(4, 2)
    for i in range(8):
        wm.pixels[i * 4:i * 4 + 4] = bytes((255, 255, 255, 255))
    config = WatermarkConfig(anchor="top-left", target_width=2)
    _, info = overlay_image(frame_4x4, wm, config)
    assert info.watermark_size == (2, 1)


def test_overlay_on_semi_transparent_frame():
    frame = Image(1, 1)
    frame.set_pixel(0, 0, 0, 0, 0, 100)
    wm = Image(1, 1)
    wm.set_pixel(0, 0, 0, 0, 0, 100)
    out, info = overlay_image(frame, wm, WatermarkConfig())
    assert info.blended_pixels == 1
    expected_alpha = 100 + 100 - round(100 * 100 / 255)
    assert out.get_pixel(0, 0)[3] == expected_alpha


def test_overlay_adaptive_differs_on_edge_vs_flat_region():
    # 左半黑、右半白的 8x4 帧
    frame = Image(8, 4)
    for y in range(4):
        for x in range(8):
            v = 255 if x >= 4 else 0
            frame.set_pixel(x, y, v, v, v, 255)
    wm = Image(8, 4)
    for i in range(32):
        wm.pixels[i * 4:i * 4 + 4] = bytes((255, 0, 0, 255))

    fixed, _ = overlay_image(frame, wm, WatermarkConfig())
    adaptive, info = overlay_image(
        frame,
        wm,
        WatermarkConfig(adaptive=True, mask_strength=0.8, detail_low=16, detail_high=96),
    )
    assert info.adaptive
    # 平坦区（x=0）两种方式结果相同；边缘列（x=3/4）自适应版红色贡献更弱
    assert adaptive.get_pixel(0, 0) == fixed.get_pixel(0, 0)
    assert adaptive.get_pixel(3, 0)[0] < fixed.get_pixel(3, 0)[0]


def test_overlay_files_roundtrip(tmp_path, frame_4x4, red_watermark):
    from watermark_overlay.image import write_png, read_png

    frame_p = tmp_path / "frame.png"
    wm_p = tmp_path / "wm.png"
    out_p = tmp_path / "out.png"
    write_png(str(frame_p), frame_4x4)
    write_png(str(wm_p), red_watermark)

    info = overlay_files(
        str(frame_p), str(wm_p), str(out_p), WatermarkConfig(anchor="bottom-right")
    )
    assert out_p.exists()
    result = read_png(str(out_p))
    assert (result.width, result.height) == (4, 4)
    assert info.overlapped
