"""生成演示用的合成帧与水印，并对比固定透明度 / 自适应蒙版 / 缩放三种效果。

运行：python demo/generate_demo.py
产物（被 .gitignore 忽略）：
  demo/frame.png       带“关键区域”（亮方块 + 硬边缘）的合成帧
  demo/watermark.png   半透明水印
  demo/out_fixed.png   固定 70% 透明度
  demo/out_adaptive.png 自适应蒙版（关键区域自动变淡）
  demo/out_scaled.png  缩小后锚定右下角
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_overlay import (  # noqa: E402
    Image,
    WatermarkConfig,
    overlay_image,
    write_png,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def build_frame() -> Image:
    """120x80 渐变背景 + 中央白色“关键区域”方块（强边缘）。"""
    width, height = 120, 80
    frame = Image(width, height)
    for y in range(height):
        for x in range(width):
            frame.set_pixel(x, y, 30 + x // 3, 40 + y // 2, 90, 255)
    # 关键区域：x 40..80, y 25..55 的纯白方块，四周都是硬边缘
    for y in range(25, 56):
        for x in range(40, 81):
            frame.set_pixel(x, y, 245, 245, 245, 255)
    return frame


def build_watermark() -> Image:
    """60x30 半透明蓝色水印，带 2px 透明边。"""
    wm = Image(60, 30)
    for y in range(30):
        for x in range(60):
            if x < 2 or y < 2 or x >= 58 or y >= 28:
                wm.set_pixel(x, y, 0, 0, 0, 0)
            else:
                wm.set_pixel(x, y, 20, 120, 230, 200)
    return wm


def main() -> None:
    frame = build_frame()
    watermark = build_watermark()
    write_png(os.path.join(HERE, "frame.png"), frame)
    write_png(os.path.join(HERE, "watermark.png"), watermark)

    fixed, info_fixed = overlay_image(
        frame, watermark, WatermarkConfig(anchor="center", opacity=0.7)
    )
    write_png(os.path.join(HERE, "out_fixed.png"), fixed)

    adaptive, info_adapt = overlay_image(
        frame,
        watermark,
        WatermarkConfig(
            anchor="center",
            opacity=0.9,
            adaptive=True,
            mask_strength=0.9,
            detail_low=16,
            detail_high=96,
            mask_radius=2,
        ),
    )
    write_png(os.path.join(HERE, "out_adaptive.png"), adaptive)

    scaled, info_scaled = overlay_image(
        frame,
        watermark,
        WatermarkConfig(anchor="bottom-right", offset_x=-4, offset_y=-4, target_width=30),
    )
    write_png(os.path.join(HERE, "out_scaled.png"), scaled)

    # 定量验证：关键区域边缘处，自适应版的水印 alpha 贡献应低于固定版
    probe = (50, 40)  # 白色方块内部
    px_fixed = fixed.get_pixel(*probe)
    px_adapt = adaptive.get_pixel(*probe)
    print(f"固定透明度  探针像素 {probe}: {px_fixed}")
    print(f"自适应蒙版  探针像素 {probe}: {px_adapt}")
    print(f"固定版混合像素数: {info_fixed.blended_pixels}")
    print(f"自适应版混合像素数: {info_adapt.blended_pixels}")
    print(f"缩放版水印尺寸/原点: {info_scaled.watermark_size} {info_scaled.origin}")
    print("演示图片已写入 demo/ 目录")


if __name__ == "__main__":
    main()
