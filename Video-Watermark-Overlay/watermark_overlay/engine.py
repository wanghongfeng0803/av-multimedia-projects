"""水印叠加引擎：把缩放、定位、自适应蒙版与 src-over 混合编排成一条流水线。

核心入口 :func:`overlay_image` 只操作内存中的 :class:`Image`，
:func:`overlay_files` 负责 PNG 文件读写与结果落盘。引擎不改输入帧，
而是返回一份合成后的新图像。
"""

from __future__ import annotations

from dataclasses import dataclass

from .blend import apply_watermark
from .config import WatermarkConfig
from .geometry import place_watermark
from .image import Image, read_png, write_png
from .interpolate import bilinear_resize, resolve_target_size
from .mask import build_alpha_map


@dataclass
class OverlayResult:
    """一次叠加的结构化结果，便于 CLI 与调用方检查。"""

    frame_width: int
    frame_height: int
    watermark_size: tuple[int, int]
    origin: tuple[int, int]
    base_alpha: int
    blended_pixels: int
    adaptive: bool

    @property
    def overlapped(self) -> bool:
        """水印与帧是否存在可见交集。"""
        return self.blended_pixels > 0


def overlay_image(
    frame: Image,
    watermark: Image,
    config: WatermarkConfig | None = None,
) -> tuple[Image, OverlayResult]:
    """按 ``config`` 把 ``watermark`` 叠加到 ``frame`` 上。

    返回 (新帧, 结果元数据)。覆盖以下边界情况：

    - 全局透明度为 0：直接返回帧拷贝，不做任何混合；
    - 水印缩放后大于帧：允许超帧，仅混合落在帧内的像素；
    - 原点为负或完全落在帧外：逐像素裁剪，无交集时返回 0 混合像素；
    - 水印自身存在全透明像素：混合层快速跳过。
    """
    config = config or WatermarkConfig()
    config.validate()

    base_alpha = round(config.opacity * 255)
    result_frame = frame.copy()

    # 全透明水印 / 全透明配置：结果与原帧完全一致
    if base_alpha == 0:
        return result_frame, OverlayResult(
            frame.width,
            frame.height,
            (watermark.width, watermark.height),
            (0, 0),
            base_alpha,
            blended_pixels=0,
            adaptive=config.adaptive,
        )

    # 1. 双线性缩放到目标尺寸（预乘 alpha 空间）
    dst_w, dst_h = resolve_target_size(
        watermark, config.target_width, config.target_height
    )
    scaled = (
        watermark.copy()
        if (dst_w, dst_h) == (watermark.width, watermark.height)
        else bilinear_resize(watermark, dst_w, dst_h)
    )

    # 2. 九宫格锚点 + 偏移定位（原点允许越界）
    origin_x, origin_y = place_watermark(
        frame.width,
        frame.height,
        scaled.width,
        scaled.height,
        config.anchor,
        config.offset_x,
        config.offset_y,
    )

    # 3. 自适应蒙版：生成帧大小的 alpha 系数图
    alpha_map = build_alpha_map(
        frame,
        origin_x,
        origin_y,
        scaled.width,
        scaled.height,
        base_alpha=base_alpha,
        strength=config.mask_strength if config.adaptive else 0.0,
        detail_low=config.detail_low,
        detail_high=config.detail_high,
        radius=config.mask_radius,
    )

    # 4. 预乘 alpha src-over 逐像素混合（内部完成越界裁剪）
    blended = apply_watermark(result_frame, scaled, origin_x, origin_y, alpha_map)

    return result_frame, OverlayResult(
        frame.width,
        frame.height,
        (scaled.width, scaled.height),
        (origin_x, origin_y),
        base_alpha,
        blended_pixels=blended,
        adaptive=config.adaptive,
    )


def overlay_files(
    frame_path: str,
    watermark_path: str,
    output_path: str,
    config: WatermarkConfig | None = None,
) -> OverlayResult:
    """读入两张 PNG，叠加后写出为 PNG，返回结果元数据。"""
    frame = read_png(frame_path)
    watermark = read_png(watermark_path)
    result, info = overlay_image(frame, watermark, config)
    write_png(output_path, result)
    return info
