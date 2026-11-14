"""水印叠加引擎：把缩放、定位、自适应蒙版与 src-over 混合编排成一条流水线。

核心入口 :func:`overlay_image` 只操作内存中的 :class:`Image`，
:func:`overlay_animated_frame` 在此基础上叠加随时间变化的运动
（关键帧轨迹求值 → 旋转/亚像素平移重采样 → 蒙版逐帧重算），
:func:`overlay_files` 负责 PNG 文件读写与结果落盘。引擎不改输入帧，
而是返回一份合成后的新图像。
"""

from __future__ import annotations

from dataclasses import dataclass

from .animation import MotionConfig, frame_to_time
from .blend import apply_watermark
from .config import WatermarkConfig
from .geometry import place_watermark
from .image import Image, read_png, write_png
from .interpolate import bilinear_resize, resolve_target_size
from .mask import build_alpha_map
from .transform import warp_watermark


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
    # 动画叠加时的实际旋转角（已归一化到 [0, 360)）与运动偏移；静态叠加恒为 0
    angle: float = 0.0
    motion_offset: tuple[float, float] = (0.0, 0.0)

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


def overlay_animated_frame(
    frame: Image,
    watermark: Image,
    config: WatermarkConfig | None = None,
    motion: MotionConfig | None = None,
    frame_index: int = 0,
    fps: float = 25.0,
) -> tuple[Image, OverlayResult]:
    """把运动水印按 ``frame_index`` / ``fps`` 对应的时刻叠加到 ``frame`` 上。

    流水线与 :func:`overlay_image` 一致（缩放 → 定位 → 蒙版 → 混合），
    区别在于定位之后按当前时刻的运动状态做旋转 + 亚像素平移重采样：

    1. ``t = frame_index / fps``，对运动轨迹求值得到 (x, y, angle)；
    2. 绕 ``motion.anchor_x/y`` 指定的锚点旋转缩放后的水印，
       旋转与亚像素位移在预乘 alpha 空间双线性重采样；
    3. 旋转锚点落在“静态锚点定位 + 运动偏移”给出的浮点位置，
       画布原点取整不引入定位误差；
    4. 自适应蒙版按当前帧的实际位置/尺寸**逐帧重算**。

    ``motion`` 为 ``None``，或求值后角度为 0、位移为 0 时，
    结果与 :func:`overlay_image` 逐字节一致（退化为静态叠加）。
    """
    # 无运动配置：完全走静态路径，保证零开销与逐字节等价
    if motion is None:
        return overlay_image(frame, watermark, config)

    config = config or WatermarkConfig()
    config.validate()
    motion.validate()
    t = frame_to_time(frame_index, fps)
    state = motion.evaluate(t)

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
            angle=state.angle,
            motion_offset=(state.x, state.y),
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

    # 2. 静态锚点定位（与静态路径相同的基准），运动偏移在其上叠加
    base_origin_x, base_origin_y = place_watermark(
        frame.width,
        frame.height,
        scaled.width,
        scaled.height,
        config.anchor,
        config.offset_x,
        config.offset_y,
    )

    # 3. 旋转 + 亚像素平移重采样（预乘 alpha 空间）
    anchor_px_x = motion.anchor_x * scaled.width
    anchor_px_y = motion.anchor_y * scaled.height
    warped, warp_ox, warp_oy = warp_watermark(
        scaled, state.angle, anchor_px_x, anchor_px_y, state.x, state.y
    )
    origin_x = base_origin_x + warp_ox
    origin_y = base_origin_y + warp_oy

    # 4. 自适应蒙版：按当前帧的实际原点与旋转后尺寸实时重算
    alpha_map = build_alpha_map(
        frame,
        origin_x,
        origin_y,
        warped.width,
        warped.height,
        base_alpha=base_alpha,
        strength=config.mask_strength if config.adaptive else 0.0,
        detail_low=config.detail_low,
        detail_high=config.detail_high,
        radius=config.mask_radius,
    )

    # 5. 预乘 alpha src-over 逐像素混合（内部完成越界裁剪）
    blended = apply_watermark(result_frame, warped, origin_x, origin_y, alpha_map)

    return result_frame, OverlayResult(
        frame.width,
        frame.height,
        (warped.width, warped.height),
        (origin_x, origin_y),
        base_alpha,
        blended_pixels=blended,
        adaptive=config.adaptive,
        angle=state.angle,
        motion_offset=(state.x, state.y),
    )


def overlay_files(
    frame_path: str,
    watermark_path: str,
    output_path: str,
    config: WatermarkConfig | None = None,
    motion: MotionConfig | None = None,
    frame_index: int = 0,
    fps: float = 25.0,
) -> OverlayResult:
    """读入两张 PNG，叠加后写出为 PNG，返回结果元数据。

    给出 ``motion`` 时按 ``frame_index`` / ``fps`` 渲染动画叠加，
    否则走静态路径（行为与旧版本完全一致）。
    """
    frame = read_png(frame_path)
    watermark = read_png(watermark_path)
    if motion is None:
        result, info = overlay_image(frame, watermark, config)
    else:
        result, info = overlay_animated_frame(
            frame, watermark, config, motion, frame_index=frame_index, fps=fps
        )
    write_png(output_path, result)
    return info
