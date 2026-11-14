"""图片水印叠加工具（纯 Python 标准库实现）。

对外主要入口：
- :class:`Image`：RGBA8888 像素缓冲，:func:`read_png` / :func:`write_png` 自研编解码
- :func:`overlay_image` / :func:`overlay_files`：水印叠加主流程
- :class:`WatermarkConfig`：位置/尺寸/透明度/自适应蒙版的可校验配置
- :func:`src_over_pixel`：预乘 alpha 的 src-over 单像素混合
- :func:`bilinear_resize`：预乘 alpha 空间的双线性插值缩放
- :func:`place_watermark`：九宫格锚点定位
"""

from .blend import apply_watermark, src_over_pixel
from .config import WatermarkConfig
from .engine import OverlayResult, overlay_files, overlay_image
from .errors import ImageDecodeError, InvalidParameterError, WatermarkError
from .geometry import ANCHORS, place_watermark
from .image import Image, read_png, write_png
from .interpolate import bilinear_resize, resolve_target_size
from .mask import build_alpha_map, build_detail_map, luma

__all__ = [
    # 主流程
    "overlay_image",
    "overlay_files",
    "OverlayResult",
    # 像素缓冲与 PNG
    "Image",
    "read_png",
    "write_png",
    # 配置与异常
    "WatermarkConfig",
    "WatermarkError",
    "InvalidParameterError",
    "ImageDecodeError",
    # 核心算法
    "src_over_pixel",
    "apply_watermark",
    "bilinear_resize",
    "resolve_target_size",
    "place_watermark",
    "ANCHORS",
    "build_alpha_map",
    "build_detail_map",
    "luma",
]
