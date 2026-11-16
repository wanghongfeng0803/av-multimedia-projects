"""vkeyer：色度键抠绿/蓝幕（零三方依赖），三层架构。

- **层 1 键控**（:mod:`vkeyer.keyer` + :mod:`vkeyer.sampler`）：
  键色从画面采样或显式指定（不写死绿色），RGB 欧氏距离双阈值
  软 alpha + 半透明区溢色抑制（despill，去残绿）；
- **层 2 alpha 后处理**（:mod:`vkeyer.matte`）：阈值硬化、
  choke 收缩、盒式羽化，固定顺序串联，可独立复用到后续羽化/
  背景替换场景；
- **层 3 输出编码**（:mod:`vkeyer.encoder` + :mod:`vkeyer.pipeline`）：
  带 alpha 的 mov/webm/mkv 或 png/webp 序列，亦可合成到纯色/
  素材背景输出普通视频，音轨 ``-c:a copy`` 直通。

整条像素管道统一 rgb24 字节序（R, G, B），alpha 平面单字节，
所有定点运算四舍五入、两端（纯透明/纯不透明）精确无误差。
"""

from .keyer import (
    MAX_DISTANCE,
    Keyer,
    KeyerConfig,
    despill_rgb,
    dominant_channel,
)
from .sampler import (
    DEFAULT_REGION,
    DEFAULT_SAMPLE_FRAMES,
    REGION_MODES,
    median_rgb,
    region_pixels,
    sample_key_color,
)
from .matte import (
    MatteConfig,
    apply_matte_to_frame,
    choke_alpha,
    feather_alpha,
    feather_alpha_row,
    postprocess_alpha,
    reflect101,
    threshold_alpha,
)
from .encoder import (
    ALPHA_UNRELIABLE_WARNING,
    ALPHA_VIDEO_SPECS,
    IMAGE_SEQUENCE_EXTS,
    RGB_VIDEO_SPECS,
    composite_over_frame,
    composite_over_solid,
    composite_rgba_frame,
    is_image_sequence,
    resolve_video_spec,
)
from .pipeline import KeyingResult, key_video
from .probe import find_tool, probe_video
from .errors import (
    BinaryNotFoundError,
    ChromaKeyError,
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    OutputFormatError,
    SampleError,
    StreamTruncatedError,
)

__all__ = [
    # 层1：键控
    "MAX_DISTANCE",
    "Keyer",
    "KeyerConfig",
    "despill_rgb",
    "dominant_channel",
    # 层1：采样
    "DEFAULT_REGION",
    "DEFAULT_SAMPLE_FRAMES",
    "REGION_MODES",
    "median_rgb",
    "region_pixels",
    "sample_key_color",
    # 层2：alpha 后处理
    "MatteConfig",
    "apply_matte_to_frame",
    "choke_alpha",
    "feather_alpha",
    "feather_alpha_row",
    "postprocess_alpha",
    "reflect101",
    "threshold_alpha",
    # 层3：编码与合成
    "ALPHA_VIDEO_SPECS",
    "ALPHA_UNRELIABLE_WARNING",
    "IMAGE_SEQUENCE_EXTS",
    "RGB_VIDEO_SPECS",
    "composite_over_frame",
    "composite_over_solid",
    "composite_rgba_frame",
    "is_image_sequence",
    "resolve_video_spec",
    # 流水线
    "KeyingResult",
    "key_video",
    # 探测
    "find_tool",
    "probe_video",
    # 错误
    "BinaryNotFoundError",
    "ChromaKeyError",
    "FFmpegRunError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "OutputFormatError",
    "SampleError",
    "StreamTruncatedError",
]
