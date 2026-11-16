"""HDR(PQ/HLG) → SDR(BT.709) 转换工具包。

对外主要入口：
- :func:`hdr_to_sdr.probe.probe_hdr`：探测并**确证** HDR 类型（PQ/HLG），不猜测
- :class:`hdr_to_sdr.frames.FrameProcessor`：解码后/编码前的逐像素转换引擎
- :func:`hdr_to_sdr.pipeline.convert_hdr_to_sdr`：完整双 ffmpeg 管线
- :func:`hdr_to_sdr.cli.main`：命令行入口（``python -m hdr_to_sdr``）
"""

from .errors import (
    HDRConversionError,
    BinaryNotFoundError,
    InputNotFoundError,
    InvalidInputError,
    UnsupportedHDRTypeError,
    FFmpegRunError,
)
from .transfer import (
    TRANSFER_PQ,
    TRANSFER_HLG,
    pq_eotf_normalized,
    hlg_inverse_oetf,
    hlg_display_linear,
    bt709_oetf,
    build_eotf_lut,
    build_oetf_lut,
    lerp_lut,
)
from .colorscience import (
    RANGE_LIMITED,
    RANGE_FULL,
    GAMUT_2020_TO_709,
    apply_gamut_matrix,
    gamut_compress,
    gamut_map_pixel,
    range_constants,
)
from .tonemap import (
    TONE_REINHARD,
    TONE_REINHARD_CLASSIC,
    TONE_HABLE,
    TONE_BT2390,
    TONE_CLIP,
    DEFAULT_TONE_METHOD,
    REINHARD_WHITE_MIN,
    REINHARD_WHITE_MAX,
    DEFAULT_REINHARD_WHITE,
    HABLE_SHARPNESS_MIN,
    HABLE_SHARPNESS_MAX,
    DEFAULT_HABLE_SHARPNESS,
    HABLE_KNEE_RATIO,
    clamp_reinhard_white,
    clamp_hable_sharpness,
    tone_map_pixel,
    reinhard_scale,
    reinhard_classic_scale,
    hable_scale,
    hable_polynomial,
    bt2390_scale,
    build_tone_curve_lut,
    tone_curve_values,
    registered_tone_methods,
    resolve_tone_method,
)
from .encoders import (
    SOFTWARE_ENCODERS,
    HARDWARE_ENCODERS,
    ALL_ENCODERS,
    EncoderSpec,
    encoder_spec,
    list_available_encoders,
    check_encoder_works,
    is_software_encoder,
)
from .probe import HDRType, HDRInfo, probe_hdr, parse_hdr_info
from .frames import FrameProcessor
from .pipeline import ConversionResult, convert_hdr_to_sdr

__all__ = [
    # 异常
    "HDRConversionError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "UnsupportedHDRTypeError",
    "FFmpegRunError",
    # 传递函数
    "TRANSFER_PQ",
    "TRANSFER_HLG",
    "pq_eotf_normalized",
    "hlg_inverse_oetf",
    "hlg_display_linear",
    "bt709_oetf",
    "build_eotf_lut",
    "build_oetf_lut",
    "lerp_lut",
    # 色彩科学
    "RANGE_LIMITED",
    "RANGE_FULL",
    "GAMUT_2020_TO_709",
    "apply_gamut_matrix",
    "gamut_compress",
    "gamut_map_pixel",
    "range_constants",
    # 色调映射
    "TONE_REINHARD",
    "TONE_REINHARD_CLASSIC",
    "TONE_HABLE",
    "TONE_BT2390",
    "TONE_CLIP",
    "DEFAULT_TONE_METHOD",
    "REINHARD_WHITE_MIN",
    "REINHARD_WHITE_MAX",
    "DEFAULT_REINHARD_WHITE",
    "HABLE_SHARPNESS_MIN",
    "HABLE_SHARPNESS_MAX",
    "DEFAULT_HABLE_SHARPNESS",
    "HABLE_KNEE_RATIO",
    "clamp_reinhard_white",
    "clamp_hable_sharpness",
    "tone_map_pixel",
    "reinhard_scale",
    "reinhard_classic_scale",
    "hable_scale",
    "hable_polynomial",
    "bt2390_scale",
    "build_tone_curve_lut",
    "tone_curve_values",
    "registered_tone_methods",
    "resolve_tone_method",
    # 编码器（软编/硬编）
    "SOFTWARE_ENCODERS",
    "HARDWARE_ENCODERS",
    "ALL_ENCODERS",
    "EncoderSpec",
    "encoder_spec",
    "list_available_encoders",
    "check_encoder_works",
    "is_software_encoder",
    # 探测 / 引擎 / 管线
    "HDRType",
    "HDRInfo",
    "probe_hdr",
    "parse_hdr_info",
    "FrameProcessor",
    "ConversionResult",
    "convert_hdr_to_sdr",
]
