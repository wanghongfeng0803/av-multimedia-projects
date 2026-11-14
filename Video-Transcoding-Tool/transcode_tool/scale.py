"""输出分辨率规划：尺寸表达式解析、宽高比保持、偶数对齐与缩放滤镜构建。

支持的分辨率表达式（大小写/空格不敏感）：

- 显式尺寸 ``"1920x1080"``：在该盒子内**等比缩放**（fit-inside），
  绝不拉伸变形；
- 单边 ``"1280"``（仅给宽）或 ``"0x720"`` / ``"x720"``（仅给高）：
  另一边按源宽高比自动推导；
- 百分比 ``"50%"``：宽高同时缩放到源的 50%；
- 预设名 ``"720p"`` / ``"1080p"`` / ``"4k"`` 等（含 fhd/qhd/uhd 别名）。

必须自行处理、不能直接丢给 ffmpeg 的边界：

1. **yuv420p 硬性要求偶数宽高**：所有计算结果在 Python 侧向下取整为偶数，
   不依赖 ffmpeg 运行期行为，结果可单测、可预期；
2. **宽高比保持**：显式 WxH 也只作为“容纳盒”，按显示宽高比（含 SAR）
   等比放入，避免画面被拉伸；
3. **默认禁止放大**：把小视频放大到 4K 只会浪费码率、不会增加真实细节，
   因此请求超过源分辨率时默认收敛回源尺寸并告警，``allow_upscale=True``
   可显式放开；
4. **极端尺寸兜底**：取整后落到 0/1 像素时抬升到最小合法偶数 2，
   超过 8K（7680）视为笔误直接报错。

纯数值逻辑，不启动外部进程。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "ResolutionPlan",
    "PRESETS",
    "MIN_DIMENSION",
    "MAX_DIMENSION",
    "parse_resolution",
    "even_floor",
    "build_scale_filter",
]

# yuv420p 下合法的最小/最大边长（均为偶数）
MIN_DIMENSION = 2
MAX_DIMENSION = 7680  # 8K 上限，超出视为参数笔误

# 预设：以“高度”为锚点，宽度按源宽高比推导，因此 4:3 源选 1080p 也不会变形。
# 键在查询前会经 _normalize_token 归一化（小写、去空格/下划线/连字符/点）。
PRESETS: dict[str, int] = {
    "240p": 240, "nhd": 240,
    "360p": 360,
    "480p": 480, "sd": 480, "vga": 480,
    "540p": 540, "qhd540": 540,
    "720p": 720, "hd": 720,
    "1080p": 1080, "fhd": 1080, "fullhd": 1080,
    "1440p": 1440, "2k": 1440, "qhd": 1440,
    "2160p": 2160, "4k": 2160, "uhd": 2160,
    "4320p": 4320, "8k": 4320,
}

_PERCENT_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*%\s*$")
_BOX_RE = re.compile(r"^\s*(?P<w>\d+)\s*[x×]\s*(?P<h>\d+)\s*$", re.IGNORECASE)
# 前导 x 的定高写法：x720 / X720（0x720 由 _BOX_RE 处理）
_HEIGHT_LEADING_X_RE = re.compile(r"^\s*[x×]\s*(?P<h>\d+)\s*$", re.IGNORECASE)
_BARE_INT_RE = re.compile(r"^\s*(?P<n>\d+)\s*$")


@dataclass(frozen=True)
class ResolutionPlan:
    """分辨率规划的归一化结果（最终像素尺寸一定是合法偶数）。"""

    width: int
    height: int
    requested: str
    upscaled: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scale_filter(self) -> str:
        """对应的 ffmpeg scale 滤镜（显式宽高，lanczos 高质量重采样）。"""
        return build_scale_filter(self.width, self.height)


def even_floor(value: float) -> int:
    """向下取整到最近的偶数；负数/0 由调用方按 MIN_DIMENSION 兜底。"""
    return int(value) // 2 * 2


def build_scale_filter(width: int, height: int) -> str:
    """构建显式宽高的 scale 滤镜字符串。"""
    return f"scale=width={width}:height={height}:flags=lanczos"


def _normalize_token(text: str) -> str:
    return (
        text.strip().lower()
        .replace("×", "x")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def parse_resolution(
    spec: str,
    src_width: int | None,
    src_height: int | None,
    *,
    allow_upscale: bool = False,
    sar: float = 1.0,
) -> ResolutionPlan:
    """把分辨率表达式解析为最终偶数像素尺寸的规划。

    :param spec: 分辨率表达式（见模块 docstring）
    :param src_width/src_height: 源视频像素尺寸（来自 ffprobe）
    :param allow_upscale: 是否允许输出大于源；默认 False（收敛回源并告警）
    :param sar: 源样本宽高比（非方形像素时用于还原真实显示宽高比）
    :raises ValueError: 表达式无法识别、源尺寸非法或结果超出 8K 上限
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("分辨率表达式必须是非空字符串。")
    if not isinstance(src_width, int) or not isinstance(src_height, int) \
            or isinstance(src_width, bool) or isinstance(src_height, bool) \
            or src_width <= 0 or src_height <= 0:
        raise ValueError(
            f"源尺寸非法，无法规划分辨率: {src_width}x{src_height}。"
        )
    if not isinstance(sar, (int, float)) or sar <= 0:
        sar = 1.0

    requested = spec.strip()
    warnings: list[str] = []
    raw_w, raw_h = _resolve_raw_box(requested, src_width, src_height, float(sar))

    if raw_w <= 0 or raw_h <= 0:
        raise ValueError(f"分辨率表达式 {requested!r} 推导出非正尺寸。")
    if max(raw_w, raw_h) > MAX_DIMENSION:
        raise ValueError(
            f"目标尺寸 {raw_w:g}x{raw_h:g} 超过 {MAX_DIMENSION}px（8K）上限，"
            "请检查分辨率参数。"
        )

    # 默认禁止放大：把等比缩放因子钳制到 <= 1.0
    upscaled = raw_w > src_width or raw_h > src_height
    if upscaled and not allow_upscale:
        factor = min(1.0, src_width / raw_w, src_height / raw_h)
        clamped_w, clamped_h = raw_w * factor, raw_h * factor
        warnings.append(
            f"请求的分辨率 {requested} 会放大源 {src_width}x{src_height}，"
            "已默认保持源尺寸（放大不增加真实细节）；"
            "确需放大请显式开启 allow_upscale。"
        )
        raw_w, raw_h = clamped_w, clamped_h
        upscaled = False

    width, height = _snap_even(raw_w, raw_h, requested, warnings)

    # 源本身奇数且结果与源一致时，沿用既有“奇数取整”告警语义
    if (src_width % 2 or src_height % 2) and (width, height) == (
        even_floor(src_width), even_floor(src_height)
    ):
        warnings.append(
            f"源视频尺寸 {src_width}x{src_height} 含奇数像素，已向下取整为 "
            f"{width}x{height}（yuv420p 要求偶数宽高）。"
        )

    return ResolutionPlan(
        width=width,
        height=height,
        requested=requested,
        upscaled=upscaled,
        warnings=tuple(warnings),
    )


def _resolve_raw_box(
    requested: str, sw: int, sh: int, sar: float
) -> tuple[float, float]:
    """把各类表达式统一换算成未取整的目标像素盒子（保持显示宽高比）。"""
    token = _normalize_token(requested)

    # 1) 预设名：以高度为锚，宽度按显示宽高比推导
    preset_h = PRESETS.get(token)
    if preset_h is not None:
        return _box_from_height(preset_h, sw, sh, sar)

    # 2) 百分比：对像素尺寸整体缩放
    pct = _PERCENT_RE.match(requested)
    if pct:
        factor = float(pct.group("num")) / 100.0
        if factor <= 0:
            raise ValueError("缩放百分比必须是正数。")
        return sw * factor, sh * factor

    # 3) 显式 WxH（含 Wx0 / 0xH）：作为容纳盒等比放入
    box = _BOX_RE.match(requested)
    if box:
        w, h = int(box.group("w")), int(box.group("h"))
        if w > 0 and h > 0:
            factor = min(w / (sw * sar), h / sh)
            return sw * sar * factor, sh * factor
        if w > 0:  # Wx0：定宽
            return _box_from_width(w, sw, sh, sar)
        if h > 0:  # 0xH：定高
            return _box_from_height(h, sw, sh, sar)
        raise ValueError(f"分辨率表达式 {requested!r} 的宽高不能同时为 0。")

    # 4) 单边：裸整数按“定宽”，x720 风格按“定高”
    h_only = _HEIGHT_LEADING_X_RE.match(requested)
    if h_only:
        return _box_from_height(int(h_only.group("h")), sw, sh, sar)
    bare = _BARE_INT_RE.match(requested)
    if bare:
        return _box_from_width(int(bare.group("n")), sw, sh, sar)

    raise ValueError(
        f"无法识别的分辨率表达式 {requested!r}。支持形如 '1920x1080'、"
        "'1280'（定宽）、'x720'（定高）、'50%' 或预设 '720p/1080p/4k'。"
    )


def _box_from_width(target_w: float, sw: int, sh: int, sar: float) -> tuple[float, float]:
    """定宽：高度按显示宽高比推导。"""
    height = target_w * sh / (sw * sar)
    return float(target_w), height


def _box_from_height(target_h: float, sw: int, sh: int, sar: float) -> tuple[float, float]:
    """定高：宽度按显示宽高比推导。"""
    width = target_h * (sw * sar) / sh
    return width, float(target_h)


def _snap_even(
    raw_w: float, raw_h: float, requested: str, warnings: list[str]
) -> tuple[int, int]:
    """向下取整为偶数，并对落到最小尺寸以下的结果抬升兜底。

    偶数取整是 yuv420p 的常规自动处理，不逐次告警；只有结果小到需要
    抬升到最小合法尺寸（属于异常输入）时才明确提示。
    """
    width = even_floor(raw_w)
    height = even_floor(raw_h)

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        old = (width, height)
        width = max(width, MIN_DIMENSION)
        height = max(height, MIN_DIMENSION)
        warnings.append(
            f"分辨率 {requested!r} 取整后为 {old[0]}x{old[1]}，"
            f"已抬升到 yuv420p 最小合法偶数尺寸 {width}x{height}。"
        )
    return width, height
