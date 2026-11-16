"""色调映射（Tone Mapping）：HDR 线性光 → SDR 归一化线性光。

设计原则（对应需求「不偏色不过曝」）：

* 所有算法都在**线性光 + 亮度加权**域工作：先算像素亮度 Y，对 Y 做压缩
  得到缩放系数，再同系数作用于 RGB 三通道。相比逐通道独立压缩，这种
  luma-weighted 做法保持通道比例，不会引入色偏或饱和度坍塌。
* 输出配合 :mod:`hdr_to_sdr.colorscience` 的保亮度色域压缩，保证没有
  通道超过 1.0，即 SDR 端不会过曝截断（硬截断仅作为显式可选项保留）。
* ``peak`` 语义统一：输入线性光归一化值达到 ``peak`` 时，输出约落在
  SDR 参考白附近；高于 peak 的高光进入软滚降区间。

两套 Reinhard
=============
* ``reinhard``（默认，**原亮度压缩模式，行为不变**）：膝盖 0.75·peak
  以下恒等、以上有理滚降，暗部零损失，是本工具的传统曲线。
* ``reinhard_classic``（**经典 Reinhard**）：全局曲线 L/(1+L)，从暗部
  起即平滑压缩，无恒等段；可调白点 ``white``（肩部位置），white 越大
  压缩越温和，white=1 即教科书 L/(1+L)。参数越界会被钳制到合法区间并
  发出 warning。
"""

from __future__ import annotations

import warnings

from .colorscience import BT709_KR, BT709_KG, BT709_KB

__all__ = [
    "TONE_REINHARD",
    "TONE_REINHARD_CLASSIC",
    "TONE_BT2390",
    "TONE_CLIP",
    "DEFAULT_TONE_METHOD",
    "REINHARD_WHITE_MIN",
    "REINHARD_WHITE_MAX",
    "DEFAULT_REINHARD_WHITE",
    "clamp_reinhard_white",
    "reinhard_scale",
    "reinhard_classic_scale",
    "bt2390_scale",
    "tone_map_pixel",
    "resolve_tone_method",
]

TONE_REINHARD = "reinhard"
TONE_REINHARD_CLASSIC = "reinhard_classic"
TONE_BT2390 = "bt2390"
TONE_CLIP = "clip"
# 默认方法保持原膝盖式 Reinhard，旧调用方行为完全不变
DEFAULT_TONE_METHOD = TONE_REINHARD

# BT.2390 EETF 膝盖起点（归一化），与常见实现（libvmaf/ffmpeg 文档示例）一致
_BT2390_KS = 0.7
# Reinhard 变体膝盖：更靠近峰值，暗部/中间调保留更完整，高光滚降更陡
_REINHARD_KS = 0.75

# 经典 Reinhard 白点（white）合法区间：相对 peak 的倍数。
# 下界取极小正数（避免除零/无压缩），上界足够覆盖任意 HDR 高光。
REINHARD_WHITE_MIN = 1e-3
REINHARD_WHITE_MAX = 100.0
# 默认白点 = peak 本身（white=1 即教科书 L/(1+L)，peak 处输出 0.5）
DEFAULT_REINHARD_WHITE = 1.0


def clamp_reinhard_white(white: float) -> float:
    """把经典 Reinhard 白点钳制到 [MIN, MAX]，越界时发出 warning。

    白点语义：相对 ``peak`` 的倍数，white 处的高光被收敛到接近 SDR 白。
    非有限值（NaN/inf）同样钳制并提示。返回值可安全用于后续计算，
    保证纯函数路径不会因异常输入产生 NaN 或除零。
    """
    try:
        w = float(white)
    except (TypeError, ValueError):
        warnings.warn(
            f"Reinhard 白点 {white!r} 不是数值，回退默认 {DEFAULT_REINHARD_WHITE}",
            stacklevel=2,
        )
        return DEFAULT_REINHARD_WHITE
    if not (w == w):  # NaN：NaN 与任何比较皆 False，单独捕获
        warnings.warn(
            f"Reinhard 白点为 NaN，回退默认 {DEFAULT_REINHARD_WHITE}", stacklevel=2
        )
        return DEFAULT_REINHARD_WHITE
    if w < REINHARD_WHITE_MIN:
        warnings.warn(
            f"Reinhard 白点 {white} 低于下限 {REINHARD_WHITE_MIN}，已钳制为 "
            f"{REINHARD_WHITE_MIN}",
            stacklevel=2,
        )
        return REINHARD_WHITE_MIN
    if w > REINHARD_WHITE_MAX:
        warnings.warn(
            f"Reinhard 白点 {white} 高于上限 {REINHARD_WHITE_MAX}，已钳制为 "
            f"{REINHARD_WHITE_MAX}",
            stacklevel=2,
        )
        return REINHARD_WHITE_MAX
    return w


def _luma(r: float, g: float, b: float) -> float:
    return BT709_KR * r + BT709_KG * g + BT709_KB * b


def _knee_curve(y: float, peak: float, ks_ratio: float) -> float:
    """膝盖式单调有界压缩的缩放系数，两种方法共用的数学核。

    归一化坐标 x=y/peak，膝盖 K=ks_ratio。数学约束决定了：要让输出严格
    单调、上界为 1（不过曝）、膝盖以下恒等，膝盖必须设在峰值**之下**，
    此时 f(peak)<1、x→∞ 时 f→1。滚降段::

        f(x) = K + (1-K)·(x-K) / ((x-K) + (1-K))
    """
    if y <= 0.0 or peak <= 0.0:
        return 1.0
    x = y / peak
    knee = ks_ratio
    if x <= knee:
        return 1.0
    over = x - knee
    mapped = knee + (1.0 - knee) * over / (over + (1.0 - knee))
    return mapped / x


def reinhard_scale(y: float, peak: float) -> float:
    """luma Reinhard 风格缩放系数（膝盖 0.75·peak 的单调有理滚降）。

    性质：膝盖以下恒等（暗部与中间调零损失）、严格单调（高光不反转）、
    输出上界为 1（不过曝，不依赖后续截断）。相比 :func:`bt2390_scale`
    膝盖更靠近峰值，正常曝光区间保留得更完整，是默认方法。
    """
    return _knee_curve(y, peak, _REINHARD_KS)


def reinhard_classic_scale(
    y: float, peak: float, white: float = DEFAULT_REINHARD_WHITE
) -> float:
    """经典 Reinhard（L/(1+L) 家族）的亮度加权缩放系数。

    先把线性亮度归一化到 peak 域 x=y/peak，再用带白点参数 w 的有界
    有理曲线（单极点 Reinhard，肩部渐近线恒为 1）::

        f(x) = w·x / (1 + w·x)
        缩放系数 s = f(x)/x = w / (1 + w·x)

    与膝盖式 :func:`reinhard_scale` 的区别：本曲线**没有恒等段**，从
    暗部起即平滑压缩，是教科书全局 Reinhard 的可调参数版本：

    * ``white=1`` 时 f(x)=x/(1+x)，即最经典的 Reinhard 全局算子；
    * ``white`` 越大（肩部越靠后）同一亮度的压缩越温和、高光保留越多，
      极限 w→∞ 趋于恒等；越小压缩越重。

    数学性质（与本项目「不过曝」硬约束一致）：x≥0 时 s∈(0, 1]，
    f(x) 严格单调递增、f(0)=0、x→∞ 时 f→1，**任意输入输出都不超过 1**，
    不依赖后续截断。注意 w>1 时暗部斜率 w 会轻微抬升暗部曝光，
    这是单极点有理曲线的固有特性，亮度加权作用于 RGB 故不引入色偏。
    """
    if y <= 0.0 or peak <= 0.0:
        return 1.0
    w = clamp_reinhard_white(white)
    x = y / peak
    return w / (1.0 + w * x)


def bt2390_scale(y: float, peak: float) -> float:
    """BT.2390 风格 EETF 缩放系数（膝盖 0.7·peak，与 Reinhard 同核不同膝）。

    膝盖 K=0.7·peak 以下信号原样通过；膝盖以上有理滚降，输出上界为 1
    （SDR 参考白），连续单调、不会硬截断。膝盖比 Reinhard 更低，
    对高光的压缩更积极、整体画面更稳。
    """
    return _knee_curve(y, peak, _BT2390_KS)


def tone_map_pixel(
    r: float,
    g: float,
    b: float,
    method: str = DEFAULT_TONE_METHOD,
    peak: float = 1.0,
    white: float = DEFAULT_REINHARD_WHITE,
) -> tuple[float, float, float]:
    """对一个线性光像素做色调映射，返回压缩后的线性 RGB。

    :param method: ``reinhard`` / ``reinhard_classic`` / ``bt2390`` / ``clip``。
    :param peak: 输入归一化峰值（PQ 由源元数据/默认 1000nit 推出，
        HLG 显示峰值归一化后为 1.0）。
    :param white: 仅 ``reinhard_classic`` 使用的白点（相对 peak 的倍数），
        越界由 :func:`clamp_reinhard_white` 钳制并告警；其它方法忽略。
    """
    if method == TONE_CLIP:
        # 显式硬截断：只截断、不做软压缩（保留给对照测试与特殊需求）
        return (
            min(max(r, 0.0), 1.0),
            min(max(g, 0.0), 1.0),
            min(max(b, 0.0), 1.0),
        )

    y = _luma(r, g, b)
    if method == TONE_REINHARD:
        s = reinhard_scale(y, peak)
    elif method == TONE_REINHARD_CLASSIC:
        s = reinhard_classic_scale(y, peak, white)
    elif method == TONE_BT2390:
        s = bt2390_scale(y, peak)
    else:
        raise ValueError(f"未知色调映射方法: {method}")
    return r * s, g * s, b * s


def resolve_tone_method(name: str) -> str:
    """校验并规范化色调映射方法名，非法值直接抛 ValueError。"""
    name = (name or "").strip().lower()
    if name not in (
        TONE_REINHARD,
        TONE_REINHARD_CLASSIC,
        TONE_BT2390,
        TONE_CLIP,
    ):
        raise ValueError(
            f"不支持的色调映射方法: {name}，可选 {TONE_REINHARD}/"
            f"{TONE_REINHARD_CLASSIC}/{TONE_BT2390}/{TONE_CLIP}"
        )
    return name
