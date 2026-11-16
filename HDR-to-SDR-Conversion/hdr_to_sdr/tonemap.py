"""色调映射（Tone Mapping）：HDR 线性光 → SDR 归一化线性光。

设计原则（对应需求「不偏色不过曝」）：

* 所有算法都在**线性光 + 亮度加权**域工作：先算像素亮度 Y，对 Y 做压缩
  得到缩放系数，再同系数作用于 RGB 三通道。相比逐通道独立压缩，这种
  luma-weighted 做法保持通道比例，不会引入色偏或饱和度坍塌。
* 输出配合 :mod:`hdr_to_sdr.colorscience` 的保亮度色域压缩，保证没有
  通道超过 1.0，即 SDR 端不会过曝截断（硬截断仅作为显式可选项保留）。
* ``peak`` 语义统一：输入线性光归一化值达到 ``peak`` 时，输出约落在
  SDR 参考白附近；高于 peak 的高光进入软滚降区间。

映射器注册机制
==============
每种曲线在 :data:`_TONE_REGISTRY` 注册一个 :class:`ToneMapperSpec`
（缩放函数 + 调谐参数的键名/钳制器/默认值）。:func:`tone_map_pixel` 与
:class:`~hdr_to_sdr.frames.FrameProcessor` 只面向注册表编程：**后续新增
曲线只需注册一个 spec，调用方代码（管线/CLI/引擎）零改动**。``clip``
是显式硬截断、不属于压缩曲线家族，保留独立分支不进注册表。

已注册映射器
============
* ``reinhard``（默认，**原亮度压缩模式，行为不变**）：膝盖 0.75·peak
  以下恒等、以上有理滚降，暗部零损失，是本工具的传统曲线。
* ``reinhard_classic``（**经典 Reinhard**）：全局曲线 L/(1+L)，从暗部
  起即平滑压缩，无恒等段；可调白点 ``white``（肩部位置），white 越大
  压缩越温和，white=1 即教科书 L/(1+L)。参数越界会被钳制到合法区间并
  发出 warning。
* ``hable``（**Hable/Filmic**）：与 Reinhard 同膝盖（0.75·peak）同
  「膝下恒等」外框，肩部换成 John Hable Uncharted2 filmic 多项式核，
  肩部更平、远端高光可分辨级数更多；锐度 ``sharpness`` 可调。
  数学形式与取舍见 :func:`hable_scale` 的 docstring。
* ``bt2390``：BT.2390 风格 EETF，膝盖 0.7·peak，与 Reinhard 同核不同膝。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

from .colorscience import BT709_KR, BT709_KG, BT709_KB

__all__ = [
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
    "reinhard_scale",
    "reinhard_classic_scale",
    "hable_scale",
    "bt2390_scale",
    "hable_polynomial",
    "build_tone_curve_lut",
    "tone_curve_values",
    "tone_map_pixel",
    "resolve_tone_method",
    "registered_tone_methods",
]

TONE_REINHARD = "reinhard"
TONE_REINHARD_CLASSIC = "reinhard_classic"
TONE_HABLE = "hable"
TONE_BT2390 = "bt2390"
TONE_CLIP = "clip"
# 默认方法保持原膝盖式 Reinhard，旧调用方行为完全不变
DEFAULT_TONE_METHOD = TONE_REINHARD

# BT.2390 EETF 膝盖起点（归一化），与常见实现（libvmaf/ffmpeg 文档示例）一致
_BT2390_KS = 0.7
# Reinhard 变体膝盖：更靠近峰值，暗部/中间调保留更完整，高光滚降更陡
_REINHARD_KS = 0.75
# Hable 与 Reinhard 共用同一膝盖，保证暗部/中间调行为逐像素一致、
# 高光对比只来自肩部核的差异（对比实验的公平前提）
HABLE_KNEE_RATIO = _REINHARD_KS

# 经典 Reinhard 白点（white）合法区间：相对 peak 的倍数。
# 下界取极小正数（避免除零/无压缩），上界足够覆盖任意 HDR 高光。
REINHARD_WHITE_MIN = 1e-3
REINHARD_WHITE_MAX = 100.0
# 默认白点 = peak 本身（white=1 即教科书 L/(1+L)，peak 处输出 0.5）
DEFAULT_REINHARD_WHITE = 1.0

# ---- Hable Uncharted2 filmic 多项式（John Hable, GDC 2010，公开规范常数） ----
_HABLE_A = 0.15   # 肩部强度（shoulder strength）
_HABLE_B = 0.50   # 线性强度（linear strength）
_HABLE_C = 0.10   # 线性角度（linear angle）
_HABLE_D = 0.20   # 线性段白（linear white）
_HABLE_E = 0.02   # 趾部强度（toe strength）
_HABLE_F = 0.30   # 趾部分母（toe numerator）
# f(x) 的水平渐近线：x→∞ 时二次项系数比 A/A=1，再减 E/F
# W_INF = 1 - 0.02/0.30 = 14/15 ≈ 0.9333333
_HABLE_W_INF = 1.0 - _HABLE_E / _HABLE_F

# Hable 肩部锐度合法区间：c 越大肩部越陡（越接近 Reinhard 的快速收敛），
# 越小肩部越平（远端高光展开越充分、峰值输出越低）。
HABLE_SHARPNESS_MIN = 0.5
HABLE_SHARPNESS_MAX = 64.0
# 默认锐度 8.0：峰值点输出为 Reinhard 的 96.7%（同量级、不更暗），
# 高光带（x∈[1,4]）几何均值局部斜率约 1.34x、可分辨 8bit 级数 33 vs 28，
# 高光细节不弱于 Reinhard（数值依据见 tests/test_tonemap.py 与
# tools/compare_tonemappers.py 的可复算对比）。
DEFAULT_HABLE_SHARPNESS = 8.0


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


def clamp_hable_sharpness(sharpness: float) -> float:
    """把 Hable 肩部锐度钳制到 [MIN, MAX]，越界时发出 warning。

    锐度语义：肩部 filmic 多项式的横向压缩系数 c，c 越大肩部越陡、
    峰值输出越高、远端高光滚降越快。非有限值回退默认
    :data:`DEFAULT_HABLE_SHARPNESS`，保证逐像素计算不产生 NaN/inf。
    """
    try:
        c = float(sharpness)
    except (TypeError, ValueError):
        warnings.warn(
            f"Hable 锐度 {sharpness!r} 不是数值，回退默认 "
            f"{DEFAULT_HABLE_SHARPNESS}",
            stacklevel=2,
        )
        return DEFAULT_HABLE_SHARPNESS
    if not (c == c):
        warnings.warn(
            f"Hable 锐度为 NaN，回退默认 {DEFAULT_HABLE_SHARPNESS}", stacklevel=2
        )
        return DEFAULT_HABLE_SHARPNESS
    if c < HABLE_SHARPNESS_MIN:
        warnings.warn(
            f"Hable 锐度 {sharpness} 低于下限 {HABLE_SHARPNESS_MIN}，已钳制为 "
            f"{HABLE_SHARPNESS_MIN}",
            stacklevel=2,
        )
        return HABLE_SHARPNESS_MIN
    if c > HABLE_SHARPNESS_MAX:
        warnings.warn(
            f"Hable 锐度 {sharpness} 高于上限 {HABLE_SHARPNESS_MAX}，已钳制为 "
            f"{HABLE_SHARPNESS_MAX}",
            stacklevel=2,
        )
        return HABLE_SHARPNESS_MAX
    return c


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


def hable_polynomial(x: float) -> float:
    """Hable Uncharted2 filmic 多项式 f(x)（规范六常数形式）。

    ::

        f(x) = ((x(Ax+CB)+DE) / (x(Ax+B)+DF)) - E/F

    性质（本项目依赖的全部数学事实）：

    * f(0)=0；x>0 时严格单调递增；
    * 水平渐近线 W_INF = 1-E/F = 14/15 ≈ 0.9333，**只趋近不到达**；
    * 原点斜率 f'(0)=5/18 ≈ 0.2778，因此直接全局使用会重度压暗暗部
      （本项目不采用全局形式，理由见 :func:`hable_scale`）。
    """
    if x <= 0.0:
        return 0.0
    numerator = x * (_HABLE_A * x + _HABLE_C * _HABLE_B) + _HABLE_D * _HABLE_E
    denominator = x * (_HABLE_A * x + _HABLE_B) + _HABLE_D * _HABLE_F
    return numerator / denominator - _HABLE_E / _HABLE_F


def hable_scale(
    y: float, peak: float, sharpness: float = DEFAULT_HABLE_SHARPNESS
) -> float:
    """Hable/Filmic 亮度加权缩放系数（膝盖恒等 + filmic 多项式肩部）。

    归一化坐标 x=y/peak，膝盖 K=0.75（与 :func:`reinhard_scale` 同膝），
    肩部锐度 c=sharpness。映射曲线 h(x) 为两段::

        x ≤ K:  h(x) = x                                  （恒等）
        x > K:  h(x) = K + (1-K)·g(x-K)
                g(u) = f(c·u) / W_INF                     （渐近线归一化）

    其中 f 是 :func:`hable_polynomial`，W_INF=14/15 是 f 的水平渐近线。
    缩放系数 s = h(x)/x。

    **为什么不是教科书的 f(kx)/f(k) 全局形式？** 实测 f 增长极慢
    （f(100)≈0.904 仍低于 W_INF），除以有限白点 f(k) 后 x→∞ 的渐近线
    W_INF/f(k) 必然大于 1，峰值之上的高光会突破 SDR 白，违背本项目
    「不过曝」硬约束；而把暗部斜率补到 1 又要求锚点趋于无穷、肩部退化
    成 Reinhard。故采用与 Reinhard 同构的膝盖外框：暗部/中间调逐像素
    恒等（零损失），只把膝盖之上的有理核换成 filmic 多项式核。

    **取舍（相对 Reinhard）**：两者膝下完全相同；肩部 g 用渐近线归一化
    （g(0)=0、g(∞)=1 不到达），默认 c=8 时峰值点 h(1)=0.8457
    （Reinhard 0.8750，为其 96.7%，同量级不更暗），而肩部更平——
    x∈[1,4] 高光带的几何均值局部斜率约 1.34 倍、8bit 可分辨级数
    33 对 28，远端高光（x>2）优势进一步放大。代价是峰值点到 x≈1.4
    区间输出略低于 Reinhard，这是用少量峰值亮度换高光层次的有意选择，
    且差距随 c 调大可收敛到零（c→∞ 时肩部趋近硬肩）。

    数学不变量：h 连续、严格单调、上界严格小于 1、膝下恒等，
    故任意 x≥0 都有 s∈(0, 1]，不依赖后续截断。
    """
    if y <= 0.0 or peak <= 0.0:
        return 1.0
    c = clamp_hable_sharpness(sharpness)
    x = y / peak
    if x <= HABLE_KNEE_RATIO:
        return 1.0
    g = hable_polynomial(c * (x - HABLE_KNEE_RATIO)) / _HABLE_W_INF
    mapped = HABLE_KNEE_RATIO + (1.0 - HABLE_KNEE_RATIO) * g
    return mapped / x


def bt2390_scale(y: float, peak: float) -> float:
    """BT.2390 风格 EETF 缩放系数（膝盖 0.7·peak，与 Reinhard 同核不同膝）。

    膝盖 K=0.7·peak 以下信号原样通过；膝盖以上有理滚降，输出上界为 1
    （SDR 参考白），连续单调、不会硬截断。膝盖比 Reinhard 更低，
    对高光的压缩更积极、整体画面更稳。
    """
    return _knee_curve(y, peak, _BT2390_KS)


# ---- 注册机制：新增曲线只改这里，调用方（pipeline/cli/frames）零改动 -------
# 缩放函数统一签名 (y, peak, **params) -> s；参数键名与 FrameProcessor
# 持有的调谐属性一致，钳制器在引擎构造期调用一次，逐像素路径拿到的
# 已是确定的合法值。
@dataclass(frozen=True)
class ToneMapperSpec:
    """一个色调映射曲线的注册描述。"""

    name: str
    scale_fn: Callable[..., float]
    param_key: str | None
    default: float
    clamp_fn: Callable[[float], float] | None


_TONE_REGISTRY: dict[str, ToneMapperSpec] = {
    TONE_REINHARD: ToneMapperSpec(
        TONE_REINHARD, reinhard_scale, None, 0.0, None
    ),
    TONE_REINHARD_CLASSIC: ToneMapperSpec(
        TONE_REINHARD_CLASSIC,
        reinhard_classic_scale,
        "reinhard_white",
        DEFAULT_REINHARD_WHITE,
        clamp_reinhard_white,
    ),
    TONE_HABLE: ToneMapperSpec(
        TONE_HABLE,
        hable_scale,
        "hable_sharpness",
        DEFAULT_HABLE_SHARPNESS,
        clamp_hable_sharpness,
    ),
    TONE_BT2390: ToneMapperSpec(TONE_BT2390, bt2390_scale, None, 0.0, None),
}


def registered_tone_methods() -> tuple[str, ...]:
    """返回所有已注册的压缩曲线名（不含独立分支的 ``clip``）。"""
    return tuple(_TONE_REGISTRY)


def _resolve_scale(method: str, y: float, peak: float, params: dict) -> float:
    """按注册表分发缩放计算，未知方法抛 ValueError（编程错误，不静默）。"""
    try:
        spec = _TONE_REGISTRY[method]
    except KeyError:
        # 与 resolve_tone_method 的错误类型保持一致，调用方只需接 ValueError
        raise ValueError(f"未知色调映射方法: {method}") from None
    if spec.param_key is None:
        return spec.scale_fn(y, peak)
    return spec.scale_fn(y, peak, params[spec.param_key])


def _is_valid_scale(s: float) -> bool:
    """缩放系数合法判定：有限、非负、不超过 1（不过曝）。"""
    return isinstance(s, float) and math.isfinite(s) and 0.0 < s <= 1.0


def tone_map_pixel(
    r: float,
    g: float,
    b: float,
    method: str = DEFAULT_TONE_METHOD,
    peak: float = 1.0,
    white: float = DEFAULT_REINHARD_WHITE,
    hable_sharpness: float = DEFAULT_HABLE_SHARPNESS,
) -> tuple[float, float, float]:
    """对一个线性光像素做色调映射，返回压缩后的线性 RGB。

    :param method: ``reinhard`` / ``reinhard_classic`` / ``hable`` /
        ``bt2390`` / ``clip``。
    :param peak: 输入归一化峰值（PQ 由源元数据/默认 1000nit 推出，
        HLG 显示峰值归一化后为 1.0）。
    :param white: 仅 ``reinhard_classic`` 使用的白点（相对 peak 的倍数），
        越界由 :func:`clamp_reinhard_white` 钳制并告警；其它方法忽略。
    :param hable_sharpness: 仅 ``hable`` 使用的肩部锐度，越界由
        :func:`clamp_hable_sharpness` 钳制并告警；其它方法忽略。

    异常输入兜底：若某映射器产出非有限/越界系数（理论上钳制后不会发生，
    此处为防御性最后一道闸），**该像素回退到默认 Reinhard 的结果**并发出
    一次 warning，保证单帧坏值不会中断整批任务。
    """
    if method == TONE_CLIP:
        # 显式硬截断：只截断、不做软压缩（保留给对照测试与特殊需求）
        return (
            min(max(r, 0.0), 1.0),
            min(max(g, 0.0), 1.0),
            min(max(b, 0.0), 1.0),
        )

    y = _luma(r, g, b)
    if not math.isfinite(y):
        # 输入像素本身已带 NaN/inf（解码脏数据等）：无法做亮度加权，
        # 警告后把各通道夹到 [0,1] 原样放行，由后续色域/量化环节收敛，
        # 绝不让单个坏像素中断整批任务。
        warnings.warn(
            f"色调映射输入亮度非有限值 ({r!r},{g!r},{b!r})，"
            "该像素按通道夹取到 [0,1] 放行",
            stacklevel=2,
        )
        return (
            min(max(r if math.isfinite(r) else 0.0, 0.0), 1.0),
            min(max(g if math.isfinite(g) else 0.0, 0.0), 1.0),
            min(max(b if math.isfinite(b) else 0.0, 0.0), 1.0),
        )
    params = {"reinhard_white": white, "hable_sharpness": hable_sharpness}
    s = _resolve_scale(method, y, peak, params)
    if not _is_valid_scale(s):
        warnings.warn(
            f"色调映射方法 {method} 在亮度 {y!r} 产出非法系数 {s!r}，"
            f"该像素回退到 {TONE_REINHARD} 结果",
            stacklevel=2,
        )
        s = reinhard_scale(y, peak)
        # 极端情况下 Reinhard 自身输入也异常（非正 peak 已在函数内返回 1.0），
        # 再兜底成恒等系数，保证返回值永远是有限数
        if not _is_valid_scale(s):
            s = 1.0
    out = (r * s, g * s, b * s)
    # 最后一道闸：正常数学路径不会触发，仅防御通道自身带 inf 的脏数据
    if not all(math.isfinite(v) for v in out):
        warnings.warn(
            f"色调映射输出非有限值 {out!r}，该像素回退到 {TONE_REINHARD} 结果",
            stacklevel=2,
        )
        s = reinhard_scale(y, peak)
        out = (
            min(max(r * s, 0.0), 1.0),
            min(max(g * s, 0.0), 1.0),
            min(max(b * s, 0.0), 1.0),
        )
    return out


def resolve_tone_method(name: str) -> str:
    """校验并规范化色调映射方法名，非法值直接抛 ValueError。"""
    name = (name or "").strip().lower()
    if name not in _TONE_REGISTRY and name != TONE_CLIP:
        allowed = "/".join([*_TONE_REGISTRY, TONE_CLIP])
        raise ValueError(f"不支持的色调映射方法: {name}，可选 {allowed}")
    return name


# ---- 公式的查找表（LUT）形态：与解析公式可互相复算 ------------------------
def tone_curve_values(
    method: str,
    samples: int = 1024,
    peak: float = 1.0,
    white: float = DEFAULT_REINHARD_WHITE,
    hable_sharpness: float = DEFAULT_HABLE_SHARPNESS,
    x_max: float = 4.0,
) -> tuple[list[float], list[float]]:
    """在均匀网格上采样整条映射曲线 h(x)，返回 (网格 x, 输出 h(x))。

    网格覆盖归一化亮度 [0, ``x_max``]（默认 4·peak，覆盖两个档的高光），
    用于文档、对比图与外部 LUT 导出；数值与逐像素解析公式同源同参数。
    ``clip`` 不提供曲线表（它不是压缩曲线）。
    """
    method = resolve_tone_method(method)
    if method == TONE_CLIP:
        raise ValueError("clip 是硬截断而非压缩曲线，无映射曲线表")
    if samples < 2:
        raise ValueError("采样数至少为 2")
    if x_max <= 0.0:
        raise ValueError("x_max 必须为正数")
    params = {"reinhard_white": white, "hable_sharpness": hable_sharpness}
    grid = [x_max * i / (samples - 1) for i in range(samples)]
    values = []
    for x in grid:
        y = x * peak
        s = _resolve_scale(method, y, peak, params)
        values.append(y * s)
    return grid, values


def build_tone_curve_lut(
    method: str,
    size: int = 4096,
    peak: float = 1.0,
    white: float = DEFAULT_REINHARD_WHITE,
    hable_sharpness: float = DEFAULT_HABLE_SHARPNESS,
    x_max: float = 4.0,
) -> list[float]:
    """构造归一化亮度网格上的 1D 映射 LUT（h(x) 的等距采样表）。

    与 :mod:`hdr_to_sdr.transfer` 的 :func:`~hdr_to_sdr.transfer.lerp_lut`
    配套：索引域为 [0, ``x_max``] 的均匀网格，查表前把输入线性亮度除以
    ``x_max`` 归一化到 [0,1] 再插值。引擎热路径默认直接用解析公式
    （单次求值仅几次乘除，LUT 主要供外部复算/导出与文档对照）。
    """
    _, values = tone_curve_values(
        method, size, peak, white, hable_sharpness, x_max
    )
    return values
