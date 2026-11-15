"""色调映射（Tone Mapping）：HDR 线性光 → SDR 归一化线性光。

设计原则（对应需求「不偏色不过曝」）：

* 所有算法都在**线性光 + 亮度加权**域工作：先算像素亮度 Y，对 Y 做压缩
  得到缩放系数，再同系数作用于 RGB 三通道。相比逐通道独立压缩，这种
  luma-weighted 做法保持通道比例，不会引入色偏或饱和度坍塌。
* 输出配合 :mod:`hdr_to_sdr.colorscience` 的保亮度色域压缩，保证没有
  通道超过 1.0，即 SDR 端不会过曝截断（硬截断仅作为显式可选项保留）。
* ``peak`` 语义统一：输入线性光归一化值达到 ``peak`` 时，输出约落在
  SDR 参考白附近；高于 peak 的高光进入软滚降区间。
"""

from __future__ import annotations

from .colorscience import BT709_KR, BT709_KG, BT709_KB

__all__ = [
    "TONE_REINHARD",
    "TONE_BT2390",
    "TONE_CLIP",
    "DEFAULT_TONE_METHOD",
    "reinhard_scale",
    "bt2390_scale",
    "tone_map_pixel",
    "resolve_tone_method",
]

TONE_REINHARD = "reinhard"
TONE_BT2390 = "bt2390"
TONE_CLIP = "clip"
DEFAULT_TONE_METHOD = TONE_REINHARD

# BT.2390 EETF 膝盖起点（归一化），与常见实现（libvmaf/ffmpeg 文档示例）一致
_BT2390_KS = 0.7


def _luma(r: float, g: float, b: float) -> float:
    return BT709_KR * r + BT709_KG * g + BT709_KB * b


# BT.2390 EETF 膝盖起点（归一化，相对 peak）
_BT2390_KS = 0.7
# Reinhard 变体膝盖：更靠近峰值，暗部/中间调保留更完整，高光滚降更陡
_REINHARD_KS = 0.75


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
) -> tuple[float, float, float]:
    """对一个线性光像素做色调映射，返回压缩后的线性 RGB。

    :param method: ``reinhard`` / ``bt2390`` / ``clip``。
    :param peak: 输入归一化峰值（PQ 由源元数据/默认 1000nit 推出，
        HLG 显示峰值归一化后为 1.0）。
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
    elif method == TONE_BT2390:
        s = bt2390_scale(y, peak)
    else:
        raise ValueError(f"未知色调映射方法: {method}")
    return r * s, g * s, b * s


def resolve_tone_method(name: str) -> str:
    """校验并规范化色调映射方法名，非法值直接抛 ValueError。"""
    name = (name or "").strip().lower()
    if name not in (TONE_REINHARD, TONE_BT2390, TONE_CLIP):
        raise ValueError(
            f"不支持的色调映射方法: {name}，"
            f"可选 {TONE_REINHARD}/{TONE_BT2390}/{TONE_CLIP}"
        )
    return name
