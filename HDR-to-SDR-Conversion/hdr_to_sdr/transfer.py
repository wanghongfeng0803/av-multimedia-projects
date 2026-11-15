"""传递函数（Transfer Characteristics）数学：PQ / HLG 的 EOTF 与 BT.709 OETF。

设计要点
========
1. 全部公式来自 SMPTE ST 2084（PQ）与 ITU-R BT.2100（HLG），不做经验近似；
   纯函数 + 预计算 1D LUT 两种形态，LUT 只用于加速、数值上可被单测对照。
2. 线性光统一使用**归一化**域：1.0 表示源信号的名义峰值
   （PQ 为 10000 nit，HLG 为标称峰值 L_W）。这样色调映射阶段不需要接触
   绝对 nit 数，峰值归一在探测/流水线层完成。
3. HLG 的 OOTF 需要三通道一起算（依赖 BT.2020 亮度 Y_s），因此 HLG 的
   场景线性→显示线性不做成单通道函数，由 :func:`hlg_display_linear` 承担。
"""

from __future__ import annotations

import math

__all__ = [
    "TRANSFER_PQ",
    "TRANSFER_HLG",
    "pq_eotf_normalized",
    "hlg_inverse_oetf",
    "hlg_display_linear",
    "bt709_oetf",
    "build_eotf_lut",
    "build_oetf_lut",
    "lerp_lut",
]

# 传递函数标识（与 probe 层探测结果对齐的内部常量）
TRANSFER_PQ = "smpte2084"
TRANSFER_HLG = "arib-std-b67"

# ---- PQ（SMPTE ST 2084）常数 ----------------------------------------------
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = (2523.0 / 4096.0) * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = (2413.0 / 4096.0) * 32.0
_PQ_C3 = (2392.0 / 4096.0) * 32.0

# ---- BT.709 OETF 常数 ------------------------------------------------------
# 规范规定的线性段上界。注意：规范两段在该点并非数学严格连续，
# 存在约 2.48e-4 的小跳变（精确连续根在 0.02），实现以规范文本为准。
_BT709_KNEE = 0.018
_HLG_A = 0.17883277
_HLG_B = 0.28466892          # = 1 - 4a
_HLG_C = 0.55991073          # = 0.5 - a*ln(4a)
_HLG_GAMMA = 1.2
# BT.2020 亮度系数（HLG OOTF 计算 Y_s 用）
_BT2020_KR, _BT2020_KG, _BT2020_KB = 0.2627, 0.6780, 0.0593


def pq_eotf_normalized(n: float) -> float:
    """PQ 电光传递逆函数：归一化信号 n∈[0,1] → 线性光（1.0 = 10000 nit）。

    先按 ST 2084 算出绝对亮度 L_p（cd/m²），再除以 10000 归一化。
    """
    n = min(max(n, 0.0), 1.0)
    if n <= 0.0:
        return 0.0
    n_pow = n ** (1.0 / _PQ_M2)
    num = max(n_pow - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * n_pow
    if den <= 0.0:
        return 1.0
    e = num / den
    lp = 10000.0 * (e ** (1.0 / _PQ_M1))
    return lp / 10000.0


def hlg_inverse_oetf(e_prime: float) -> float:
    """HLG 逆 OETF：非线性信号 E'∈[0,1] → 场景线性光 E_s（相对值）。

    分段点 E'=0.5 对应 E_s=1/12。
    """
    e_prime = min(max(e_prime, 0.0), 1.0)
    if e_prime <= 0.5:
        return (e_prime * e_prime) / 3.0
    return (math.exp((e_prime - _HLG_C) / _HLG_A) + _HLG_B) / 12.0


def hlg_display_linear(r_s: float, g_s: float, b_s: float) -> tuple[float, float, float]:
    """HLG OOTF：场景线性 RGB → 显示端归一化线性 RGB。

    BT.2100 定义 L_D = α·Y_s^(γ−1)·E_s，其中 α=L_W/1.2^γ。本函数输出再
    除以 L_W 归一化，α 中的 L_W 恰好约掉，最终形式为
    ``L_norm = Y_s^(γ−1) · E_s / 1.2^γ``；参考白（Y_s=1）落在 1/1.2^γ≈0.80，
    与「HLG 参考白约为峰值 75–80%」的业界共识一致。
    """
    y_s = _BT2020_KR * r_s + _BT2020_KG * g_s + _BT2020_KB * b_s
    if y_s <= 0.0:
        return 0.0, 0.0, 0.0
    factor = (y_s ** (_HLG_GAMMA - 1.0)) / (1.2 ** _HLG_GAMMA)
    return r_s * factor, g_s * factor, b_s * factor


def bt709_oetf(linear: float) -> float:
    """BT.709 OETF：线性光 L∈[0,1] → SDR 非线性信号 V∈[0,1]。

    两段的精确交点由 4.5·L = 1.099·L^0.45−0.099 解出，
    L0 ≈ 0.0180539685（规范文本写的 0.018 是四舍五入值，直接用会产生
    一个码值量级的不连续）。
    """
    if linear <= 0.0:
        return 0.0
    if linear <= _BT709_KNEE:
        return 4.5 * linear
    return 1.099 * (linear ** 0.45) - 0.099


def build_eotf_lut(kind: str, bits_per_sample: int) -> list[float]:
    """预构建逆传递函数 LUT：索引为整型码值，值为归一化线性光。

    :param kind: ``TRANSFER_PQ`` 或 ``TRANSFER_HLG``。
    :param bits_per_sample: 码深 8/10/12/16，LUT 长度为 2^bits。
    :returns: 长度 ``2**bits_per_sample`` 的浮点列表。

    HLG 的 LUT 只覆盖逆 OETF（单通道部分）；OOTF 三通道运算在像素循环里
    按 :func:`hlg_display_linear` 完成，因为它依赖像素亮度，无法 1D 化。
    """
    if bits_per_sample not in (8, 10, 12, 16):
        raise ValueError(f"不支持的码深: {bits_per_sample}")
    size = 1 << bits_per_sample
    lut = [0.0] * size
    if kind == TRANSFER_PQ:
        for code in range(size):
            lut[code] = pq_eotf_normalized(code / (size - 1))
    elif kind == TRANSFER_HLG:
        for code in range(size):
            lut[code] = hlg_inverse_oetf(code / (size - 1))
    else:
        raise ValueError(f"未知传递函数类型: {kind}")
    return lut


def build_oetf_lut(size: int = 4096) -> list[float]:
    """预构建 BT.709 正 OETF LUT，网格覆盖线性光 [0,1]，配合线性插值使用。"""
    if size < 2:
        raise ValueError("OETF LUT 至少需要 2 个表项")
    return [bt709_oetf(i / (size - 1)) for i in range(size)]


def lerp_lut(lut: list[float], x: float) -> float:
    """对均匀网格 LUT 做线性插值。x 会被夹到 [0,1]。"""
    x = min(max(x, 0.0), 1.0)
    pos = x * (len(lut) - 1)
    lo = int(pos)
    if lo >= len(lut) - 1:
        return lut[-1]
    frac = pos - lo
    return lut[lo] * (1.0 - frac) + lut[lo + 1] * frac
