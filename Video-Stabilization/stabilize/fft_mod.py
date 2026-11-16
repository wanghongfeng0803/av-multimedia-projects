"""一维快速傅里叶变换（纯标准库，零三方依赖）。

供运动估计的环形互相关使用（旋转角度的环形签名匹配）。只实现
迭代版 Cooley–Tukey 基-2 FFT：输入长度补零到 2 的幂，位反转置换 +
原地蝶形运算。旋转签名只有几十到几百点，Python 复数运算完全够用。

约定：
- 正变换不带归一化，逆变换整体除以 N（与 numpy 等主流实现一致）；
- 指数符号：正变换 ``exp(-2πi·kn/N)``，逆变换取共轭；
- 所有输入先补零到不短于自身的最小 2 的幂。
"""

import cmath
import math

__all__ = [
    "next_pow2",
    "fft",
    "ifft",
    "circular_correlation",
    "peak_lag",
]


def next_pow2(n):
    """返回 >= n 的最小 2 的幂；n<=1 时返回 1。"""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_reverse_indices(n):
    """长度 n（2 的幂）的位反转排列下标。"""
    bits = n.bit_length() - 1  # n=8 为 3 位，bit_length()=4 需减 1
    out = [0] * n
    for i in range(n):
        r = 0
        x = i
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        out[i] = r
    return out


def fft(x, inverse=False):
    """一维基-2 迭代 FFT。

    参数
    ----
    x:
        实数或复数序列；长度非 2 的幂时自动补零到下一个 2 的幂。
    inverse:
        ``True`` 执行逆变换（共轭指数，结果整体除以 N）。

    返回
    ----
    长度为 2 的幂的复数 list。
    """
    n = next_pow2(len(x))
    # 用实数 ±1 作指数符号：sign·2j 才是纯虚指数；直接用 ±1j
    # 与 2j 相乘会先消掉虚部（(-1j)(2j)=2），旋转因子退化为实数。
    sign = 1.0 if inverse else -1.0
    a = [complex(v) for v in x] + [0j] * (n - len(x))

    # 位反转置换
    rev = _bit_reverse_indices(n)
    for i, r in enumerate(rev):
        if i < r:
            a[i], a[r] = a[r], a[i]

    # 蝶形：跨度 length 从 2 倍增到 n
    length = 2
    while length <= n:
        half = length // 2
        # 本级基频因子 w=exp(±2πi/length)；第 k 个蝶形用 w^k。
        for start in range(0, n, length):
            for k in range(half):
                w_k = cmath.exp(1j * sign * 2.0 * math.pi * k / length)
                u = a[start + k]
                t = w_k * a[start + k + half]
                a[start + k] = u + t
                a[start + k + half] = u - t
        length <<= 1

    if inverse:
        a = [v / n for v in a]
    return a


def ifft(x):
    """一维逆 FFT，见 :func:`fft`。"""
    return fft(x, inverse=True)


def circular_correlation(a, b):
    """循环互相关 R[k] = Σ_n conj(a[n])·b[n+k]（下标按 N 取模）。

    用 FFT 计算：``R = IFFT(conj(FFT(a)) * FFT(b))``。``b`` 相对
    ``a`` 发生循环右移 d 时，R 在 lag=d 处取得主峰。两序列等长
    （不等长时补零到相同的 2 的幂长度，补零不改变原区间内的峰值位置）。
    """
    n = next_pow2(max(len(a), len(b)))
    fa = fft(list(a) + [0] * (n - len(a)))
    fb = fft(list(b) + [0] * (n - len(b)))
    spec = [conj_a * bk for conj_a, bk in zip((x.conjugate() for x in fa), fb)]
    return [v.real for v in ifft(spec)]


def peak_lag(corr, max_lag=None):
    """在循环互相关序列中找主峰，返回 ``(lag, 峰值, 峰均比)``。

    - lag 取 ``[-N/2, N/2)`` 的循环位移语义（超过半圈的正 lag 折算
      为负 lag，与真实角度范围对齐）；
    - ``max_lag`` 给出时只在 ``|lag| <= max_lag`` 内搜索；
    - 峰均比 = 峰值 / 相关序列绝对值均值，作为置信度：平坦噪声下
      趋近 1，峰越尖锐越大。
    """
    n = len(corr)
    half = n // 2
    limit = half if max_lag is None else min(max_lag, half)
    best_i, best_v = 0, None
    for i, v in enumerate(corr):
        lag = i if i < half else i - n
        if abs(lag) > limit:
            continue
        if best_v is None or v > best_v:
            best_v, best_i = v, i
    lag = best_i if best_i < half else best_i - n
    mean_abs = sum(abs(v) for v in corr) / n
    ratio = (best_v / mean_abs) if mean_abs > 0 else 1.0
    return lag, best_v, ratio
