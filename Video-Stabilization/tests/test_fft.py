"""FFT 与循环互相关的纯逻辑测试（不依赖 ffmpeg）。"""

import cmath
import math

import pytest

from stabilize.fft_mod import (
    circular_correlation,
    fft,
    ifft,
    next_pow2,
    peak_lag,
)


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(5) == 8
    assert next_pow2(128) == 128
    assert next_pow2(129) == 256


def test_fft_dc_and_sine():
    # 直流：能量全部落在 0 号 bin
    spec = fft([1.0] * 8)
    assert abs(spec[0] - 8.0) < 1e-9
    assert all(abs(spec[k]) < 1e-9 for k in range(1, 8))

    # 单频正弦：fft[1] 与 fft[N-1] 为共轭峰，其余接近 0
    n = 16
    sig = [math.sin(2 * math.pi * k / n) for k in range(n)]
    spec = fft(sig)
    assert abs(abs(spec[1]) - n / 2) < 1e-9
    assert abs(abs(spec[n - 1]) - n / 2) < 1e-9
    assert all(abs(spec[k]) < 1e-9
               for k in range(n) if k not in (1, n - 1))


def test_fft_ifft_roundtrip():
    x = [0.5, -1.25, 3.0, 4.75, -2.0, 0.0, 1.5, -3.25]
    rec = ifft(fft(x))
    assert len(rec) == len(x)
    for a, b in zip(x, rec):
        assert abs(a - b.real) < 1e-10
        assert abs(b.imag) < 1e-10


def test_fft_pads_non_pow2():
    x = [1.0, 2.0, 3.0]
    spec = fft(x)
    assert len(spec) == 4
    # 补零等价于直接传 4 点
    assert spec == pytest.approx(fft([1.0, 2.0, 3.0, 0.0]), abs=1e-12)


def test_circular_correlation_shift_peak():
    # b 是 a 的循环右移 3 位 → 主峰 lag=+3
    a = [0.1, 0.5, -0.8, 1.2, -0.3, 0.9, -1.1, 0.4,
         0.7, -0.6, 1.0, -0.2, 0.3, -0.9, 0.8, -0.4]
    shift = 3
    b = [a[(k - shift) % len(a)] for k in range(len(a))]
    corr = circular_correlation(a, b)
    lag, peak, ratio = peak_lag(corr)
    assert lag == shift
    assert ratio > 2.0  # 主峰显著高于相关序列均值


def test_circular_correlation_negative_lag():
    # 左移 2 位（右移 N-2）→ 折算后 lag=-2
    a = [math.sin(k * 1.7) + 0.3 * math.cos(k * 0.4) for k in range(32)]
    b = [a[(k + 2) % len(a)] for k in range(len(a))]
    lag, _, _ = peak_lag(circular_correlation(a, b))
    assert lag == -2


def test_peak_lag_respects_max_lag():
    a = [math.sin(k * 0.9) for k in range(32)]
    b = [a[(k - 6) % len(a)] for k in range(len(a))]
    corr = circular_correlation(a, b)
    lag_all, _, _ = peak_lag(corr)
    assert lag_all == 6
    lag_lim, _, _ = peak_lag(corr, max_lag=3)
    assert abs(lag_lim) <= 3


def test_fft_matches_direct_dft():
    # 与暴力 DFT 定义逐 bin 对照（含复数输入）
    x = [0.1 + 0.2j, -1.0, 2.5, -0.75, 1.25, 0.0, -2.0, 0.5]
    n = len(x)
    spec = fft(x)
    for k in range(n):
        expect = sum(
            x[j] * cmath.exp(-2j * math.pi * k * j / n) for j in range(n)
        )
        assert abs(spec[k] - expect) < 1e-9
