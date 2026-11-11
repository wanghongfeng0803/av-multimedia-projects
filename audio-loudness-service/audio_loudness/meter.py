"""BS.1770 响度 / LRA / 真峰值测量.

综合响度 (Integrated LUFS) 与 LRA 使用 pyloudnorm (BS.1770-4 合规实现),
真峰值按 BS.1770 建议做 4 倍过采样后取峰值 (dBTP)。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

import pyloudnorm

OVERSAMPLE_FACTOR = 4


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """综合响度 (LUFS)。输入 (n, ch) float。静音/过短返回 -inf。"""
    if x.shape[0] < int(0.4 * sr):
        return float("-inf")
    meter = pyloudnorm.Meter(sr, block_size=0.400)
    try:
        return float(meter.integrated_loudness(x))
    except ValueError:
        return float("-inf")


def loudness_range(x: np.ndarray, sr: int) -> float:
    """LRA (LU)。不足 3 秒返回 nan。"""
    if x.shape[0] < int(3.0 * sr):
        return float("nan")
    meter = pyloudnorm.Meter(sr)
    try:
        return float(meter.loudness_range(x))
    except ValueError:
        return float("nan")


def true_peak_db(x: np.ndarray, sr: int) -> float:
    """真峰值 (dBTP)，4 倍过采样。全零信号返回 -inf。"""
    if x.shape[0] < 8:
        return float("-inf")
    x_os = signal.resample_poly(x, OVERSAMPLE_FACTOR, 1, axis=0)
    peak = float(np.max(np.abs(x_os)))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


def measure_all(x: np.ndarray, sr: int) -> dict:
    """一次返回 LUFS / LRA / dBTP。"""
    return {
        "lufs": integrated_loudness(x, sr),
        "lra": loudness_range(x, sr),
        "true_peak_db": true_peak_db(x, sr),
    }
