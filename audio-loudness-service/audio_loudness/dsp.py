"""音频 DSP: 5.1 下混、增益、真峰值限幅器."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

# ITU-R BS.775 下混系数
DOWNMIX_CENTER = 0.7071   # C -> L/R
DOWNMIX_SURROUND = 0.7071  # Ls/Rs -> L/R


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(lin: float) -> float:
    return 20.0 * np.log10(lin) if lin > 0 else float("-inf")


def downmix_to_stereo(x: np.ndarray, include_lfe: bool = False) -> np.ndarray:
    """把 (n, ch) 信号下混为 (n, 2) 立体声。

    5.1 按 ITU-R BS.775:
        Lo = L + 0.7071*C + 0.7071*Ls
        Ro = R + 0.7071*C + 0.7071*Rs
    LFE 默认丢弃 (BS.775 不参与响度下混), include_lfe=True 时以 0.7071 混入。

    声道顺序假定 WAV 标准: FL FR FC LFE SL/BL SR/BR。
    """
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    if ch == 1:
        return np.repeat(x, 2, axis=1)
    if ch == 2:
        return x.copy()
    if ch == 6:
        lfe = DOWNMIX_CENTER * x[:, 3] if include_lfe else 0.0
        lo = x[:, 0] + DOWNMIX_CENTER * x[:, 2] + DOWNMIX_SURROUND * x[:, 4] + lfe
        ro = x[:, 1] + DOWNMIX_CENTER * x[:, 2] + DOWNMIX_SURROUND * x[:, 5] + lfe
        return np.stack([lo, ro], axis=1)
    raise ValueError(f"不支持的声道数: {ch} (支持 1/2/6)")


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return x * db_to_lin(gain_db)


def true_peak_limit(
    x: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 1.5,
    release_ms: float = 100.0,
    os_factor: int = 4,
) -> tuple[np.ndarray, float]:
    """真峰值限幅器 (立体声联动)。

    在 4 倍过采样域计算包络与所需增益, 带前视 (lookahead) 与指数释放,
    增益施加在过采样域后降采样回原采样率, 保证输出真峰值 <= ceiling。

    返回 (限幅后信号, 最大增益衰减量 dB)。若原本未超天花板则原样返回, 衰减 0。
    """
    ceiling_lin = db_to_lin(ceiling_db)
    x_os = signal.resample_poly(x, os_factor, 1, axis=0)
    env = np.max(np.abs(x_os), axis=1)
    if env.max() <= ceiling_lin:
        return x, 0.0

    sr_os = sr * os_factor
    n_os = x_os.shape[0]

    # 瞬时所需增益 (超过天花板处 < 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_inst = np.where(env > ceiling_lin, ceiling_lin / np.maximum(env, 1e-12), 1.0)

    # 前视: 窗口内取最小增益, 让衰减提前到来, 避免硬拐点
    la = max(1, int(lookahead_ms / 1000.0 * sr_os))
    g_look = minimum_filter1d(g_inst, size=2 * la + 1, mode="nearest")

    # 指数释放 (控制率下循环, 避免逐样本 Python 循环)
    hop = 32
    n_ctrl = (n_os + hop - 1) // hop
    g_ctrl_in = g_look[::hop][:n_ctrl]
    decay = np.exp(-hop / (release_ms / 1000.0 * sr_os))
    r = 1.0 - g_ctrl_in  # 衰减量 >= 0
    r_out = np.empty_like(r)
    prev = 0.0
    for i in range(n_ctrl):
        cur = r[i]
        prev = cur if cur > prev * decay else prev * decay
        r_out[i] = prev
    g_ctrl = 1.0 - r_out

    # 控制率增益插值回过采样率
    t_ctrl = np.minimum(np.arange(n_ctrl) * hop, n_os - 1)
    g_os = np.interp(np.arange(n_os), t_ctrl, g_ctrl)

    y_os = x_os * g_os[:, None]
    y = signal.resample_poly(y_os, 1, os_factor, axis=0)
    # resample_poly 长度对齐
    if y.shape[0] > x.shape[0]:
        y = y[: x.shape[0]]
    elif y.shape[0] < x.shape[0]:
        y = np.pad(y, ((0, x.shape[0] - y.shape[0]), (0, 0)))

    max_gr_db = -lin_to_db(float(g_os.min()))
    return y, max_gr_db
