"""单条文件处理流水线: 测量 -> 增益 -> 真峰值限幅 -> 复测 -> 写盘."""

from __future__ import annotations

import math
from pathlib import Path

from . import dsp, meter
from .io import load_audio, save_wav

LUFS_TOLERANCE = 0.5  # 验收容差 ±0.5 LU


def process_one(
    input_path: str | Path,
    output_path: str | Path,
    target_lufs: float = -23.0,
    true_peak_db: float = -1.0,
    include_lfe: bool = False,
) -> dict:
    """处理单条音频, 返回报告条目 (dict)。

    一次成片: 只测一次输入、施加一次增益、做一次限幅, 不做 loudnorm 式两遍处理。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    entry: dict = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "status": "failed",
    }
    try:
        x, sr = load_audio(input_path)
        n, ch = x.shape
        entry["input"] = {
            "duration_s": round(n / sr, 3),
            "sample_rate": sr,
            "channels": ch,
        }

        # 1. 测量 (对将要处理的立体声信号测量, 5.1 先下混)
        x_st = dsp.downmix_to_stereo(x, include_lfe=include_lfe)
        m_in = meter.measure_all(x_st, sr)
        entry["input"].update({k: _round(v) for k, v in m_in.items()})

        if not math.isfinite(m_in["lufs"]):
            entry["status"] = "skipped"
            entry["reason"] = "静音或过短, 无法测量综合响度"
            return entry

        # 2. 计算增益
        gain_db = target_lufs - m_in["lufs"]
        y = dsp.apply_gain(x_st, gain_db)

        # 3. 真峰值限幅 (仅当超天花板时介入)
        limiter_gr_db = 0.0
        if meter.true_peak_db(y, sr) > true_peak_db:
            y, limiter_gr_db = dsp.true_peak_limit(y, sr, ceiling_db=true_peak_db)
            # 限幅后仍微量超标 (滤波器边界) 时做静态微调兜底
            tp_after = meter.true_peak_db(y, sr)
            if tp_after > true_peak_db + 0.05:
                trim = true_peak_db - tp_after
                y = dsp.apply_gain(y, trim)
                gain_db += trim

        # 4. 复测输出 (报告用)
        m_out = meter.measure_all(y, sr)
        entry["output"] = {k: _round(v) for k, v in m_out.items()}
        entry["output"]["applied_gain_db"] = round(gain_db, 2)
        entry["output"]["limiter_max_reduction_db"] = round(limiter_gr_db, 2)

        # 5. 写盘
        save_wav(output_path, y, sr)

        entry["status"] = "success"
        dev = m_out["lufs"] - target_lufs
        if math.isfinite(dev) and abs(dev) > LUFS_TOLERANCE:
            entry["warning"] = (
                f"输出响度 {m_out['lufs']:.1f} LUFS 超出目标 ±{LUFS_TOLERANCE} LU "
                f"(限幅介入 {limiter_gr_db:.1f} dB 所致)"
            )
    except Exception as e:
        entry["status"] = "failed"
        entry["error"] = f"{type(e).__name__}: {e}"
    return entry


def _round(v: float) -> float | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 2)
