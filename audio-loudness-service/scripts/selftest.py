"""端到端自检: 生成合成音频 -> 跑批量处理 -> 校验输出指标.

运行: python scripts/selftest.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_loudness import dsp, meter  # noqa: E402
from audio_loudness.batch import run_batch  # noqa: E402

SR = 48000
TARGET = -23.0
TP_CEIL = -1.0


def gen_quiet_sine(path: Path, dur: float = 8.0) -> None:
    """低响度正弦, 需要提增益。"""
    t = np.arange(int(dur * SR)) / SR
    x = 0.01 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), np.stack([x, x], 1), SR, subtype="PCM_24")


def gen_loud_music(path: Path, dur: float = 8.0) -> None:
    """高响度多音叠加, 需要降增益。"""
    t = np.arange(int(dur * SR)) / SR
    x = sum(0.25 * np.sin(2 * np.pi * f * t) for f in (110, 220, 440, 880))
    x /= np.max(np.abs(x))
    x *= 0.9
    sf.write(str(path), np.stack([x, 0.9 * x], 1), SR, subtype="PCM_24")


def gen_51(path: Path, dur: float = 8.0) -> np.ndarray:
    """5.1 白噪声 (各声道不同电平), 返回原始信号用于下混校验。"""
    rng = np.random.default_rng(42)
    n = int(dur * SR)
    gains = [0.5, 0.4, 0.3, 0.2, 0.25, 0.35]  # FL FR C LFE SL SR
    x = np.stack([rng.standard_normal(n) * g * 0.05 for g in gains], 1)
    sf.write(str(path), x, SR, subtype="PCM_24")
    return x


def gen_hot_peaks(path: Path, dur: float = 8.0) -> None:
    """中等响度主体 + 超幅短脉冲: 提增益后真峰值超 -1 dBTP, 考验限幅器。

    主体噪声承担大部分响度, 限幅只削少量短峰, 响度与真峰值应同时达标。
    """
    rng = np.random.default_rng(7)
    n = int(dur * SR)
    x = rng.standard_normal(n) * 0.02  # 主体约 -31 LUFS, 需要 +8 dB 左右增益
    # 短脉冲 (0.17ms, 每 4s 一个): 提增益后真峰值约 +1.5 dBTP, 能量占比极低
    for start in range(0, n - 8, 4 * SR):
        x[start : start + 8] += 0.75 * np.sin(2 * np.pi * 3000 * np.arange(8) / SR)
    sf.write(str(path), np.stack([x, x], 1), SR, subtype="PCM_24")


def gen_extreme_peaks(path: Path, dur: float = 8.0) -> None:
    """峰值主导响度的极端信号: -1 dBTP 与 -23 LUFS 物理上不可兼得。

    期望行为: 真峰值必须达标 (硬指标), 响度偏差以 warning 形式出现在报告中。
    """
    rng = np.random.default_rng(11)
    n = int(dur * SR)
    x = rng.standard_normal(n) * 0.002
    for start in range(0, n - 32, 2 * SR):
        x[start : start + 32] += 0.98 * np.sin(2 * np.pi * 3000 * np.arange(32) / SR)
    sf.write(str(path), np.stack([x, x], 1), SR, subtype="PCM_24")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="loudness_selftest_"))
    in_dir, out_dir = tmp / "in", tmp / "out"
    in_dir.mkdir(parents=True)

    gen_quiet_sine(in_dir / "01_quiet_sine.wav")
    gen_loud_music(in_dir / "02_loud_music.wav")
    x51 = gen_51(in_dir / "03_surround_51.wav")
    gen_hot_peaks(in_dir / "04_hot_peaks.wav")
    gen_extreme_peaks(in_dir / "05_extreme_peaks.wav")

    # --- 单元校验: 下混公式 ---
    ref_l = x51[:, 0] + 0.7071 * x51[:, 2] + 0.7071 * x51[:, 4]
    ref_r = x51[:, 1] + 0.7071 * x51[:, 2] + 0.7071 * x51[:, 5]
    dm = dsp.downmix_to_stereo(x51)
    assert np.allclose(dm[:, 0], ref_l, atol=1e-6), "下混 L 声道不符"
    assert np.allclose(dm[:, 1], ref_r, atol=1e-6), "下混 R 声道不符"
    print("[OK] 5.1 下混公式校验通过 (ITU-R BS.775, LFE 丢弃)")

    # --- 批量处理 ---
    results = run_batch(
        in_dir, out_dir, target=TARGET, true_peak=TP_CEIL,
        concurrency=4, retries=1, report_path=out_dir / "report",
    )

    failures = []
    for e in results:
        name = Path(e["input_file"]).name
        if e["status"] != "success":
            failures.append(f"{name}: 状态 {e['status']} {e.get('error', '')}")
            continue
        o = e["output"]
        if o["true_peak_db"] > TP_CEIL + 0.05:
            failures.append(f"{name}: 真峰值 {o['true_peak_db']} 超 -1 dBTP")
        if name == "05_extreme_peaks.wav":
            # 极端信号: 真峰值必须达标, 响度偏差必须有 warning 说明
            if "warning" not in e:
                failures.append(f"{name}: 响度超差但报告未给出 warning")
        elif not (TARGET - 0.5 <= o["lufs"] <= TARGET + 0.5):
            failures.append(f"{name}: 输出响度 {o['lufs']} 超出 -23±0.5")
        if name == "04_hot_peaks.wav" and o["limiter_max_reduction_db"] <= 0:
            failures.append(f"{name}: 限幅器未介入, 测试用例无效")
        # 输出确实写盘
        assert Path(e["output_file"]).exists(), f"{name}: 输出文件缺失"

    if failures:
        print("\n[FAIL]")
        for f in failures:
            print(" -", f)
        return 1

    print(f"\n[OK] 全部 {len(results)} 条处理完成, 指标符合预期 (详见报告)")
    print(f"自检目录: {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
