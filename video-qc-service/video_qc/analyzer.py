# -*- coding: utf-8 -*-
"""检测编排：解码 -> 逐帧度量 -> 四类检测 -> JSON 报告。"""
import time
from pathlib import Path

from .detectors import (Thresholds, detect_audio, detect_black,
                        detect_blocky, detect_freeze)
from .pipeline import compute_metrics

ISSUE_ORDER = {
    "black_frame": 0,
    "freeze_frame": 1,
    "blocky": 2,
    "audio_silence": 3,
    "audio_clipping": 4,
}


def analyze(path, thresholds: Thresholds | None = None) -> dict:
    th = thresholds or Thresholds()
    start = time.perf_counter()
    metrics, info = compute_metrics(Path(path), th)
    fps, n = metrics["fps"], metrics["n_frames"]

    black_issues, black_flag = detect_black(
        metrics["mean_luma"], metrics["p99_luma"], fps, th)
    freeze_issues, _ = detect_freeze(metrics["mad"], fps, th, black_flag)
    blocky_issues, _ = detect_blocky(metrics["anomaly_ratio"], fps, th)
    silence_issues, clip_issues, _, _ = detect_audio(
        metrics["rms_db"], metrics["peak"], fps, n, metrics["audio_ok"], th)

    issues = sorted(
        black_issues + freeze_issues + blocky_issues
        + silence_issues + clip_issues,
        key=lambda d: (d["start_frame"], ISSUE_ORDER[d["type"]]))

    elapsed = time.perf_counter() - start
    duration = info.duration_sec or (n / fps if fps else 0.0)
    return {
        "file": str(path),
        "meta": {
            "duration_sec": round(duration, 3),
            "fps": round(fps, 3),
            "width": info.width,
            "height": info.height,
            "total_frames": n,
            "has_audio": info.has_audio,
        },
        "issues": issues,
        "stats": {
            "processing_time_sec": round(elapsed, 3),
            "speed_factor": round(duration / elapsed, 2) if elapsed > 0 else None,
            "analyze_resolution": list(metrics["analyze_resolution"]),
        },
    }
