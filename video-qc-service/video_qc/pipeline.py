# -*- coding: utf-8 -*-
"""解码管道与逐帧度量计算。

架构：
- ffmpeg 子进程 1：解码视频并缩放到分析分辨率（默认宽 480），输出灰度 rawvideo；
  缩放/色彩转换由 ffmpeg 多线程完成，Python 侧只做轻量 numpy 统计。
- ffmpeg 子进程 2（后台线程）：解码音频为 f32le 单声道 PCM。
- 两路解码与原图分辨率解耦，4K 素材的单帧分析成本与 480p 相同。
"""
import os
import subprocess
import threading

import numpy as np

from .detectors import Thresholds
from .probe import MediaInfo, probe

ANALYZE_WIDTH = 480
# 目标部署规格为 8 核；ffmpeg 解码/缩放线程数封顶到 8，避免与同机其他任务争抢
MAX_THREADS = min(8, os.cpu_count() or 8)


def _analyze_height(src_w, src_h):
    h = max(2, round(ANALYZE_WIDTH * src_h / src_w))
    return h // 2 * 2  # rawvideo 要求偶数高度更稳妥


def _read_exact(pipe, size):
    """管道一次 read 可能返回部分数据，循环读满。"""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if not chunks:
        return b""
    return b"".join(chunks)


def _block_grids(frame, block):
    """把帧切成 block×block 网格，返回每个块的 (块内标准差, 块均值)。"""
    h, w = frame.shape
    h2, w2 = (h // block) * block, (w // block) * block
    g = frame[:h2, :w2].astype(np.float32).reshape(
        h2 // block, block, w2 // block, block)
    return g.std(axis=(1, 3)), g.mean(axis=(1, 3))


def _block_anomaly_ratio(bstd, bmean, prev_std, prev_mean):
    """花屏度量（时空联合）。

    合法的高对比内容（彩条、棋盘格、物体边缘）空间上可能很锐利，但块纹理/位置
    在时间上是稳定的；传输/解码损坏产生的噪点块和平铺纯色块每帧随机变化。
    因此异常块必须同时满足"绝对强度异常"和"时间不稳定"：
    - 噪点块：块内标准差绝对高（>=25）且相对上一帧剧变（>=6）；
    - 纯色损坏块：块均值相对邻域突兀、块内平滑，且块均值相对上一帧剧变。
    首帧没有上一帧参照，返回 0。
    """
    if prev_std is None:
        return 0.0
    d_std = np.abs(bstd - prev_std)
    d_mean = np.abs(bmean - prev_mean)
    padded = np.pad(bmean, 1, mode="edge")
    neighbors = np.stack([
        padded[:-2, 1:-1], padded[2:, 1:-1],
        padded[1:-1, :-2], padded[1:-1, 2:],
    ], axis=-1)
    contrast = np.abs(bmean - np.median(neighbors, axis=-1))
    med_contrast = np.median(contrast) + 1e-6

    noisy = (bstd >= 25.0) & (d_std >= 6.0)
    solid = ((contrast > med_contrast * 6) & (bstd < 8.0) & (d_mean >= 6.0))
    return float((noisy | solid).mean())


def _start_video_process(path, height):
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-threads", str(MAX_THREADS),
        "-i", str(path),
        "-an",
        "-vf", f"scale={ANALYZE_WIDTH}:{height}:flags=bilinear,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ], stdout=subprocess.PIPE)


def _start_audio_process(path):
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-threads", str(MAX_THREADS),
        "-i", str(path),
        "-vn", "-ac", "1", "-f", "f32le", "-",
    ], stdout=subprocess.PIPE)


def _audio_metrics(raw, fps, n_frames):
    """把 PCM 按视频帧时长切窗，计算逐帧 RMS(dB) 与峰值。"""
    samples = np.frombuffer(raw, dtype=np.float32)
    if len(samples) == 0:
        return (np.full(n_frames, -np.inf), np.zeros(n_frames))
    spf = len(samples) / max(n_frames, 1)
    win = max(1, int(round(spf)))
    n = min(n_frames, len(samples) // win) if win else 0
    rms_db = np.full(n_frames, -np.inf)
    peak = np.zeros(n_frames)
    for i in range(n):
        seg = samples[i * win:(i + 1) * win]
        rms = np.sqrt((seg ** 2).mean())
        rms_db[i] = 20 * np.log10(rms + 1e-12)
        peak[i] = float(np.abs(seg).max())
    return rms_db, peak


def compute_metrics(path, th: Thresholds):
    """返回 (逐帧度量字典, MediaInfo)。"""
    info: MediaInfo = probe(path)
    fps = info.fps or 25.0
    height = _analyze_height(info.width, info.height)
    frame_size = ANALYZE_WIDTH * height

    vproc = _start_video_process(path, height)

    audio_buf = {}

    def drain_audio():
        if not info.has_audio:
            return
        aproc = _start_audio_process(path)
        raw, _ = aproc.communicate()
        # 解码失败时按"音频不可用"处理，避免把空 PCM 误判为整段静音
        audio_buf["ok"] = aproc.returncode == 0
        audio_buf["raw"] = raw if aproc.returncode == 0 else b""

    athread = threading.Thread(target=drain_audio, daemon=True)
    athread.start()

    mean_luma, p99_luma, mad_series, ratio_series = [], [], [], []
    prev_frame = None
    prev_std = prev_mean = None
    idx = 0
    try:
        while True:
            raw = _read_exact(vproc.stdout, frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, ANALYZE_WIDTH)
            mean_luma.append(float(frame.mean()))
            p99_luma.append(float(np.percentile(frame, 99)))
            if prev_frame is None:
                mad_series.append(0.0)
            else:
                mad_series.append(float(np.abs(
                    frame.astype(np.int16) - prev_frame.astype(np.int16)).mean()))
            bstd, bmean = _block_grids(frame, th.block_size)
            ratio_series.append(
                _block_anomaly_ratio(bstd, bmean, prev_std, prev_mean))
            prev_std, prev_mean = bstd, bmean
            prev_frame = frame
            idx += 1
    finally:
        vproc.stdout.close()
        vproc.wait()
        athread.join()

    n_frames = idx
    if n_frames == 0:
        raise RuntimeError(f"未能从视频解码出任何帧: {path}")

    audio_ok = bool(audio_buf.get("ok", False))
    if audio_ok:
        rms_db, peak = _audio_metrics(audio_buf.get("raw", b""), fps, n_frames)
    else:
        rms_db = np.full(n_frames, -np.inf)
        peak = np.zeros(n_frames)

    metrics = {
        "mean_luma": np.array(mean_luma),
        "p99_luma": np.array(p99_luma),
        "mad": np.array(mad_series),
        "anomaly_ratio": np.array(ratio_series),
        "rms_db": rms_db,
        "peak": peak,
        "fps": fps,
        "n_frames": n_frames,
        "analyze_resolution": (ANALYZE_WIDTH, height),
        "audio_ok": audio_ok,
    }
    return metrics, info
