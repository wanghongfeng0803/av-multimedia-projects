# -*- coding: utf-8 -*-
"""四类缺陷的帧级检测逻辑。

所有检测器都工作在统一的逐帧度量数组上（见 video_qc.pipeline），
只负责"阈值判定 + 连续段聚合 + 置信度计算"，不做任何解码。
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Thresholds:
    # ---- 纯黑帧 ----
    black_mean_luma: float = 8.0        # 帧平均亮度上限（0-255）
    black_p99_luma: float = 16.0        # 帧 99 分位亮度上限：夜景有光斑，p99 远高于此
    black_min_sec: float = 0.5          # 最短持续：淡入淡出中的瞬时黑帧短于此值
    # ---- 静帧卡死 ----
    freeze_mad: float = 0.05            # 帧间平均绝对差上限（分析分辨率灰度，0-255）
    freeze_min_sec: float = 1.0         # 慢镜头帧间差虽小，但不会持续 1 秒低于该阈值
    # ---- 花屏马赛克 ----
    blocky_ratio: float = 0.10          # 时空异常块占比阈值（干净/高对比合法内容实测上限约 0.046）
    blocky_min_sec: float = 0.2
    fade_ramp_dmean: float = 2.0        # 全局亮度单帧变化超过此值视为淡入淡出斜坡
    fade_ramp_min_sec: float = 0.3      # 斜坡持续下限，避免把普通镜头切换误判为 fade
    fade_guard_sec: float = 0.5         # 斜坡前后保护窗口
    # ---- 音频 ----
    silence_rms_db: float = -45.0       # 静音 RMS 上限（正常混音约 -13 dB）
    silence_min_sec: float = 1.0
    clip_peak: float = 0.90             # 单帧峰值上限（0 dBFS=1.0，留 1dB 工程余量）
    clip_min_sec: float = 0.2
    block_size: int = 16


def _runs(flag: np.ndarray):
    """把布尔逐帧标记聚合为 (start, length) 连续段。"""
    out = []
    i, n = 0, len(flag)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i + 1
        while j < n and flag[j]:
            j += 1
        out.append((i, j - i))
        i = j
    return out


def _min_frames(min_sec, fps):
    return max(1, int(round(min_sec * fps)))


def _confidence(x, edge, strong, cap=0.99):
    """把度量值线性映射到 0.6~cap 的置信度；edge 为判定边界，strong 为高置信边界。"""
    if strong == edge:
        return cap
    c = 0.6 + 0.39 * min(1.0, abs(x - edge) / abs(strong - edge))
    return float(round(min(cap, c), 3))


def detect_black(mean_luma, p99_luma, fps, th: Thresholds):
    flag = (mean_luma < th.black_mean_luma) & (p99_luma < th.black_p99_luma)
    fade_guard = _fade_guard_mask(mean_luma, fps, th)
    min_frames = _min_frames(th.black_min_sec, fps)
    window = max(1, int(round(fps)))  # 黑段前后各看 1 秒
    issues = []
    for start, length in _runs(flag):
        if length < min_frames:
            continue
        # 淡入淡出保护：黑段前后窗口若被亮度斜坡覆盖，说明是 fade-to-black 而非异常黑帧
        left = fade_guard[max(0, start - window):start]
        right = fade_guard[start + length:start + length + window]
        flanks = np.concatenate([left, right]) if len(left) + len(right) else np.array([False])
        if flanks.mean() >= 0.5:
            continue
        worst_p99 = float(p99_luma[start:start + length].max())
        conf = _confidence(worst_p99, th.black_p99_luma, 2.0)
        issues.append({
            "type": "black_frame",
            "start_frame": int(start),
            "duration_frames": int(length),
            "confidence": conf,
        })
    return issues, flag


def detect_freeze(mad, fps, th: Thresholds, black_flag=None):
    flag = mad < th.freeze_mad
    if black_flag is not None:
        # 黑帧段内帧间差天然为 0，不作为静帧重复上报
        flag = flag & (~black_flag)
    min_frames = _min_frames(th.freeze_min_sec, fps)
    issues = []
    for start, length in _runs(flag):
        if length < min_frames:
            continue
        # 用段内中位 MAD 评估置信度，避免个别编码噪声帧拉低真实静帧的置信度
        median_mad = float(np.median(mad[start:start + length]))
        conf = _confidence(median_mad, th.freeze_mad, 0.005)
        issues.append({
            "type": "freeze_frame",
            "start_frame": int(start),
            "duration_frames": int(length),
            "confidence": conf,
        })
    return issues, flag


def _fade_guard_mask(mean_luma, fps, th: Thresholds):
    """标记处于淡入淡出斜坡及其保护窗口内的帧。

    淡入淡出的本质是全局亮度沿单一方向持续变化：单帧 |Δmean| 持续偏大，
    且相邻帧变化方向一致。花屏是局部块损坏，全局亮度基本稳定。
    """
    n = len(mean_luma)
    guard = np.zeros(n, dtype=bool)
    if n < 2:
        return guard
    dmean = np.diff(mean_luma)
    ramp = np.abs(dmean) > th.fade_ramp_dmean
    ramp_min = _min_frames(th.fade_ramp_min_sec, fps)
    guard_half = max(1, int(round(th.fade_guard_sec * fps)))
    for start, length in _runs(ramp):
        if length < ramp_min:
            continue
        lo = max(0, start - guard_half)
        hi = min(n, start + length + 1 + guard_half)
        guard[lo:hi] = True
    return guard


def detect_blocky(anomaly_ratio, fps, th: Thresholds):
    """花屏判定。度量本身已排除稳定的高对比内容与淡入淡出，这里只做阈值+时长聚合。"""
    flag = anomaly_ratio > th.blocky_ratio
    min_frames = _min_frames(th.blocky_min_sec, fps)
    issues = []
    for start, length in _runs(flag):
        if length < min_frames:
            continue
        peak_ratio = float(anomaly_ratio[start:start + length].max())
        conf = _confidence(peak_ratio, th.blocky_ratio, 0.25)
        issues.append({
            "type": "blocky",
            "start_frame": int(start),
            "duration_frames": int(length),
            "confidence": conf,
        })
    return issues, flag


def detect_audio(rms_db, peak, fps, n_frames, has_audio, th: Thresholds):
    """rms_db / peak 为逐视频帧对齐的音频度量，长度可能略少于视频帧数。"""
    if not has_audio:
        return [], [], np.zeros(n_frames, bool), np.zeros(n_frames, bool)

    silence_flag = rms_db < th.silence_rms_db
    clip_flag = peak >= th.clip_peak
    silence_issues, clip_issues = [], []

    min_sil = _min_frames(th.silence_min_sec, fps)
    for start, length in _runs(silence_flag):
        if length < min_sil:
            continue
        best_rms = float(rms_db[start:start + length].max())
        conf = _confidence(best_rms, th.silence_rms_db, -60.0)
        silence_issues.append({
            "type": "audio_silence",
            "start_frame": int(start),
            "duration_frames": int(length),
            "confidence": conf,
        })

    min_clip = _min_frames(th.clip_min_sec, fps)
    for start, length in _runs(clip_flag):
        if length < min_clip:
            continue
        peak_val = float(peak[start:start + length].max())
        conf = _confidence(peak_val, th.clip_peak, 1.0)
        clip_issues.append({
            "type": "audio_clipping",
            "start_frame": int(start),
            "duration_frames": int(length),
            "confidence": conf,
        })

    pad = n_frames - len(rms_db)
    if pad > 0:
        silence_flag = np.concatenate([silence_flag, np.zeros(pad, bool)])
        clip_flag = np.concatenate([clip_flag, np.zeros(pad, bool)])
    return silence_issues, clip_issues, silence_flag, clip_flag
