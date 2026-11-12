"""Objective quality evaluation (VMAF + SSIM).

Two views are computed for every rung:

* **native** — the encoded rung compared against the source scaled to that
  rung's resolution.  This measures coding quality at delivery resolution.
* **same-screen / display** — the encoded rung upscaled to the top rung's
  resolution and compared against the source at that resolution.  This
  captures resolution loss as a viewer on a large 1080p screen would see it.
  Native-resolution VMAF alone tends to favour lower rungs (downscaling hides
  detail); the display view makes adjacent-rung deltas meaningful.

VMAF uses the libvmaf built-in vmaf_v0.6.1 model by default.  SSIM is computed
in the same decode pass.  Long clips subsample VMAF frames (default every 5th)
so the 10-minute ladder fits comfortably in the time budget.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from . import ff
from .config import RungSpec, Settings, SourceInfo, fit_dimensions
from .keyframes import KeyframePlan, SceneCut

_MODEL_COPY_LOCK = threading.Lock()


@dataclass
class QualityResult:
    rung_name: str
    metric: str
    mean: float
    min: float
    min_time: float
    harmonic_mean: float
    p1: float
    sample_count: int
    subsample: int
    ssim_mean: Optional[float] = None
    ssim_min: Optional[float] = None
    ssim_min_time: Optional[float] = None
    worst_windows: List[dict] = field(default_factory=list)
    worst_scene: Optional[dict] = None
    scenes: List[dict] = field(default_factory=list)
    vmaf_version: Optional[str] = None
    model: str = "built-in vmaf_v0.6.1"
    elapsed_seconds: float = 0.0
    files: dict = field(default_factory=dict)
    warning: Optional[str] = None
    display: Optional[dict] = None

    def to_dict(self) -> dict:
        out = {
            "metric": self.metric,
            "mean": round(self.mean, 3) if self.mean == self.mean else None,
            "harmonic_mean": round(self.harmonic_mean, 3)
            if self.harmonic_mean == self.harmonic_mean else None,
            "min": round(self.min, 3) if self.min == self.min else None,
            "p1": round(self.p1, 3) if self.p1 == self.p1 else None,
            "min_frame_time": round(self.min_time, 3),
            "sample_count": self.sample_count,
            "subsample": self.subsample,
            "model": self.model,
            "vmaf_lib_version": self.vmaf_version,
            "ssim_mean": round(self.ssim_mean, 5) if self.ssim_mean is not None else None,
            "ssim_min": round(self.ssim_min, 5) if self.ssim_min is not None else None,
            "ssim_min_frame_time": round(self.ssim_min_time, 3)
            if self.ssim_min_time is not None else None,
            "worst_windows": self.worst_windows,
            "worst_scene": self.worst_scene,
            "scenes": self.scenes,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "files": self.files,
            "display": self.display,
        }
        if self.warning:
            out["warning"] = self.warning
        return out


def _window_summary(frames: List[Tuple[float, float]], window_seconds: float = 2.0,
                    keep: int = 3) -> List[dict]:
    if not frames:
        return []
    groups: dict = {}
    for t, score in frames:
        key = int(math.floor(t / window_seconds))
        groups.setdefault(key, []).append(score)
    windows = []
    for key, scores in groups.items():
        windows.append({
            "start": round(key * window_seconds, 3),
            "end": round((key + 1) * window_seconds, 3),
            "mean": round(sum(scores) / len(scores), 3),
            "samples": len(scores),
        })
    windows.sort(key=lambda w: w["mean"])
    return windows[:keep]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1,
                     int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _harm_mean(values: List[float]) -> float:
    eps = 1e-6
    if not values:
        return float("nan")
    return len(values) / sum(1.0 / max(v, eps) for v in values)


def _scene_summary(frames: List[Tuple[float, float]], scene_cuts: List[SceneCut],
                   duration: float) -> Tuple[List[dict], Optional[dict]]:
    if not frames:
        return [], None
    boundaries = [0.0]
    for cut in scene_cuts:
        if 0.0 < cut.time < duration:
            boundaries.append(cut.time)
    boundaries.append(duration)
    boundaries = sorted(set(round(b, 4) for b in boundaries))
    scenes = []
    for start, end in zip(boundaries, boundaries[1:]):
        scores = [s for t, s in frames if start <= t < end]
        if not scores:
            continue
        scenes.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "mean": round(sum(scores) / len(scores), 3),
            "min": round(min(scores), 3),
            "samples": len(scores),
        })
    worst = min(scenes, key=lambda s: s["mean"]) if scenes else None
    return scenes, worst


def _parse_ssim(path: str, fps: float) -> List[Tuple[float, float]]:
    frames: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return frames
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = re.search(r"n:(\d+).*All:([0-9.]+)", line)
            if not match:
                continue
            try:
                n = int(match.group(1))
                value = float(match.group(2))
            except ValueError:
                continue
            frames.append(((n - 1) / fps, value))
    return frames


def _vmaf_filter_args(model_arg: Optional[str], log_name: str,
                      threads: int, subsample: int) -> str:
    parts = ["log_fmt=json", f"log_path={log_name}",
             f"n_threads={max(1, threads)}"]
    if subsample > 1:
        parts.append(f"n_subsample={subsample}")
    if model_arg:
        parts.append(f"model=path={model_arg}")
    return "libvmaf=" + ":".join(parts)


def _run_pass(source: SourceInfo, dist_path: str, target_w: int, target_h: int,
              settings: Settings, work_dir: str, threads: int, tag: str,
              scale_dist: bool, use_vmaf: bool, model_arg: Optional[str],
              subsample: int, log_path: str,
              log_fn: Callable[[str], None]) -> dict:
    """Run one quality pass (SSIM always, VMAF when requested)."""
    ssim_name = f"ssim_{tag}.log"
    vmaf_name = f"vmaf_{tag}.json"
    ssim_path = os.path.join(work_dir, ssim_name)
    vmaf_path = os.path.join(work_dir, vmaf_name)
    for p in (ssim_path, vmaf_path):
        if os.path.exists(p):
            os.remove(p)

    fps = source.fps_expr
    fit_w, fit_h = fit_dimensions(source.display_aspect, target_w, target_h)
    pad = ""
    if (fit_w, fit_h) != (target_w, target_h):
        pad = (f",pad={target_w}:{target_h}:"
               f"(ow-iw)/2:(oh-ih)/2:color=black")

    ref = (f"[1:v]fps={fps},"
           f"scale={fit_w}:{fit_h}:flags=bicubic{pad},setsar=1,"
           f"format=yuv420p")
    if scale_dist:
        dis = (f"[0:v]fps={fps},"
               f"scale={target_w}:{target_h}:flags=bicubic,"
               f"setsar=1,format=yuv420p")
    else:
        dis = f"[0:v]fps={fps},setsar=1,format=yuv420p"

    if use_vmaf:
        graph = (
            f"{ref},split=2[r1][r2];"
            f"{dis},split=2[d1][d2];"
            f"[d1][r1]ssim=stats_file={ssim_name}[sout];"
            f"[d2][r2]{_vmaf_filter_args(model_arg, vmaf_name, threads, subsample)}[vout]"
        )
        maps = ["-map", "[sout]", "-map", "[vout]"]
    else:
        graph = f"{ref}[r1];{dis}[d1];[d1][r1]ssim=stats_file={ssim_name}[sout]"
        maps = ["-map", "[sout]"]

    cmd = [
        "-hide_banner", "-y",
        "-threads", str(max(1, threads)), "-i", dist_path,
        "-threads", str(max(1, threads)), "-i", source.path,
        "-filter_complex", graph,
        "-filter_complex_threads", str(max(1, threads)),
    ] + maps + ["-f", "null", ff.NULL_DEVICE]

    t0 = time.perf_counter()
    ff.run_ffmpeg(cmd, log_path=log_path, cwd=work_dir)
    elapsed = time.perf_counter() - t0

    ssim_frames = _parse_ssim(ssim_path, source.fps_float)
    ssim_mean = (sum(v for _, v in ssim_frames) / len(ssim_frames)
                 if ssim_frames else None)
    ssim_min = min((v for _, v in ssim_frames), default=None)
    ssim_min_time = None
    if ssim_frames:
        ssim_min_time = min(ssim_frames, key=lambda p: p[1])[0]

    vmaf_version = None
    if use_vmaf:
        if not os.path.exists(vmaf_path):
            raise ff.FFError(f"{tag}: libvmaf did not write {vmaf_path}")
        with open(vmaf_path, "r", encoding="utf-8") as f:
            vmaf = json.load(f)
        vmaf_version = vmaf.get("version")
        frames: List[Tuple[float, float]] = []
        for fr in vmaf.get("frames", []):
            value = fr.get("metrics", {}).get("vmaf")
            num = fr.get("frameNum")
            if value is None or num is None:
                continue
            try:
                frames.append((float(num) / source.fps_float, float(value)))
            except (TypeError, ValueError):
                continue
        if not frames:
            raise ff.FFError(f"{tag}: empty VMAF log")
        metric = "vmaf"
        scores = [s for _, s in frames]
        mean = sum(scores) / len(scores)
        minimum = min(scores)
        min_time = min(frames, key=lambda p: p[1])[0]
        harmonic = _harm_mean(scores)
        p1 = _percentile(scores, 1)
        worst_windows = _window_summary(frames)
        files = {"vmaf_log": vmaf_name, "ssim_log": ssim_name if ssim_frames else None}
    else:
        metric = "ssim"
        frames = ssim_frames
        scores = [v for _, v in frames]
        if not scores:
            raise ff.FFError(f"{tag}: empty SSIM stats")
        mean = sum(scores) / len(scores)
        minimum = min(scores)
        min_time = ssim_min_time or 0.0
        harmonic = _harm_mean(scores)
        p1 = _percentile(scores, 1)
        worst_windows = _window_summary(frames)
        files = {"ssim_log": ssim_name}

    return {
        "metric": metric,
        "mean": mean,
        "min": minimum,
        "min_time": min_time,
        "harmonic_mean": harmonic,
        "p1": p1,
        "sample_count": len(frames),
        "subsample": subsample if use_vmaf else 1,
        "worst_windows": worst_windows,
        "ssim_mean": ssim_mean,
        "ssim_min": ssim_min,
        "ssim_min_time": ssim_min_time,
        "vmaf_version": vmaf_version,
        "elapsed_seconds": elapsed,
        "files": files,
        "frames": frames,
    }


def evaluate(source: SourceInfo, dist_path: str, rung: RungSpec,
             plan: KeyframePlan, settings: Settings, work_dir: str,
             logs_dir: str, threads: int,
             log_fn: Callable[[str], None],
             vmaf_model_rel: Optional[str] = None,
             display_size: Optional[Tuple[int, int]] = None) -> QualityResult:
    """Compute native and same-screen quality for one rung."""
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"quality_{rung.name}.log")
    name = rung.name

    use_vmaf = (settings.metric == "vmaf" and ff.has_filter("libvmaf"))
    warning = None
    if settings.metric == "vmaf" and not use_vmaf:
        warning = ("libvmaf is not available in this ffmpeg build; "
                   "falling back to SSIM")

    subsample = settings.vmaf_subsample
    if subsample <= 0:
        subsample = 5 if source.duration >= 60 else 1

    model_arg = None
    model_label = "built-in vmaf_v0.6.1"
    if use_vmaf and settings.vmaf_model and settings.vmaf_model not in ("default", "builtin"):
        if not os.path.exists(settings.vmaf_model):
            raise ff.FFError(f"VMAF model not found: {settings.vmaf_model}")
        if vmaf_model_rel and os.path.exists(os.path.join(work_dir, vmaf_model_rel)):
            model_arg = vmaf_model_rel
        else:
            with _MODEL_COPY_LOCK:
                model_copy = os.path.join(work_dir, "vmaf_model.json")
                if not os.path.exists(model_copy):
                    shutil.copyfile(settings.vmaf_model, model_copy)
            model_arg = "vmaf_model.json"
        model_label = os.path.abspath(settings.vmaf_model)

    log_fn(f"[quality] {rung.name}: "
           + ("VMAF (native" + (f", subsample {subsample}" if subsample > 1 else "")
              + " + same-screen)" if use_vmaf else "SSIM native + same-screen")
           + f" with {threads} threads")

    native = _run_pass(
        source, dist_path, rung.width, rung.height, settings, work_dir,
        threads, tag=f"native_{name}", scale_dist=False, use_vmaf=use_vmaf,
        model_arg=model_arg, subsample=subsample, log_path=log_path,
        log_fn=log_fn,
    )

    display_target = display_size or (rung.width, rung.height)
    display_elapsed = 0.0
    if (display_target[0], display_target[1]) == (rung.width, rung.height):
        display = dict(native)
        display["label"] = "顶档原生分辨率"
        display["target_width"] = rung.width
        display["target_height"] = rung.height
    else:
        display = _run_pass(
            source, dist_path, display_target[0], display_target[1],
            settings, work_dir, threads, tag=f"display_{name}",
            scale_dist=True, use_vmaf=use_vmaf, model_arg=model_arg,
            subsample=subsample, log_path=log_path, log_fn=log_fn,
        )
        display_elapsed = display["elapsed_seconds"]
        display["label"] = f"放大到 {display_target[0]}x{display_target[1]}"
        display["target_width"] = display_target[0]
        display["target_height"] = display_target[1]

    scenes, worst_scene = _scene_summary(native["frames"], plan.scene_cuts,
                                         source.duration)
    display_summary = {
        "metric": display["metric"],
        "target_width": display["target_width"],
        "target_height": display["target_height"],
        "label": display["label"],
        "mean": round(display["mean"], 4),
        "harmonic_mean": round(display["harmonic_mean"], 4),
        "min": round(display["min"], 4),
        "min_frame_time": round(display["min_time"], 3),
        "p1": round(display["p1"], 4),
        "sample_count": display["sample_count"],
        "subsample": display["subsample"],
        "worst_windows": display["worst_windows"],
        "ssim_mean": round(display["ssim_mean"], 5)
        if display["ssim_mean"] is not None else None,
        "ssim_min": round(display["ssim_min"], 5)
        if display["ssim_min"] is not None else None,
        "files": display["files"],
    }

    log_fn(
        f"[quality] {rung.name}: {native['metric'].upper()} native "
        f"{native['mean']:.2f} (min {native['min']:.2f} @ "
        f"{native['min_time']:.1f}s); same-screen "
        f"{display['mean']:.2f}; SSIM {native['ssim_mean']:.4f}"
        if native["ssim_mean"] is not None else
        f"[quality] {rung.name}: {native['metric'].upper()} native "
        f"{native['mean']:.2f}; same-screen {display['mean']:.2f}"
    )

    return QualityResult(
        rung_name=rung.name,
        metric=native["metric"],
        mean=native["mean"],
        min=native["min"],
        min_time=native["min_time"],
        harmonic_mean=native["harmonic_mean"],
        p1=native["p1"],
        sample_count=native["sample_count"],
        subsample=native["subsample"],
        ssim_mean=native["ssim_mean"],
        ssim_min=native["ssim_min"],
        ssim_min_time=native["ssim_min_time"],
        worst_windows=native["worst_windows"],
        worst_scene=worst_scene,
        scenes=scenes,
        vmaf_version=native["vmaf_version"],
        model=model_label if use_vmaf else "n/a",
        elapsed_seconds=native["elapsed_seconds"] + display_elapsed,
        files=native["files"],
        warning=warning,
        display=display_summary,
    )
