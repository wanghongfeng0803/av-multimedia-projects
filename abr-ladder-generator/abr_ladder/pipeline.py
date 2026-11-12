"""Pipeline orchestration: probe -> keyframes -> encode -> quality -> verify.

Encodes and quality evaluations share one thread budget.  A slot-based
scheduler runs all four rung encodes concurrently (1080p receives the largest
share) and starts each rung's quality evaluation as soon as that rung's encode
finishes and slots are free.  This overlaps VMAF with the slower 1080p encode
instead of serialising the two phases.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import __version__, ff
from .config import RungSpec, Settings
from .encode import EncodeResult, encode_rung, prepare_audio
from .keyframes import KeyframePlan, plan_from_source
from .quality import QualityResult, evaluate
from .report import build_report, write_report
from .verify import print_summary, verify_ladder


class Logger:
    def __init__(self, path: Optional[str] = None, verbose: bool = True):
        self._lock = threading.Lock()
        self._fh = None
        self._verbose = verbose
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        with self._lock:
            if self._verbose:
                print(message, flush=True)
            if self._fh:
                self._fh.write(message + "\n")
                self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class SlotPool:
    """A budget of N reusable thread slots."""

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.free = self.total
        self._cond = threading.Condition()

    def acquire(self, slots: int) -> int:
        slots = max(1, min(int(slots), self.total))
        with self._cond:
            while self.free < slots:
                self._cond.wait()
            self.free -= slots
        return slots

    def release(self, slots: int) -> None:
        with self._cond:
            self.free += max(1, min(int(slots), self.total))
            self._cond.notify_all()


@dataclass
class Job:
    name: str
    slots: int
    deps: List[str]
    fn: Callable[[], None]
    event: threading.Event = field(default_factory=threading.Event)
    failed: bool = False
    error: Optional[BaseException] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    def to_timing(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 2)


def _run_jobs(jobs: Dict[str, Job], pool: SlotPool,
              log_fn: Callable[[str], None]) -> None:
    threads: List[threading.Thread] = []

    def worker(job: Job):
        for dep in job.deps:
            dep_job = jobs[dep]
            dep_job.event.wait()
            if dep_job.failed:
                job.failed = True
                job.event.set()
                return
        held = pool.acquire(job.slots)
        job.started_at = time.perf_counter()
        try:
            job.fn()
        except BaseException as exc:  # noqa: BLE001 - propagate after join
            job.failed = True
            job.error = exc
            log_fn(f"[error] {job.name}: {exc}")
        finally:
            job.ended_at = time.perf_counter()
            pool.release(held)
            job.event.set()

    for job in jobs.values():
        t = threading.Thread(target=worker, args=(job,), name=job.name)
        t.daemon = True
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for job in jobs.values():
        if job.failed and job.error is not None:
            raise job.error
        if job.failed:
            raise ff.FFError(f"job {job.name} was skipped because a dependency failed")


def allocate_encode_slots(rungs: List[RungSpec], budget: int) -> List[int]:
    """Greedy pixel-proportional thread allocation with a minimum of one."""
    budget = max(1, int(budget))
    alloc = [1] * len(rungs)
    remaining = budget - len(alloc)
    if remaining > 0:
        for _ in range(remaining):
            best = max(range(len(rungs)),
                       key=lambda i: rungs[i].pixels / alloc[i])
            alloc[best] += 1
    return [min(a, budget) for a in alloc]


def _verify_expected_files(output_dir: str, rungs: List[RungSpec]) -> List[str]:
    paths = [os.path.join(output_dir, f"{r.name}.mp4") for r in rungs]
    paths += [os.path.join(output_dir, "report.json"),
              os.path.join(output_dir, "report.md")]
    return paths


def prepare_output_dir(output_dir: str, rungs: List[RungSpec],
                       overwrite: bool) -> None:
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ff.FFError(f"output path is not a directory: {output_dir}")
    if os.path.isdir(output_dir) and not overwrite:
        existing = [p for p in _verify_expected_files(output_dir, rungs)
                    if os.path.exists(p)]
        if existing:
            raise ff.FFError(
                "output directory already contains ladder files "
                f"({', '.join(os.path.basename(p) for p in existing)}); "
                "use --overwrite to replace them"
            )
    os.makedirs(output_dir, exist_ok=True)
    if overwrite:
        for path in _verify_expected_files(output_dir, rungs):
            if os.path.exists(path):
                os.remove(path)
        for sub in ("logs", "metrics", ".work"):
            sub_path = os.path.join(output_dir, sub)
            if os.path.isdir(sub_path):
                shutil.rmtree(sub_path, ignore_errors=True)


def run_pipeline(source_path: str, output_dir: str, settings: Settings,
                 log: Optional[Callable[[str], None]] = None,
                 verbose: bool = True) -> dict:
    pipeline_start = time.perf_counter()

    # ---- pre-flight --------------------------------------------------------
    if not os.path.exists(source_path):
        raise ff.FFError(f"source not found: {source_path}")
    if not ff.has_encoder("libx264"):
        raise ff.FFError(
            "this ffmpeg build has no libx264 encoder; install a full build"
        )
    if settings.metric == "vmaf" and not ff.has_filter("libvmaf"):
        log_fn("[warn] libvmaf filter not found; quality will fall back to SSIM")

    prepare_output_dir(output_dir, settings.rungs, settings.overwrite)
    output_dir = os.path.abspath(output_dir)
    work_dir = os.path.join(output_dir, ".work")
    logs_dir = os.path.join(output_dir, "logs")
    metrics_dir = os.path.join(output_dir, "metrics")
    for d in (work_dir, logs_dir, metrics_dir):
        os.makedirs(d, exist_ok=True)
    run_log = os.path.join(logs_dir, "run.log")
    log = Logger(run_log, verbose=verbose)

    version = ff.version_info()
    log(f"[abr-ladder] {__version__} | {version.get('ffmpeg')}")
    log(f"[abr-ladder] source: {source_path}")
    log(f"[abr-ladder] output: {output_dir}")
    log(f"[abr-ladder] threads budget: {settings.threads}, preset: "
        f"{settings.preset}, GOP: {settings.gop_seconds}s, "
        f"metric: {settings.metric}")

    timings: dict = {}

    # ---- probe -------------------------------------------------------------
    t0 = time.perf_counter()
    source = ff.source_info(source_path)
    timings["probe"] = round(time.perf_counter() - t0, 2)
    log(f"[probe] {source.width}x{source.height} @ {source.fps_expr} fps, "
        f"{source.duration:.2f}s, {source.frame_count} frames, "
        f"{source.video_codec}/{source.pix_fmt}, "
        f"audio={source.audio_codec or 'none'}")
    if source.bit_depth > 8:
        log(f"[warn] source is {source.bit_depth}-bit; output ladder is 8-bit "
            f"yuv420p and will not preserve HDR/10-bit quality")
    if source.height < 1080:
        log(f"[warn] source height {source.height} < 1080; the 1080p rung "
            f"will be upscaled and cannot add detail")
    if source.duration < 5:
        log(f"[warn] very short source ({source.duration:.1f}s); rate and "
            f"quality statistics will be noisy")

    # ---- scene detection + keyframe plan -----------------------------------
    t0 = time.perf_counter()
    plan = plan_from_source(
        source, work_dir, gop_seconds=settings.gop_seconds,
        scene_threshold=settings.scene_threshold,
        use_scene_cuts=settings.scene_cuts,
        threads=max(1, min(4, settings.threads)),
        log_path=os.path.join(logs_dir, "scene_detect.log"),
    )
    timings["keyframe_plan"] = round(time.perf_counter() - t0, 2)
    log(f"[keyframes] {len(plan.frame_indices)} keyframes "
        f"(grid step {plan.grid_step_frames} frames, "
        f"{len(plan.scene_cuts)} scene cuts, max gap {plan.max_gap_frames} "
        f"frames / {plan.max_gap_frames / plan.fps:.2f}s)")

    # ---- audio (once, bit-identical in every rung) -------------------------
    t0 = time.perf_counter()
    audio_path, audio_codec = prepare_audio(
        source, work_dir, os.path.join(logs_dir, "audio_prepare.log"),
        threads=max(1, min(2, settings.threads)),
    )
    timings["prepare_audio"] = round(time.perf_counter() - t0, 2)
    log(f"[audio] {'prepared ' + str(audio_codec) if audio_path else 'no audio'}")

    # Custom VMAF model: copy once into the metrics dir under a simple
    # relative name (avoids Windows drive-colon parsing in filter options).
    vmaf_model_rel = None
    if (settings.metric == "vmaf" and settings.vmaf_model
            and settings.vmaf_model not in ("default", "builtin")):
        if not os.path.exists(settings.vmaf_model):
            raise ff.FFError(f"VMAF model not found: {settings.vmaf_model}")
        vmaf_model_rel = "vmaf_model.json"
        shutil.copyfile(settings.vmaf_model,
                        os.path.join(metrics_dir, vmaf_model_rel))
        log(f"[quality] using custom VMAF model {settings.vmaf_model}")

    # ---- encode + quality, sharing the thread budget -----------------------
    encode_slots = allocate_encode_slots(settings.rungs, settings.threads)
    quality_slots = max(1, settings.threads // 4)
    pool = SlotPool(settings.threads)
    log(f"[schedule] encode threads per rung: "
        + ", ".join(f"{r.name}={s}" for r, s in zip(settings.rungs, encode_slots))
        + f"; quality threads: {quality_slots}")

    encode_results: Dict[str, EncodeResult] = {}
    quality_results: Dict[str, QualityResult] = {}
    jobs: Dict[str, Job] = {}

    def make_encode(rung: RungSpec, slots: int):
        out_path = os.path.join(output_dir, f"{rung.name}.mp4")

        def _fn():
            result = encode_rung(
                source, rung, plan, settings, out_path, work_dir, logs_dir,
                threads=slots, audio_path=audio_path, audio_codec=audio_codec,
                log_fn=log,
            )
            encode_results[rung.name] = result

        return _fn

    top_rung = max(settings.rungs, key=lambda r: r.pixels)
    display_size = (top_rung.width, top_rung.height)

    def make_quality(rung: RungSpec, slots: int):
        def _fn():
            result = evaluate(
                source, os.path.join(output_dir, f"{rung.name}.mp4"), rung,
                plan, settings, metrics_dir, logs_dir,
                threads=slots, log_fn=log,
                vmaf_model_rel=vmaf_model_rel,
                display_size=display_size,
            )
            quality_results[rung.name] = result

        return _fn

    for rung, slots in zip(settings.rungs, encode_slots):
        jobs[f"encode:{rung.name}"] = Job(
            name=f"encode:{rung.name}", slots=slots, deps=[],
            fn=make_encode(rung, slots),
        )
    for rung in settings.rungs:
        jobs[f"quality:{rung.name}"] = Job(
            name=f"quality:{rung.name}", slots=quality_slots,
            deps=[f"encode:{rung.name}"],
            fn=make_quality(rung, quality_slots),
        )

    log("[schedule] starting encodes and quality evaluation")
    _run_jobs(jobs, pool, log)

    timings["encodes"] = {
        r.rung.name: round(r.encode_seconds, 2) for r in encode_results.values()
    }
    timings["quality"] = {
        q.rung_name: round(q.elapsed_seconds, 2) for q in quality_results.values()
    }

    encodes = [encode_results[r.name] for r in settings.rungs
               if r.name in encode_results]
    qualities = [quality_results[r.name] for r in settings.rungs
                 if r.name in quality_results]
    if len(encodes) != len(settings.rungs) or len(qualities) != len(settings.rungs):
        missing = [r.name for r in settings.rungs
                   if r.name not in encode_results or r.name not in quality_results]
        raise ff.FFError(f"pipeline did not complete for: {', '.join(missing)}")

    # ---- verification ------------------------------------------------------
    t0 = time.perf_counter()
    verification = verify_ladder(
        output_dir,
        rungs=[{"name": e.rung.name, "width": e.rung.width,
                "height": e.rung.height,
                "nominal_kbps": e.rung.nominal_kbps,
                "output": f"{e.rung.name}.mp4"} for e in encodes],
        fps=source.fps_float,
        source_path=source.path,
        tolerance_pct=settings.rate_tolerance_pct,
        expected_keyframes=plan.frame_indices,
        gop_seconds=settings.gop_seconds,
        threads=max(1, min(4, settings.threads)),
        sample_points=2,
        log_dir=logs_dir,
        log_fn=log,
    )
    timings["verify"] = round(time.perf_counter() - t0, 2)
    timings["total_wall_seconds"] = round(time.perf_counter() - pipeline_start, 2)

    # ---- report ------------------------------------------------------------
    report = build_report(source, settings, plan, encodes, qualities,
                          verification, timings, output_dir)
    json_path, md_path = write_report(report, output_dir)
    log(f"[report] {json_path}")
    log(f"[report] {md_path}")

    if not settings.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        log(f"[work] kept temp/statistics files in {work_dir}")

    elapsed = timings["total_wall_seconds"]
    log(f"[done] total wall time {elapsed:.1f}s "
        f"(budget {settings.time_budget_seconds:.0f}s), "
        f"verification {'PASS' if verification['passed'] else 'FAIL'}")

    if verbose:
        print_summary(verification)
    log.close()
    return report
