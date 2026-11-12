"""Independent ladder verification.

Checks, from the produced files only:

  * video-stream bitrate deviation against the nominal value (±3% by default)
  * frame-exact keyframe alignment across all rungs (and against the plan)
  * GOP length sanity
  * resolution / duration / frame-count consistency
  * switch-in decode health: starting at each switch keyframe, decode the
    target rung without errors and compare the first frames against the
    source (SSIM) so a green/corrupted IDR cannot pass silently
  * audio identity: byte-identical audio packets in every rung
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Callable, List, Optional, Tuple

from . import ff
from .config import SourceInfo, fit_dimensions


DEFAULT_SWITCH_SSIM_MIN = 0.50


def _load_report(output_dir: str) -> dict:
    report_path = os.path.join(output_dir, "report.json")
    if not os.path.exists(report_path):
        raise ff.FFError(
            f"{report_path} not found; run abr-ladder first or point the "
            f"verifier at its output directory"
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rungs_from_report(report: dict) -> List[dict]:
    out = []
    for rung in report.get("rungs", []):
        out.append({
            "name": rung["name"],
            "width": rung["width"],
            "height": rung["height"],
            "nominal_kbps": rung["nominal_kbps"],
            "output": rung.get("output") or f"{rung['name']}.mp4",
        })
    return out


def _pick_switch_times(keyframes: List[Tuple[int, float]],
                       sample_points: int, duration: float) -> List[float]:
    if len(keyframes) <= 1:
        return []
    candidates = [k for k in keyframes[1:] if k[1] < duration - 1.0]
    if not candidates:
        return []
    points = []
    for i in range(1, sample_points + 1):
        pos = int(round(i * len(candidates) / (sample_points + 1)))
        pos = max(0, min(len(candidates) - 1, pos))
        points.append(candidates[pos][1])
    return sorted(set(points))


def _boundary_ssim(dist_path: str, source_path: Optional[str], rung: dict,
                   source_info: Optional[SourceInfo], t: float,
                   frames: int, work_dir: str, threads: int,
                   log_path: str) -> Tuple[bool, Optional[float], str]:
    """Decode <frames> frames starting at keyframe t; compare to source.

    Returns (ok, mean_ssim, detail).  With -xerror, a broken/corrupt IDR
    makes the command fail instead of silently producing green frames.
    """
    rung_w, rung_h = int(rung["width"]), int(rung["height"])
    if source_path and source_info is not None:
        fit_w, fit_h = fit_dimensions(source_info.display_aspect, rung_w, rung_h)
        pad = ""
        if (fit_w, fit_h) != (rung_w, rung_h):
            pad = (f",pad={rung_w}:{rung_h}:"
                   f"(ow-iw)/2:(oh-ih)/2:color=black")
        ref = (f"[1:v]fps={source_info.fps_expr},"
               f"scale={fit_w}:{fit_h}:flags=bicubic{pad},setsar=1,"
               f"format=yuv420p[ref]")
        dis = "[0:v]setsar=1,format=yuv420p[dis]"
        ssim_name = f"switch_{rung['name']}_{int(round(t * 1000))}.log"
        ssim_path = os.path.join(work_dir, ssim_name)
        if os.path.exists(ssim_path):
            os.remove(ssim_path)
        graph = f"{ref};{dis};[dis][ref]ssim=stats_file={ssim_name}"
        cmd = [
            "-hide_banner", "-y", "-xerror",
            "-threads", str(max(1, threads)), "-ss", f"{t:.6f}", "-i", dist_path,
            "-threads", str(max(1, threads)), "-ss", f"{t:.6f}", "-i", source_path,
            "-filter_complex", graph,
            "-filter_complex_threads", str(max(1, threads)),
            "-frames:v", str(frames), "-f", "null", ff.NULL_DEVICE,
        ]
        try:
            ff.run_ffmpeg(cmd, log_path=log_path, cwd=work_dir)
        except ff.FFError as exc:
            return False, None, f"decode/SSIM failed: {str(exc)[:400]}"
        values = []
        if os.path.exists(ssim_path):
            with open(ssim_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.search(r"All:([0-9.]+)", line)
                    if m:
                        values.append(float(m.group(1)))
        if not values:
            return False, None, "no SSIM frames written (decode produced nothing)"
        mean = sum(values) / len(values)
        return True, mean, f"{len(values)} frames decoded, ssim_mean={mean:.4f}"
    else:
        # Without the source we can still prove the switch-in decodes cleanly.
        cmd = [
            "-hide_banner", "-y", "-xerror",
            "-threads", str(max(1, threads)), "-ss", f"{t:.6f}", "-i", dist_path,
            "-frames:v", str(frames), "-f", "null", ff.NULL_DEVICE,
        ]
        try:
            ff.run_ffmpeg(cmd, log_path=log_path)
        except ff.FFError as exc:
            return False, None, f"decode failed: {str(exc)[:400]}"
        return True, None, f"{frames} frames decoded cleanly"


def verify_ladder(output_dir: str, rungs: List[dict], fps: float,
                  source_path: Optional[str], tolerance_pct: float = 3.0,
                  expected_keyframes: Optional[List[int]] = None,
                  gop_seconds: Optional[float] = None,
                  threads: int = 2, sample_points: int = 2,
                  log_dir: Optional[str] = None,
                  log_fn: Callable[[str], None] = print) -> dict:
    out_log_dir = log_dir or os.path.join(output_dir, "logs")
    os.makedirs(out_log_dir, exist_ok=True)
    log_path = os.path.join(out_log_dir, "verify.log")
    source_info: Optional[SourceInfo] = None
    if source_path and os.path.exists(source_path):
        try:
            source_info = ff.source_info(source_path)
        except ff.FFError:
            source_info = None

    errors: List[str] = []
    warnings: List[str] = []
    rung_checks: List[dict] = []
    keyframe_sets: List[List[int]] = []
    keyframe_times: List[List[float]] = []
    audio_hashes: List[Tuple[str, Optional[str]]] = []
    durations: List[float] = []
    frame_counts: List[int] = []
    fps_value = float(fps) if fps and fps > 0 else 30.0

    for rung in rungs:
        path = os.path.join(output_dir, rung["output"])
        entry: dict = {
            "name": rung["name"],
            "file": rung["output"],
            "exists": os.path.exists(path),
        }
        if not os.path.exists(path):
            entry["ok"] = False
            entry["error"] = "file missing"
            errors.append(f"{rung['name']}: output file missing ({path})")
            rung_checks.append(entry)
            keyframe_sets.append([])
            keyframe_times.append([])
            durations.append(0.0)
            frame_counts.append(0)
            continue

        measured, detail = ff.measure_video_bitrate_kbps(path, fps_value)
        nominal = float(rung["nominal_kbps"])
        deviation = (measured - nominal) / nominal * 100.0
        within = abs(deviation) <= tolerance_pct + 1e-9
        if not within:
            errors.append(
                f"{rung['name']}: bitrate {measured:.1f} kb/s vs nominal "
                f"{nominal:.0f} kb/s ({deviation:+.2f}% > +/-{tolerance_pct}%)"
            )
        try:
            width, height = ff.frame_size(path)
        except ff.FFError as exc:
            width = height = 0
            errors.append(f"{rung['name']}: cannot probe size: {exc}")
        if (width, height) != (int(rung["width"]), int(rung["height"])):
            resolution_ok = False
            errors.append(
                f"{rung['name']}: resolution {width}x{height} does not match "
                f"{rung['width']}x{rung['height']}"
            )
        else:
            resolution_ok = True
        duration, frame_count, _ = ff.video_stream_duration_and_frames(
            path, fps_value
        )
        durations.append(duration)
        frame_counts.append(frame_count)

        kfs = ff.keyframes(path, fps_value)
        indices = [k[0] for k in kfs]
        times = [k[1] for k in kfs]
        keyframe_sets.append(indices)
        keyframe_times.append(times)

        gaps = [b - a for a, b in zip(indices, indices[1:])] if len(indices) > 1 else []
        entry.update({
            "ok": within and resolution_ok,
            "nominal_kbps": nominal,
            "measured_kbps": round(measured, 2),
            "deviation_pct": round(deviation, 3),
            "within_tolerance": within,
            "measured_detail": detail,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 4),
            "frame_count": frame_count,
            "keyframe_count": len(indices),
            "max_gop_frames": max(gaps) if gaps else 0,
            "first_keyframe": indices[0] if indices else None,
            "last_keyframe": indices[-1] if indices else None,
        })

        if gop_seconds:
            max_allowed = int(round(gop_seconds * fps_value)) + 3
            if gaps and max(gaps) > max_allowed:
                warnings.append(
                    f"{rung['name']}: GOP gap {max(gaps)} frames exceeds "
                    f"configured {gop_seconds}s (~{max_allowed} frames)"
                )
        if not indices or indices[0] != 0:
            errors.append(f"{rung['name']}: first frame is not a keyframe")

        audio_hash = ff.audio_md5(path)
        audio_hashes.append((rung["name"], audio_hash))
        rung_checks.append(entry)
        log_fn(
            f"[verify] {rung['name']}: {measured:.1f} kb/s "
            f"({deviation:+.2f}%), {len(indices)} keyframes, "
            f"{duration:.2f}s"
        )

    # ---- keyframe alignment across rungs ----------------------------------
    aligned = True
    alignment: dict = {"aligned": True, "rungs": []}
    if keyframe_sets and all(keyframe_sets):
        ref = keyframe_sets[0]
        ref_times = keyframe_times[0]
        ref_name = rungs[0]["name"]
        for rung, indices, times in zip(rungs, keyframe_sets, keyframe_times):
            entry = {
                "name": rung["name"],
                "keyframe_count": len(indices),
                "matches_reference": indices == ref,
                "extra_vs_reference": len(set(indices) - set(ref)),
                "missing_vs_reference": len(set(ref) - set(indices)),
            }
            max_time_diff = None
            if len(times) == len(ref_times):
                max_time_diff = max((abs(a - b) for a, b in zip(times, ref_times)),
                                    default=0.0)
                entry["max_pts_diff_seconds"] = round(max_time_diff, 6)
                if max_time_diff > 0.5 / fps_value:
                    aligned = False
                    errors.append(
                        f"{rung['name']}: keyframe PTS differs from {ref_name} "
                        f"by up to {max_time_diff:.3f}s (> half a frame)"
                    )
            if indices != ref:
                aligned = False
                errors.append(
                    f"{rung['name']}: keyframes differ from {ref_name} "
                    f"({len(set(indices) - set(ref))} extra, "
                    f"{len(set(ref) - set(indices))} missing)"
                )
            if expected_keyframes:
                expected = set(expected_keyframes)
                actual = set(indices)
                entry["matches_plan"] = actual == expected
                if actual != expected:
                    aligned = False
                    errors.append(
                        f"{rung['name']}: keyframes differ from the planned "
                        f"schedule ({len(actual - expected)} extra, "
                        f"{len(expected - actual)} missing)"
                    )
            alignment["rungs"].append(entry)
    else:
        aligned = False
        errors.append("could not collect keyframes from every rung")
    alignment["aligned"] = aligned
    alignment["reference"] = rungs[0]["name"] if rungs else None
    alignment["reference_keyframe_count"] = len(keyframe_sets[0]) if keyframe_sets else 0

    # ---- duration / frame count consistency --------------------------------
    consistent = True
    if durations and max(durations) - min(durations) > (1.5 / fps_value):
        consistent = False
        errors.append(
            f"rung durations differ: {min(durations):.3f}s..{max(durations):.3f}s"
        )
    if frame_counts and max(frame_counts) - min(frame_counts) > 1:
        consistent = False
        errors.append(
            f"rung frame counts differ: {min(frame_counts)}..{max(frame_counts)}"
        )

    # ---- audio identity -----------------------------------------------------
    audio_present = any(h is not None for _, h in audio_hashes)
    audio_same = True
    if audio_present:
        hashes = [h for _, h in audio_hashes]
        if any(h is None for h in hashes):
            audio_same = False
            errors.append("audio present in only some rungs")
        elif len(set(hashes)) != 1:
            audio_same = False
            errors.append("audio packets differ between rungs")
    audio_info = {
        "present": audio_present,
        "identical": audio_same,
        "md5": {name: h for name, h in audio_hashes},
    }

    # ---- switch-in health ---------------------------------------------------
    switch_tests: List[dict] = []
    switch_ok = True
    switch_ssim_min = None
    if len(rungs) >= 2 and len(keyframe_sets[0]) > 1:
        # Use actual keyframes (already verified aligned).
        times = _pick_switch_times(
            [(i, t) for i, t in zip(keyframe_sets[0], keyframe_times[0])],
            sample_points, durations[0] if durations else 0.0,
        )
        for idx in range(len(rungs) - 1):
            lower = rungs[idx + 1]
            lower_path = os.path.join(output_dir, lower["output"])
            if not os.path.exists(lower_path):
                continue
            for t in times:
                ok, ssim_mean, detail = _boundary_ssim(
                    lower_path, source_path, lower, source_info, t, 8,
                    out_log_dir, threads, log_path,
                )
                if ssim_mean is not None:
                    if switch_ssim_min is None or ssim_mean < switch_ssim_min:
                        switch_ssim_min = ssim_mean
                    if ssim_mean < DEFAULT_SWITCH_SSIM_MIN:
                        ok = False
                        detail += (f" (below threshold "
                                   f"{DEFAULT_SWITCH_SSIM_MIN})")
                test = {
                    "from": rungs[idx]["name"],
                    "to": lower["name"],
                    "at_seconds": round(t, 4),
                    "frame_index": int(round(t * fps_value)),
                    "ok": ok,
                    "ssim_mean": round(ssim_mean, 4) if ssim_mean is not None else None,
                    "detail": detail,
                }
                switch_tests.append(test)
                if not ok:
                    switch_ok = False
                    errors.append(
                        f"switch {rungs[idx]['name']}->{lower['name']} at "
                        f"{t:.2f}s failed: {detail}"
                    )
                else:
                    log_fn(
                        f"[verify] switch {rungs[idx]['name']}->{lower['name']} "
                        f"@ {t:.2f}s: OK ({detail})"
                    )
    else:
        warnings.append("switch test skipped (fewer than two rungs/keyframes)")

    passed = (
        not [e for e in rung_checks if not e.get("ok")]
        and aligned and consistent and audio_same and switch_ok
        and len(errors) == 0
    )

    return {
        "passed": passed,
        "tolerance_pct": tolerance_pct,
        "rungs": rung_checks,
        "keyframe_alignment": alignment,
        "duration_consistent": consistent,
        "audio": audio_info,
        "switch_tests": switch_tests,
        "switch_ssim_min": round(switch_ssim_min, 4)
        if switch_ssim_min is not None else None,
        "switch_ssim_threshold": DEFAULT_SWITCH_SSIM_MIN,
        "errors": errors,
        "warnings": warnings,
        "verified_at_source": source_path if source_info else None,
    }


def print_summary(result: dict) -> None:
    print()
    print("=" * 78)
    print("ABR ladder verification")
    print("=" * 78)
    print(f"{'rung':<10}{'measured':>12}{'nominal':>10}{'dev':>9}"
          f"{'keyframes':>11}{'duration':>10}  status")
    for rung in result["rungs"]:
        if not rung.get("exists"):
            print(f"{rung['name']:<10}{'MISSING':>12}")
            continue
        status = "OK" if rung.get("ok") else "FAIL"
        print(f"{rung['name']:<10}"
              f"{rung['measured_kbps']:>10.1f}k"
              f"{rung['nominal_kbps']:>9.0f}k"
              f"{rung['deviation_pct']:>8.2f}%"
              f"{rung['keyframe_count']:>11}"
              f"{rung['duration_seconds']:>9.2f}s  {status}")
    align = result["keyframe_alignment"]
    print()
    print(f"keyframe alignment : {'OK' if align['aligned'] else 'FAIL'} "
          f"({align.get('reference_keyframe_count', 0)} keyframes)")
    print(f"duration/frames    : {'OK' if result['duration_consistent'] else 'FAIL'}")
    audio = result["audio"]
    print(f"audio identity     : "
          f"{'OK' if audio['identical'] else 'FAIL'}"
          f"{'' if audio['present'] else ' (no audio)'}")
    switch = result["switch_tests"]
    if switch:
        bad = sum(1 for t in switch if not t["ok"])
        print(f"switch-in decode   : {'OK' if bad == 0 else 'FAIL'} "
              f"({len(switch)} tests, min SSIM "
              f"{result['switch_ssim_min']})")
    for err in result["errors"]:
        print(f"  ERROR: {err}")
    for warn in result["warnings"]:
        print(f"  WARN: {warn}")
    print()
    print("RESULT:", "PASS" if result["passed"] else "FAIL")
    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="abr-ladder-verify",
        description=("Verify bitrate tolerance, keyframe alignment and "
                     "switch-in health of an ABR ladder output directory."),
    )
    parser.add_argument("output_dir", help="output directory produced by abr-ladder")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="allowed bitrate deviation in percent (default 3.0)")
    parser.add_argument("--threads", type=int, default=2,
                        help="decode/filter threads for boundary tests")
    parser.add_argument("--sample-points", type=int, default=2,
                        help="switch points sampled per adjacent rung pair")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON")
    parser.add_argument("--quiet", action="store_true", help="only print errors")
    args = parser.parse_args(argv)

    try:
        report = _load_report(args.output_dir)
    except ff.FFError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rungs = _rungs_from_report(report)
    source = report.get("source", {})
    fps = float(source.get("fps_float") or 30.0)
    source_path = source.get("path")
    expected = report.get("keyframes", {}).get("frame_indices")
    gop = report.get("settings", {}).get("gop_seconds")

    try:
        result = verify_ladder(
            args.output_dir, rungs, fps, source_path,
            tolerance_pct=args.tolerance, expected_keyframes=expected,
            gop_seconds=gop, threads=args.threads,
            sample_points=args.sample_points,
            log_fn=(lambda *_a, **_k: None) if args.quiet else print,
        )
    except ff.FFError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif not args.quiet:
        print_summary(result)
    elif not result["passed"]:
        for err in result["errors"]:
            print(f"ERROR: {err}", file=sys.stderr)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
