"""Command-line interface for the ABR ladder generator."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__, ff
from .config import DEFAULT_RUNGS, Settings, load_ladder
from .pipeline import run_pipeline


PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower", "veryslow"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abr-ladder",
        description=(
            "Generate a 1080p/720p/480p/360p ABR ladder from one source clip. "
            "Encodes use frame-exact aligned keyframes, closed-loop bitrate "
            "targeting (+/-3%), and VMAF/SSIM quality reporting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="source video (1080p or higher recommended)")
    parser.add_argument("output_dir", help="directory for the four MP4s and reports")
    parser.add_argument("--preset", choices=PRESETS, default="fast",
                        help="libx264 preset")
    parser.add_argument("--vmaf-model", default=None,
                        help="path to a VMAF model JSON; default uses the "
                             "libvmaf built-in vmaf_v0.6.1")
    parser.add_argument("--metric", choices=["vmaf", "ssim"], default="vmaf",
                        help="primary quality metric (SSIM is always computed "
                             "alongside VMAF)")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 8, 8),
                        help="total encode+quality thread budget; the 8-core "
                             "target uses 8")
    parser.add_argument("--gop-seconds", type=float, default=2.0,
                        help="maximum distance between forced keyframes")
    parser.add_argument("--scene-threshold", type=float, default=0.35,
                        help="ffmpeg scene score threshold for extra keyframes")
    parser.add_argument("--no-scene-cuts", action="store_true",
                        help="use only the fixed GOP grid, skip scene-cut keyframes")
    parser.add_argument("--vmaf-subsample", type=int, default=0,
                        help="compute VMAF on every Nth frame; 0 = auto "
                             "(1 for short clips, 5 for >=60s)")
    parser.add_argument("--rate-tolerance", type=float, default=3.0,
                        help="allowed measured-vs-nominal bitrate deviation (%%)")
    parser.add_argument("--maxrate-ratio", type=float, default=1.5,
                        help="VBR maxrate as a multiple of the nominal bitrate")
    parser.add_argument("--bufsize-ratio", type=float, default=3.0,
                        help="VBR bufsize as a multiple of the nominal bitrate")
    parser.add_argument("--profile", default="high",
                        help="H.264 profile")
    parser.add_argument("--ladder", default=None,
                        help="custom ladder JSON (defaults to the brief's four rungs)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing ladder files in output_dir")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep .work (2-pass stats, temp audio, scene metadata)")
    parser.add_argument("--time-budget", type=float, default=1500.0,
                        help="report whether wall time fits this many seconds")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    rungs = DEFAULT_RUNGS
    if args.ladder:
        rungs = load_ladder(args.ladder)
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if not (0.2 <= args.gop_seconds <= 30):
        raise ValueError("--gop-seconds must be between 0.2 and 30")
    if not (0.1 <= args.rate_tolerance <= 10):
        raise ValueError("--rate-tolerance must be between 0.1 and 10")
    if args.vmaf_subsample < 0:
        raise ValueError("--vmaf-subsample must be >= 0")
    return Settings(
        preset=args.preset,
        threads=args.threads,
        gop_seconds=args.gop_seconds,
        scene_threshold=args.scene_threshold,
        scene_cuts=not args.no_scene_cuts,
        metric=args.metric,
        vmaf_model=args.vmaf_model,
        vmaf_subsample=args.vmaf_subsample,
        rate_tolerance_pct=args.rate_tolerance,
        maxrate_ratio=args.maxrate_ratio,
        bufsize_ratio=args.bufsize_ratio,
        profile=args.profile,
        overwrite=args.overwrite,
        keep_work=args.keep_work,
        time_budget_seconds=args.time_budget,
        rungs=list(rungs),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        report = run_pipeline(args.input, args.output_dir, settings)
    except ff.FFError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    verification = report.get("verification", {})
    if not verification.get("passed", False):
        print("error: ladder verification failed; see report.json for details",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
