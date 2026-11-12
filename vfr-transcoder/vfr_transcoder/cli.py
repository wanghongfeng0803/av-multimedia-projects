"""Command line entry points.

Three modes::

    python -m vfr_transcoder transcode input.mp4 out.mp4 [--target-fps 25]
    python -m vfr_transcoder serve   [--host 0.0.0.0] [--port 8080]
    python -m vfr_transcoder worker  [--once]

The direct ``transcode`` mode still uses the JobStore, so it gets the same
segmented, resumable pipeline (its job lives under the work directory).
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings
from .media import analyze_source
from .pipeline import Transcoder
from .state import JobStore, JobState
from .worker import Worker


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vfr-transcoder",
                                description="VFR -> CFR transcoder with copyts and resume")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--work-root", default=None, help="state/work directory (default ./work)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcode", help="run a single transcode in this process")
    t.add_argument("input")
    t.add_argument("output")
    t.add_argument("--target-fps", type=float, default=None,
                   help="CFR target; defaults to the source dominant frame rate")
    t.add_argument("--segment-seconds", type=float, default=None)
    t.add_argument("--job-id", default=None, help="resume an existing job id instead of new")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)

    w = sub.add_parser("worker", help="run a worker process")
    w.add_argument("--once", action="store_true", help="process one job then exit")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    cfg = Settings()
    if getattr(args, "work_root", None):
        cfg.work_root = __import__("pathlib").Path(args.work_root)
        cfg.db_path = cfg.work_root / "jobs.db"
    if getattr(args, "segment_seconds", None):
        cfg.segment_seconds = args.segment_seconds
    return cfg


def cmd_transcode(args: argparse.Namespace, cfg: Settings) -> int:
    store = JobStore(cfg.db_path)
    try:
        if args.job_id:
            job = store.get_job(args.job_id)
            if job is None:
                print(f"no such job: {args.job_id}", file=sys.stderr)
                return 2
        else:
            info = analyze_source(args.input, cfg)
            fps = args.target_fps or info.dominant_fps
            job = store.create_job(args.input, args.output, fps)
            print(f"job {job.id}: {args.input} ({'VFR' if info.is_vfr else 'CFR'}, "
                  f"tb={info.video_time_base}, bframes={info.has_b_frames}, "
                  f"{info.duration:.1f}s) -> {fps} fps")
        worker = Worker(store, cfg)
        ok = worker.run_one(job.id)
        final = store.get_job(job.id)
        if not ok:
            print(f"FAILED: {final.error if final else 'unknown'}", file=sys.stderr)
            return 1
        print(f"completed: {final.output}  progress=100%  origin_pts={final.origin_pts}")
        return 0
    finally:
        store.close()


def cmd_serve(args: argparse.Namespace, cfg: Settings) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(cfg=cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_worker(args: argparse.Namespace, cfg: Settings) -> int:
    store = JobStore(cfg.db_path)
    try:
        worker = Worker(store, cfg)
        worker.run_forever(max_jobs=1 if args.once else None)
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = _settings(args)
    if args.command == "transcode":
        return cmd_transcode(args, cfg)
    if args.command == "serve":
        return cmd_serve(args, cfg)
    if args.command == "worker":
        return cmd_worker(args, cfg)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
