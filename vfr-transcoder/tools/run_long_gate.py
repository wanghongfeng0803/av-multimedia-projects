#!/usr/bin/env python3
"""Long-form qualification gate.

The brief is explicit: a 30 second sample does not count. This script runs the
complete chain on a genuinely long VFR file and fails unless every head /
middle / tail sync and cumulative-drift check passes.

    # minimum valid gate (30 minutes)
    python tools/run_long_gate.py --work-dir work/long --duration 1800

    # full 3 hour qualification
    python tools/run_long_gate.py --work-dir work/long3h --duration 10800

It writes a machine-readable report next to the output and prints the same
PASS/FAIL lines verify_sync produces, so it can serve as a CI gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import GENERATOR  # noqa: E402
from tools.verify_sync import parse_args as verify_parse, verify  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--duration", type=float, default=1800.0,
                   help="sample duration seconds; 1800=30min, 10800=3h")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--segment-seconds", type=float, default=120.0)
    p.add_argument("--marker-interval", type=float, default=300.0)
    args = p.parse_args(argv)

    if args.duration < 1800.0:
        print(f"REFUSING: duration {args.duration}s < 1800s; 30s samples do not "
              "count. Use --duration 1800 or more.", file=sys.stderr)
        return 2

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"vfr_{int(args.duration)}s.mkv"
    out = work / f"cfr_{int(args.duration)}s.mp4"
    report = work / "gate_report.json"

    steps: list[dict] = []

    def run_step(name: str, cmd: list[str]) -> None:
        print(f"\n=== {name} ===")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        elapsed = time.time() - t0
        steps.append({"step": name, "seconds": round(elapsed, 1),
                      "rc": proc.returncode})
        print(f"({elapsed:.0f}s, rc={proc.returncode})")
        if proc.returncode != 0:
            print(proc.stderr[-3000:], file=sys.stderr)
            report.write_text(json.dumps({"ok": False, "steps": steps}, indent=2))
            raise SystemExit(f"step failed: {name}")

    if not src.exists():
        run_step("generate VFR long sample", [
            sys.executable, str(GENERATOR),
            "--duration", str(args.duration), "--fps", str(args.fps),
            "--marker-interval", str(args.marker_interval),
            str(src)])
    else:
        print(f"reusing existing sample {src}")

    run_step("transcode (resumable pipeline)", [
        sys.executable, "-m", "vfr_transcoder",
        "--work-root", str(work / "state"),
        "transcode", str(src), str(out),
        "--segment-seconds", str(args.segment_seconds)])

    vargs = verify_parse([str(out), "--source", str(src)])
    ok, lines = verify(vargs)
    print()
    print("\n".join(lines))

    report.write_text(json.dumps({
        "ok": ok, "duration_seconds": args.duration,
        "source": str(src), "output": str(out), "steps": steps,
        "checks": lines,
    }, indent=2))
    print(f"\nreport written to {report}")
    print("LONG GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
