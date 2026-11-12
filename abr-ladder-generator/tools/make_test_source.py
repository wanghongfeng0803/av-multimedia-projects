#!/usr/bin/env python3
"""Generate a reproducible synthetic test source for the ABR benchmark.

The clip is built from seven 5-second lavfi patterns (testsrc2, SMPTE bars,
testsrc, RGB testsrc, mandelbrot, PAL bars, grainy testsrc2), spliced together
so it contains real scene cuts, then looped to the requested duration and
given a sine audio track.  Grain is added to make 5 Mb/s 1080p non-trivial.

Usage:
    python tools/make_test_source.py bench10m.mp4 --minutes 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


PATTERNS = [
    "testsrc2",
    "smptebars",
    "testsrc",
    "rgbtestsrc",
    "mandelbrot",
    "pal75bars",
    "testsrc2",
]


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def make_source(out_path: str, minutes: float = 10.0, width: int = 1920,
                height: int = 1080, fps: int = 30, grain: int = 3) -> None:
    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found (set FFMPEG_BIN)")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    duration = minutes * 60.0

    with tempfile.TemporaryDirectory(prefix="abr_source_") as tmp:
        segments = []
        for i, pattern in enumerate(PATTERNS):
            seg = os.path.join(tmp, f"seg{i}.mp4")
            run([ffmpeg, "-hide_banner", "-y", "-f", "lavfi",
                 "-i", f"{pattern}=size={width}x{height}:rate={fps}",
                 "-t", "5", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "18", "-pix_fmt", "yuv420p", "-an", seg])
            segments.append(seg)

        concat_file = os.path.join(tmp, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write("file '" + seg.replace("\\", "/") + "'\n")
        joined = os.path.join(tmp, "joined.mp4")
        run([ffmpeg, "-hide_banner", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_file, "-c", "copy", joined])

        grainy = os.path.join(tmp, "grainy.mp4")
        run([ffmpeg, "-hide_banner", "-y", "-i", joined,
             "-vf", f"noise=alls={grain}:allf=t",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-an", grainy])

        tmp_out = os.path.join(tmp, "loop.mp4")
        run([ffmpeg, "-hide_banner", "-y", "-stream_loop", "-1",
             "-i", grainy,
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
             "-t", f"{duration:.3f}",
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ac", "2",
             "-movflags", "+faststart", tmp_out])
        shutil.move(tmp_out, out_path)

    print(f"wrote {out_path} ({duration:.0f}s, {width}x{height}@{fps}, "
          f"grain={grain})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="output MP4 path")
    parser.add_argument("--minutes", type=float, default=10.0,
                        help="duration in minutes (default 10)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grain", type=int, default=3,
                        help="noise strength; 0 disables grain")
    args = parser.parse_args(argv)
    make_source(args.output, minutes=args.minutes, width=args.width,
                height=args.height, fps=args.fps, grain=args.grain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
