#!/usr/bin/env python3
"""Generate synthetic VFR long-form test material with B-frame reorder.

The generated file deliberately exhibits every property called out in the
brief's "old material" description:

* video time base exactly ``1/1000`` (millisecond PTS, like the legacy sources)
* genuine VFR: per-frame deltas jitter by +/-12 ms around the nominal frame
  duration using a deterministic bounded function (no external RNG, fully
  reproducible), mean rate equal to the chosen nominal fps
* libx264 with B-frames so packet PTS/DTS are reordered in the file
* a non-zero start offset so copyts handling is actually exercised
* synchronized content markers: a flashing red box in the video and a 1 kHz
  beep in the audio at the same *content* times (head / every interval / tail),
  giving verify_sync.py a content-level A/V sync signal independent of packet
  timestamps

Examples
--------
30 minute gate sample (default nominal 25 fps)::

    python tools/make_vfr_sample.py --duration 1800 samples/vfr_30min.mkv

Full 3 hour qualification sample::

    python tools/make_vfr_sample.py --duration 10800 --marker-interval 300 \\
        samples/vfr_3h.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Bounded jitter in MILLISECONDS. Alternating +10/-10 ms around the nominal
# frame duration produces genuine VFR (deltas alternate 30 ms / 50 ms at
# 25 fps) with zero long-run mean, so the stream's average rate stays exactly
# nominal and the fps filter maps it to a drift-free CFR grid. Marker periods
# use an even frame count so the alternation is balanced across every block.
JITTER_EXPR = "10*(2*mod(N\\,2)-1)"


def build_command(args: argparse.Namespace, out: Path) -> list[str]:
    fps = args.fps
    frame_dur_ms = 1000.0 / fps
    marker = args.marker_interval
    # Choose an EVEN frame count per marker period so the alternating jitter
    # zero-sums exactly on every boundary. The audio beep then uses the SAME
    # period expressed in seconds (period_frames / fps), so the frame-gated
    # video markers and the wall-clock audio beeps share one clock and never
    # drift apart, even across a 3 hour file.
    period_frames = max(2, int(round(marker * fps)))
    if period_frames % 2:
        period_frames += 1
    period_seconds = period_frames / fps
    hold_frames = max(1, int(round(args.marker_hold * fps)))
    hold_seconds = hold_frames / fps
    box_x = max(0, min(20, args.width - 60))
    box_y = max(0, min(20, args.height - 60))

    # Video: marker box first (frame-index gated), then millisecond time base
    # and VFR PTS. settb MUST precede setpts (see ffmpeg_utils docs).
    # Two expression-language traps here:
    #   * drawbox's enable uses lowercase frame number ``n`` (uppercase N only
    #     exists inside setpts);
    #   * function-arg commas stay backslash-escaped even inside the quotes.
    video_chain = (
        f"drawbox=x={box_x}:y={box_y}:w=50:h=50:color=red:t=fill:"
        f"enable='lt(mod(n\\,{period_frames})\\,{hold_frames})',"
        f"settb=1/1000,"
        f"setpts=N*{frame_dur_ms:.6f}+({JITTER_EXPR})+{args.start_offset_ms}"
    )
    # Audio marker mirrors the video marker period exactly (same clock).
    audio_src = (
        f"aevalsrc=0.3*sin(2*PI*1000*t)*lt(mod(t\\,{period_seconds:.6f})\\,{hold_seconds:.6f})"
        f":s={args.sample_rate}:d={args.duration}"
    )

    cmd = [
        args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        f"testsrc=size={args.width}x{args.height}:rate={fps}:duration={args.duration}",
        "-f", "lavfi", "-i", audio_src,
        "-filter_complex",
        f"[0:v]{video_chain}[v];"
        # Audio marker mirrors the video box on the same content-time clock;
        # audio gets the same start offset so source A/V is aligned at the
        # origin. "0.120*SR" (samples) is time-base independent -- inside a
        # filter_complex the audio tb is 1/sample_rate, so a "/TB" form would
        # be interpreted as a 120 *second* shift.
        f"[1:a]asetpts=PTS+{args.start_offset_ms/1000.0}*SR[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
        "-bf", str(args.bframes), "-g", str(int(fps * args.gop_seconds)),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", str(args.sample_rate), "-ac", "2",
        # without these two, the encoder time base (1/fps) quantizes the
        # millisecond PTS back onto the coarse CFR grid and VFR disappears
        "-enc_time_base", "1/1000",
        "-fps_mode", "vfr",
    ]
    if out.suffix.lower() == ".mp4":
        cmd += ["-video_track_timescale", "1000"]
    cmd.append(str(out))
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--duration", type=float, default=1800.0,
                   help="content duration in seconds (default 1800 = 30 min; use 10800 for 3 h)")
    p.add_argument("--fps", type=float, default=25.0, help="nominal frame rate")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--start-offset-ms", type=int, default=120,
                   help="non-zero first PTS to exercise copyts")
    p.add_argument("--marker-interval", type=float, default=60.0,
                   help="seconds between A/V sync markers")
    p.add_argument("--marker-hold", type=float, default=0.2,
                   help="duration of each marker (box/beep) in seconds")
    p.add_argument("--bframes", type=int, default=2)
    p.add_argument("--gop-seconds", type=float, default=2.0)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--preset", default="ultrafast")
    p.add_argument("--crf", type=int, default=28, help="low quality is fine for tests")
    p.add_argument("--ffmpeg", default="ffmpeg")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, args.output)
    print("generating", args.output, f"({args.duration:.0f}s, nominal {args.fps} fps)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        return proc.returncode
    print("done ->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
