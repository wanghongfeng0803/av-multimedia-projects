"""Low level ffmpeg / ffprobe wrappers.

Design notes (validated against ffmpeg 9.x):

* ``-vsync`` was removed; use ``-fps_mode`` (``vfr`` / ``cfr`` / ``passthrough``).
* A filter's ``setpts`` works in *input* time-base ticks, not seconds. To author
  millisecond timestamps you must put ``settb=1/1000`` *before* ``setpts`` and
  also force ``-enc_time_base 1/1000`` -- otherwise the encoder time base
  (defaulting to the frame rate, e.g. 1/25) quantizes the millisecond pts back
  onto the coarse grid and silently destroys VFR.
* Inside a filtergraph, commas that belong to a function argument must be
  escaped (``mod(N+1\\,2)``); this module builds every command as an argv list
  so shell quoting is never involved.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import Settings, settings as default_settings

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str):
        super().__init__(
            f"ffmpeg failed rc={returncode}: {' '.join(str(c) for c in cmd[:12])} ...\n"
            f"{stderr[-2000:]}")
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr


def run_ffprobe(args: Sequence[str], cfg: Settings | None = None) -> dict:
    cfg = cfg or default_settings
    cmd = [cfg.ffprobe_bin, "-v", "error", *args, "-of", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr)
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(cmd, proc.returncode, f"bad ffprobe json: {exc}\n{proc.stdout}")


_PROGRESS_OUT_TIME = re.compile(r"^out_time_ms=(\d+)", re.MULTILINE)
_PROGRESS_FRAME = re.compile(r"^frame=(\d+)", re.MULTILINE)
_PROGRESS_DONE = re.compile(r"^progress=(end|continue)", re.MULTILINE)


@dataclass
class Progress:
    frame: int = 0
    out_time_ms: int = 0
    fps: float = 0.0
    speed: float = 0.0


def run_ffmpeg(
    args: Sequence[str],
    cfg: Settings | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Progress:
    """Run ffmpeg, parsing its machine-readable ``-progress`` channel.

    ``on_progress`` is invoked as frames are muxed; ``should_cancel`` polling
    True terminates the process (used by worker shutdown / kill simulation).
    The process is always waited on so no zombies are left behind.
    """
    cfg = cfg or default_settings
    cmd = [
        cfg.ffmpeg_bin, "-y", "-hide_banner", "-nostdin",
        "-loglevel", "error",
        *args,
        "-progress", "pipe:1",
    ]
    log.debug("ffmpeg argv: %s", cmd)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    prog = Progress()
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("frame="):
            try:
                prog.frame = int(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("out_time_ms="):
            # ffmpeg writes microseconds here despite the historical name.
            try:
                prog.out_time_ms = int(line.split("=", 1)[1]) // 1000
            except ValueError:
                pass
        elif line.startswith("fps="):
            try:
                prog.fps = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("speed="):
            val = line.split("=", 1)[1].rstrip("x")
            try:
                prog.speed = float(val)
            except ValueError:
                pass
        elif line.startswith("progress="):
            if on_progress is not None:
                on_progress(prog)
            if should_cancel is not None and should_cancel():
                proc.terminate()
                break
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    proc.wait()
    # release the pipe wrappers explicitly (avoids ResourceWarning on Windows)
    if proc.stdout is not None:
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, stderr)
    return prog


def probe_streams(path: str | Path, cfg: Settings | None = None) -> list[dict]:
    data = run_ffprobe(
        ["-show_entries",
         "stream=index,codec_type,codec_name,time_base,r_frame_rate,avg_frame_rate,"
         "start_time,duration,nb_frames,width,height,channels,sample_rate",
         str(path)], cfg)
    return data.get("streams", [])


def probe_format(path: str | Path, cfg: Settings | None = None) -> dict:
    data = run_ffprobe(
        ["-show_entries", "format=start_time,duration,bit_rate,format_name", str(path)],
        cfg)
    return data.get("format", {})


def read_packets(path: str | Path, stream: str, cfg: Settings | None = None) -> list[dict]:
    """Packet-level (file/decode order) pts/dts -- correct for DTS monotonicity."""
    data = run_ffprobe(
        ["-select_streams", stream,
         "-show_entries", "packet=pts_time,dts_time,duration_time,flags",
         str(path)], cfg)
    out = []
    for p in data.get("packets", []):
        if "pts_time" not in p:
            continue
        pts = float(p["pts_time"])
        dts_raw = p.get("dts_time")
        # Matroska omits DTS when it equals PTS; treat them as equal.
        dts = pts if dts_raw in (None, "N/A") else float(dts_raw)
        out.append({"pts": pts, "dts": dts,
                    "keyframe": "K" in (p.get("flags") or "")})
    return out


def read_frames(path: str | Path, stream: str, cfg: Settings | None = None) -> list[float]:
    """Frame-level presentation timestamps (presentation order)."""
    data = run_ffprobe(
        ["-select_streams", stream,
         "-show_entries", "frame=pts_time,pict_type", str(path)], cfg)
    return [float(f["pts_time"]) for f in data.get("frames", []) if "pts_time" in f]
