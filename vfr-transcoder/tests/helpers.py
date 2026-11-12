"""Shared helpers for the test suite."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

GENERATOR = TOOLS / "make_vfr_sample.py"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_vfr_sample(path: Path, duration: float = 12.0, fps: float = 25.0,
                    marker_interval: float = 4.0, container: str = "mkv") -> Path:
    """Generate a short VFR sample via the production generator script."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--duration", str(duration),
         "--fps", str(fps), "--marker-interval", str(marker_interval),
         "--width", "160", "--height", "120", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sample generation failed: {proc.stderr[-1500:]}")
    return out


def make_workdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="vfr_test_"))
    return d


def probe_json(path: Path, *args: str) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    import json
    return json.loads(proc.stdout)


def video_frame_pts(path: Path) -> list[float]:
    data = probe_json(path, "-select_streams", "v:0",
                      "-show_entries", "frame=pts_time")
    return [float(f["pts_time"]) for f in data["frames"]]


def is_process_alive(pid: int) -> bool:
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                           capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid: int) -> None:
    """Hard kill (SIGKILL semantics) -- simulates `kill -9` on every OS."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
    else:
        os.kill(pid, 9)
