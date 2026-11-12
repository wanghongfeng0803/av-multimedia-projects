"""Thin wrappers around the ffmpeg/ffprobe binaries.

The tool deliberately has no third-party Python dependencies: everything is
driven through the ffmpeg CLI.  Logs are written next to the output so a run
can be audited afterwards.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple

from .config import SourceInfo

NULL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"

_ffmpeg_bin: Optional[str] = None
_ffprobe_bin: Optional[str] = None
_version_info: Optional[dict] = None
_tool_lock = threading.Lock()


class FFError(RuntimeError):
    """ffmpeg/ffprobe invocation failed."""

    def __init__(self, message: str, command: Optional[List[str]] = None,
                 output: str = ""):
        super().__init__(message)
        self.command = command
        self.output = output


def _resolve_binary(env_var: str, default: str) -> str:
    candidate = os.environ.get(env_var) or default
    path = shutil.which(candidate)
    if path:
        return path
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate
    raise FFError(
        f"cannot find '{candidate}' (set {env_var} to the full binary path)"
    )


def ffmpeg_bin() -> str:
    global _ffmpeg_bin
    if _ffmpeg_bin is None:
        _ffmpeg_bin = _resolve_binary("FFMPEG_BIN", "ffmpeg")
    return _ffmpeg_bin


def ffprobe_bin() -> str:
    global _ffprobe_bin
    if _ffprobe_bin is None:
        _ffprobe_bin = _resolve_binary("FFPROBE_BIN", "ffprobe")
    return _ffprobe_bin


def version_info() -> dict:
    """Cached ffmpeg version + build configuration."""
    global _version_info
    with _tool_lock:
        if _version_info is not None:
            return _version_info
        try:
            proc = subprocess.run(
                [ffmpeg_bin(), "-hide_banner", "-version"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30,
            )
        except OSError as exc:
            raise FFError(f"failed to run ffmpeg: {exc}") from exc
        text = (proc.stdout or "") + (proc.stderr or "")
        first = text.splitlines()[0] if text else "unknown"
        config_line = ""
        for line in text.splitlines():
            if line.startswith("configuration:"):
                config_line = line[len("configuration:"):].strip()
                break
        _version_info = {
            "ffmpeg": first.strip(),
            "configuration": config_line,
        }
        return _version_info


def has_encoder(name: str) -> bool:
    try:
        proc = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
    except OSError:
        return False
    return re.search(rf"\b{re.escape(name)}\b", (proc.stdout or "")) is not None


def has_filter(name: str) -> bool:
    try:
        proc = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-filters"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
    except OSError:
        return False
    return re.search(rf"\b{re.escape(name)}\b", (proc.stdout or "")) is not None


@dataclass
class RunResult:
    args: List[str]
    returncode: int
    stdout: str
    stderr: str
    log_path: Optional[str] = None


def run_ffmpeg(args: List[str], log_path: Optional[str] = None,
               cwd: Optional[str] = None, timeout: Optional[float] = None,
               check: bool = True) -> RunResult:
    """Run ffmpeg, tee stdout/stderr to a log file, and raise on failure."""
    cmd = [ffmpeg_bin()] + args
    display = " ".join(_quote_arg(a) for a in cmd)
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n$ {display}\n")
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write(proc.stderr)
    result = RunResult(args=args, returncode=proc.returncode,
                       stdout=proc.stdout or "", stderr=proc.stderr or "",
                       log_path=log_path)
    if check and proc.returncode != 0:
        tail = _tail((proc.stdout or "") + (proc.stderr or ""), 2500)
        raise FFError(
            f"ffmpeg exited with code {proc.returncode}\n{tail}",
            command=cmd, output=(proc.stdout or "") + (proc.stderr or ""),
        )
    return result


def _quote_arg(arg: str) -> str:
    if any(ch in arg for ch in " \t\"'&|<>"):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "... (truncated) ...\n" + text[-limit:]


def probe(path: str, timeout: Optional[float] = 120) -> dict:
    try:
        proc = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except OSError as exc:
        raise FFError(f"failed to run ffprobe: {exc}") from exc
    if proc.returncode != 0:
        raise FFError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFError(f"invalid ffprobe JSON for {path}: {exc}") from exc


def _parse_fraction(text: Optional[str], fallback: Fraction) -> Fraction:
    if not text:
        return fallback
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            n, d = int(num), int(den)
            if n > 0 and d > 0:
                return Fraction(n, d)
        else:
            val = float(text)
            if val > 0:
                return Fraction(val).limit_denominator(100000)
    except (ValueError, ZeroDivisionError):
        pass
    return fallback


def _rotation_of(stream: dict) -> int:
    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotation = int(round(float(side["rotation"])))
            except (TypeError, ValueError):
                rotation = 0
    if not rotation:
        try:
            rotation = int(round(float(stream.get("tags", {}).get("rotate", 0))))
        except (TypeError, ValueError):
            rotation = 0
    return rotation % 360


def source_info(path: str) -> SourceInfo:
    data = probe(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFError(f"no video stream found in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise FFError(f"invalid video dimensions for {path}: {width}x{height}")

    rotation = _rotation_of(video)
    display_w, display_h = width, height
    if rotation in (90, 270):
        display_w, display_h = height, width

    sar = video.get("sample_aspect_ratio") or "1:1"
    try:
        sar_num, sar_den = (int(x) for x in sar.split(":"))
        if sar_num <= 0 or sar_den <= 0:
            raise ValueError
    except (ValueError, AttributeError):
        sar_num, sar_den = 1, 1

    fps = _parse_fraction(video.get("avg_frame_rate"), Fraction(0))
    if fps <= 0:
        fps = _parse_fraction(video.get("r_frame_rate"), Fraction(0))
    if fps <= 0:
        fps = Fraction(25)

    duration = 0.0
    for candidate in (video.get("duration"), fmt.get("duration")):
        try:
            value = float(candidate)
            if value > 0:
                duration = value
                break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        try:
            duration = float(video.get("nb_frames")) / float(fps)
        except (TypeError, ValueError):
            raise FFError(f"cannot determine duration of {path}")

    frame_count = 0
    try:
        frame_count = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0:
        frame_count = int(round(duration * float(fps)))

    try:
        start_time = float(video.get("start_time") or fmt.get("start_time") or 0.0)
    except (TypeError, ValueError):
        start_time = 0.0

    pix_fmt = str(video.get("pix_fmt") or "yuv420p")
    if "10" in pix_fmt:
        bit_depth = 10
    elif "12" in pix_fmt:
        bit_depth = 12
    elif "9" in pix_fmt:
        bit_depth = 9
    else:
        bit_depth = 8

    try:
        br = float(video.get("bit_rate"))
        br_kbps = br / 1000.0 if br > 0 else None
    except (TypeError, ValueError):
        br_kbps = None

    color = {}
    for key in ("color_range", "color_space", "color_primaries",
                "color_transfer", "color_trc"):
        value = video.get(key)
        if value and value not in ("unknown", "unspecified", "reserved"):
            color[key.replace("color_transfer", "color_trc")] = value

    return SourceInfo(
        path=os.path.abspath(path),
        width=display_w,
        height=display_h,
        coded_width=width,
        coded_height=height,
        sar_num=sar_num,
        sar_den=sar_den,
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        video_codec=str(video.get("codec_name") or "unknown"),
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        audio_channels=int(audio.get("channels")) if audio and audio.get("channels") else None,
        start_time=start_time,
        rotation=rotation,
        color=color,
        bit_rate_kbps=br_kbps,
    )


def video_stream_duration_and_frames(path: str, fps_fallback: float) -> Tuple[float, int, Optional[float]]:
    """Return (duration, frame_count, stream bit_rate_kbps) for the video stream."""
    data = probe(path)
    video = next((s for s in data.get("streams", [])
                  if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFError(f"no video stream found in {path}")
    duration = 0.0
    for candidate in (video.get("duration"), data.get("format", {}).get("duration")):
        try:
            value = float(candidate)
            if value > 0:
                duration = value
                break
        except (TypeError, ValueError):
            continue
    try:
        frame_count = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if duration <= 0 and frame_count > 0 and fps_fallback > 0:
        duration = frame_count / fps_fallback
    if frame_count <= 0 and duration > 0 and fps_fallback > 0:
        frame_count = int(round(duration * fps_fallback))
    br = None
    try:
        val = float(video.get("bit_rate"))
        br = val / 1000.0 if val > 0 else None
    except (TypeError, ValueError):
        br = None
    return duration, frame_count, br


def measure_video_bitrate_kbps(path: str, fps_fallback: float) -> Tuple[float, dict]:
    """Measure the *video elementary stream* bitrate from packet sizes.

    Container files include audio and muxing overhead; the brief's per-rung
    bitrate is about the video stream, so packets of stream v:0 are summed.
    """
    data = probe(path, timeout=300)
    video = next((s for s in data.get("streams", [])
                  if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFError(f"no video stream found in {path}")

    duration = 0.0
    for candidate in (video.get("duration"), data.get("format", {}).get("duration")):
        try:
            value = float(candidate)
            if value > 0:
                duration = value
                break
        except (TypeError, ValueError):
            continue
    try:
        frame_count = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if duration <= 0 and frame_count > 0 and fps_fallback > 0:
        duration = frame_count / fps_fallback

    try:
        proc = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
             "-show_packets", "-show_entries", "packet=size",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600,
        )
    except OSError as exc:
        raise FFError(f"ffprobe packet scan failed: {exc}") from exc
    if proc.returncode != 0:
        raise FFError(f"ffprobe packet scan failed for {path}: {proc.stderr.strip()}")
    packets = json.loads(proc.stdout or "{}").get("packets", [])
    total_bytes = 0
    for p in packets:
        try:
            total_bytes += int(p.get("size") or 0)
        except (TypeError, ValueError):
            pass
    if duration <= 0:
        raise FFError(f"cannot determine video duration for {path}")
    kbps = (total_bytes * 8.0) / duration / 1000.0
    detail = {
        "video_bytes": total_bytes,
        "video_duration_seconds": round(duration, 6),
        "frame_count": frame_count,
        "packet_count": len(packets),
        "stream_bit_rate_kbps": None,
    }
    try:
        val = float(video.get("bit_rate"))
        if val > 0:
            detail["stream_bit_rate_kbps"] = round(val / 1000.0, 3)
    except (TypeError, ValueError):
        pass
    return kbps, detail


def keyframes(path: str, fps: float) -> List[Tuple[int, float]]:
    """Return [(frame_index, pts_time)] for every keyframe, via skip_frame nokey."""
    try:
        proc = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
             "-skip_frame", "nokey", "-show_frames",
             "-show_entries", "frame=key_frame,pts_time",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300,
        )
    except OSError as exc:
        raise FFError(f"ffprobe keyframe scan failed: {exc}") from exc
    if proc.returncode != 0:
        raise FFError(f"ffprobe keyframe scan failed for {path}: {proc.stderr.strip()}")
    frames = json.loads(proc.stdout or "{}").get("frames", [])
    out: List[Tuple[int, float]] = []
    for fr in frames:
        if fr.get("key_frame") != 1:
            continue
        try:
            pts = float(fr.get("pts_time"))
        except (TypeError, ValueError):
            continue
        idx = int(round(pts * fps))
        out.append((idx, pts))
    out.sort(key=lambda t: t[0])
    return out


def audio_md5(path: str) -> Optional[str]:
    """MD5 over the first audio stream's packets (None when no audio)."""
    try:
        info = probe(path)
    except FFError:
        return None
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        return None
    proc = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-c", "copy", "-f", "md5", "-"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    if proc.returncode != 0:
        return None
    match = re.search(r"MD5=([0-9a-fA-F]+)", proc.stdout or "")
    return match.group(1).lower() if match else None


def audio_stream_info(path: str) -> Optional[dict]:
    data = probe(path)
    audio = next((s for s in data.get("streams", [])
                  if s.get("codec_type") == "audio"), None)
    if not audio:
        return None
    return {
        "codec": audio.get("codec_name"),
        "sample_rate": audio.get("sample_rate"),
        "channels": audio.get("channels"),
        "channel_layout": audio.get("channel_layout"),
    }


def frame_size(path: str) -> Tuple[int, int]:
    data = probe(path)
    video = next((s for s in data.get("streams", [])
                  if s.get("codec_type") == "video"), None)
    if not video:
        raise FFError(f"no video stream in {path}")
    return int(video.get("width") or 0), int(video.get("height") or 0)
