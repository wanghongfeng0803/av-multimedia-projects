"""Source media analysis.

The hard case from the brief: VFR video with a 1/1000 time base and B-frame
reorder. ``r_frame_rate`` on such files still reports the *nominal* rate and is
not trustworthy for choosing the CFR target, so we estimate the dominant frame
rate from the actual packet deltas (median, snapped to common rates) and verify
that the stream really is VFR by measuring delta spread.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import ffmpeg_utils
from .config import Settings, settings as default_settings

COMMON_FPS = (12, 15, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60)


def parse_fraction(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def snap_to_common_fps(fps: float) -> float:
    return min(COMMON_FPS, key=lambda c: abs(c - fps))


@dataclass
class SourceInfo:
    path: Path
    duration: float
    video_index: int
    audio_index: int | None
    video_time_base: str
    video_start: float          # first video packet pts (seconds)
    audio_start: float | None   # first audio packet pts (seconds)
    start_origin: float         # copyts origin to re-apply on output
    r_frame_rate: float | None
    avg_frame_rate: float | None
    measured_fps: float         # median reciprocal delta
    dominant_fps: float         # snapped to a common rate
    is_vfr: bool
    has_b_frames: bool
    video_packets: int

    @property
    def default_target_fps(self) -> float:
        return self.dominant_fps


def _first_video_stream(streams: list[dict]) -> dict:
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    raise ValueError("no video stream found in source")


def analyze_source(path: str | Path, cfg: Settings | None = None) -> SourceInfo:
    cfg = cfg or default_settings
    path = Path(path)
    streams = ffmpeg_utils.probe_streams(path, cfg)
    fmt = ffmpeg_utils.probe_format(path, cfg)
    vstream = _first_video_stream(streams)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # ffmpeg 9 matroska rejects numeric "0:N" specifiers; use type-based ones.
    packets = ffmpeg_utils.read_packets(path, "v:0", cfg)
    if len(packets) < 2:
        raise ValueError("not enough video packets to analyze frame timing")

    # Packets are in FILE order, which is DTS order; B-frame reorder makes the
    # pts sequence non-monotonic there. Sort by presentation time before
    # measuring frame intervals, otherwise the "deltas" are decode reorder
    # distances and the fps estimate is garbage.
    pts = sorted(p["pts"] for p in packets)
    positive_deltas = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)
                       if pts[i + 1] > pts[i]]
    if not positive_deltas:
        raise ValueError("no positive presentation intervals found")
    # For symmetric VFR jitter the intervals alternate short/long around the
    # nominal period (e.g. 20 ms / 60 ms around a 40 ms nominal frame). The
    # median is biased to whichever alternation has one more sample; the MEAN
    # is unbiased because the jitter zero-sums, so use it for the nominal rate.
    mean_delta = statistics.fmean(positive_deltas)
    measured_fps = 1.0 / mean_delta if mean_delta > 0 else 0.0
    rfr = parse_fraction(vstream.get("r_frame_rate"))
    afr = parse_fraction(vstream.get("avg_frame_rate"))
    # Prefer the container's declared nominal rate when it is plausible and
    # close to the measured mean; fall back to the measured value otherwise.
    if rfr and 5.0 <= float(rfr) <= 240.0 and (
            not (5.0 <= measured_fps <= 240.0)
            or abs(float(rfr) - measured_fps) <= 1.5):
        measured_fps = float(rfr)
    elif not (5.0 <= measured_fps <= 240.0) and rfr:
        measured_fps = float(rfr)
    dominant = snap_to_common_fps(measured_fps)
    # VFR if intervals vary relative to their mean by more than a couple of
    # percent. An absolute 1-2 ms spread on a high-frame-rate CFR must not be
    # misclassified; the 1 ms time-base quantization alone stays well under 2%.
    rel_spread = (max(positive_deltas) - min(positive_deltas)) / max(mean_delta, 1e-9)
    is_vfr = rel_spread > 0.02

    has_b = any(p["pts"] != p["dts"] for p in packets)

    duration = float(fmt.get("duration") or vstream.get("duration") or 0.0)
    video_start = float(vstream.get("start_time") or pts[0])
    audio_start = float(astream["start_time"]) if astream and astream.get("start_time") else None

    # copyts origin: earliest presentation time across the streams we carry,
    # so the output timeline starts exactly where the source timeline started.
    origins = [video_start]
    if audio_start is not None:
        origins.append(audio_start)
    start_origin = min(origins)

    return SourceInfo(
        path=path,
        duration=duration,
        video_index=int(vstream["index"]),
        audio_index=int(astream["index"]) if astream else None,
        video_time_base=vstream.get("time_base", "1/1000"),
        video_start=video_start,
        audio_start=audio_start,
        start_origin=start_origin,
        r_frame_rate=float(rfr) if rfr else None,
        avg_frame_rate=float(afr) if afr else None,
        measured_fps=measured_fps,
        dominant_fps=dominant,
        is_vfr=is_vfr,
        has_b_frames=has_b,
        video_packets=len(packets),
    )
