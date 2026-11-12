"""Ladder specification and shared data types."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import List, Optional


@dataclass(frozen=True)
class RungSpec:
    """One rung of the ladder."""

    name: str
    width: int
    height: int
    nominal_kbps: int

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def label(self) -> str:
        return f"{self.name} ({self.width}x{self.height}, {self.nominal_kbps} kb/s)"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "nominal_kbps": self.nominal_kbps,
        }


# The four delivery rungs required by the brief.
DEFAULT_RUNGS: List[RungSpec] = [
    RungSpec("1080p", 1920, 1080, 5000),
    RungSpec("720p", 1280, 720, 2500),
    RungSpec("480p", 854, 480, 1000),
    RungSpec("360p", 640, 360, 600),
]


@dataclass
class SourceInfo:
    """Everything the pipeline needs to know about the source video."""

    path: str
    width: int                 # display width (rotation applied)
    height: int                # display height (rotation applied)
    coded_width: int
    coded_height: int
    sar_num: int
    sar_den: int
    fps: Fraction
    duration: float            # seconds
    frame_count: int
    video_codec: str
    pix_fmt: str
    bit_depth: int
    has_audio: bool
    audio_codec: Optional[str]
    audio_channels: Optional[int]
    start_time: float
    rotation: int
    color: dict = field(default_factory=dict)
    bit_rate_kbps: Optional[float] = None

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    @property
    def fps_expr(self) -> str:
        """FPS as an ffmpeg filter expression, preserving NTSC fractions."""
        if self.fps.denominator == 1:
            return str(self.fps.numerator)
        return f"{self.fps.numerator}/{self.fps.denominator}"

    @property
    def display_aspect(self) -> float:
        sar = self.sar_num / self.sar_den if self.sar_den else 1.0
        return (self.width * sar) / self.height if self.height else 16 / 9

    def frame_duration(self) -> float:
        return 1.0 / self.fps_float

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "coded_width": self.coded_width,
            "coded_height": self.coded_height,
            "sample_aspect_ratio": f"{self.sar_num}:{self.sar_den}",
            "display_aspect_ratio": round(self.display_aspect, 4),
            "fps": self.fps_expr,
            "fps_float": round(self.fps_float, 6),
            "duration_seconds": round(self.duration, 4),
            "frame_count": self.frame_count,
            "video_codec": self.video_codec,
            "pix_fmt": self.pix_fmt,
            "bit_depth": self.bit_depth,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "start_time": self.start_time,
            "rotation": self.rotation,
            "bit_rate_kbps": self.bit_rate_kbps,
            "color": self.color,
        }


@dataclass
class Settings:
    """Run-wide settings."""

    preset: str = "fast"
    threads: int = 8
    gop_seconds: float = 2.0
    scene_threshold: float = 0.35
    scene_cuts: bool = True
    metric: str = "vmaf"               # "vmaf" | "ssim"
    vmaf_model: Optional[str] = None
    vmaf_subsample: int = 0            # 0 = auto
    rate_tolerance_pct: float = 3.0
    maxrate_ratio: float = 1.5
    bufsize_ratio: float = 3.0
    profile: str = "high"
    pix_fmt: str = "yuv420p"
    overwrite: bool = False
    keep_work: bool = False
    time_budget_seconds: float = 1500.0
    rungs: List[RungSpec] = field(default_factory=lambda: list(DEFAULT_RUNGS))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rungs"] = [r.to_dict() for r in self.rungs]
        return d


def load_ladder(path: str) -> List[RungSpec]:
    """Load a custom ladder from JSON.

    Accepted shapes:
      {"rungs": [{"name": ..., "width": ..., "height": ..., "kbps": ...}, ...]}
      [{"name": ..., "width": ..., "height": ..., "kbps": ...}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("rungs") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError("ladder JSON must contain a non-empty list of rungs")

    rungs: List[RungSpec] = []
    for i, item in enumerate(items):
        try:
            width = int(item["width"])
            height = int(item["height"])
            kbps = int(item.get("kbps", item.get("nominal_kbps")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid rung #{i + 1}: {item!r}") from exc
        if width <= 0 or height <= 0 or kbps <= 0:
            raise ValueError(f"invalid dimensions/bitrate in rung #{i + 1}: {item!r}")
        if width % 2 or height % 2:
            raise ValueError(f"rung #{i + 1} dimensions must be even: {width}x{height}")
        name = str(item.get("name") or f"{height}p")
        rungs.append(RungSpec(name=name, width=width, height=height, nominal_kbps=kbps))

    names = [r.name for r in rungs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate rung names in ladder: {names}")
    return rungs


def bpp(kbps: float, width: int, height: int, fps: float) -> float:
    """Bits per pixel per frame (compression density)."""
    if width <= 0 or height <= 0 or fps <= 0:
        return float("nan")
    return (kbps * 1000.0) / (width * height * fps)


def fit_dimensions(src_aspect: float, target_w: int, target_h: int) -> tuple:
    """Fit the source aspect inside target_w x target_h, preserving aspect.

    Returns even (width, height) values <= target.  A 16:9 source scaled to
    854x480 or 640x360 fits exactly (854 is the conventional 480p width).
    """
    if src_aspect <= 0 or not math.isfinite(src_aspect):
        src_aspect = 16 / 9
    target_aspect = target_w / target_h
    # 854x480 and 640x360 are the conventional 16:9 lower rungs; treat any
    # source within 1% of that ratio as an exact fit (sub-pixel stretch only)
    # instead of adding pillarbox bars.
    if abs(src_aspect - target_aspect) / target_aspect < 0.01:
        return target_w, target_h
    if src_aspect > target_aspect:
        fit_w = target_w
        fit_h = int(round(target_w / src_aspect))
    else:
        fit_h = target_h
        fit_w = int(round(target_h * src_aspect))
    fit_w = max(2, min(target_w, fit_w) // 2 * 2)
    fit_h = max(2, min(target_h, fit_h) // 2 * 2)
    return fit_w, fit_h
