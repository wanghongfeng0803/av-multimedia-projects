"""Process-wide tunables. Everything can be overridden via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # storage
    work_root: Path = field(default_factory=lambda: Path(os.environ.get("VFR_WORK_ROOT", "./work")))
    db_path: Path | None = None  # derived: work_root/jobs.db

    # pipeline defaults
    segment_seconds: float = field(default_factory=lambda: _env_float("VFR_SEGMENT_SECONDS", 60.0))
    video_codec: str = field(default_factory=lambda: os.environ.get("VFR_VCODEC", "libx264"))
    audio_codec: str = field(default_factory=lambda: os.environ.get("VFR_ACODEC", "aac"))
    preset: str = field(default_factory=lambda: os.environ.get("VFR_PRESET", "medium"))
    crf: int = field(default_factory=lambda: _env_int("VFR_CRF", 23))
    audio_bitrate: str = field(default_factory=lambda: os.environ.get("VFR_ABR", "192k"))
    gop_seconds: float = field(default_factory=lambda: _env_float("VFR_GOP_SECONDS", 2.0))
    b_frames: int = field(default_factory=lambda: _env_int("VFR_BFRAMES", 2))
    ffmpeg_bin: str = field(default_factory=lambda: os.environ.get("VFR_FFMPEG", "ffmpeg"))
    ffprobe_bin: str = field(default_factory=lambda: os.environ.get("VFR_FFPROBE", "ffprobe"))

    # state machine / worker
    lease_seconds: float = field(default_factory=lambda: _env_float("VFR_LEASE_SECONDS", 90.0))
    heartbeat_seconds: float = field(default_factory=lambda: _env_float("VFR_HEARTBEAT_SECONDS", 10.0))
    poll_seconds: float = field(default_factory=lambda: _env_float("VFR_POLL_SECONDS", 2.0))
    # ffmpeg progress heartbeat counts as alive at least this often
    progress_stall_timeout: float = field(
        default_factory=lambda: _env_float("VFR_STALL_TIMEOUT", 300.0))

    def __post_init__(self) -> None:
        self.work_root = Path(self.work_root)
        if self.db_path is None:
            self.db_path = self.work_root / "jobs.db"


settings = Settings()
