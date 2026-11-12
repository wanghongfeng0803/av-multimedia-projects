"""Frame-exact shared keyframe scheduling.

All rungs are encoded from the same frames at the same frame rate, so a
keyframe schedule expressed as *frame indices* (rather than wall-clock
timestamps) is guaranteed to be identical in every rung.  ffmpeg's
``force_key_frames`` expression is evaluated on the frame counter (``n``),
which lets us force keyframes at exact frame numbers.

The schedule is:

  * a fixed grid every ``gop_seconds`` (default 2 s), and
  * content scene cuts detected once on the source at 320x180,

unioned together.  Keyframes are closed-GOP IDRs; scene-cut keyframes improve
quality right after cuts without letting GOPs drift between rungs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import ff


@dataclass
class SceneCut:
    time: float
    score: float
    frame_index: int = 0


@dataclass
class KeyframePlan:
    fps: float
    gop_seconds: float
    grid_step_frames: int
    frame_indices: List[int]
    scene_cuts: List[SceneCut] = field(default_factory=list)
    grid_count: int = 0
    max_gap_frames: int = 0

    def expr(self) -> str:
        """ffmpeg -force_key_frames expression, evaluated per frame."""
        parts = [f"not(mod(n,{self.grid_step_frames}))"]
        for idx in self.frame_indices:
            if idx % self.grid_step_frames != 0:
                parts.append(f"eq(n,{idx})")
        return "expr:" + "+".join(parts)

    def times(self) -> List[float]:
        return [idx / self.fps for idx in self.frame_indices]

    def to_dict(self) -> dict:
        return {
            "gop_seconds": self.gop_seconds,
            "fps": self.fps,
            "grid_step_frames": self.grid_step_frames,
            "keyframe_count": len(self.frame_indices),
            "first_keyframe_frame": self.frame_indices[0] if self.frame_indices else None,
            "last_keyframe_frame": self.frame_indices[-1] if self.frame_indices else None,
            "max_gap_frames": self.max_gap_frames,
            "max_gap_seconds": round(self.max_gap_frames / self.fps, 4) if self.fps else None,
            "frame_indices": self.frame_indices,
            "scene_cuts": [
                {"time": round(c.time, 4), "score": round(c.score, 4),
                 "frame_index": c.frame_index}
                for c in self.scene_cuts
            ],
        }


def detect_scene_cuts(source: str, fps_expr: str, fps_float: float,
                      work_dir: str, threshold: float = 0.35,
                      analysis_width: int = 320,
                      decode_threads: int = 2,
                      log_path: Optional[str] = None) -> List[SceneCut]:
    """Detect hard cuts on a downscaled analysis stream.

    Runs ``select='gt(scene,T)'`` at 320x180 (roughly 50x realtime) and reads
    the frame timestamps that survive.  The time mapping matches the encode
    timeline because the analysis chain applies the same CFR ``fps`` filter.
    """
    os.makedirs(work_dir, exist_ok=True)
    meta_name = "scene_cuts.txt"
    meta_path = os.path.join(work_dir, meta_name)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    vf = (
        f"fps={fps_expr},"
        f"scale={analysis_width}:-2:flags=fast_bilinear,"
        f"select='gt(scene,{threshold})',"
        f"metadata=print:file={meta_name}"
    )
    ff.run_ffmpeg(
        ["-hide_banner", "-y", "-threads", str(decode_threads),
         "-i", source, "-map", "0:v:0", "-vf", vf, "-an",
         "-f", "null", ff.NULL_DEVICE],
        log_path=log_path, cwd=work_dir,
    )

    cuts: List[SceneCut] = []
    if not os.path.exists(meta_path):
        return cuts
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    pending: Optional[SceneCut] = None
    for line in lines:
        stripped = line.strip()
        m = re.match(r"frame:\d+\s+pts:\S+\s+pts_time:([0-9.]+)", stripped)
        if m:
            if pending is not None:
                cuts.append(pending)
            try:
                t = float(m.group(1))
            except ValueError:
                t = 0.0
            pending = SceneCut(time=t, score=0.0,
                               frame_index=int(round(t * fps_float)))
            continue
        m2 = re.match(r"lavfi\.scene_score=([0-9.]+)", stripped)
        if m2 and pending is not None:
            try:
                pending.score = float(m2.group(1))
            except ValueError:
                pass
    if pending is not None:
        cuts.append(pending)
    return cuts


def build_keyframe_plan(scene_cuts: List[SceneCut], fps: float, frame_count: int,
                        gop_seconds: float = 2.0,
                        min_cut_spacing: float = 0.5,
                        close_to_grid: float = 0.15) -> KeyframePlan:
    """Union of a fixed GOP grid and scene cuts, on frame indices.

    A cut is skipped when it is within ``close_to_grid`` seconds of a grid
    keyframe (the grid keyframe already covers it) or within
    ``min_cut_spacing`` of an accepted cut.  The grid is never removed, so
    the maximum GOP stays at ``gop_seconds``.
    """
    if fps <= 0:
        fps = 30.0
    step = max(1, int(round(gop_seconds * fps)))
    frame_count = max(1, int(frame_count))

    grid = list(range(0, frame_count, step))
    if grid and grid[-1] >= frame_count:
        grid = grid[:-1]
    if not grid:
        grid = [0]

    close_frames = max(1, int(round(close_to_grid * fps)))
    spacing_frames = max(1, int(round(min_cut_spacing * fps)))

    accepted: List[SceneCut] = []
    for cut in sorted(scene_cuts, key=lambda c: c.time):
        idx = int(round(cut.time * fps))
        if idx <= 0 or idx >= frame_count:
            continue
        if any(abs(idx - g) <= close_frames for g in grid):
            continue
        if accepted and idx - accepted[-1].frame_index < spacing_frames:
            continue
        cut.frame_index = idx
        accepted.append(cut)

    indices = sorted(set(grid) | {c.frame_index for c in accepted})
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    max_gap = max(gaps) if gaps else 0

    return KeyframePlan(
        fps=fps,
        gop_seconds=gop_seconds,
        grid_step_frames=step,
        frame_indices=indices,
        scene_cuts=accepted,
        grid_count=len(grid),
        max_gap_frames=max_gap,
    )


def plan_from_source(source_info, work_dir: str, gop_seconds: float = 2.0,
                     scene_threshold: float = 0.35, use_scene_cuts: bool = True,
                     threads: int = 2,
                     log_path: Optional[str] = None) -> KeyframePlan:
    cuts: List[SceneCut] = []
    if use_scene_cuts:
        cuts = detect_scene_cuts(
            source_info.path, source_info.fps_expr, source_info.fps_float,
            work_dir, threshold=scene_threshold,
            decode_threads=max(1, threads), log_path=log_path,
        )
    # The encode normalises to CFR, so the output frame count is duration*fps
    # (VFR sources would otherwise get a plan that does not match the output).
    cfr_frames = int(round(source_info.duration * source_info.fps_float))
    return build_keyframe_plan(
        cuts, source_info.fps_float, cfr_frames,
        gop_seconds=gop_seconds,
    )
