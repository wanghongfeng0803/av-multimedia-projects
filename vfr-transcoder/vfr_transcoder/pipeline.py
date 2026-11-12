"""Resumable VFR -> CFR transcode pipeline.

Why segments are MPEG-TS and why there is exactly one final remux
(validated in prototypes):

1. Each segment is encoded independently with ``-ss/-t`` before the input and
   the ``fps`` filter does the VFR->CFR timestamp remapping. The filter emits
   frames on an ideal CFR grid, so there is **no per-frame rounding that can
   accumulate** -- the error at frame N is the same bounded value as frame 0,
   which is what keeps a 3 hour source inside the same 40 ms budget as a short
   clip.
2. Segments are MPEG-TS: every segment starts on a keyframe and the concat
   demuxer continues timestamps across files instead of resetting them, so DTS
   stays strictly increasing across segment boundaries.
3. Audio is extracted once, whole, with normalized timestamps. Splitting audio
   would add encoder priming gaps at every boundary; a single extraction cannot.
4. One final mux from concatenated video + whole audio re-applies the source's
   original origin via ``-output_ts_offset`` (the copyts semantic) and writes
   atomically. An extra intermediate remux was shown to duplicate initial DTS,
   so it is deliberately avoided.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import ffmpeg_utils
from .config import Settings, settings as default_settings
from .ffmpeg_utils import Progress
from .media import SourceInfo, analyze_source
from .state import Job, JobStore

log = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]
"""progress fraction in [0,1], human message."""


@dataclass
class TranscodeResult:
    output: Path
    origin_pts: float
    target_fps: float
    segments: int
    duration: float


class Transcoder:
    def __init__(self, store: JobStore, cfg: Settings | None = None):
        self.store = store
        self.cfg = cfg or default_settings

    # -------------------------------------------------- paths

    def _job_dir(self, job: Job) -> Path:
        d = self.cfg.work_root / "jobs" / job.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _segment_path(self, job: Job, index: int) -> Path:
        return self._job_dir(job) / f"seg_{index:05d}.ts"

    def _audio_path(self, job: Job) -> Path:
        return self._job_dir(job) / "audio.m4a"

    def _concat_path(self, job: Job) -> Path:
        return self._job_dir(job) / "concat.txt"

    # -------------------------------------------------- segment plan

    def _plan_segments(self, duration: float) -> list[tuple[int, float, float]]:
        seg = max(1.0, float(self.cfg.segment_seconds))
        total = max(1, math.ceil(duration / seg))
        plan = []
        for i in range(total):
            start = i * seg
            length = min(seg, max(0.0, duration - start))
            if length <= 0:
                break
            plan.append((i, start, length))
        return plan

    # -------------------------------------------------- ffmpeg command builders

    def _segment_cmd(self, src: Path, index: int, start: float, length: float,
                     fps: float, out: Path) -> list[str]:
        gop = max(1, int(round(fps * self.cfg.gop_seconds)))
        return [
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
            "-map", "0:v:0", "-an",
            "-vf", f"fps={fps}",
            "-c:v", self.cfg.video_codec,
            "-preset", self.cfg.preset, "-crf", str(self.cfg.crf),
            "-g", str(gop), "-bf", str(self.cfg.b_frames),
            "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr",
            # fps filter already established the CFR rate; the encoder picks a
            # matching time base automatically (no -enc_time_base needed here,
            # and -1 is rejected by libx264 on ffmpeg 9). It matters only when
            # preserving VFR, which the sample generator does explicitly.
            "-muxdelay", "0",
            "-f", "mpegts",
            str(out),
        ]

    def _audio_cmd(self, src: Path, audio_index: int, out: Path) -> list[str]:
        return [
            "-i", str(src),
            "-map", "0:a:0",
            # normalize whole-file audio timeline to zero; single priming point
            "-af", "asetpts=N/SR/TB",
            "-c:a", self.cfg.audio_codec,
            "-b:a", self.cfg.audio_bitrate,
            "-f", "ipod",
            str(out),
        ]

    _MUXERS = {".mkv": "matroska", ".webm": "matroska", ".mp4": "mp4",
               ".mov": "mov", ".m4v": "mp4", ".ts": "mpegts", ".m2ts": "mpegts"}

    def _muxer_for(self, path: Path) -> str:
        return self._MUXERS.get(path.suffix.lower(), "matroska")

    def _final_cmd(self, video: Path, audio: Path | None, origin: float,
                   out_tmp: Path) -> list[str]:
        cmd = ["-i", str(video)]
        maps = ["-map", "0:v:0"]
        if audio is not None:
            cmd += ["-i", str(audio)]
            maps += ["-map", "1:a:0"]
        cmd += maps + [
            "-c", "copy",
            "-muxdelay", "0",
            # copyts: restore the original timeline origin for both streams
            "-output_ts_offset", f"{origin:.6f}",
            "-f", self._muxer_for(out_tmp),
            str(out_tmp),
        ]
        return cmd

    # -------------------------------------------------- main entry

    def run(
        self,
        job: Job,
        on_progress: ProgressCb | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> TranscodeResult:
        def report(frac: float, msg: str) -> None:
            log.info("[%s] %.1f%% %s", job.id, frac * 100, msg)
            if on_progress:
                on_progress(frac, msg)

        cfg = self.cfg
        src = Path(job.source)
        if not src.exists():
            raise FileNotFoundError(f"source missing: {src}")
        output = Path(job.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        info: SourceInfo = analyze_source(src, cfg)
        fps = job.target_fps or info.dominant_fps
        origin = info.start_origin
        plan = self._plan_segments(info.duration)
        self.store.init_segment_plan(job.id, len(plan), origin_pts=origin)
        report(0.0, f"source {info.duration:.1f}s vfr={info.is_vfr} "
                    f"bframes={info.has_b_frames} tb={info.video_time_base} "
                    f"-> {fps} fps, {len(plan)} segments, origin={origin:.3f}s")

        existing = {s["segment_index"]: s for s in self.store.get_segments(job.id)}

        # ---------------- video segments (the resumable unit) ----------------
        for idx, start, length in plan:
            if should_cancel and should_cancel():
                raise InterruptedError("cancelled")
            seg_path = self._segment_path(job, idx)
            done = idx in existing and Path(existing[idx]["path"]).exists() and \
                existing[idx]["path"] == str(seg_path)
            if done:
                report(self._video_frac(idx, len(plan)),
                       f"segment {idx+1}/{len(plan)} already encoded (resume)")
                continue

            # always write to a temp file then atomically replace, so a kill
            # mid-encode can never leave a plausible-but-truncated segment.
            # Keep the real extension BEFORE the .tmp suffix so the muxer is
            # still inferred from the format.
            tmp = seg_path.with_name(seg_path.name + f".tmp{os.getpid()}")
            cmd = self._segment_cmd(src, idx, start, length, fps, tmp)

            def seg_prog(p: Progress, _idx=idx, _start=start, _length=length):
                within = min(1.0, max(0.0, p.out_time_ms / 1000.0 / max(_length, 1e-6)))
                frac = self._video_frac(_idx + within, len(plan))
                self.store.heartbeat(job.id, cfg.lease_seconds, progress=frac)

            ffmpeg_utils.run_ffmpeg(cmd, cfg, on_progress=seg_prog,
                                    should_cancel=should_cancel)
            os.replace(tmp, seg_path)
            self.store.record_segment(job.id, idx, str(seg_path), start, length)
            report(self._video_frac(idx + 1, len(plan)),
                   f"segment {idx+1}/{len(plan)} done")

        # ---------------- audio (single whole-file extraction) ----------------
        audio_path: Path | None = None
        if info.audio_index is not None:
            audio_path = self._audio_path(job)
            if job.audio_done and audio_path.exists():
                report(0.92, "audio already extracted (resume)")
            else:
                tmp = audio_path.with_name(audio_path.name + f".tmp{os.getpid()}")
                ffmpeg_utils.run_ffmpeg(
                    self._audio_cmd(src, info.audio_index, tmp), cfg,
                    should_cancel=should_cancel)
                os.replace(tmp, audio_path)
                self.store.mark_audio_done(job.id)
                report(0.92, "audio extracted")

        # ---------------- concat + final atomic mux ----------------
        segments = self.store.get_segments(job.id)
        if len(segments) != len(plan):
            raise RuntimeError(
                f"segment count mismatch: {len(segments)} != {len(plan)}")
        concat = self._concat_path(job)
        with open(concat, "w", encoding="utf-8") as fh:
            fh.write("ffconcat version 1.0\n")
            for s in segments:
                # concat demuxer resolves entries relative to the LIST file's
                # directory, so a relative stored path would be doubled. Always
                # emit an absolute, forward-slashed path.
                fh.write(f"file '{Path(s['path']).resolve().as_posix()}'\n")

        # concat to a temp TS (stream copy, continuous timestamps)
        video_ts = self._job_dir(job) / "video.ts"
        ffmpeg_utils.run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-map", "0:v:0", "-c", "copy", "-muxdelay", "0",
            "-f", "mpegts", str(video_ts),
        ], cfg, should_cancel=should_cancel)

        out_tmp = output.with_name(output.name + f".tmp{os.getpid()}")
        ffmpeg_utils.run_ffmpeg(
            self._final_cmd(video_ts, audio_path, origin, out_tmp), cfg,
            should_cancel=should_cancel)
        os.replace(out_tmp, output)
        report(1.0, f"finalized {output}")

        return TranscodeResult(
            output=output, origin_pts=origin, target_fps=fps,
            segments=len(plan), duration=info.duration)

    @staticmethod
    def _video_frac(done: float, total: int) -> float:
        # video occupies 90% of the progress bar; audio 2%, mux 8%
        return 0.90 * min(1.0, done / max(1, total))
