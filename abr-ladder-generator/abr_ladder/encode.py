"""Two-pass encoding with closed-loop bitrate targeting.

Strategy
--------
1. Encode once with 2-pass VBR at the nominal bitrate (capped VBR with
   maxrate/bufsize so the result is stream-friendly).
2. Measure the *video stream* bitrate from packet sizes and correct the pass-2
   target by ``nominal / measured``.  Pass-1 statistics are target-independent,
   so each correction only costs one more pass-2 encode.
3. If the content is too easy to consume even the nominal rate (static scenes),
   VBR cannot spend the requested bits.  In that case fall back to CBR with
   x264 filler NALs, which guarantees the nominal rate.
4. Mux the prepared audio (identical bytes in every rung) into the final MP4.

Keyframe forcing and all structural encoder settings are identical between
pass 1, pass 2 and every rung; only resolution/bitrate differ.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from . import ff
from .config import RungSpec, Settings, SourceInfo, fit_dimensions
from .keyframes import KeyframePlan


@dataclass
class Attempt:
    mode: str
    target_kbps: float
    measured_kbps: float
    deviation_pct: float
    seconds: float

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target_kbps": round(self.target_kbps, 2),
            "measured_kbps": round(self.measured_kbps, 2),
            "deviation_pct": round(self.deviation_pct, 3),
            "seconds": round(self.seconds, 2),
        }


@dataclass
class EncodeResult:
    rung: RungSpec
    output_path: str
    measured_kbps: float
    deviation_pct: float
    mode: str
    target_kbps: float
    attempts: List[Attempt] = field(default_factory=list)
    encode_seconds: float = 0.0
    size_bytes: int = 0
    video_duration: float = 0.0
    frame_count: int = 0
    commands: List[str] = field(default_factory=list)
    audio_codec: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.rung.name,
            "width": self.rung.width,
            "height": self.rung.height,
            "nominal_kbps": self.rung.nominal_kbps,
            "measured_kbps": round(self.measured_kbps, 2),
            "deviation_pct": round(self.deviation_pct, 3),
            "target_kbps": round(self.target_kbps, 2),
            "mode": self.mode,
            "attempts": [a.to_dict() for a in self.attempts],
            "encode_seconds": round(self.encode_seconds, 2),
            "file_size_bytes": self.size_bytes,
            "video_duration_seconds": round(self.video_duration, 4),
            "frame_count": self.frame_count,
            "audio_codec": self.audio_codec,
            "output": os.path.basename(self.output_path),
            "commands": self.commands,
        }


def build_video_filter(source: SourceInfo, rung: RungSpec) -> str:
    fit_w, fit_h = fit_dimensions(source.display_aspect, rung.width, rung.height)
    parts = [f"scale={fit_w}:{fit_h}:flags=bicubic"]
    if (fit_w, fit_h) != (rung.width, rung.height):
        parts.append(
            f"pad={rung.width}:{rung.height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    parts += [
        "setsar=1",
        "format=yuv420p",
    ]
    return ",".join(parts)


def prepare_audio(source: SourceInfo, work_dir: str, log_path: str,
                  threads: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """Extract/copy the first audio track once and reuse it in every rung.

    Returns (audio_path, codec) or (None, None).  AAC is copied bit-exactly;
    everything else is transcoded to AAC so the mux step is uniform.  Using
    the same bytes in every rung means audio never jumps when a player
    switches representation.
    """
    if not source.has_audio:
        return None, None
    audio_path = os.path.join(work_dir, "audio.m4a")
    codec = "aac"

    def _transcode():
        bitrate = "192k" if (source.audio_channels or 2) > 2 else "128k"
        ff.run_ffmpeg(
            ["-hide_banner", "-y", "-threads", str(max(1, threads)),
             "-i", source.path, "-vn", "-sn", "-dn", "-map", "0:a:0",
             "-c:a", "aac", "-b:a", bitrate, "-ar", "48000",
             "-movflags", "+faststart", audio_path],
            log_path=log_path,
        )

    if source.audio_codec == "aac":
        try:
            ff.run_ffmpeg(
                ["-hide_banner", "-y", "-threads", str(max(1, threads)),
                 "-i", source.path, "-vn", "-sn", "-dn", "-map", "0:a:0",
                 "-c:a", "copy", "-movflags", "+faststart", audio_path],
                log_path=log_path,
            )
            return audio_path, codec
        except ff.FFError:
            # AAC copy can fail for unusual container layouts; transcode instead.
            _transcode()
            return audio_path, codec
    _transcode()
    return audio_path, codec


def _color_args(source: SourceInfo) -> List[str]:
    mapping = [
        ("color_range", "-color_range"),
        ("color_space", "-colorspace"),
        ("color_primaries", "-color_primaries"),
        ("color_trc", "-color_trc"),
    ]
    args: List[str] = []
    for key, flag in mapping:
        value = source.color.get(key)
        if value:
            args += [flag, value]
    return args


_VBR_X264_PARAMS = "keyint=9999:min-keyint=1:scenecut=0:open-gop=0"
_CBR_X264_PARAMS = "nal-hrd=cbr:filler=1:" + _VBR_X264_PARAMS


def _structural_args(source: SourceInfo, plan: KeyframePlan,
                     settings: Settings, threads: int,
                     cbr: bool = False) -> List[str]:
    return [
        "-c:v", "libx264",
        "-preset", settings.preset,
        "-profile:v", settings.profile,
        "-pix_fmt", settings.pix_fmt,
        "-refs", "4",
        "-bf", "3",
        "-threads", str(max(1, threads)),
        "-filter_threads", str(max(1, threads // 2)),
        "-force_key_frames", plan.expr(),
        "-x264-params", _CBR_X264_PARAMS if cbr else _VBR_X264_PARAMS,
    ]


def _base_args(source: SourceInfo, rung: RungSpec, plan: KeyframePlan,
               settings: Settings, threads: int,
               cbr: bool = False) -> List[str]:
    return (
        ["-hide_banner", "-y",
         "-threads", str(max(1, threads)),
         "-i", source.path,
         "-map", "0:v:0",
         "-vf", build_video_filter(source, rung),
         "-fps_mode", "cfr",
         "-r", source.fps_expr,
         "-video_track_timescale", "90000"]
        + _structural_args(source, plan, settings, threads, cbr=cbr)
        + _color_args(source)
        + ["-an", "-sn", "-dn"]
    )


def _encode_vbr_video(source: SourceInfo, rung: RungSpec, plan: KeyframePlan,
                      settings: Settings, threads: int, temp_video: str,
                      best_video: str, pass_prefix: str, log_path: str,
                      log_fn: Callable[[str], None],
                      commands: List[str]) -> List[Attempt]:
    """Run pass 1 once and pass 2 until the measured rate is in tolerance."""
    nominal = float(rung.nominal_kbps)
    target = nominal
    maxrate = int(round(nominal * settings.maxrate_ratio))
    bufsize = int(round(nominal * settings.bufsize_ratio))
    accept = min(settings.rate_tolerance_pct, 2.8)
    attempts: List[Attempt] = []
    have_best = False

    base = _base_args(source, rung, plan, settings, threads)
    pass1 = base + [
        "-b:v", f"{int(round(target))}k",
        "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        "-pass", "1", "-passlogfile", pass_prefix,
        "-f", "null", ff.NULL_DEVICE,
    ]
    log_fn(f"[encode] {rung.name}: pass 1 (analysis)")
    ff.run_ffmpeg(pass1, log_path=log_path)

    for attempt_no in range(1, 4):
        if os.path.exists(temp_video):
            os.remove(temp_video)
        pass2 = base + [
            "-b:v", f"{int(round(target))}k",
            "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
            "-pass", "2", "-passlogfile", pass_prefix,
            temp_video,
        ]
        commands.append(" ".join(pass2))
        log_fn(
            f"[encode] {rung.name}: pass 2 attempt {attempt_no} "
            f"(target {int(round(target))} kb/s)"
        )
        t0 = time.perf_counter()
        ff.run_ffmpeg(pass2, log_path=log_path)
        elapsed = time.perf_counter() - t0

        measured, _ = ff.measure_video_bitrate_kbps(temp_video,
                                                    source.fps_float)
        deviation = (measured - nominal) / nominal * 100.0
        attempt = Attempt(
            mode="2pass-vbr",
            target_kbps=target,
            measured_kbps=measured,
            deviation_pct=deviation,
            seconds=elapsed,
        )
        attempts.append(attempt)
        log_fn(
            f"[rate] {rung.name}: measured {measured:.1f} kb/s "
            f"({deviation:+.2f}%) at target {target:.0f} kb/s"
        )

        best = min(attempts, key=lambda a: abs(a.deviation_pct))
        if attempt is best or not have_best:
            os.replace(temp_video, best_video)
            have_best = True
        else:
            os.remove(temp_video)

        if abs(deviation) <= accept:
            break

        # A large first-pass undershoot on easy content means the encoder hit
        # its quality floor and cannot spend the target no matter what; skip
        # the corrective reruns and go straight to CBR+filler.
        if attempt_no == 1 and deviation < -8.0:
            log_fn(
                f"[rate] {rung.name}: first pass is {deviation:+.1f}% under "
                f"target (quality floor); switching to CBR+filler"
            )
            break

        # Detect a quality-floor plateau: the encoder cannot spend more bits
        # no matter how much the target is raised (very easy content).  Stop
        # paying for pass-2 reruns and let the CBR+filler fallback take over.
        if attempt_no > 1:
            prev = attempts[-2]
            target_change = abs(target - prev.target_kbps) / prev.target_kbps
            measured_change = abs(measured - prev.measured_kbps) / nominal
            if target_change > 0.02 and measured_change < 0.005:
                log_fn(
                    f"[rate] {rung.name}: bitrate plateau detected "
                    f"(target {prev.target_kbps:.0f}->{target:.0f} kb/s but "
                    f"measured {prev.measured_kbps:.0f}->{measured:.0f} kb/s)"
                )
                break

        # Closed-loop correction.  The pass-1 stats do not depend on the
        # bitrate, so only pass 2 needs to run again.
        corrected = target * (nominal / measured)
        corrected = max(nominal * 0.4, min(nominal * 2.0, corrected))
        if abs(corrected - target) / target < 0.005:
            break
        target = corrected

    return attempts


def _encode_cbr_video(source: SourceInfo, rung: RungSpec, plan: KeyframePlan,
                      settings: Settings, threads: int, temp_video: str,
                      best_video: str, log_path: str,
                      log_fn: Callable[[str], None],
                      commands: List[str],
                      attempts: List[Attempt],
                      current_best: Attempt) -> None:
    """CBR+filler fallback; filler guarantees the target average rate."""
    nominal = float(rung.nominal_kbps)
    target = nominal
    best_dev = abs(current_best.deviation_pct)
    base = _base_args(source, rung, plan, settings, threads, cbr=True)

    for attempt_no in range(1, 3):
        if os.path.exists(temp_video):
            os.remove(temp_video)
        cbr_target = int(round(target))
        cbr = base + [
            "-b:v", f"{cbr_target}k",
            "-maxrate", f"{cbr_target}k",
            "-minrate", f"{cbr_target}k",
            "-bufsize", f"{int(round(cbr_target * 2))}k",
            temp_video,
        ]
        commands.append(" ".join(cbr))
        log_fn(
            f"[rate] {rung.name}: CBR+filler attempt {attempt_no} "
            f"(target {cbr_target} kb/s)"
        )
        t0 = time.perf_counter()
        ff.run_ffmpeg(cbr, log_path=log_path)
        elapsed = time.perf_counter() - t0
        measured, _ = ff.measure_video_bitrate_kbps(temp_video,
                                                    source.fps_float)
        deviation = (measured - nominal) / nominal * 100.0
        attempt = Attempt(
            mode="cbr-filler",
            target_kbps=target,
            measured_kbps=measured,
            deviation_pct=deviation,
            seconds=elapsed,
        )
        attempts.append(attempt)
        log_fn(
            f"[rate] {rung.name}: CBR+filler measured {measured:.1f} kb/s "
            f"({deviation:+.2f}%)"
        )
        if abs(deviation) < best_dev:
            os.replace(temp_video, best_video)
            best_dev = abs(deviation)
            best_attempt = attempt
        else:
            os.remove(temp_video)
        if abs(deviation) <= min(settings.rate_tolerance_pct, 2.8):
            break
        target = target * (nominal / measured)
        target = max(nominal * 0.95, min(nominal * 1.05, target))


def encode_rung(source: SourceInfo, rung: RungSpec, plan: KeyframePlan,
               settings: Settings, output_path: str, work_dir: str,
               logs_dir: str, threads: int, audio_path: Optional[str],
               audio_codec: Optional[str],
               log_fn: Callable[[str], None]) -> EncodeResult:
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"encode_{rung.name}.log")
    pass_prefix = os.path.join(work_dir, f"{rung.name}_2pass")
    temp_video = os.path.join(work_dir, f"{rung.name}_video.mp4")
    best_video = os.path.join(work_dir, f"{rung.name}_best.mp4")
    started = time.perf_counter()
    commands: List[str] = []

    vbr_attempts = _encode_vbr_video(
        source, rung, plan, settings, threads, temp_video, best_video,
        pass_prefix, log_path, log_fn, commands,
    )
    best = min(vbr_attempts, key=lambda a: abs(a.deviation_pct))

    if abs(best.deviation_pct) > settings.rate_tolerance_pct:
        log_fn(
            f"[rate] {rung.name}: VBR best is {best.deviation_pct:+.2f}%, "
            f"outside +/-{settings.rate_tolerance_pct}%; trying CBR+filler"
        )
        _encode_cbr_video(
            source, rung, plan, settings, threads, temp_video, best_video,
            log_path, log_fn, commands, vbr_attempts, best,
        )
        best = min(vbr_attempts, key=lambda a: abs(a.deviation_pct))

    if not os.path.exists(best_video):
        raise ff.FFError(f"{rung.name}: no encoded output was kept")

    measured, _ = ff.measure_video_bitrate_kbps(best_video,
                                                source.fps_float)
    deviation = (measured - rung.nominal_kbps) / rung.nominal_kbps * 100.0

    # ---- final mux: identical audio bytes in every rung ---------------------
    mux_args = ["-hide_banner", "-y", "-i", best_video]
    if audio_path:
        mux_args += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0"]
    else:
        mux_args += ["-map", "0:v:0"]
    mux_args += [
        "-c", "copy",
        "-video_track_timescale", "90000",
        "-movflags", "+faststart",
        output_path,
    ]
    commands.append(" ".join(mux_args))
    ff.run_ffmpeg(mux_args, log_path=log_path)
    os.remove(best_video)
    if os.path.exists(temp_video):
        os.remove(temp_video)

    size_bytes = os.path.getsize(output_path)
    video_duration, frame_count, _ = ff.video_stream_duration_and_frames(
        output_path, source.fps_float
    )
    total_seconds = time.perf_counter() - started
    log_fn(
        f"[done] {rung.name}: {measured:.1f} kb/s ({deviation:+.2f}%), "
        f"{size_bytes / 1e6:.1f} MB, {total_seconds:.1f}s, mode={best.mode}"
    )

    return EncodeResult(
        rung=rung,
        output_path=output_path,
        measured_kbps=measured,
        deviation_pct=deviation,
        mode=best.mode,
        target_kbps=best.target_kbps,
        attempts=vbr_attempts,
        encode_seconds=total_seconds,
        size_bytes=size_bytes,
        video_duration=video_duration,
        frame_count=frame_count,
        commands=commands,
        audio_codec=audio_codec,
    )
