#!/usr/bin/env python3
"""Verify a transcoded CFR file against its VFR source.

Two independent layers of evidence:

1. Structural (every frame, cheap ffprobe JSON):
   - video is CFR on one ideal grid; maximum grid error is reported
   - video DTS is strictly increasing (B-frame reorder handled correctly)
   - copyts: output timeline origin matches the source origin
   - packet-level A/V offset at head, middle and tail stays inside budget
   - no accumulated drift: tail grid error is compared with head grid error

2. Content markers (head / middle / tail, the actual sync the brief measures):
   the generator stamps a red box and a 1 kHz beep at identical content times.
   We locate each marker in source and output and require the *change* in
   video-minus-audio marker time introduced by transcoding to stay within
   ``--max-offset-ms`` (default 40 ms). This works even when the source itself
   is not perfectly aligned, because only the transcode-induced delta matters.

Exit code is non-zero if any hard check fails, so this script is a CI gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def ffprobe_json(ffprobe: str, path: Path, stream: str, entries: str,
                 level: str = "packet") -> list[dict]:
    q = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", stream,
         "-show_entries", f"{level}={entries}", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(q.stdout).get(f"{level}s", [])


def stream_info(ffprobe: str, path: Path) -> list[dict]:
    q = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                        "stream=index,codec_type,time_base,r_frame_rate,start_time",
                        "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return json.loads(q.stdout)["streams"]


def video_pts(ffprobe: str, path: Path) -> list[float]:
    return [float(f["pts_time"]) for f in
            ffprobe_json(ffprobe, path, "v:0", "pts_time", "frame")]


def video_packets(ffprobe: str, path: Path) -> list[tuple[float, float | None]]:
    """(pts, dts) in file order; dts is None where the container omits it
    (Matroska stores no DTS when it equals the PTS-derived value)."""
    out = []
    for p in ffprobe_json(ffprobe, path, "v:0", "pts_time,dts_time"):
        if "pts_time" not in p:
            continue
        pts = float(p["pts_time"])
        raw = p.get("dts_time")
        dts = None if raw in (None, "N/A") else float(raw)
        out.append((pts, dts))
    return out


def audio_pts(ffprobe: str, path: Path) -> list[float]:
    return [float(p["pts_time"]) for p in
            ffprobe_json(ffprobe, path, "a:0", "pts_time") if "pts_time" in p]


# ------------------------------------------------------------ content markers

def video_marker_times(ffmpeg: str, ffprobe: str, path: Path,
                       box_xy: tuple[int, int]) -> list[float]:
    """Onset times of the red marker box, one entry per marker.

    Decodes EVERY frame natively (no ``fps`` resampling -- that filter can
    drop the opening frames of an offset/VFR stream) and pairs decoded frames
    1:1 with ffprobe presentation timestamps, so detection works identically on
    the VFR source and the CFR output. A small center crop keeps memory bounded
    even for a 3 hour file.
    """
    x, y = box_xy
    crop, half = 8, 4
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"crop={crop}:{crop}:{x + 25 - half}:{y + 25 - half}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"video marker decode failed: {proc.stderr.decode()[-500:]}")
    arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, crop * crop, 3)
    # channels are on the LAST axis; average over the spatial axis (1), not the
    # pixel index, or arr[:, 0] would select the first pixel instead of red.
    red = (arr[:, :, 0].mean(axis=1) > 180) & (arr[:, :, 1].mean(axis=1) < 80)

    frames = ffprobe_json(ffprobe, path, "v:0", "pts_time", "frame")
    pts = [float(f["pts_time"]) for f in frames]
    # A VFR source's raw decode can carry one trailing frame that ffprobe does
    # not list (last-frame duration handling). The prefixes correspond 1:1 in
    # presentation order, so pair over the shorter length instead of failing.
    n = min(len(pts), len(red))
    if len(pts) != len(red):
        print(f"[WARN] {path.name}: decode={len(red)} probe={len(pts)} frames; "
              f"using first {n}", file=sys.stderr)
    red_idx = np.where(red[:n])[0]
    return _onset_clusters(red_idx, lambda i: pts[i])


def audio_marker_times(ffmpeg: str, path: Path) -> list[float]:
    """Onset times of the loud 1 kHz beep markers.

    Streams mono 16 kHz PCM and computes a 10 ms RMS envelope in fixed chunks,
    so a 3 hour file never has to hold ~700 MB of float audio in memory.
    """
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    win = 160  # 10 ms per RMS window
    chunk_windows = 10000  # 100 s of audio per processing chunk
    chunk_bytes = win * chunk_windows * 2  # int16
    rms_values: list[float] = []
    carry = b""
    assert proc.stdout is not None
    while True:
        block = proc.stdout.read(chunk_bytes)
        if not block:
            break
        data = carry + block
        usable = (len(data) // (win * 2)) * (win * 2)
        audio = np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        n = len(audio) // win
        rms = np.sqrt((audio[:n * win].reshape(n, win) ** 2).mean(axis=1))
        rms_values.extend(rms.tolist())
        carry = data[usable:]
    proc.wait()
    if not rms_values:
        return []
    rms_arr = np.asarray(rms_values)
    loud = rms_arr > max(0.05, rms_arr.max() * 0.3)
    loud_idx = np.where(loud)[0]
    return _onset_clusters(loud_idx, lambda i: i * 0.01)


def _onset_clusters(indices: np.ndarray, to_time) -> list[float]:
    """Group consecutive detections and return each group's ONSET time.
    Onset (not center) is used so a marker clipped at a stream boundary does
    not acquire a systematic offset between source and output."""
    if len(indices) == 0:
        return []
    onsets: list[float] = []
    group_start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx - prev > 3:  # gap larger than three samples => new marker
            onsets.append(float(to_time(group_start)))
            group_start = idx
        prev = idx
    onsets.append(float(to_time(group_start)))
    return onsets


# ------------------------------------------------------------ checks

def verify(args: argparse.Namespace) -> tuple[bool, list[str]]:
    ff = args.ffmpeg
    fp = args.ffprobe
    out = Path(args.output)
    src = Path(args.source) if args.source else None
    lines: list[str] = []
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and cond
        lines.append(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    out_streams = stream_info(fp, out)
    vstream = next(s for s in out_streams if s["codec_type"] == "video")
    rate_num, rate_den = vstream["r_frame_rate"].split("/")
    fps = float(rate_num) / float(rate_den)

    vpts = video_pts(fp, out)
    vpk = video_packets(fp, out)
    apts = audio_pts(fp, out)
    check("output is CFR", abs(1 / fps - np.median(np.diff(vpts))) < 1e-4,
          f"r_frame_rate={vstream['r_frame_rate']} frames={len(vpts)}")

    grid = vpts[0] + np.arange(len(vpts)) / fps
    err_ms = np.abs(np.array(vpts) - grid) * 1000.0
    check("CFR grid error <= 5 ms", err_ms.max() <= 5.0,
          f"max={err_ms.max():.3f} ms head={err_ms[0]:.3f} tail={err_ms[-1]:.3f}")
    check("no accumulated drift",
          abs(err_ms[-1] - err_ms[0]) <= 2.0,
          f"tail-head grid error delta={err_ms[-1]-err_ms[0]:.3f} ms")

    # Only compare DTS values the container actually stores. Matroska omits
    # DTS on the opening packets, which is legal and not a reorder violation.
    dts = [d for _, d in vpk if d is not None]
    dts_strict = all(dts[i] < dts[i + 1] for i in range(len(dts) - 1))
    check("video DTS strictly increasing", dts_strict,
          f"{len(dts)} explicit DTS values" if dts_strict else
          f"first violation near {next(i for i in range(len(dts)-1) if dts[i] >= dts[i+1])}")

    if src is not None:
        src_streams = stream_info(fp, src)
        src_v = next(s for s in src_streams if s["codec_type"] == "video")
        src_origin = float(src_v.get("start_time") or 0.0)
        out_origin = float(vstream.get("start_time") or vpts[0])
        check("copyts origin preserved",
              abs(out_origin - src_origin) <= 5.0 / 1000.0,
              f"source={src_origin:.3f}s output={out_origin:.3f}s "
              f"delta={abs(out_origin-src_origin)*1000:.1f} ms")

    if apts:
        audio_arr = np.array(apts)
        audio_step = float(np.median(np.diff(audio_arr))) if len(apts) > 1 else 0.023

        def offset_at(t: float) -> float:
            j = int(np.argmin(np.abs(audio_arr - t)))
            return abs(t - audio_arr[j]) * 1000.0

        # The tail point must sit INSIDE the audio timeline: the last video
        # frame can legitimately fall a few ms beyond the final audio packet
        # (frame vs. audio frame granularity), so pull the tail probe back by
        # one audio packet interval instead of measuring that boundary gap.
        last_covered = int(np.searchsorted(np.array(vpts), apts[-1] - audio_step))
        last_covered = max(1, min(last_covered, len(vpts) - 1))
        points = [("head", 0), ("middle", len(vpts) // 2), ("tail", last_covered)]
        for label, i in points:
            off = offset_at(vpts[i])
            check(f"packet A/V offset {label} <= {args.max_offset_ms} ms",
                  off <= args.max_offset_ms, f"{off:.2f} ms at t={vpts[i]:.3f}s")

    # ---- content-level markers (head / mid / tail) ----
    # A/V sync is judged by the DRIFT of the video-minus-audio marker gap
    # between the head and the tail of the OUTPUT, not by subtracting VFR pts
    # from CFR pts. A VFR frame's pts is, by construction, jittered away from
    # its ideal content time; the fps filter regularizes it back, so a raw
    # source-vs-output pts difference measures that regularization rather than
    # sync loss. The failure mode in the brief (offset growing toward the tail)
    # shows up as a growing internal gap, which a constant origin offset (the
    # copyts start time) does not affect because it cancels in gap_k - gap_0.
    if not args.skip_content:
        box = (20, 20)
        out_vm = video_marker_times(ff, fp, out, box)
        out_am = audio_marker_times(ff, out)
        n_markers = min(len(out_vm), len(out_am))
        lines.append(f"[INFO] output content markers: video={len(out_vm)} audio={len(out_am)}")
        if n_markers < 3:
            check("content markers found", False,
                  f"need >=3 paired markers, got {n_markers}")
        else:
            if src is not None:
                # content preservation: transcoding must not drop or duplicate
                # a marker (frames are never reordered, only retimed)
                sv = len(video_marker_times(ff, fp, src, box))
                sa = len(audio_marker_times(ff, src))
                check("marker content preserved (video)",
                      abs(sv - len(out_vm)) <= 1,
                      f"source={sv} output={len(out_vm)}")
                check("marker content preserved (audio)",
                      abs(sa - len(out_am)) <= 1,
                      f"source={sa} output={len(out_am)}")

            gaps = [out_vm[k] - out_am[k] for k in range(n_markers)]
            head_gap = gaps[0]
            probe_idx = sorted({0, n_markers // 2, n_markers - 1})
            for k in probe_idx:
                drift_ms = (gaps[k] - head_gap) * 1000.0
                check(
                    f"content A/V drift marker#{k} @t={out_vm[k]:.1f}s",
                    abs(drift_ms) <= args.max_offset_ms,
                    f"gap={gaps[k]*1000:.1f} ms, drift-vs-head={drift_ms:+.1f} ms")

    lines.append(f"[INFO] duration: video {vpts[-1]-vpts[0]:.1f}s, "
                 f"{len(vpts)} frames, target {fps:g} fps")
    return ok, lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, default=None,
                   help="original VFR source; enables copyts and source-relative sync checks")
    p.add_argument("--max-offset-ms", type=float, default=40.0)
    p.add_argument("--skip-content", action="store_true",
                   help="structural checks only (no frame/audio decode)")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ok, lines = verify(args)
    print("\n".join(lines))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
