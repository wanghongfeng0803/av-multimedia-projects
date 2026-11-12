# -*- coding: utf-8 -*-
"""ffprobe 元数据查询。"""
import json
import subprocess
from dataclasses import dataclass


@dataclass
class MediaInfo:
    width: int
    height: int
    fps: float
    duration_sec: float
    has_audio: bool
    audio_sample_rate: int
    audio_channels: int


def _ffprobe_json(path):
    return subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,"
                         "sample_rate,channels:format=duration",
        "-of", "json", str(path),
    ])


def probe(path) -> MediaInfo:
    data = json.loads(_ffprobe_json(path))
    vstream = next((s for s in data.get("streams", [])
                    if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise ValueError(f"文件中没有视频流: {path}")
    num, den = vstream["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    astream = next((s for s in data.get("streams", [])
                    if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    return MediaInfo(
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        fps=fps,
        duration_sec=duration,
        has_audio=astream is not None,
        audio_sample_rate=int(astream["sample_rate"]) if astream else 0,
        audio_channels=int(astream["channels"]) if astream else 0,
    )
