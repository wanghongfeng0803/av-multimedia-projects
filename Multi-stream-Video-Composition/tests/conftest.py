"""测试夹具：构造 MediaInfo 的辅助函数 + 需要真实 ffmpeg 的样本生成。"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from compose_tool.binaries import find_ffmpeg, find_ffprobe
from compose_tool.models import MediaInfo, StreamInfo


def make_media(
    *,
    width: int = 640,
    height: int = 360,
    duration: float = 2.0,
    fps: str = "25/1",
    sar: str = "1:1",
    has_audio: bool = True,
    format_duration: float | None = None,
) -> MediaInfo:
    """不依赖 ffprobe，直接构造一份探测结果用于纯逻辑测试。"""
    streams = [
        StreamInfo(
            index=0,
            codec_type="video",
            codec_name="h264",
            width=width,
            height=height,
            pix_fmt="yuv420p",
            duration_seconds=duration,
            avg_frame_rate=fps,
            r_frame_rate=fps,
            sample_aspect_ratio=sar,
        )
    ]
    if has_audio:
        streams.append(
            StreamInfo(
                index=1,
                codec_type="audio",
                codec_name="aac",
                sample_rate=44100,
                channels=2,
                duration_seconds=duration,
            )
        )
    return MediaInfo(
        file_path="fake.mp4",
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=format_duration if format_duration is not None else duration,
        streams=streams,
    )


# ---- 以下夹具需要真实 ffmpeg，无 ffmpeg 时自动 skip ----

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"夹具生成失败: {' '.join(cmd)}\n{proc.stderr[-800:]}")


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    return find_ffmpeg()


@pytest.fixture(scope="session")
def ffprobe() -> str:
    return find_ffprobe()


@pytest.fixture()
def work_dir(tmp_path) -> str:
    return str(tmp_path)


@pytest.fixture()
def left_clip(work_dir, ffmpeg) -> str:
    """左路样本：320x240 @ 15fps，2 秒，带 440Hz 音频。"""
    path = os.path.join(work_dir, "left.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture()
def right_clip(work_dir, ffmpeg) -> str:
    """右路样本：640x360 @ 25fps，3.5 秒，无音频（分辨率/帧率/时长/音轨全不一致）。"""
    path = os.path.join(work_dir, "right.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3.5",
        "-c:v", "libx264", "-preset", "ultrafast", "-an", path,
    ])
    return path


@pytest.fixture()
def right_audio_clip(work_dir, ffmpeg) -> str:
    """右路样本（带 880Hz 音频）：用于混音场景。"""
    path = os.path.join(work_dir, "right_audio.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3.5",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=3.5",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-shortest", path,
    ])
    return path
