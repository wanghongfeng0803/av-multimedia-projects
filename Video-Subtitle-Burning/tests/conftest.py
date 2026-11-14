"""pytest 共享夹具：探测 ffmpeg，不可用时跳过集成测试。"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def ffmpeg():
    if not ffmpeg_available():
        pytest.skip("系统未找到 ffmpeg/ffprobe，跳过管道集成测试")


def make_test_video(path, duration=2, size="320x240", rate=25, with_audio=True):
    """用 lavfi 源生成确定性测试视频（可选正弦音轨）。"""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc=duration={duration}:size={size}:rate={rate}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency=440:duration={duration}",
                "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
