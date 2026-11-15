"""pytest 共享夹具：探测 ffmpeg，不可用时跳过集成测试。"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def ffmpeg_available() -> bool:
    """ffmpeg 与 ffprobe 是否都在 PATH。"""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def ffmpeg():
    """会话级夹具：无 ffmpeg 时自动 skip 集成测试。"""
    if not ffmpeg_available():
        pytest.skip("系统未找到 ffmpeg/ffprobe，跳过管道集成测试")


def make_clip(
    path,
    *,
    duration=2,
    size="320x240",
    rate=25,
    with_audio=True,
    audio_freq=440,
    pattern="testsrc",
    color=None,
):
    """用 lavfi 源生成确定性测试素材（可选正弦音轨）。

    返回素材的理论帧数 ``round(duration * rate)``，供测试直接做账本核对。
    ``color`` 给定时改用 ``color`` 源生成纯色素材（如 ``red``/``0xff0000``）。
    """
    if color is not None:
        source = f"color=c={color}:duration={duration}:size={size}:rate={rate}"
    else:
        source = f"{pattern}=duration={duration}:size={size}:rate={rate}"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", source,
    ]
    if with_audio:
        cmd += [
            "-f", "lavfi", "-i",
            f"sine=frequency={audio_freq}:duration={duration}",
            "-c:a", "aac",
        ]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return round(duration * rate)
