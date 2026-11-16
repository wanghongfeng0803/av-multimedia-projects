"""ffmpeg 集成测试夹具：用 ffmpeg 生成带音轨的样本。

纯逻辑测试（lut / chunks）不依赖 ffmpeg；只有 test_pipeline 使用
这里的夹具，并在模块级 skipif 中处理无 ffmpeg 的环境。
"""

from __future__ import annotations

import subprocess

import pytest

from color_grading.binaries import find_ffmpeg


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"夹具生成失败: {' '.join(cmd)}\n{proc.stderr[-800:]}")


@pytest.fixture()
def sample_av(tmp_path) -> str:
    """1 秒、64x48、10fps、H.264 + AAC 的音视频样本。"""
    path = tmp_path / "sample_av.mp4"
    _run([
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ])
    return str(path)


@pytest.fixture()
def sample_no_audio(tmp_path) -> str:
    """纯视频、无音频流的样本。"""
    path = tmp_path / "sample_noaudio.mp4"
    _run([
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-an", str(path),
    ])
    return str(path)
