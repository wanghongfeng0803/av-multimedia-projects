"""测试公共夹具。

纯逻辑测试（test_histogram / test_detector / test_output 等）不依赖
ffmpeg，任何环境都能运行；需要真实 ffmpeg 的集成测试在模块内用
``pytestmark`` 声明整体 skip，夹具本身只负责按需生成样本。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from scene_seg_tool.binaries import find_ffmpeg, find_ffprobe


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"夹具生成失败: {' '.join(cmd)}\n{proc.stderr[-800:]}"
        )


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
def hard_cut_video(work_dir, ffmpeg) -> str:
    """两段恒定灰场拼接：深灰 25 帧 + 浅灰 25 帧，切点恰在帧 25。"""
    path = os.path.join(work_dir, "hard_cut.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=0x202020:s=160x120:r=25:d=1",
        "-f", "lavfi", "-i", "color=c=0xD0D0D0:s=160x120:r=25:d=1",
        "-filter_complex", "[0][1]concat=n=2:v=1[v]",
        "-map", "[v]", "-pix_fmt", "gray",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-an", path,
    ])
    return path


@pytest.fixture()
def single_frame_video(work_dir, ffmpeg) -> str:
    """只有一帧的视频（单帧边界条件）。"""
    path = os.path.join(work_dir, "single_frame.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=0x404040:s=64x48:r=25:d=1",
        "-frames:v", "1", "-pix_fmt", "gray",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-an", path,
    ])
    return path


@pytest.fixture()
def static_video(work_dir, ffmpeg) -> str:
    """长时间无变化：50 帧完全相同的灰场。"""
    path = os.path.join(work_dir, "static.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=0x555555:s=64x48:r=25:d=2",
        "-pix_fmt", "gray",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-an", path,
    ])
    return path


@pytest.fixture()
def flash_video(work_dir, ffmpeg) -> str:
    """单帧白闪：25 帧常态灰 + 1 帧纯白 + 25 帧常态灰，共 51 帧。"""
    path = os.path.join(work_dir, "flash.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=0x303030:s=160x120:r=25:d=1",
        "-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=0.04",
        "-f", "lavfi", "-i", "color=c=0x303030:s=160x120:r=25:d=1",
        "-filter_complex", "[0][1][2]concat=n=3:v=1[v]",
        "-map", "[v]", "-pix_fmt", "gray",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-an", path,
    ])
    return path


@pytest.fixture()
def text_disguised_file(work_dir) -> str:
    """纯文本伪装成 .mp4 —— 扩展名欺骗。"""
    path = os.path.join(work_dir, "pretend.mp4")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("this is definitely not a video file\n" * 20)
    return path


@pytest.fixture()
def truncated_video(work_dir, hard_cut_video) -> str:
    """截断的 MP4：只保留前半字节，属于损坏输入。"""
    path = os.path.join(work_dir, "truncated.mp4")
    size = os.path.getsize(hard_cut_video)
    with open(hard_cut_video, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read(size // 2))
    return path


@pytest.fixture()
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
