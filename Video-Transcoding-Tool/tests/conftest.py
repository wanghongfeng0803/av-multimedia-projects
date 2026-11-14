"""测试夹具：用 ffmpeg 生成各类规范与不规范样本。"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from transcode_tool.binaries import find_ffmpeg

pytestmark = pytest.mark.skipif(
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


@pytest.fixture()
def work_dir(tmp_path) -> str:
    return str(tmp_path)


@pytest.fixture()
def hevc_file(work_dir, ffmpeg) -> str:
    """规范但非 H.264 的输入：H.265 + AAC，MP4 容器。"""
    path = os.path.join(work_dir, "sample_hevc.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx265", "-preset", "ultrafast",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture()
def odd_size_file(work_dir, ffmpeg) -> str:
    """奇数尺寸的 H.264 输入（yuv420p 无法直接承载奇数宽/高）。"""
    path = os.path.join(work_dir, "sample_odd.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=641x363:rate=25:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture()
def rgb_file(work_dir, ffmpeg) -> str:
    """RGB 像素格式 + 非 AAC 音频（mp3），需要像素格式与音频双重转换。"""
    path = os.path.join(work_dir, "sample_rgb.avi")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        # rawvideo + rgb24 保留真正的 RGB 像素格式
        "-c:v", "rawvideo", "-pix_fmt", "rgb24",
        "-c:a", "libmp3lame", "-shortest", path,
    ])
    return path


@pytest.fixture()
def no_audio_file(work_dir, ffmpeg) -> str:
    """纯视频、无音频流。"""
    path = os.path.join(work_dir, "sample_noaudio.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-an", path,
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
def truncated_file(hevc_file, work_dir) -> str:
    """截断的 MP4：保留前半字节，moov 在尾部会丢失，属于损坏输入。"""
    path = os.path.join(work_dir, "truncated.mp4")
    size = os.path.getsize(hevc_file)
    with open(hevc_file, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read(size // 2))
    return path


@pytest.fixture()
def multi_stream_file(work_dir, ffmpeg) -> str:
    """含两条音频流的输入，验证只保留主音轨且不报错。"""
    path = os.path.join(work_dir, "sample_multi.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", path,
    ])
    return path


@pytest.fixture()
def long_file(work_dir, ffmpeg) -> str:
    """4 秒带音频的 H.264 输入，用于分段转码测试。"""
    path = os.path.join(work_dir, "sample_long.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture()
def tenbit_file(work_dir, ffmpeg) -> str:
    """10bit 像素格式输入，验证向 yuv420p 的降级转换。"""
    path = os.path.join(work_dir, "sample_10bit.mkv")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
        "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        "-an", path,
    ])
    return path


@pytest.fixture()
def sparse_gop_file(work_dir, ffmpeg) -> str:
    """稀疏关键帧输入（GOP 远长于分段长度），用于暴露分段寻址的时间戳错位。

    30fps、-g 250 使关键帧间隔约 8.3s，而分段只有 1-2s，
    输入寻址必然回退到更早的关键帧，是检验双 -ss 时间戳重置的严苛样本。
    """
    path = os.path.join(work_dir, "sample_sparse.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-preset", "ultrafast", "-g", "250",
        "-c:a", "aac", "-shortest", path,
    ])
    return path
