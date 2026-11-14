"""测试夹具：用 ffmpeg 合成各类规范素材，并提供 ffprobe 查询助手。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

# 整个模块的夹具都依赖本机 ffmpeg；缺失时集成测试整体跳过
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="环境中未安装 ffmpeg/ffprobe",
)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"夹具生成失败: {' '.join(cmd)}\n{tail}")


@pytest.fixture(scope="session")
def work_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("media"))


@pytest.fixture()
def out_dir(tmp_path) -> str:
    path = str(tmp_path / "hls_out")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def h264_file(work_root) -> str:
    """30s H.264 + AAC，24fps，GOP 固定 2s（g=48），含 B 帧。"""
    path = os.path.join(work_root, "h264.mp4")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
        "-c:v", "libx264", "-preset", "veryfast",
        "-g", "48", "-keyint_min", "48", "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        path,
    ])
    return path


@pytest.fixture(scope="session")
def audio_only_file(work_root) -> str:
    """12s 纯 AAC 音频。"""
    path = os.path.join(work_root, "audio.m4a")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=12",
        "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="session")
def big_gop_file(work_root) -> str:
    """25s 视频，GOP 固定 10s（g=240），用于验证大 GOP 对齐。"""
    path = os.path.join(work_root, "biggop.mp4")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=25",
        "-c:v", "libx264", "-preset", "veryfast",
        "-g", "240", "-keyint_min", "240", "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="session")
def hevc_file(work_root) -> str:
    """12s HEVC + AAC，验证非 H.264 编码也能流拷贝进 TS。"""
    path = os.path.join(work_root, "hevc.mp4")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-c:v", "libx265", "-preset", "ultrafast", "-g", "48",
        "-c:a", "aac", path,
    ])
    return path


def ffprobe_json(path: str, *args: str) -> dict:
    """集成测试专用：运行 ffprobe 并返回解析后的 JSON。"""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", *args, "-print_format", "json", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    return json.loads(proc.stdout.decode("utf-8", errors="replace"))
