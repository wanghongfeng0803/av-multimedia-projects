"""pytest 公共夹具：样本生成在无 ffmpeg 时自动 skip 相关集成测试。

纯逻辑测试（契约/扫描/规划/清单/复核/批量注入）不使用本文件中的 ffmpeg
夹具，因此在未安装 ffmpeg 的环境也能完整运行；只有 ``test_integration_*``
里真实转码的用例才依赖 ffmpeg。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from bvp.binaries import find_ffmpeg

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG, reason="环境中未安装 ffmpeg/ffprobe"
)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"夹具生成失败: {' '.join(cmd)}\n{proc.stderr[-800:]}"
        )


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    return find_ffmpeg()


@pytest.fixture()
def work_dir(tmp_path) -> str:
    return str(tmp_path)


def make_video(path: str, ffmpeg: str, *, size: str = "320x240",
               duration: int = 1, codec: str = "libx264",
               ext: str | None = None) -> str:
    """生成一个可被 ffprobe 识别的最小视频样本（测试辅助函数）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size={size}:rate=15:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", codec, "-preset", "ultrafast",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture()
def video_factory(ffmpeg):
    """返回生成视频样本的工厂函数。"""
    def _factory(path: str, **kwargs) -> str:
        return make_video(path, ffmpeg, **kwargs)
    return _factory


@pytest.fixture()
def fake_video(work_dir, video_factory) -> str:
    """一个规范的最小 mp4 视频。"""
    return video_factory(os.path.join(work_dir, "sample.mp4"))
