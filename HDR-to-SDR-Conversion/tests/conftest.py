"""测试夹具：用 ffmpeg 生成真实 PQ / HLG / SDR / RGB 样本。

纯数学测试不依赖本夹具；需要真实 ffmpeg 的用例在无 ffmpeg 时整体 skip。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from hdr_to_sdr.binaries import find_ffmpeg

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)

# x265 内联 VUI 参数：经实测 -color_trc 等外层标签不会写进 HEVC bitstream，
# 必须通过 -x265-params 内联 colorprim/transfer/colormatrix 才能被 ffprobe 确证。
_X265_HDR_BASE = "libx265"
_MASTER_DISPLAY = (
    "master-display=G(13250,34500)B(7500,3000)R(34000,16000)"
    "WP(15635,16450)L(10000000,1):max-cll=1000,400"
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


@pytest.fixture()
def pq_file(work_dir, ffmpeg) -> str:
    """10bit PQ(SMPTE2084) + BT.2020，带 mastering/CLL 元数据的 MKV。"""
    path = os.path.join(work_dir, "sample_pq.mkv")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", _X265_HDR_BASE, "-preset", "ultrafast",
        "-pix_fmt", "yuv420p10le",
        "-x265-params",
        f"hdr10=1:colorprim=bt2020:transfer=smpte2084:"
        f"colormatrix=bt2020nc:{_MASTER_DISPLAY}",
        "-c:a", "aac", "-shortest", path,
    ])
    return path


@pytest.fixture()
def hlg_file(work_dir, ffmpeg) -> str:
    """10bit HLG(ARIB STD-B67) + BT.2020 的 MKV。"""
    path = os.path.join(work_dir, "sample_hlg.mkv")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=2",
        "-c:v", _X265_HDR_BASE, "-preset", "ultrafast",
        "-pix_fmt", "yuv420p10le",
        "-x265-params",
        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
        "-an", path,
    ])
    return path


@pytest.fixture()
def sdr_file(work_dir, ffmpeg) -> str:
    """普通 8bit BT.709 SDR 文件，用于验证「不靠猜」拒绝逻辑。"""
    path = os.path.join(work_dir, "sample_sdr.mp4")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an", path,
    ])
    return path


@pytest.fixture()
def rgb_hdr_file(work_dir, ffmpeg) -> str:
    """RGB 像素族（gbrp 浮点经 rawvideo 承载）的 HDR 标签样本。

    rawvideo 无法承载 VUI 标签，因此本夹具用于 RGB 路径的帧引擎测试时
    由调用方手动构造 HDRInfo；这里仅产出 gbrpf32 可读的原始素材。
    """
    path = os.path.join(work_dir, "sample_rgb.nut")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
        "-c:v", "rawvideo", "-pix_fmt", "gbrpf32le", "-an", path,
    ])
    return path
