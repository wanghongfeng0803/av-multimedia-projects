"""流水线集成测试：真实 ffmpeg 解码→滤波→编码。

无 ffmpeg 时整个模块自动 skip（见 conftest 的 needs_ffmpeg）。
"""

import subprocess

import pytest

from vdenoise import denoise_video, pix_fmt_matches, probe

from conftest import FFMPEG, count_frames, needs_ffmpeg, probe_streams

pytestmark = needs_ffmpeg


def decode_raw(path, pix_fmt="gray"):
    """把视频首路流解码成 gray 原始字节，用于内容级断言。"""
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True, check=True,
    )
    return out.stdout


# ---------- probe ----------

def test_probe_fields(video_factory):
    src = video_factory(frames=5, audio=True)
    info = probe(src)
    assert (info["width"], info["height"]) == (64, 48)
    assert info["pix_fmt"] == "yuv420p"
    assert info["has_audio"] is True
    # ffprobe 以分数形式报告帧率
    num, _, den = info["fps"].partition("/")
    assert int(num) / int(den or 1) == 10


def test_probe_no_audio(video_factory):
    src = video_factory(frames=5, audio=False)
    assert probe(src)["has_audio"] is False


# ---------- 端到端 ----------

def test_denoise_mp4_keeps_audio_and_frames(video_factory, tmp_path):
    src = video_factory(frames=8, audio=True)
    dst = tmp_path / "out.mp4"
    stats = denoise_video(src, dst, radius=1)
    assert stats["frames"] == 8
    streams = probe_streams(dst)
    assert "audio" in streams, "输出必须保留音轨"
    video = streams["video"][0]
    assert (video["width"], video["height"]) == (64, 48)
    assert pix_fmt_matches(video["pix_fmt"], "yuv420p")
    assert count_frames(dst) == 8


def test_denoise_no_audio_input(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "out.mp4"
    stats = denoise_video(src, dst, radius=2)
    assert stats["frames"] == 6
    assert stats["has_audio"] is False
    assert "audio" not in probe_streams(dst)
    assert count_frames(dst) == 6


def test_denoise_actually_smooths(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    dst = tmp_path / "out.mp4"
    denoise_video(src, dst, radius=2)
    before = decode_raw(src)
    after = decode_raw(dst)
    assert len(before) == len(after)
    assert before != after  # 滤波确实改变了画面


def test_denoise_uniform_input_stays_uniform(tmp_path):
    # 纯色输入经均值滤波应近似不变（有损编码允许小误差）
    src = tmp_path / "flat.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=gray:size=64x48:rate=10",
         "-frames:v", "5", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    dst = tmp_path / "out.mp4"
    denoise_video(src, dst, radius=2)
    after = decode_raw(dst)
    assert max(after) - min(after) <= 8


def test_tiny_budget_many_bands(video_factory, tmp_path):
    # 4 KiB 预算迫使 48 行帧分成核心行高=1 的多带，结果仍须完整
    src = video_factory(frames=5, audio=False)
    dst = tmp_path / "out.mp4"
    stats = denoise_video(src, dst, radius=2, budget_bytes=4096)
    assert stats["frames"] == 5
    assert count_frames(dst) == 5


def test_progress_callback(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "out.mp4"
    calls = []
    denoise_video(src, dst, radius=1, progress=calls.append)
    assert calls == [1, 2, 3, 4, 5, 6]


def test_rgb_input_nut_container(video_factory, tmp_path):
    # rawvideo rgb24 只有 nut 能封装；RGB 工作格式须自动换 libx264rgb
    src = video_factory(name="in.nut", pix_fmt="rgb24", frames=5, audio=False)
    assert probe(src)["pix_fmt"] == "rgb24"
    dst = tmp_path / "out.mp4"
    stats = denoise_video(src, dst, radius=1)
    assert stats["pix_fmt"] == "rgb24"
    video = probe_streams(dst)["video"][0]
    # H.264 RGB 产物被探为 gbrp 而非 rgb24，两种都要认
    assert pix_fmt_matches(video["pix_fmt"], "rgb24")
    assert count_frames(dst) == 5


def test_gray_input_outputs_yuv420p(tmp_path):
    # gray 无法直接进 libx264，输出转 yuv420p
    src = tmp_path / "gray.mkv"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
         "-frames:v", "5", "-pix_fmt", "gray", "-c:v", "ffv1", str(src)],
        check=True,
    )
    assert probe(src)["pix_fmt"] == "gray"
    dst = tmp_path / "out.mp4"
    stats = denoise_video(src, dst, radius=1)
    assert stats["pix_fmt"] == "gray"
    video = probe_streams(dst)["video"][0]
    assert video["pix_fmt"] == "yuv420p"
    assert count_frames(dst) == 5
