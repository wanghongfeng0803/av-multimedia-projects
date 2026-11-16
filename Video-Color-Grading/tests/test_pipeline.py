"""ffmpeg 端到端集成测试：管道、音轨保留、钳制提示、建表次数。"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

import color_grading.pipeline as pipeline_mod
from color_grading import adjust
from color_grading.binaries import find_ffmpeg, find_ffprobe
from color_grading.cli import main
from color_grading.errors import OutputExistsError
from color_grading.lut import build_lut as _real_build_lut

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)


def _probe_streams(path: str) -> list[dict]:
    proc = subprocess.run(
        [find_ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["streams"]


def _decode_rgb_mean(path: str) -> float:
    """把视频解码为 rgb24 裸流并返回全通道字节均值。"""
    proc = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-loglevel", "error",
         "-i", path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    data = proc.stdout
    assert data, "解码输出为空"
    return sum(data) / len(data)


def test_default_call_preserves_audio_and_geometry(sample_av, tmp_path):
    out = str(tmp_path / "out.mp4")
    result = adjust(sample_av, out, workers=2)
    assert result.frames == 10
    assert result.messages == []
    assert result.has_audio

    streams = _probe_streams(out)
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    assert video["width"] == 64 and video["height"] == 48
    assert audio is not None, "音轨必须保留"
    assert audio["codec_name"] == "aac"  # -c:a copy 不重编码


def test_default_call_pixels_match_input(sample_av, tmp_path):
    """无参数调用的回归基线：输出像素统计与原片一致（有损编码容差内）。"""
    out = str(tmp_path / "out.mp4")
    adjust(sample_av, out, workers=2)
    src_mean = _decode_rgb_mean(sample_av)
    out_mean = _decode_rgb_mean(out)
    assert abs(src_mean - out_mean) < 2.0


def test_brightness_raises_pixel_mean(sample_av, tmp_path):
    out = str(tmp_path / "bright.mp4")
    adjust(sample_av, out, brightness=60, workers=2)
    assert _decode_rgb_mean(out) > _decode_rgb_mean(sample_av) + 20


def test_out_of_range_is_clamped_and_reported(sample_av, tmp_path):
    out = str(tmp_path / "clamped.mp4")
    with pytest.warns(UserWarning, match="钳制"):
        result = adjust(sample_av, out, brightness=500, contrast=-999, workers=2)
    assert result.brightness == 100.0
    assert result.contrast == -100.0
    assert len(result.messages) == 2
    assert all("钳制" in m for m in result.messages)


def test_lut_is_built_exactly_once(sample_av, tmp_path, monkeypatch):
    """映射表只算一次：整个 adjust 流程只调用 build_lut 一回。"""
    calls = []

    def spy(brightness=0.0, contrast=0.0):
        calls.append((brightness, contrast))
        return _real_build_lut(brightness, contrast)

    monkeypatch.setattr(pipeline_mod, "build_lut", spy)
    adjust(sample_av, str(tmp_path / "out.mp4"), brightness=10, contrast=20, workers=2)
    assert len(calls) == 1


def test_no_audio_input_still_works(sample_no_audio, tmp_path):
    out = str(tmp_path / "out.mp4")
    result = adjust(sample_no_audio, out, brightness=10, workers=2)
    assert result.frames == 10
    assert not result.has_audio
    streams = _probe_streams(out)
    assert all(s["codec_type"] != "audio" for s in streams)


def test_existing_output_requires_overwrite(sample_av, tmp_path):
    out = tmp_path / "exists.mp4"
    out.write_bytes(b"occupied")
    with pytest.raises(OutputExistsError):
        adjust(sample_av, str(out))
    assert out.read_bytes() == b"occupied"  # 原文件未被破坏


def test_cli_end_to_end(sample_av, tmp_path, capsys):
    out = str(tmp_path / "cli.mp4")
    rc = main([sample_av, "-o", out, "-b", "20", "-c", "10", "--workers", "2"])
    assert rc == 0
    assert "完成" in capsys.readouterr().out
    streams = _probe_streams(out)
    assert any(s["codec_type"] == "audio" for s in streams)
