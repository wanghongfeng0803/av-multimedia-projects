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
    adjust(
        sample_av, str(tmp_path / "out.mp4"),
        brightness=10, contrast=20, hue=30, saturation=-15, workers=2,
    )
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


# ------------------------------------------------- 色相 / 饱和度端到端


def _decode_rgb24(path: str) -> bytes:
    proc = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-loglevel", "error",
         "-i", path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert proc.stdout, "解码输出为空"
    return proc.stdout


def _make_wheel_frame(tmp_path, size=96) -> str:
    """生成一帧色相环测试图（满饱和度、满亮度，角度绕一周）。

    每个像素的色相由其在帧内的水平位置线性铺满 0-360°，
    便于用数值而非肉眼核对旋转方向。
    """
    import colorsys

    path = tmp_path / "wheel.mp4"
    raw = bytearray()
    for _ in range(size):  # 高度
        for x in range(size):  # 宽度即色相角
            angle = x / size  # 0..1 对应 0..360°
            r, g, b = colorsys.hsv_to_rgb(angle, 1.0, 1.0)
            raw += bytes((round(r * 255), round(g * 255), round(b * 255)))
    proc = subprocess.run(
        [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}",
         "-r", "10", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(path)],
        input=bytes(raw), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return str(path)


def _dominant_hue_degrees(rgb: bytes, size: int, x: int) -> float:
    """取第 x 列（所有行平均）的平均 RGB，返回其 HSV 色相角（度）。"""
    import colorsys

    rs = gs = bs = 0
    count = 0
    for y in range(size):
        i = (y * size + x) * 3
        rs += rgb[i]
        gs += rgb[i + 1]
        bs += rgb[i + 2]
        count += 1
    h, _, _ = colorsys.rgb_to_hsv(rs / count / 255, gs / count / 255, bs / count / 255)
    return h * 360.0


def test_hue_direction_matches_ffmpeg_native_filter(tmp_path):
    """色相 +60° 的方向与 ffmpeg 原生 ``hue=h=60`` 一致（逆时针为正）。

    不用肉眼：比较旋转后色相环上同一采样列的色相角，容差考虑
    yuv420p 有损压缩与 HSV/YUV 色彩空间差异。
    """
    size = 96
    wheel = _make_wheel_frame(tmp_path, size)
    shifted = str(tmp_path / "shifted.mp4")
    adjust(wheel, shifted, hue=60, workers=2, overwrite=True)

    # ffmpeg 原生滤镜作为外部参考
    ref = str(tmp_path / "ref.mp4")
    proc = subprocess.run(
        [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
         "-i", wheel, "-vf", "hue=h=60", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", ref],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")

    ours = _decode_rgb24(shifted)
    ref_bytes = _decode_rgb24(ref)
    # 在环上取若干采样列，比较两者色相角（环上角度差按最短弧归一）
    for x in (10, 30, 50, 70, 88):
        a = _dominant_hue_degrees(ours, size, x)
        b = _dominant_hue_degrees(ref_bytes, size, x)
        diff = abs((a - b + 180) % 360 - 180)
        assert diff < 12.0, f"列 {x} 色相方向偏差过大: ours={a:.1f} ffmpeg={b:.1f}"


def test_hue_rotation_advances_angle_on_wheel(tmp_path):
    """数值核对：+60° 后，原本 0°（红）所在列的色相角应推进到约 60°。"""
    size = 96
    wheel = _make_wheel_frame(tmp_path, size)
    shifted = str(tmp_path / "shifted.mp4")
    adjust(wheel, shifted, hue=60, workers=2, overwrite=True)
    out = _decode_rgb24(shifted)
    # 最左列原始为红（0°），旋转后应接近黄（60°）。取列平均抗压缩误差。
    angle = _dominant_hue_degrees(out, size, 2)
    assert abs(angle - 60.0) < 15.0


def test_saturation_minus_100_produces_gray(tmp_path):
    """饱和度 -100：输出每个像素 R≈G≈B（去色），用通道极差数值核对。"""
    wheel = _make_wheel_frame(tmp_path, 48)
    out_path = str(tmp_path / "gray.mp4")
    adjust(wheel, out_path, saturation=-100, workers=2, overwrite=True)
    rgb = _decode_rgb24(out_path)
    max_spread = 0
    for i in range(0, len(rgb), 3):
        max_spread = max(max_spread, max(rgb[i:i + 3]) - min(rgb[i:i + 3]))
    assert max_spread <= 6  # yuv420p 有损，允许极小残差


def test_hsv_clamp_is_reported(sample_av, tmp_path):
    out = str(tmp_path / "clamped.mp4")
    with pytest.warns(UserWarning, match="钳制"):
        result = adjust(sample_av, out, hue=999, saturation=-500, workers=2)
    assert result.hue == 180.0
    assert result.saturation == -100.0
    assert any("hue" in m for m in result.messages)
    assert any("saturation" in m for m in result.messages)


def test_hsv_defaults_keep_legacy_identity(sample_av, tmp_path):
    """不传色相/饱和度时默认 0，结果与旧的纯默认调用一致（恒等基线不破坏）。"""
    out = str(tmp_path / "legacy.mp4")
    result = adjust(sample_av, out, workers=2)
    assert result.hue == 0.0
    assert result.saturation == 0.0
    src_mean = _decode_rgb_mean(sample_av)
    assert abs(_decode_rgb_mean(out) - src_mean) < 2.0


def test_cli_accepts_hue_and_saturation(sample_av, tmp_path, capsys):
    out = str(tmp_path / "cli_hsv.mp4")
    rc = main([sample_av, "-o", out, "-H", "45", "-s", "-30",
               "--workers", "2", "--overwrite"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "色相 45" in text and "饱和度 -30" in text
