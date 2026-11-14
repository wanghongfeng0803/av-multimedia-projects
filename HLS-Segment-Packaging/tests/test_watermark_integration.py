"""文字水印端到端集成测试：真实 ffmpeg 重编码 + overlay + segment。

验收口径：
* 水印分片首帧仍为 IDR、可独立解码、切点仍对齐规划；
* 帧画面的水印锚点区域确实出现了文字颜色像素；
* 含空格 / 特殊字符的输入输出路径可正常工作；
* CLI --watermark-* 参数端到端打通。
"""

from __future__ import annotations

import os
import subprocess

import pytest

from hls_packager.errors import UnsupportedInputError
from hls_packager.packager import package
from hls_packager.watermark.config import WatermarkConfig
from tests.conftest import ffprobe_json

pytestmark = pytest.mark.integration


def _extract_frame(path: str, out_png: str, at: float | None = None) -> None:
    """抽取一帧为 PNG。

    分片保留全局连续时间戳（不 reset），首帧 PTS 可能大于 0，因此默认
    直接取首帧；需要指定时刻时用输出侧寻址（-ss 放在 -i 之后）。
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    if at is not None:
        cmd += ["-ss", str(at)]
    cmd += ["-frames:v", "1", out_png]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")


def _read_png_rgb(path: str) -> tuple[int, int, bytearray]:
    import struct
    import zlib

    data = open(path, "rb").read()
    pos = 8
    width = height = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", data[pos + 8:pos + 16])
        elif tag == b"IDAT":
            idat += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    raw = zlib.decompress(bytes(idat))
    channels = 4 if data[25] == 6 else 3
    stride = width * channels
    pixels = bytearray(width * height * 4)
    for row in range(height):
        line = raw[row * (stride + 1) + 1: row * (stride + 1) + 1 + stride]
        for x in range(width):
            base = row * width * 4 + x * 4
            if channels == 4:
                pixels[base:base + 4] = line[x * 4:x * 4 + 4]
            else:
                pixels[base:base + 3] = line[x * 3:x * 3 + 3]
                pixels[base + 3] = 255
    return width, height, pixels


def _count_bright_pixels(pixels: bytearray, x0: int, y0: int, x1: int, y1: int, width: int) -> int:
    """统计区域内接近纯白的像素数（默认水印为白色文字）。"""
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = (y * width + x) * 4
            if pixels[i] > 220 and pixels[i + 1] > 220 and pixels[i + 2] > 220:
                count += 1
    return count


def test_watermark_packages_and_every_segment_starts_idr(h264_file, out_dir):
    config = WatermarkConfig(text="HLS 水印 2026", font_size=28, anchor="bottom-right")
    result = package(h264_file, out_dir, target_duration=4.0, watermark=config, overwrite=True)

    assert len(result.segments) >= 2
    for seg in result.segments:
        path = os.path.join(out_dir, seg.filename)
        data = ffprobe_json(
            path, "-select_streams", "v:0", "-show_entries", "packet=flags"
        )
        assert "K" in data["packets"][0]["flags"], f"{seg.filename} 首帧不是 IDR"
    # 水印 PNG 已落盘并记录在结果中
    assert os.path.isfile(result.extra["watermark_png"])


def test_watermark_is_actually_visible_in_frame(h264_file, out_dir, tmp_path):
    config = WatermarkConfig(text="VISIBLE_WATERMARK", font_size=32, anchor="bottom-right")
    result = package(h264_file, out_dir, target_duration=4.0, watermark=config, overwrite=True)

    first_seg = os.path.join(out_dir, result.segments[0].filename)
    frame_png = str(tmp_path / "frame.png")
    _extract_frame(first_seg, frame_png)
    width, height, pixels = _read_png_rgb(frame_png)

    # 右下角锚点区域（margin 24，向内取 220x60 窗口）应有明显白色文字像素
    bright = _count_bright_pixels(
        pixels,
        max(0, width - 244), max(0, height - 84),
        max(1, width - 24), max(1, height - 24),
        width,
    )
    assert bright > 50, f"水印区域亮像素仅 {bright} 个，疑似水印未生效"


def test_watermark_top_left_anchor(h264_file, out_dir, tmp_path):
    config = WatermarkConfig(text="TOPLEFT", font_size=28, anchor="top-left", margin_x=16, margin_y=16)
    result = package(h264_file, out_dir, target_duration=6.0, watermark=config, overwrite=True)
    frame_png = str(tmp_path / "frame_tl.png")
    _extract_frame(os.path.join(out_dir, result.segments[0].filename), frame_png)
    width, _, pixels = _read_png_rgb(frame_png)
    bright = _count_bright_pixels(pixels, 16, 16, 260, 70, width)
    assert bright > 30


def test_watermark_special_character_paths(h264_file, tmp_path):
    # 输出目录含空格、#、分号、中文；文件名也含空格
    special_out = tmp_path / "out dir; 水印#1"
    special_out.mkdir()
    config = WatermarkConfig(text="path; injection `whoami`", font_size=24)
    result = package(
        h264_file, str(special_out), target_duration=6.0,
        watermark=config, overwrite=True,
    )
    assert len(result.segments) >= 1
    assert os.path.isfile(os.path.join(str(special_out), "playlist.m3u8"))
    # 分片都能被 ffprobe 独立解析
    for seg in result.segments:
        ffprobe_json(os.path.join(str(special_out), seg.filename), "-show_entries", "format=duration")


def test_watermark_rejects_audio_only_input(audio_only_file, out_dir):
    config = WatermarkConfig(text="no video")
    with pytest.raises(UnsupportedInputError):
        package(audio_only_file, out_dir, target_duration=3.0, watermark=config, overwrite=True)


def test_watermark_keeps_global_timestamps_continuous(h264_file, out_dir):
    config = WatermarkConfig(text="TS", font_size=20)
    result = package(h264_file, out_dir, target_duration=4.0, watermark=config, overwrite=True)
    first_pts = []
    for seg in result.segments:
        data = ffprobe_json(
            os.path.join(out_dir, seg.filename),
            "-select_streams", "v:0", "-show_entries", "packet=pts_time",
        )
        first_pts.append(min(float(p["pts_time"]) for p in data["packets"]))
    # 不重置时间戳：逐片首帧 PTS 单调递增
    assert all(cur > prev for prev, cur in zip(first_pts, first_pts[1:]))


def test_cli_watermark_end_to_end(h264_file, out_dir):
    from hls_packager.cli import main

    exit_code = main([
        h264_file, out_dir, "-d", "6", "-y",
        "--watermark-text", "CLI 水印",
        "--watermark-size", "26",
        "--watermark-color", "white",
        "--watermark-anchor", "bottom-right",
    ])
    assert exit_code == 0
    assert os.path.isfile(os.path.join(out_dir, "playlist.m3u8"))
    assert os.path.isfile(os.path.join(out_dir, ".watermark_text.png"))


def test_no_watermark_path_unchanged(h264_file, out_dir):
    # 不传 watermark 时不应产生水印 PNG，行为与旧版本一致
    result = package(h264_file, out_dir, target_duration=6.0, overwrite=True)
    assert "watermark_png" not in result.extra
    assert not os.path.isfile(os.path.join(out_dir, ".watermark_text.png"))
