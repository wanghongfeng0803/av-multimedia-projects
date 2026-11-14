"""目标码率与输出分辨率的端到端测试（真实启动 ffmpeg）。

无 ffmpeg 的环境整体 skip；纯算法边界见 test_bitrate.py / test_scale.py。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from transcode_tool.binaries import find_ffprobe
from transcode_tool.cli import main
from transcode_tool.errors import EncoderUnavailableError
from transcode_tool.probe import probe_media
from transcode_tool.transcoder import transcode

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)


def _format_bit_rate(path: str) -> int | None:
    proc = subprocess.run(
        [find_ffprobe(), "-v", "error", "-show_entries", "format=bit_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return int(text) if text.isdigit() else None


# ---------------------------------------------------------------------------
# 目标码率
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["vbr", "abr", "cbr"])
def test_bitrate_modes_produce_valid_h264(hevc_file, work_dir, mode):
    out = os.path.join(work_dir, f"br_{mode}.mp4")
    result = transcode(
        hevc_file, out, preset="ultrafast",
        target_bitrate="400k", rate_mode=mode,
    )
    assert result.success
    assert result.target_bitrate_bps == 400_000
    assert result.rate_mode == mode
    info = probe_media(out)
    assert info.primary_video.codec_name == "h264"
    assert info.primary_video.width % 2 == 0
    actual = _format_bit_rate(out)
    # 短样本码率偏差较大，只做宽松的数量级校验（含音频开销）
    assert actual is None or 50_000 < actual < 2_000_000


def test_bitrate_target_recorded_in_json_cli(hevc_file, work_dir, capsys):
    out = os.path.join(work_dir, "cli_br.mp4")
    code = main([hevc_file, "-o", out, "--preset", "ultrafast",
                 "--bitrate", "500k", "--rate-mode", "cbr", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_bitrate_bps"] == 500_000
    assert payload["rate_mode"] == "cbr"


def test_crf_and_bitrate_are_mutually_exclusive(hevc_file, work_dir):
    with pytest.raises(ValueError, match="互斥"):
        transcode(hevc_file, os.path.join(work_dir, "x.mp4"),
                  crf=23, target_bitrate="400k")


def test_invalid_bitrate_rejected_before_ffmpeg(hevc_file, work_dir):
    out = os.path.join(work_dir, "bad_br.mp4")
    with pytest.raises(ValueError):
        transcode(hevc_file, out, target_bitrate="0")
    assert not os.path.exists(out)  # 参数非法时不启动 ffmpeg、不留产物


def test_cli_invalid_bitrate_exit_1(hevc_file, work_dir):
    code = main([hevc_file, "-o", os.path.join(work_dir, "o.mp4"),
                 "--bitrate", "abc"])
    assert code == 1


def test_vp9_bitrate_into_webm(hevc_file, work_dir):
    out = os.path.join(work_dir, "br_vp9.webm")
    result = transcode(
        hevc_file, out, codec="vp9", preset="ultrafast",
        target_bitrate="350k", rate_mode="vbr",
    )
    assert result.success
    assert probe_media(out).primary_video.codec_name == "vp9"


def test_av1_bitrate_uses_available_encoder(hevc_file, work_dir):
    out = os.path.join(work_dir, "br_av1.mp4")
    try:
        result = transcode(
            hevc_file, out, codec="av1", preset="ultrafast",
            target_bitrate="350k", rate_mode="abr",
        )
    except EncoderUnavailableError:
        pytest.skip("本机 ffmpeg 未包含任何 AV1 编码器")
    assert result.success
    assert probe_media(out).primary_video.codec_name == "av1"


# ---------------------------------------------------------------------------
# 输出分辨率
# ---------------------------------------------------------------------------

def test_resolution_downscale_by_height(hevc_file, work_dir):
    # 源 640x360，定高 180 → 宽 320
    out = os.path.join(work_dir, "res_h.mp4")
    result = transcode(hevc_file, out, preset="ultrafast", resolution="x180")
    assert result.success
    video = probe_media(out).primary_video
    assert (video.width, video.height) == (320, 180)


def test_resolution_percent(hevc_file, work_dir):
    out = os.path.join(work_dir, "res_pct.mp4")
    transcode(hevc_file, out, preset="ultrafast", resolution="50%")
    video = probe_media(out).primary_video
    assert (video.width, video.height) == (320, 180)


def test_resolution_preset_and_even_dimensions(odd_size_file, work_dir):
    # 641x363 的奇数源，缩到 720p 会触发放大收敛；这里用定宽 320 验证偶数对齐
    out = os.path.join(work_dir, "res_odd.mp4")
    transcode(odd_size_file, out, preset="ultrafast", resolution="320")
    video = probe_media(out).primary_video
    assert video.width % 2 == 0 and video.height % 2 == 0


def test_upscale_blocked_by_default(hevc_file, work_dir):
    out = os.path.join(work_dir, "no_up.mp4")
    result = transcode(hevc_file, out, preset="ultrafast", resolution="1080p")
    video = probe_media(out).primary_video
    assert (video.width, video.height) == (640, 360)  # 收敛回源
    assert any("放大" in w for w in result.warnings)


def test_upscale_allowed(hevc_file, work_dir):
    out = os.path.join(work_dir, "up.mp4")
    result = transcode(hevc_file, out, preset="ultrafast",
                       resolution="1080p", allow_upscale=True)
    video = probe_media(out).primary_video
    assert (video.width, video.height) == (1920, 1080)
    assert not any("放大" in w for w in result.warnings)


def test_invalid_resolution_rejected(hevc_file, work_dir):
    out = os.path.join(work_dir, "bad_res.mp4")
    with pytest.raises(ValueError):
        transcode(hevc_file, out, resolution="1920y1080")
    assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# 组合场景：码率 + 分辨率 + 分段
# ---------------------------------------------------------------------------

def test_bitrate_and_resolution_combined(hevc_file, work_dir):
    out = os.path.join(work_dir, "combo.mp4")
    result = transcode(
        hevc_file, out, preset="ultrafast",
        target_bitrate="250k", rate_mode="cbr", resolution="x180",
    )
    info = probe_media(out)
    assert (info.primary_video.width, info.primary_video.height) == (320, 180)
    assert result.target_bitrate_bps == 250_000


def test_bitrate_resolution_with_segmentation(long_file, work_dir):
    # 320x240、4s，分 2 段，同时施加码率与定高 120（→160x120）
    out = os.path.join(work_dir, "seg_combo.mp4")
    result = transcode(
        long_file, out, preset="ultrafast", segment_count=2,
        target_bitrate="200k", resolution="x120",
    )
    assert result.success
    video = probe_media(out).primary_video
    assert (video.width, video.height) == (160, 120)
    assert os.path.getsize(out) > 0


def test_cli_resolution_flag(hevc_file, work_dir):
    out = os.path.join(work_dir, "cli_res.mp4")
    assert main([hevc_file, "-o", out, "--preset", "ultrafast",
                 "--resolution", "50%"]) == 0
    video = probe_media(out).primary_video
    assert (video.width, video.height) == (320, 180)
