"""转码核心逻辑端到端测试。"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from transcode_tool.binaries import find_ffprobe
from transcode_tool.errors import (
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
)
from transcode_tool.probe import probe_media
from transcode_tool.transcoder import transcode


def _assert_valid_h264_mp4(path: str):
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0
    info = probe_media(path)
    video = info.primary_video
    assert video is not None
    assert video.codec_name == "h264"
    assert video.pix_fmt == "yuv420p"
    # H.264 yuv420p 要求偶数尺寸
    assert video.width % 2 == 0 and video.height % 2 == 0
    return info


def test_hevc_to_h264(hevc_file, work_dir):
    out = os.path.join(work_dir, "out", "hevc_h264.mp4")  # 目录不存在，应自动创建
    result = transcode(hevc_file, out, preset="ultrafast")
    assert result.success
    info = _assert_valid_h264_mp4(out)
    # AAC 音频应直接保留
    assert info.primary_audio is not None
    assert info.primary_audio.codec_name == "aac"
    assert result.input_path == os.path.abspath(hevc_file)


def test_odd_dimensions_are_rounded(odd_size_file, work_dir):
    out = os.path.join(work_dir, "odd.mp4")
    result = transcode(odd_size_file, out, preset="ultrafast")
    info = _assert_valid_h264_mp4(out)
    assert info.primary_video.width == 640
    assert info.primary_video.height == 362
    assert any("奇数" in w for w in result.warnings)


def test_rgb_and_mp3_audio_converted(rgb_file, work_dir):
    out = os.path.join(work_dir, "rgb.mp4")
    result = transcode(rgb_file, out, preset="ultrafast")
    info = _assert_valid_h264_mp4(out)
    # mp3 属于 MP4 可直通编码，允许 copy；这里主要验证像素格式已转换且产物可播放
    assert info.primary_video.pix_fmt == "yuv420p"
    assert any("yuv420p" in w for w in result.warnings)


def test_no_audio_input_produces_silent_video(no_audio_file, work_dir):
    out = os.path.join(work_dir, "silent.mp4")
    result = transcode(no_audio_file, out, preset="ultrafast")
    info = _assert_valid_h264_mp4(out)
    assert info.primary_audio is None
    assert any("无音频" in w for w in result.warnings)


def test_multi_stream_keeps_one_audio(multi_stream_file, work_dir):
    out = os.path.join(work_dir, "multi.mp4")
    result = transcode(multi_stream_file, out, preset="ultrafast")
    info = _assert_valid_h264_mp4(out)
    assert len(info.audio_streams) == 1
    assert any("丢弃" in w for w in result.warnings)


def test_text_disguised_input_raises(text_disguised_file, work_dir):
    out = os.path.join(work_dir, "x.mp4")
    with pytest.raises(InvalidInputError):
        transcode(text_disguised_file, out, preset="ultrafast")
    assert not os.path.exists(out)  # 失败时不能留下半成品


def test_missing_input_raises(work_dir):
    with pytest.raises(InputNotFoundError):
        transcode(os.path.join(work_dir, "ghost.mp4"),
                  os.path.join(work_dir, "o.mp4"))


def test_existing_output_refused_then_overwritten(hevc_file, work_dir):
    out = os.path.join(work_dir, "exists.mp4")
    transcode(hevc_file, out, preset="ultrafast")
    with pytest.raises(OutputExistsError):
        transcode(hevc_file, out, preset="ultrafast")
    # 显式覆盖后成功
    result = transcode(hevc_file, out, preset="ultrafast", overwrite=True)
    assert result.success


def test_default_output_path_beside_input(hevc_file, work_dir):
    result = transcode(hevc_file, preset="ultrafast")
    expected = os.path.join(work_dir, "sample_hevc_h264.mp4")
    assert os.path.abspath(result.output_path) == os.path.abspath(expected)
    _assert_valid_h264_mp4(expected)


def test_output_extension_normalized(hevc_file, work_dir):
    out = os.path.join(work_dir, "noext")  # 没有扩展名
    result = transcode(hevc_file, out, preset="ultrafast")
    assert result.output_path.endswith(".mp4")
    _assert_valid_h264_mp4(result.output_path)


def test_output_directory_path(hevc_file, work_dir):
    target_dir = os.path.join(work_dir, "bucket")
    os.makedirs(target_dir)
    result = transcode(hevc_file, target_dir + os.sep, preset="ultrafast")
    assert os.path.dirname(result.output_path) == os.path.abspath(target_dir)
    _assert_valid_h264_mp4(result.output_path)


def test_truncated_input_behaviour(truncated_file, work_dir):
    out = os.path.join(work_dir, "recovered.mp4")
    try:
        result = transcode(truncated_file, out, preset="ultrafast")
    except (InvalidInputError, FFmpegRunError):
        # 损坏严重到无法挽救时，允许明确失败，但不能留下半成品
        assert not os.path.exists(out)
        return
    assert result.success
    _assert_valid_h264_mp4(out)
    assert result.warnings


def test_invalid_crf_and_preset(hevc_file, work_dir):
    with pytest.raises(ValueError):
        transcode(hevc_file, os.path.join(work_dir, "a.mp4"), crf=99)
    with pytest.raises(ValueError):
        transcode(hevc_file, os.path.join(work_dir, "b.mp4"), preset="turbo")


def test_invalid_input_leaves_no_artifacts(text_disguised_file, work_dir):
    out = os.path.join(work_dir, "fail.mp4")
    with pytest.raises(InvalidInputError):
        transcode(text_disguised_file, out)
    # InvalidInput 在 ffmpeg 启动前就拦截，因此既无输出也无日志残留
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".ffmpeg.log")


def test_ffmpeg_failure_writes_log_and_cleans(hevc_file, work_dir, monkeypatch):
    """模拟 ffmpeg 自身崩溃：应抛 FFmpegRunError、落盘日志、清理半成品。"""
    fake_ffmpeg = os.path.join(work_dir, "fake_ffmpeg.bat")
    with open(fake_ffmpeg, "w", encoding="utf-8") as fh:
        fh.write("@echo simulated ffmpeg crash 1>&2\n@exit 1\n")
    monkeypatch.setattr(
        "transcode_tool.transcoder.find_ffmpeg", lambda: fake_ffmpeg
    )
    out = os.path.join(work_dir, "boom.mp4")
    with pytest.raises(FFmpegRunError, match="simulated ffmpeg crash") as exc:
        transcode(hevc_file, out, preset="ultrafast")
    assert not os.path.exists(out)  # 半成品已清理
    assert os.path.isfile(out + ".ffmpeg.log")  # 完整日志已落盘
    assert "simulated ffmpeg crash" in exc.value.stderr


def test_output_playable_with_ffprobe_json(hevc_file, work_dir):
    """用独立 ffprobe 进程交叉验证产物。"""
    out = os.path.join(work_dir, "cross.mp4")
    transcode(hevc_file, out, preset="ultrafast")
    proc = subprocess.run(
        [find_ffprobe(), "-v", "error", "-print_format", "json",
         "-show_streams", out],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    codecs = {s["codec_type"]: s["codec_name"] for s in data["streams"]}
    assert codecs["video"] == "h264"
