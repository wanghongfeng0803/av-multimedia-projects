"""ffprobe 探测模块测试。"""

from __future__ import annotations

import os

import pytest

from transcode_tool.errors import InputNotFoundError, InvalidInputError
from transcode_tool.probe import probe_media


def test_probe_hevc_reports_streams(hevc_file):
    info = probe_media(hevc_file)
    assert info.primary_video is not None
    assert info.primary_video.codec_name == "hevc"
    assert info.primary_audio is not None
    assert info.primary_audio.codec_name == "aac"
    assert info.duration_seconds and info.duration_seconds > 1


def test_probe_missing_file_raises(work_dir):
    with pytest.raises(InputNotFoundError):
        probe_media(os.path.join(work_dir, "nope.mp4"))


def test_probe_text_disguised_as_mp4_raises(text_disguised_file):
    with pytest.raises(InvalidInputError, match="无法解析|没有任何媒体流"):
        probe_media(text_disguised_file)


def test_probe_no_audio_file(no_audio_file):
    info = probe_media(no_audio_file)
    assert info.primary_video is not None
    assert info.primary_audio is None


def test_probe_multi_stream(multi_stream_file):
    info = probe_media(multi_stream_file)
    assert len(info.audio_streams) == 2
    assert len(info.video_streams) == 1


def test_probe_truncated_does_not_crash_or_raises_invalid(truncated_file):
    # 截断文件：要么带着警告探测成功，要么明确报 InvalidInputError，
    # 绝不允许出现其它未受控异常。
    try:
        info = probe_media(truncated_file)
    except InvalidInputError:
        return
    assert info.primary_video is not None
    assert info.warnings  # 损坏情况必须留下警告
