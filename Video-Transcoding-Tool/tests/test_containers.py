"""容器兼容矩阵与音频策略的单元测试（不启动 ffmpeg）。"""

from __future__ import annotations

import os

import pytest

from transcode_tool.codecs import AV1, H264, H265, VP8, VP9
from transcode_tool.containers import (
    MKV,
    MP4,
    WEBM,
    ContainerError,
    audio_strategy,
    container_from_extension,
    default_container_for,
    ensure_video_compatible,
    list_containers,
    normalize_output_extension,
    resolve_container,
)


def test_container_registry():
    assert {c.canonical for c in list_containers()} == {"mp4", "webm", "mkv"}


@pytest.mark.parametrize(
    "name,expected",
    [("mp4", MP4), ("MP4", MP4), (".mp4", MP4), ("mov", MP4), ("m4v", MP4),
     ("webm", WEBM), ("WEBM", WEBM), ("mkv", MKV), ("matroska", MKV)],
)
def test_resolve_container_aliases(name, expected):
    assert resolve_container(name) is expected


def test_resolve_unknown_container_raises():
    with pytest.raises(ContainerError, match="未知容器"):
        resolve_container("avi")


@pytest.mark.parametrize(
    "path,expected",
    [("a.mp4", MP4), ("b.MKV", MKV), ("c.webm", WEBM),
     ("d.avi", None), ("noext", None), ("", None)],
)
def test_container_from_extension(path, expected):
    assert container_from_extension(path) is expected


@pytest.mark.parametrize(
    "container,codec,ok",
    [(MP4, H264, True), (MP4, H265, True), (MP4, AV1, True),
     (MP4, VP9, False), (MP4, VP8, False),
     (WEBM, VP9, True), (WEBM, VP8, True), (WEBM, AV1, True),
     (WEBM, H264, False), (WEBM, H265, False),
     (MKV, H264, True), (MKV, H265, True), (MKV, VP9, True),
     (MKV, VP8, True), (MKV, AV1, True)],
)
def test_video_compatibility_matrix(container, codec, ok):
    if ok:
        ensure_video_compatible(container, codec)  # 不抛异常
    else:
        with pytest.raises(ContainerError, match="不能承载"):
            ensure_video_compatible(container, codec)


def test_default_container_per_codec():
    assert default_container_for(H264) is MP4
    assert default_container_for(H265) is MP4
    assert default_container_for(VP9) is WEBM
    assert default_container_for(VP8) is WEBM
    assert default_container_for(AV1) is MP4


@pytest.mark.parametrize(
    "container,src,mode",
    [(MP4, "aac", "copy"), (MP4, "mp3", "copy"), (MP4, "opus", "reencode"),
     (MP4, "flac", "reencode"), (MP4, None, "drop"),
     (WEBM, "opus", "copy"), (WEBM, "vorbis", "copy"),
     (WEBM, "aac", "reencode"), (WEBM, "mp3", "reencode"), (WEBM, None, "drop"),
     (MKV, "aac", "copy"), (MKV, "flac", "copy"), (MKV, "opus", "copy"),
     (MKV, "wmapro", "reencode"), (MKV, None, "drop")],
)
def test_audio_strategy(container, src, mode):
    result_mode, encoder = audio_strategy(container, src)
    assert result_mode == mode
    if mode == "reencode":
        assert encoder == container.audio_encoder
    else:
        assert encoder is None


def test_audio_reencode_target_codec():
    assert WEBM.audio_encoder == "libopus"
    assert WEBM.audio_encoder_fallbacks == ("libvorbis",)


@pytest.mark.parametrize(
    "path,container,expected",
    [("out.avi", MP4, "out.mp4"),
     ("out", WEBM, "out.webm"),
     (os.path.join("d", "x.mkv"), WEBM, os.path.join("d", "x.webm")),
     ("a.b.c.mkv", MKV, "a.b.c.mkv")],
)
def test_normalize_output_extension(path, container, expected):
    assert normalize_output_extension(path, container) == expected


def test_mp4_uses_faststart_others_do_not():
    assert MP4.faststart is True
    assert WEBM.faststart is False
    assert MKV.faststart is False
