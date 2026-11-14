"""容器规格与多容器切片命令测试（纯逻辑，不执行 ffmpeg）。"""

from __future__ import annotations

import pytest

from hls_packager.containers import (
    CONTAINER_SPECS,
    DEFAULT_CONTAINER,
    SUPPORTED_CONTAINERS,
    get_container,
)
from hls_packager.errors import PackagerError, UnsupportedInputError
from hls_packager.models import MediaInfo, SegmentInfo, StreamInfo
from hls_packager.naming import NamingConfig
from hls_packager.ts_writer import build_segmentation_command, validate_codecs


def _media(codecs: list[tuple[str, str]]) -> MediaInfo:
    streams = tuple(
        StreamInfo(index=i, codec_type=kind, codec_name=name)
        for i, (kind, name) in enumerate(codecs)
    )
    return MediaInfo(
        path="X:/input.mp4",
        format_name="mov,mp4,m4a",
        duration=30.0,
        bit_rate=None,
        streams=streams,
    )


def _planned(ext: str = "ts") -> list[SegmentInfo]:
    return [
        SegmentInfo(0, 0.0, 8.0, f"seg_0000.{ext}", source_start=0.0),
        SegmentInfo(1, 8.0, 8.0, f"seg_0001.{ext}", source_start=8.0),
    ]


def test_supported_container_table():
    assert DEFAULT_CONTAINER == "ts"
    assert set(SUPPORTED_CONTAINERS) == {"ts", "fmp4", "mp4", "mkv"}
    ts = get_container("ts")
    assert ts.segment_format == "mpegts"
    assert ts.extension == "ts"
    assert ts.hls_version == 3
    assert ts.is_hls_standard is True


def test_fmp4_uses_fragmented_movflags():
    spec = get_container("fmp4")
    assert spec.segment_format == "mp4"
    assert spec.extension == "m4s"
    assert spec.hls_version == 7
    assert spec.is_hls_standard is True
    assert "-movflags" in spec.muxer_options
    movflags = spec.muxer_options[spec.muxer_options.index("-movflags") + 1]
    assert "empty_moov" in movflags
    assert "frag_keyframe" in movflags


def test_mp4_and_mkv_specs():
    mp4 = get_container("mp4")
    assert mp4.extension == "mp4"
    assert mp4.is_hls_standard is False
    mkv = get_container("mkv")
    assert mkv.segment_format == "matroska"
    assert mkv.extension == "mkv"


def test_unknown_container_raises_with_legal_values():
    with pytest.raises(PackagerError) as exc_info:
        get_container("avi")
    message = str(exc_info.value)
    for name in SUPPORTED_CONTAINERS:
        assert name in message


def test_default_container_matches_ts_whitelist():
    # 不传容器时 validate_codecs 的行为与历史 MPEG-TS 白名单一致
    media = _media([("video", "h264"), ("audio", "aac")])
    assert validate_codecs(media) == validate_codecs(media, get_container("ts"))


def test_container_specific_codec_whitelist():
    # MPEG-4 Part 2（ffprobe 名 mpeg4）不能进 TS，但可以进 mp4/mkv
    media = _media([("video", "mpeg4"), ("audio", "aac")])
    with pytest.raises(UnsupportedInputError):
        validate_codecs(media, get_container("ts"))
    assert "0:v:0?" in validate_codecs(media, get_container("mp4"))
    assert "0:v:0?" in validate_codecs(media, get_container("mkv"))


def test_mkv_accepts_wider_audio_codecs():
    media = _media([("video", "h264"), ("audio", "flac")])
    # TS 白名单不含 flac：音频被跳过，仅剩视频
    ts_maps = validate_codecs(media, get_container("ts"))
    assert ts_maps == ["0:v:0?"]
    # MKV 可承载 flac
    mkv_maps = validate_codecs(media, get_container("mkv"))
    assert "0:1?" in mkv_maps


def test_fmp4_command_uses_mp4_muxer_and_movflags():
    media = _media([("video", "h264"), ("audio", "aac")])
    spec = get_container("fmp4")
    naming = NamingConfig(name_prefix="seg", extension=spec.extension)
    argv = build_segmentation_command(
        media, _planned("m4s"), "X:/out", "seg", ["0:v:0?", "0:1?"],
        container=spec, naming=naming,
    )
    fmt_idx = argv.index("-segment_format")
    assert argv[fmt_idx + 1] == "mp4"
    assert "-movflags" in argv
    assert argv[-1].replace("\\", "/").endswith("X:/out/seg_%04d.m4s")


def test_mkv_command_pattern_extension():
    media = _media([("video", "h264")])
    spec = get_container("mkv")
    naming = NamingConfig(name_prefix="clip", extension=spec.extension)
    argv = build_segmentation_command(
        media, _planned("mkv"), "X:/out", "clip", ["0:v:0?"],
        container=spec, naming=naming,
    )
    assert argv[argv.index("-segment_format") + 1] == "matroska"
    assert argv[-1].replace("\\", "/").endswith("X:/out/clip_%04d.mkv")


def test_custom_naming_template_in_command():
    media = _media([("video", "h264")])
    spec = get_container("fmp4")
    naming = NamingConfig(
        template="{name}-{index:03d}.{ext}",
        name_prefix="ep",
        extension=spec.extension,
    )
    argv = build_segmentation_command(
        media, _planned("m4s"), "X:/out", "ep", ["0:v:0?"],
        container=spec, naming=naming,
    )
    # index 模式：计数器从 0 起，模式串带三位补零
    assert argv[argv.index("-segment_start_number") + 1] == "0"
    assert argv[-1].replace("\\", "/").endswith("X:/out/ep-%03d.m4s")


def test_legacy_command_without_container_args_unchanged():
    media = _media([("video", "h264"), ("audio", "aac")])
    argv = build_segmentation_command(
        media, _planned(), "X:/out", "seg", ["0:v:0?", "0:1?"]
    )
    # 缺省路径必须与历史命令一致：mpegts + seg_%04d.ts + 起始序号 0
    assert argv[argv.index("-segment_format") + 1] == "mpegts"
    assert argv[-1].replace("\\", "/").endswith("X:/out/seg_%04d.ts")
    assert "-movflags" not in argv


def test_all_specs_have_consistent_extensions():
    for name, spec in CONTAINER_SPECS.items():
        assert spec.name == name
        assert spec.extension
        assert spec.hls_version >= 3
        assert spec.video_codecs and spec.audio_codecs
