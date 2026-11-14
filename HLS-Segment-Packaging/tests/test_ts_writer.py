"""切片命令构造与编码白名单测试（不实际执行 ffmpeg）。"""

from __future__ import annotations

import pytest

from hls_packager.errors import UnsupportedInputError
from hls_packager.models import MediaInfo, SegmentInfo, StreamInfo
from hls_packager.ts_writer import build_segmentation_command, validate_codecs


def _media(codecs: list[tuple[str, str]]) -> MediaInfo:
    """codecs: (codec_type, codec_name) 列表。"""
    streams = tuple(
        StreamInfo(index=i, codec_type=kind, codec_name=name)
        for i, (kind, name) in enumerate(codecs)
    )
    return MediaInfo(
        path="X:/input.mp4",
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration=30.0,
        bit_rate=None,
        streams=streams,
    )


def _planned() -> list[SegmentInfo]:
    return [
        SegmentInfo(0, 0.0, 8.0, "seg_0000.ts", source_start=0.0),
        SegmentInfo(1, 8.0, 8.0, "seg_0001.ts", source_start=8.0),
        SegmentInfo(2, 16.0, 6.0, "seg_0002.ts", source_start=16.0),
    ]


def test_validate_codecs_accepts_h264_aac():
    media = _media([("video", "h264"), ("audio", "aac")])
    maps = validate_codecs(media)
    assert "0:v:0?" in maps
    assert "0:1?" in maps


def test_validate_codecs_accepts_hevc_mp3():
    media = _media([("video", "hevc"), ("audio", "mp3")])
    assert validate_codecs(media)  # 不抛异常即可


def test_validate_codecs_rejects_unsupported_video():
    # av1 无法直接流拷贝进 MPEG-TS
    media = _media([("video", "av1"), ("audio", "aac")])
    with pytest.raises(UnsupportedInputError):
        validate_codecs(media)


def test_validate_codecs_rejects_no_av_streams():
    media = _media([("subtitle", "webvtt"), ("data", "bin_data")])
    with pytest.raises(UnsupportedInputError):
        validate_codecs(media)


def test_unsupported_audio_skipped_but_video_kept():
    # 音频不支持时跳过该音频，视频仍可打包
    media = _media([("video", "h264"), ("audio", "unknown_codec")])
    maps = validate_codecs(media)
    assert maps == ["0:v:0?"]


def test_command_uses_segment_muxer_and_copy():
    media = _media([("video", "h264"), ("audio", "aac")])
    argv = build_segmentation_command(
        media, _planned(), "X:/out", "seg", ["0:v:0?", "0:1?"]
    )
    joined = " ".join(argv)

    assert "-f" in argv and "segment" in argv
    assert "-segment_format" in argv and "mpegts" in argv
    assert "-c" in argv and "copy" in argv
    # 切点只含中间边界（首片起点 0 不列出）
    idx = argv.index("-segment_times")
    assert argv[idx + 1] == "8.000000,16.000000"
    # 起始序号对齐
    sn_idx = argv.index("-segment_start_number")
    assert argv[sn_idx + 1] == "0"
    # 输出模式串
    assert argv[-1].replace("\\", "/").endswith("X:/out/seg_%04d.ts")
    # 不能使用 ffmpeg 自带的 hls muxer
    assert "hls" not in argv


def test_command_single_segment_omits_segment_times():
    media = _media([("video", "h264")])
    single = [SegmentInfo(0, 0.0, 30.0, "seg_0000.ts", source_start=0.0)]
    argv = build_segmentation_command(
        media, single, "X:/out", "seg", ["0:v:0?"]
    )
    assert "-segment_times" not in argv


def test_command_respects_start_sequence_number():
    media = _media([("video", "h264")])
    planned = [
        SegmentInfo(12, 0.0, 4.0, "seg_0012.ts", source_start=0.0),
        SegmentInfo(13, 4.0, 4.0, "seg_0013.ts", source_start=4.0),
    ]
    argv = build_segmentation_command(
        media, planned, "X:/out", "seg", ["0:v:0?"]
    )
    sn_idx = argv.index("-segment_start_number")
    assert argv[sn_idx + 1] == "12"


def test_command_rejects_empty_plan():
    media = _media([("video", "h264")])
    with pytest.raises(Exception):
        build_segmentation_command(media, [], "X:/out", "seg", ["0:v:0?"])
