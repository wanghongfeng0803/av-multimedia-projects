"""编码注册表与参数构建的单元测试（不启动 ffmpeg）。"""

from __future__ import annotations

import pytest

from transcode_tool.codecs import (
    AV1,
    H264,
    H265,
    VP8,
    VP9,
    build_video_encoder_args,
    list_codecs,
    resolve_codec,
    validate_crf,
    validate_preset,
)


def test_list_codecs_covers_registry():
    canonicals = {c.canonical for c in list_codecs()}
    assert canonicals == {"h264", "h265", "vp9", "vp8", "av1"}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("h264", H264), ("H264", H264), (" h264 ", H264),
        ("avc", H264), ("x264", H264), ("h.264", H264),
        ("h265", H265), ("HEVC", H265), ("hevc", H265), ("x265", H265),
        ("vp9", VP9), ("VP09", VP9),
        ("vp8", VP8), ("av1", AV1), ("AV01", AV1),
    ],
)
def test_resolve_canonical_and_alias(name, expected):
    assert resolve_codec(name) is expected


def test_resolve_default_is_h264():
    assert resolve_codec(None) is H264


def test_resolve_unknown_codec_raises():
    with pytest.raises(ValueError, match="未知视频编码"):
        resolve_codec("mpeg2")


def test_crf_range_per_codec():
    assert (H264.crf_min, H264.crf_max) == (0, 51)
    assert (H265.crf_min, H265.crf_max) == (0, 51)
    assert (VP9.crf_min, VP9.crf_max) == (0, 63)
    assert (AV1.crf_min, AV1.crf_max) == (0, 63)
    assert VP8.crf_min == 4  # libvpx 常量质量下限是 4


@pytest.mark.parametrize(
    "codec,good,bad",
    [(H264, 0, 52), (H264, 51, -1), (VP9, 63, 64), (VP8, 4, 3), (AV1, 30, 100)],
)
def test_validate_crf_bounds(codec, good, bad):
    validate_crf(codec, good)  # 不抛异常
    with pytest.raises(ValueError):
        validate_crf(codec, bad)


def test_validate_crf_rejects_non_int_and_bool():
    with pytest.raises(ValueError):
        validate_crf(H264, 23.0)
    with pytest.raises(ValueError):
        validate_crf(H264, True)  # bool 是 int 子类，必须显式拒绝
    with pytest.raises(ValueError):
        validate_crf(H264, "23")  # type: ignore[arg-type]


def test_default_crf_per_codec():
    assert H264.default_crf == 23
    assert H265.default_crf == 28
    assert VP9.default_crf == 31
    assert AV1.default_crf == 30


def test_validate_preset_named_and_numeric():
    validate_preset(VP9, "medium")
    validate_preset(VP9, "ultrafast")
    validate_preset(AV1, "8")   # 数字档位
    with pytest.raises(ValueError):
        validate_preset(H264, "turbo")
    with pytest.raises(ValueError):
        validate_preset(AV1, "99")
    with pytest.raises(ValueError):
        validate_preset(VP9, "")


def test_x265_args_use_named_preset():
    enc = H265.encoder_chain[0]
    args = build_video_encoder_args(H265, enc, 28, "slow")
    assert args == ["-preset", "slow", "-crf", "28", "-tag:v", "hvc1"]


def test_x264_args_basic():
    enc = H264.encoder_chain[0]
    args = build_video_encoder_args(H264, enc, 23, "medium")
    assert args == ["-preset", "medium", "-crf", "23"]


def test_libvpx_args_constant_quality():
    enc = VP9.encoder_chain[0]
    args = build_video_encoder_args(VP9, enc, 31, "ultrafast")
    # 常量质量：-b:v 0；ultrafast 映射 cpu-used 8；开启 row-mt
    assert args[:4] == ["-b:v", "0", "-crf", "31"]
    assert "-cpu-used" in args and args[args.index("-cpu-used") + 1] == "8"
    assert "-deadline" in args and "good" in args
    assert "-row-mt" in args


def test_libvpx_accepts_numeric_cpu_used():
    enc = VP9.encoder_chain[0]
    args = build_video_encoder_args(VP9, enc, 31, "4")
    assert args[args.index("-cpu-used") + 1] == "4"


def test_libvpx_rejects_out_of_range_numeric():
    enc = VP9.encoder_chain[0]
    with pytest.raises(ValueError, match="数字 preset"):
        build_video_encoder_args(VP9, enc, 31, "9")


def test_svtav1_args_numeric_preset():
    enc = AV1.resolve_encoder("libsvtav1")
    args = build_video_encoder_args(AV1, enc, 30, "ultrafast")
    assert args == ["-crf", "30", "-preset", "13"]
    args_medium = build_video_encoder_args(AV1, enc, 30, "medium")
    assert args_medium[args_medium.index("-preset") + 1] == "8"


def test_libaom_args_cpu_used_range():
    enc = AV1.resolve_encoder("libaom-av1")
    args = build_video_encoder_args(AV1, enc, 30, "fast")
    assert args[args.index("-cpu-used") + 1] == "5"
    with pytest.raises(ValueError):
        build_video_encoder_args(AV1, enc, 30, "14")


def test_rav1e_maps_crf_to_qp():
    enc = AV1.resolve_encoder("librav1e")
    args_max = build_video_encoder_args(AV1, enc, 63, "ultrafast")
    assert args_max[args_max.index("-qp") + 1] == "255"
    args_min = build_video_encoder_args(AV1, enc, 0, "placebo")
    assert args_min[args_min.index("-qp") + 1] == "0"
    assert args_min[args_min.index("-speed") + 1] == "0"


def test_av1_encoder_chain_order_and_fallback():
    names = [e.name for e in AV1.encoder_chain]
    assert names == ["libsvtav1", "libaom-av1", "librav1e"]


def test_build_args_rejects_crf_outside_range():
    enc = H264.encoder_chain[0]
    with pytest.raises(ValueError):
        build_video_encoder_args(H264, enc, 99, "medium")
