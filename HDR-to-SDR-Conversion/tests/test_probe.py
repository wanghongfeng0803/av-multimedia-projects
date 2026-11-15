"""探测逻辑测试。

parse_hdr_info 用合成 ffprobe JSON 做纯函数测试（不依赖 ffmpeg）；
真实文件的端到端探测在类尾部、有 ffmpeg 时才运行。
"""

from __future__ import annotations

import pytest

from hdr_to_sdr.probe import parse_hdr_info, probe_hdr, HDRInfo
from hdr_to_sdr.errors import (
    UnsupportedHDRTypeError,
    InputNotFoundError,
    InvalidInputError,
)
from hdr_to_sdr.transfer import TRANSFER_PQ, TRANSFER_HLG


def _stream(transfer="smpte2084", pix_fmt="yuv420p10le", **over):
    stream = {
        "codec_type": "video",
        "width": 320,
        "height": 240,
        "pix_fmt": pix_fmt,
        "color_transfer": transfer,
        "color_primaries": "bt2020",
        "color_range": "tv",
        "avg_frame_rate": "30/1",
    }
    stream.update(over)
    return {"streams": [stream]}


class TestParseHDRType:
    def test_pq_detected(self):
        info = parse_hdr_info(_stream("smpte2084"))
        assert info.is_pq and not info.is_hlg
        assert info.transfer == TRANSFER_PQ
        assert info.assumed is False

    def test_hlg_detected(self):
        info = parse_hdr_info(_stream("arib-std-b67"))
        assert info.is_hlg and not info.is_pq
        assert info.transfer == TRANSFER_HLG

    def test_unknown_transfer_rejected(self):
        """核心需求：不能靠猜。未知/缺失 transfer 必须报错。"""
        with pytest.raises(UnsupportedHDRTypeError):
            parse_hdr_info(_stream("bt709"))
        with pytest.raises(UnsupportedHDRTypeError):
            parse_hdr_info(_stream("unknown"))

    def test_missing_transfer_rejected(self):
        raw = _stream()
        del raw["streams"][0]["color_transfer"]
        with pytest.raises(UnsupportedHDRTypeError):
            parse_hdr_info(raw)

    def test_sdr_not_silently_treated_as_hdr(self):
        with pytest.raises(UnsupportedHDRTypeError):
            parse_hdr_info(_stream("bt709", pix_fmt="yuv420p"))

    def test_assume_pq_overrides_unknown(self):
        info = parse_hdr_info(_stream("unknown"), assume="pq")
        assert info.is_pq and info.assumed is True

    def test_assume_hlg_overrides_missing(self):
        raw = _stream()
        del raw["streams"][0]["color_transfer"]
        info = parse_hdr_info(raw, assume="hlg")
        assert info.is_hlg and info.assumed is True

    def test_assume_ignored_when_metadata_certain(self):
        # 元数据已确证 HLG 时，assume 不能改变结论
        info = parse_hdr_info(_stream("arib-std-b67"), assume="pq")
        assert info.is_hlg and info.assumed is False

    def test_bad_assume_value(self):
        with pytest.raises(UnsupportedHDRTypeError):
            parse_hdr_info(_stream("unknown"), assume="nope")

    def test_assume_aliases(self):
        assert parse_hdr_info(_stream("unknown"), assume="st2084").is_pq
        assert parse_hdr_info(_stream("unknown"), assume="b67").is_hlg


class TestParseFields:
    def test_bits_from_pix_fmt(self):
        info = parse_hdr_info(_stream(pix_fmt="yuv420p10le"))
        assert info.bits_per_raw_sample == 10

    def test_bits_from_raw_sample_field(self):
        info = parse_hdr_info(_stream(pix_fmt="yuv420p10le",
                                      bits_per_raw_sample="12"))
        assert info.bits_per_raw_sample == 12

    def test_rgb_pix_fmt_flag(self):
        info = parse_hdr_info(_stream(pix_fmt="gbrpf32le"))
        assert info.is_rgb is True
        info2 = parse_hdr_info(_stream(pix_fmt="yuv444p10le"))
        assert info2.is_rgb is False

    def test_full_range(self):
        info = parse_hdr_info(_stream(color_range="pc"))
        assert info.pixel_range == "full"

    def test_garbled_range_defaults_limited(self):
        info = parse_hdr_info(_stream(color_range="weird"))
        assert info.pixel_range == "limited"

    def test_frame_rate(self):
        info = parse_hdr_info(_stream(avg_frame_rate="24000/1001"))
        assert info.frame_rate == pytest.approx(23.976, abs=1e-3)

    def test_peak_from_stream_side_data(self):
        raw = _stream()
        raw["streams"][0]["side_data_list"] = [
            {"side_data_type": "Mastering display metadata",
             "max_luminance": "10000000/10000"},
        ]
        info = parse_hdr_info(raw)
        assert info.peak_nits == pytest.approx(1000.0)

    def test_peak_from_frame_side_data(self):
        raw = _stream()
        frame_raw = {"frames": [{"side_data_list": [
            {"side_data_type": "Content light level metadata",
             "max_content": 1000, "max_average": 400},
        ]}]}
        info = parse_hdr_info(raw, frame_raw=frame_raw)
        assert info.peak_nits == 1000.0

    def test_peak_missing_is_none(self):
        assert parse_hdr_info(_stream()).peak_nits is None

    def test_no_video_stream_raises(self):
        with pytest.raises(InvalidInputError):
            parse_hdr_info({"streams": [{"codec_type": "audio"}]})


class TestProbeFile:
    def test_missing_file(self):
        with pytest.raises(InputNotFoundError):
            probe_hdr("definitely_not_exist_12345.mkv")

    @pytest.fixture(autouse=True)
    def _require_ffmpeg(self):
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("环境中未安装 ffmpeg")

    def test_real_pq_file(self, pq_file):
        info = probe_hdr(pq_file)
        assert info.is_pq
        assert info.color_primaries == "bt2020"
        assert info.bits_per_raw_sample == 10
        assert info.peak_nits == pytest.approx(1000.0)
        assert (info.width, info.height) == (160, 120)

    def test_real_hlg_file(self, hlg_file):
        info = probe_hdr(hlg_file)
        assert info.is_hlg
        assert info.peak_nits is None  # HLG 样本不带 mastering metadata

    def test_real_sdr_rejected(self, sdr_file):
        with pytest.raises(UnsupportedHDRTypeError):
            probe_hdr(sdr_file)

    def test_sdr_with_assume(self, sdr_file):
        info = probe_hdr(sdr_file, assume="pq")
        assert info.is_pq and info.assumed is True
