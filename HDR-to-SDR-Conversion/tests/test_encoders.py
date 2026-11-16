"""编码器规格单测：软/硬编参数翻译、VUI 内联差异、白名单校验。

不依赖真实 ffmpeg（list_available_encoders 的探测部分在集成测试覆盖）。
"""

from __future__ import annotations

import pytest

from hdr_to_sdr.encoders import (
    ALL_ENCODERS,
    HARDWARE_ENCODERS,
    SOFTWARE_ENCODERS,
    build_quality_args,
    build_vui_args,
    encoder_spec,
    is_software_encoder,
)


class TestEncoderTable:
    def test_eight_encoders_covered(self):
        # 2 软编 + 6 硬编（nvenc/qsv/amf 各 H.264/HEVC）
        assert set(SOFTWARE_ENCODERS) == {"libx264", "libx265"}
        assert len(HARDWARE_ENCODERS) == 6
        assert len(ALL_ENCODERS) == 8

    @pytest.mark.parametrize("name", ALL_ENCODERS)
    def test_spec_lookup(self, name):
        spec = encoder_spec(name)
        assert spec.name == name
        assert spec.codec in ("h264", "hevc")
        assert spec.family in ("software", "nvenc", "qsv", "amf")

    def test_unknown_encoder_raises(self):
        with pytest.raises(ValueError):
            encoder_spec("rm -rf /")

    @pytest.mark.parametrize("name", SOFTWARE_ENCODERS)
    def test_software_flag(self, name):
        assert is_software_encoder(name) is True

    @pytest.mark.parametrize("name", HARDWARE_ENCODERS)
    def test_hardware_flag(self, name):
        assert is_software_encoder(name) is False


class TestQualityArgs:
    def test_software_uses_crf(self):
        args = build_quality_args("libx264", 18, "medium")
        assert "-crf" in args and "18" in args
        args265 = build_quality_args("libx265", 23, "fast")
        assert "-crf" in args265 and "23" in args265

    def test_nvenc_uses_cq(self):
        args = build_quality_args("h264_nvenc", 24, "medium")
        assert "-cq" in args and "24" in args
        # 固定质量而非码率模式
        assert "vbr" in args

    def test_qsv_uses_global_quality(self):
        args = build_quality_args("hevc_qsv", 20, "medium")
        assert "-global_quality" in args and "20" in args

    def test_amf_uses_three_qp(self):
        args = build_quality_args("h264_amf", 26, "medium")
        for flag in ("-qp_i", "-qp_p", "-qp_b"):
            assert flag in args and "26" in args

    def test_crf_clamped_to_valid_range(self):
        # 越界质量值钳到 0..51，不允许把异常数字透传给 ffmpeg
        args = build_quality_args("h264_nvenc", 999, "medium")
        assert args[args.index("-cq") + 1] == "51"
        args0 = build_quality_args("libx264", -10, "medium")
        assert args0[args0.index("-crf") + 1] == "0"


class TestVUIArgs:
    @pytest.mark.parametrize("name", SOFTWARE_ENCODERS)
    def test_software_inline_vui(self, name):
        inline, outer = build_vui_args(name)
        assert any("x264-params" in tok or "x265-params" in tok for tok in inline)
        joined = " ".join(inline)
        assert "colorprim=bt709" in joined
        assert "-color_trc" in outer  # 外层兜底标签始终保留

    @pytest.mark.parametrize("name", HARDWARE_ENCODERS)
    def test_hardware_outer_vui_only(self, name):
        inline, outer = build_vui_args(name)
        assert inline == []  # 硬编外层标签即写码流，不发 x264 私有参数
        assert outer == [
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
        ]
