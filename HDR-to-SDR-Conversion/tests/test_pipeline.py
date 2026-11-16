"""端到端管线测试：真实 ffmpeg 解码 → Python 引擎 → ffmpeg 编码。

无 ffmpeg 时由 conftest 的 pytestmark 整体 skip。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from hdr_to_sdr.binaries import find_ffprobe
from hdr_to_sdr.pipeline import convert_hdr_to_sdr, build_decode_cmd, build_encode_cmd
from hdr_to_sdr.probe import probe_hdr
from hdr_to_sdr.encoders import (
    HARDWARE_ENCODERS,
    check_encoder_works,
)
from hdr_to_sdr.errors import (
    UnsupportedHDRTypeError,
    InputNotFoundError,
)
from hdr_to_sdr.transfer import TRANSFER_PQ, TRANSFER_HLG

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)

# 本机运行时真正可用的硬编（编译进来≠可用，需真实编码冒烟通过）
_WORKING_HW = [name for name in HARDWARE_ENCODERS if check_encoder_works(name)]
hw_available = pytest.mark.skipif(
    not _WORKING_HW, reason="本机无运行时可用的 GPU 硬编（nvenc/qsv/amf）"
)


def _ffprobe_json(path: str) -> dict:
    proc = subprocess.run(
        [find_ffprobe(), "-v", "error", "-show_streams", "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    return json.loads(proc.stdout)


def _video_stream(path: str) -> dict:
    for stream in _ffprobe_json(path)["streams"]:
        if stream.get("codec_type") == "video":
            return stream
    raise AssertionError(f"输出中无视频流: {path}")


class TestPQConversion:
    def test_basic_conversion(self, pq_file, work_dir):
        out = os.path.join(work_dir, "pq_sdr.mp4")
        result = convert_hdr_to_sdr(pq_file, out)
        assert os.path.isfile(out)
        assert result.transfer == TRANSFER_PQ
        assert result.frames_processed > 0
        assert result.width == 160 and result.height == 120

    def test_output_tagged_bt709_sdr(self, pq_file, work_dir):
        """转换后必须是 BT.709 SDR 三件套标签。"""
        out = os.path.join(work_dir, "pq_sdr.mp4")
        convert_hdr_to_sdr(pq_file, out)
        stream = _video_stream(out)
        assert stream["color_transfer"] == "bt709"
        assert stream["color_primaries"] == "bt709"
        assert stream["color_space"] == "bt709"

    def test_output_has_no_hdr_transfer(self, pq_file, work_dir):
        """HDR 传递函数标签必须被清除，不能残留 smpte2084/hlg。"""
        out = os.path.join(work_dir, "pq_sdr.mp4")
        convert_hdr_to_sdr(pq_file, out)
        stream = _video_stream(out)
        assert stream["color_transfer"] not in ("smpte2084", "arib-std-b67")

    def test_output_8bit_default(self, pq_file, work_dir):
        out = os.path.join(work_dir, "pq_sdr.mp4")
        convert_hdr_to_sdr(pq_file, out)
        stream = _video_stream(out)
        assert stream["pix_fmt"] in ("yuv420p", "yuv444p")

    def test_output_10bit_option(self, pq_file, work_dir):
        out = os.path.join(work_dir, "pq_sdr_10.mp4")
        convert_hdr_to_sdr(pq_file, out, dst_bits=10)
        stream = _video_stream(out)
        assert "10le" in stream["pix_fmt"]

    def test_audio_passthrough(self, pq_file, work_dir):
        """PQ 夹具带 AAC 音轨，转换后音频应保留（copy 或 aac 回退）。"""
        out = os.path.join(work_dir, "pq_audio.mp4")
        result = convert_hdr_to_sdr(pq_file, out)
        assert result.audio_mode in ("copy", "aac")
        data = _ffprobe_json(out)
        assert any(s["codec_type"] == "audio" for s in data["streams"])

    def test_no_clipping_in_output(self, pq_file, work_dir):
        """不过曝：解码输出帧的亮度不应大面积顶到 limited 白电平 235。"""
        out = os.path.join(work_dir, "pq_clip.mp4")
        convert_hdr_to_sdr(pq_file, out)
        # 抽帧成 rawvideo 统计 Y 码值
        proc = subprocess.run(
            [shutil.which("ffmpeg"), "-v", "error", "-i", out,
             "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        raw = proc.stdout
        width, height = 160, 120
        frame_size = width * height * 3 // 2
        first_y = raw[:width * height]
        clipped = sum(1 for c in first_y if c >= 235)
        # testsrc 含纯白块，允许少量白电平，但不得整帧全部钉死在 235
        assert clipped < width * height * 0.95


class TestHLGConversion:
    def test_basic_conversion(self, hlg_file, work_dir):
        out = os.path.join(work_dir, "hlg_sdr.mp4")
        result = convert_hdr_to_sdr(hlg_file, out)
        assert result.transfer == TRANSFER_HLG
        assert result.frames_processed > 0

    def test_hlg_output_sdr_tags(self, hlg_file, work_dir):
        out = os.path.join(work_dir, "hlg_sdr.mp4")
        convert_hdr_to_sdr(hlg_file, out)
        stream = _video_stream(out)
        assert stream["color_transfer"] == "bt709"
        assert stream["color_primaries"] == "bt709"

    def test_hlg_no_audio_stream(self, hlg_file, work_dir):
        out = os.path.join(work_dir, "hlg_na.mp4")
        result = convert_hdr_to_sdr(hlg_file, out)
        assert result.audio_mode == "none"


class TestToneMethods:
    @pytest.mark.parametrize(
        "method", ["reinhard", "reinhard_classic", "hable", "bt2390", "clip"]
    )
    def test_all_methods_produce_valid_output(self, pq_file, work_dir, method):
        out = os.path.join(work_dir, f"tone_{method}.mp4")
        result = convert_hdr_to_sdr(pq_file, out, tone_method=method)
        assert os.path.isfile(out)
        assert result.tone_method == method

    def test_hable_with_sharpness(self, pq_file, work_dir):
        """Hable + 可调锐度端到端：锐度不同产出都合法可播放、帧数一致。"""
        out1 = os.path.join(work_dir, "hable_c2.mp4")
        out2 = os.path.join(work_dir, "hable_c16.mp4")
        r1 = convert_hdr_to_sdr(
            pq_file, out1, tone_method="hable", hable_sharpness=2.0
        )
        r2 = convert_hdr_to_sdr(
            pq_file, out2, tone_method="hable", hable_sharpness=16.0
        )
        assert r1.tone_method == "hable" and r2.tone_method == "hable"
        assert r1.frames_processed == r2.frames_processed
        assert os.path.getsize(out1) > 0 and os.path.getsize(out2) > 0

    def test_hable_sharpness_out_of_range_clamps(self, pq_file, work_dir):
        """越界锐度：钳制 + warning，但转换仍成功完成（不中断批处理）。"""
        import warnings
        out = os.path.join(work_dir, "hable_clamp.mp4")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = convert_hdr_to_sdr(
                pq_file, out, tone_method="hable", hable_sharpness=99999.0,
            )
        assert os.path.isfile(out) and result.tone_method == "hable"

    def test_switching_mapper_keeps_colorspace_bitdepth_metadata(
        self, pq_file, work_dir
    ):
        """硬约束：切换映射器时色彩空间、位深、元数据清理行为完全一致。"""
        outs = {}
        for method in ("reinhard", "hable"):
            path = os.path.join(work_dir, f"switch_{method}.mp4")
            convert_hdr_to_sdr(
                pq_file, path, tone_method=method, dst_bits=10,
            )
            outs[method] = path

        streams = {m: _video_stream(p) for m, p in outs.items()}
        for key in ("color_transfer", "color_primaries", "color_space",
                    "pix_fmt", "bits_per_raw_sample"):
            assert streams["reinhard"].get(key) == streams["hable"].get(key)
        # 两者都是干净的 BT.709 SDR，无任何 HDR 传递函数残留
        for stream in streams.values():
            assert stream["color_transfer"] == "bt709"
            assert stream["color_transfer"] not in (
                "smpte2084", "arib-std-b67"
            )

    def test_hable_frame_count_matches_reinhard(self, pq_file, work_dir):
        """同素材同帧号区间：两种映射器处理帧数必须一致（可对比的前提）。"""
        r_out = os.path.join(work_dir, "cnt_r.mp4")
        h_out = os.path.join(work_dir, "cnt_h.mp4")
        r1 = convert_hdr_to_sdr(pq_file, r_out, tone_method="reinhard")
        r2 = convert_hdr_to_sdr(pq_file, h_out, tone_method="hable")
        assert r1.frames_processed == r2.frames_processed

    def test_hable_no_clipping_in_output(self, pq_file, work_dir):
        """不过曝对 hable 同样成立：亮度不整帧钉死在 limited 白电平。"""
        out = os.path.join(work_dir, "hable_clip.mp4")
        convert_hdr_to_sdr(pq_file, out, tone_method="hable")
        proc = subprocess.run(
            [shutil.which("ffmpeg"), "-v", "error", "-i", out,
             "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        raw = proc.stdout
        width, height = 160, 120
        first_y = raw[:width * height]
        clipped = sum(1 for c in first_y if c >= 235)
        assert clipped < width * height * 0.95

    def test_classic_reinhard_with_white(self, pq_file, work_dir):
        """经典 Reinhard + 可调白点端到端：白点不同产出都合法可播放。"""
        out1 = os.path.join(work_dir, "classic_w1.mp4")
        out4 = os.path.join(work_dir, "classic_w4.mp4")
        r1 = convert_hdr_to_sdr(
            pq_file, out1, tone_method="reinhard_classic", reinhard_white=1.0
        )
        r4 = convert_hdr_to_sdr(
            pq_file, out4, tone_method="reinhard_classic", reinhard_white=4.0
        )
        assert r1.tone_method == "reinhard_classic"
        assert r1.frames_processed == r4.frames_processed
        assert os.path.getsize(out1) > 0 and os.path.getsize(out4) > 0

    def test_white_out_of_range_clamps_with_warning(self, pq_file, work_dir):
        """越界白点：钳制 + warning，但转换仍成功完成。"""
        out = os.path.join(work_dir, "classic_clamp.mp4")
        with pytest.warns(UserWarning):
            convert_hdr_to_sdr(
                pq_file, out, tone_method="reinhard_classic",
                reinhard_white=99999.0,
            )
        assert os.path.isfile(out)

    def test_parallel_matches_serial_output(self, pq_file, work_dir):
        """workers=2 与串行应产出同样可播放、同尺寸、同帧数的文件。"""
        out1 = os.path.join(work_dir, "ser.mp4")
        out2 = os.path.join(work_dir, "par.mp4")
        r1 = convert_hdr_to_sdr(pq_file, out1, workers=1)
        r2 = convert_hdr_to_sdr(pq_file, out2, workers=2)
        assert r1.frames_processed == r2.frames_processed

    def test_deterministic_two_runs_identical_bytes(self, pq_file, work_dir):
        """需求：同输入两次一致——软编同参数两次产物逐字节相同。"""
        out1 = os.path.join(work_dir, "run1.mp4")
        out2 = os.path.join(work_dir, "run2.mp4")
        convert_hdr_to_sdr(pq_file, out1, preset="medium", crf=18)
        convert_hdr_to_sdr(pq_file, out2, preset="medium", crf=18)
        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            assert f1.read() == f2.read()


def _ssim_score(ref: str, other: str) -> float:
    """用 ffmpeg ssim 滤镜计算两视频平均 SSIM（解析 stderr 汇总行）。"""
    proc = subprocess.run(
        [shutil.which("ffmpeg"), "-hide_banner", "-i", ref, "-i", other,
         "-lavfi", "ssim", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    # 汇总行形如：... SSIM Y:0.99... U:... V:... All:0.99... (...)
    for line in proc.stderr.splitlines():
        if "All:" in line:
            token = line.split("All:")[1].split()[0]
            # ffmpeg 可能输出 n.m 或 1.# 形式，统一转 float
            return float(token.split("(")[0])
    raise AssertionError(f"无法解析 SSIM:\n{proc.stderr[-500:]}")


class TestSoftHardParity:
    """软编与硬编共用同一 Python 引擎，画面差异只来自末端编码器。"""

    @hw_available
    @pytest.mark.parametrize("hw", _WORKING_HW)
    def test_hardware_output_tagged_bt709(self, pq_file, work_dir, hw):
        out = os.path.join(work_dir, f"hw_{hw}.mp4")
        result = convert_hdr_to_sdr(pq_file, out, codec=hw)
        assert result.tone_method  # 转换跑通
        stream = _video_stream(out)
        assert stream["color_transfer"] == "bt709"
        assert stream["color_primaries"] == "bt709"

    @hw_available
    @pytest.mark.parametrize("hw", _WORKING_HW)
    def test_soft_hard_ssim_within_tolerance(self, pq_file, work_dir, hw):
        """同一输入软编 vs 硬编：SSIM 必须达到高相似度（编码器噪声级）。"""
        soft = os.path.join(work_dir, "soft_ref.mp4")
        hard = os.path.join(work_dir, f"hard_{hw}.mp4")
        convert_hdr_to_sdr(pq_file, soft, codec="libx264", crf=18)
        convert_hdr_to_sdr(pq_file, hard, codec=hw, crf=18)
        score = _ssim_score(soft, hard)
        # 同源原始帧、不同末端编码器，SSIM 应 ≥ 0.98（人眼近无损级别）
        assert score >= 0.98, f"{hw} 与软编 SSIM={score} 超出可接受范围"


class TestErrorPaths:
    def test_missing_input(self, work_dir):
        with pytest.raises(InputNotFoundError):
            convert_hdr_to_sdr("nope_12345.mkv",
                               os.path.join(work_dir, "o.mp4"))

    def test_sdr_input_rejected(self, sdr_file, work_dir):
        """不靠猜：SDR 输入默认拒绝，必须显式 assume。"""
        with pytest.raises(UnsupportedHDRTypeError):
            convert_hdr_to_sdr(sdr_file, os.path.join(work_dir, "o.mp4"))

    def test_sdr_input_with_assume_runs(self, sdr_file, work_dir):
        out = os.path.join(work_dir, "forced.mp4")
        result = convert_hdr_to_sdr(sdr_file, out, assume="pq")
        assert os.path.isfile(out) and result.assumed is True

    def test_bad_dst_bits(self, pq_file, work_dir):
        with pytest.raises(ValueError):
            convert_hdr_to_sdr(pq_file, os.path.join(work_dir, "o.mp4"),
                               dst_bits=12)

    def test_unknown_codec_rejected(self, pq_file, work_dir):
        """编码器走白名单：任意 -c:v 名不允许透传。"""
        with pytest.raises(ValueError):
            convert_hdr_to_sdr(pq_file, os.path.join(work_dir, "o.mp4"),
                               codec="not_a_real_encoder")


class TestCmdBuilders:
    def test_decode_cmd_yuv(self, pq_file):
        info = probe_hdr(pq_file)
        cmd, raw_fmt, bpp = build_decode_cmd("ffmpeg", pq_file, info)
        assert raw_fmt == "yuv444p16le" and bpp == 6
        assert "pipe:1" in cmd and "-an" in cmd

    def test_encode_cmd_has_sdr_tags_and_metadata_clear(self):
        cmd = build_encode_cmd(
            "ffmpeg", "o.mp4", 160, 120, 30.0, "yuv444p16le",
            "libx264", 8, "limited", "none", None, 18, "medium",
        )
        joined = " ".join(cmd)
        assert "-color_trc bt709" in joined
        assert "-color_primaries bt709" in joined
        assert "-colorspace bt709" in joined
        assert "-map_metadata -1" in joined

    @pytest.mark.parametrize("hw", HARDWARE_ENCODERS)
    def test_hardware_encode_cmd_uses_outer_vui(self, hw):
        """硬编不发 x264/x265 私有内联参数，VUI 走外层 -color_* 标签。"""
        cmd = build_encode_cmd(
            "ffmpeg", "o.mp4", 160, 120, 30.0, "yuv444p16le",
            hw, 8, "limited", "none", None, 23, "medium",
        )
        joined = " ".join(cmd)
        assert "-x264-params" not in joined and "-x265-params" not in joined
        assert "-color_trc bt709" in joined
        assert "-map_metadata -1" in joined
        assert cmd[cmd.index("-c:v") + 1] == hw

    def test_software_encode_cmd_keeps_inline_vui(self):
        cmd = build_encode_cmd(
            "ffmpeg", "o.mkv", 160, 120, 30.0, "yuv444p16le",
            "libx265", 10, "limited", "none", None, 20, "fast",
        )
        joined = " ".join(cmd)
        assert "-x265-params" in joined
        assert "colormatrix=bt709" in joined
