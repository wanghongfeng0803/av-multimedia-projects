"""CLI 测试：参数解析、probe-only、错误退出码、端到端调用。"""

from __future__ import annotations

import os
import shutil
import warnings

import pytest

from hdr_to_sdr.cli import build_parser, main

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args(["in.mkv", "out.mp4"])
        assert args.tone == "reinhard"
        assert args.codec == "libx264"
        assert args.ten_bit is False
        assert args.full_range is False
        assert args.assume is None
        assert args.workers == 1
        assert args.reinhard_white == 1.0

    def test_flags(self):
        args = build_parser().parse_args([
            "in.mkv", "out.mkv", "--tone", "bt2390", "--codec", "libx265",
            "--10bit", "--full-range", "--assume", "hlg", "--workers", "4",
        ])
        assert args.tone == "bt2390"
        assert args.codec == "libx265"
        assert args.ten_bit and args.full_range
        assert args.assume == "hlg" and args.workers == 4

    def test_classic_reinhard_with_white(self):
        args = build_parser().parse_args([
            "in.mkv", "out.mp4",
            "--tone", "reinhard_classic", "--white", "2.5",
            "--codec", "h264_nvenc",
        ])
        assert args.tone == "reinhard_classic"
        assert args.reinhard_white == 2.5
        assert args.codec == "h264_nvenc"

    def test_hable_with_sharpness(self):
        args = build_parser().parse_args([
            "in.mkv", "out.mp4", "--tone", "hable", "--sharpness", "12.5",
        ])
        assert args.tone == "hable"
        assert args.hable_sharpness == 12.5

    def test_default_sharpness(self):
        args = build_parser().parse_args(["in.mkv", "out.mp4"])
        from hdr_to_sdr.tonemap import DEFAULT_HABLE_SHARPNESS
        assert args.hable_sharpness == DEFAULT_HABLE_SHARPNESS

    def test_all_hardware_codecs_accepted(self):
        for hw in (
            "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf",
        ):
            args = build_parser().parse_args(["in.mkv", "o.mp4", "--codec", hw])
            assert args.codec == hw

    def test_bad_tone_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["in.mkv", "o.mp4", "--tone", "nope"])

    def test_bad_codec_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["in.mkv", "o.mp4", "--codec", "nope"])


class TestCLIEndToEnd:
    def test_probe_only_pq(self, pq_file, capsys):
        rc = main([pq_file, "unused.mp4", "--probe-only"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PQ" in out
        assert "1000" in out  # 母版峰值

    def test_probe_only_hlg(self, hlg_file, capsys):
        rc = main([hlg_file, "unused.mp4", "--probe-only"])
        assert rc == 0
        assert "HLG" in capsys.readouterr().out

    def test_probe_sdr_rejected_exit_code(self, sdr_file, capsys):
        """不靠猜：SDR 输入 CLI 必须以非零码退出并提示。"""
        rc = main([sdr_file, "o.mp4"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "HDR" in err or "smpte2084" in err

    def test_conversion_cli(self, pq_file, work_dir, capsys):
        out = os.path.join(work_dir, "cli_out.mp4")
        rc = main([pq_file, out])
        assert rc == 0
        assert os.path.isfile(out)
        assert "转换完成" in capsys.readouterr().out

    def test_conversion_with_assume(self, sdr_file, work_dir):
        out = os.path.join(work_dir, "forced.mp4")
        rc = main([sdr_file, out, "--assume", "pq"])
        assert rc == 0
        assert os.path.isfile(out)

    def test_missing_file_exit_code(self, work_dir, capsys):
        rc = main(["no_such_12345.mkv", os.path.join(work_dir, "o.mp4")])
        assert rc == 2

    def test_10bit_hlg_cli(self, hlg_file, work_dir):
        out = os.path.join(work_dir, "hlg10.mp4")
        rc = main([hlg_file, out, "--10bit", "--codec", "libx265"])
        assert rc == 0
        assert os.path.isfile(out)

    def test_classic_reinhard_cli(self, pq_file, work_dir, capsys):
        out = os.path.join(work_dir, "classic.mp4")
        rc = main([
            pq_file, out,
            "--tone", "reinhard_classic", "--white", "2.0",
        ])
        assert rc == 0
        assert os.path.isfile(out)
        assert "转换完成" in capsys.readouterr().out

    def test_hable_cli(self, pq_file, work_dir, capsys):
        out = os.path.join(work_dir, "hable.mp4")
        rc = main([pq_file, out, "--tone", "hable", "--sharpness", "8.0"])
        assert rc == 0
        assert os.path.isfile(out)
        assert "转换完成" in capsys.readouterr().out

    def test_clamp_warning_surfaces(self, pq_file, work_dir, capsys):
        """越界白点钳制时用户应能看到提示，且转换不中断。"""
        out = os.path.join(work_dir, "clamp.mp4")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc = main([pq_file, out, "--tone", "reinhard_classic",
                       "--white", "99999"])
        assert rc == 0
        assert any("白点" in str(w.message) for w in caught)
