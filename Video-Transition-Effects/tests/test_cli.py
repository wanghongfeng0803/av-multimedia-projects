"""cli.py 测试：参数解析与错误退出码（不依赖 ffmpeg）。"""

from __future__ import annotations

import pytest

from video_transition.cli import build_parser, main


class TestParser:
    def test_overlap_seconds(self):
        args = build_parser().parse_args(
            ["a.mp4", "b.mp4", "-o", "out.mp4", "--overlap", "0.5"]
        )
        assert args.overlap == 0.5
        assert args.frames is None

    def test_overlap_frames(self):
        args = build_parser().parse_args(
            ["a.mp4", "b.mp4", "-o", "out.mp4", "--frames", "12"]
        )
        assert args.frames == 12
        assert args.overlap is None

    def test_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["a.mp4", "b.mp4", "-o", "o.mp4", "--overlap", "0.5", "--frames", "5"]
            )

    def test_one_overlap_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["a.mp4", "b.mp4", "-o", "o.mp4"])


class TestMainErrors:
    def test_missing_input_returns_code_2(self, tmp_path):
        out = tmp_path / "o.mp4"
        code = main([
            str(tmp_path / "nope-a.mp4"), str(tmp_path / "nope-b.mp4"),
            "-o", str(out), "--frames", "5",
        ])
        assert code == 2

    def test_existing_output_without_overwrite(self, tmp_path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        out = tmp_path / "o.mp4"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        out.write_bytes(b"x")
        code = main([str(a), str(b), "-o", str(out), "--frames", "5"])
        assert code == 2
