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

    def test_transition_defaults_to_fade(self):
        args = build_parser().parse_args(
            ["a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5"]
        )
        assert args.transition == "fade"
        assert args.direction == "right"
        assert args.easing == "linear"

    def test_slide_args_parsed(self):
        args = build_parser().parse_args([
            "a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5",
            "--transition", "slide",
            "--direction", "up",
            "--easing", "ease_in_out",
        ])
        assert args.transition == "slide"
        assert args.direction == "up"
        assert args.easing == "ease_in_out"

    def test_bad_direction_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5",
                "--transition", "slide", "--direction", "north",
            ])

    def test_bad_easing_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5",
                "--easing", "bounce",
            ])

    def test_soft_width_defaults_to_zero(self):
        args = build_parser().parse_args(
            ["a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5"]
        )
        assert args.soft_width == 0

    def test_wipe_args_parsed(self):
        args = build_parser().parse_args([
            "a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5",
            "--transition", "wipe",
            "--direction", "down",
            "--soft-width", "12",
            "--easing", "ease_out",
        ])
        assert args.transition == "wipe"
        assert args.direction == "down"
        assert args.soft_width == 12
        assert args.easing == "ease_out"

    def test_bad_transition_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "a.mp4", "b.mp4", "-o", "o.mp4", "--frames", "5",
                "--transition", "dissolve",
            ])


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
