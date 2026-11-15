"""CLI 的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import pytest

from compose_tool.cli import build_parser, main
from compose_tool.errors import (
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
)


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args(["l.mp4", "r.mp4"])
        assert args.left == "l.mp4"
        assert args.right == "r.mp4"
        assert args.duration_mode == "pad"
        assert args.layout == "auto"
        assert args.audio_mode == "auto"
        assert args.overwrite is False
        assert args.dry_run is False

    def test_all_options(self):
        args = build_parser().parse_args([
            "l.mp4", "r.mp4", "-o", "out.mp4",
            "--duration-mode", "trim", "--layout", "fit",
            "--height", "720", "--fps", "30000/1001",
            "--audio-mode", "left", "--crf", "18", "--preset", "slow",
            "--overwrite", "--progress", "--json",
        ])
        assert args.output == "out.mp4"
        assert args.duration_mode == "trim"
        assert args.layout == "fit"
        assert args.height == 720
        assert args.fps == "30000/1001"
        assert args.audio_mode == "left"
        assert args.crf == 18
        assert args.preset == "slow"
        assert args.overwrite and args.progress and args.json


class TestErrorExitCodes:
    def test_missing_input_exits_3(self, tmp_path, capsys):
        code = main([str(tmp_path / "nope.mp4"), str(tmp_path / "nope2.mp4")])
        assert code == InputNotFoundError.exit_code
        assert "不存在" in capsys.readouterr().err

    def test_existing_output_exits_5(self, tmp_path, monkeypatch, capsys):
        left = tmp_path / "l.mp4"
        right = tmp_path / "r.mp4"
        left.write_bytes(b"x")
        right.write_bytes(b"x")
        out = tmp_path / "out.mp4"
        out.write_bytes(b"exists")
        code = main([str(left), str(right), "-o", str(out)])
        assert code == OutputExistsError.exit_code
        assert "已存在" in capsys.readouterr().err

    def test_invalid_param_exits_4(self, tmp_path, monkeypatch, capsys):
        """非法规划参数在调用 ffmpeg 之前就被拦截。"""
        left = tmp_path / "l.mp4"
        right = tmp_path / "r.mp4"
        left.write_bytes(b"x")
        right.write_bytes(b"x")
        # 跳过真实 ffprobe，直接喂探测结果
        from conftest import make_media
        monkeypatch.setattr(
            "compose_tool.cli.probe_media", lambda p: make_media()
        )
        code = main([str(left), str(right), "--fps", "bogus", "--dry-run"])
        assert code == InvalidInputError.exit_code
