"""命令行入口测试。"""

from __future__ import annotations

import pytest

from subburn.cli import _build_parser, main

from .conftest import make_test_video


class TestParser:
    def test_returns_root_parser_with_subcommand(self):
        parser = _build_parser()
        ns = parser.parse_args(
            ["burn", "v.mp4", "s.srt", "-o", "o.mp4", "--crf", "20"]
        )
        assert ns.command == "burn"
        assert ns.video == "v.mp4" and ns.srt == "s.srt"
        assert ns.output == "o.mp4" and ns.crf == 20

    def test_default_output_path_is_none_until_main(self):
        ns = _build_parser().parse_args(["burn", "v.mp4", "s.srt"])
        assert ns.output is None


@pytest.mark.usefixtures("ffmpeg")
class TestCliMain:
    def test_burn_via_main(self, tmp_path, monkeypatch):
        video = tmp_path / "v.mp4"
        srt = tmp_path / "s.srt"
        out = tmp_path / "o.mp4"
        make_test_video(video)
        srt.write_text("1\n00:00:00,100 --> 00:00:01,000\nHELLO\n",
                       encoding="utf-8")
        rc = main(["burn", str(video), str(srt), "-o", str(out), "-y"])
        assert rc == 0 and out.exists()

    def test_missing_cues_returns_error(self, tmp_path):
        video = tmp_path / "v.mp4"
        srt = tmp_path / "empty.srt"
        srt.write_text("\n\n  \n", encoding="utf-8")
        rc = main(["burn", str(video), str(srt), "-o", str(tmp_path / "o.mp4")])
        assert rc == 2
