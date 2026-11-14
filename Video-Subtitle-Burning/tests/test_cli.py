"""命令行入口测试。"""

from __future__ import annotations

import json

import pytest

from subburn.cli import _build_parser, _build_style, main
from subburn.render import Style as LegacyStyle
from subburn.style import (
    FontWeight,
    HAlign,
    Margins,
    SubtitleStyle,
    VAlign,
)

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


class TestStyleArgs:
    def test_no_style_args_keeps_legacy_style(self):
        ns = _build_parser().parse_args(["burn", "v.mp4", "s.srt"])
        style = _build_style(ns)
        assert isinstance(style, LegacyStyle)
        assert style.box is True  # 默认底盒的旧行为

    def test_legacy_box_flags(self):
        ns = _build_parser().parse_args(
            ["burn", "v.mp4", "s.srt", "--no-box"])
        assert _build_style(ns).box is False

    def test_individual_style_args_build_subtitle_style(self):
        ns = _build_parser().parse_args([
            "burn", "v.mp4", "s.srt",
            "--font-family", "Arial", "--font-family", "微软雅黑",
            "--font-weight", "bold", "--font-size", "30",
            "--color", "#ff8800", "--alpha", "200",
            "--halign", "left", "--valign", "top",
            "--margin", "24", "--no-box",
        ])
        st = _build_style(ns)
        assert isinstance(st, SubtitleStyle)
        assert st.font.families == ("Arial", "微软雅黑")
        assert st.font.weight is FontWeight.BOLD
        assert st.font.size == 30
        assert st.color.rgba == (255, 136, 0, 200)
        assert st.position.halign is HAlign.LEFT
        assert st.position.valign is VAlign.TOP
        assert st.position.margins == Margins.uniform(24)
        assert st.box is False

    def test_numeric_weight_and_hsl_color(self):
        ns = _build_parser().parse_args([
            "burn", "v", "s", "--font-weight", "700",
            "--color", "hsl(240, 100%, 50%)"])
        st = _build_style(ns)
        assert st.font.weight is FontWeight.BOLD
        assert st.color.rgb == (0, 0, 255)

    def test_out_of_bounds_alpha_is_clamped(self):
        ns = _build_parser().parse_args(
            ["burn", "v", "s", "--alpha", "999"])
        assert _build_style(ns).color.a == 255

    def test_font_file_heads_family_chain(self):
        ns = _build_parser().parse_args([
            "burn", "v", "s", "--font", "C:/fonts/x.ttf",
            "--font-family", "Arial"])
        st = _build_style(ns)
        assert st.font.families[0] == "C:/fonts/x.ttf"
        assert st.font.families[1] == "Arial"

    def test_style_json_merged_with_overrides(self, tmp_path):
        cfg = tmp_path / "style.json"
        cfg.write_text(json.dumps({
            "color": "#00ff00",
            "position": {"halign": "right", "valign": "bottom"},
        }), encoding="utf-8")
        ns = _build_parser().parse_args([
            "burn", "v", "s", "--style", str(cfg), "--alpha", "128"])
        st = _build_style(ns)
        assert st.color.rgb == (0, 255, 0) and st.color.a == 128
        assert st.position.halign is HAlign.RIGHT
        assert st.position.valign is VAlign.BOTTOM

    def test_invalid_color_returns_error_path(self):
        ns = _build_parser().parse_args(
            ["burn", "v", "s", "--color", "garbage"])
        with pytest.raises(ValueError):
            _build_style(ns)

    def test_broken_style_json_raises(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("{not json", encoding="utf-8")
        ns = _build_parser().parse_args(
            ["burn", "v", "s", "--style", str(cfg)])
        with pytest.raises(json.JSONDecodeError):
            _build_style(ns)


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

    def test_burn_via_main_with_new_style(self, tmp_path):
        video = tmp_path / "v.mp4"
        srt = tmp_path / "s.srt"
        out = tmp_path / "o.mp4"
        make_test_video(video)
        srt.write_text("1\n00:00:00,100 --> 00:00:01,000\nHELLO\n",
                       encoding="utf-8")
        rc = main(["burn", str(video), str(srt), "-o", str(out),
                   "--halign", "left", "--color", "#ff0000",
                   "--font-weight", "bold", "-y"])
        assert rc == 0 and out.exists()

    def test_missing_cues_returns_error(self, tmp_path):
        video = tmp_path / "v.mp4"
        srt = tmp_path / "empty.srt"
        srt.write_text("\n\n  \n", encoding="utf-8")
        rc = main(["burn", str(video), str(srt), "-o", str(tmp_path / "o.mp4")])
        assert rc == 2

    def test_bad_style_returns_rc_2(self, tmp_path):
        video = tmp_path / "v.mp4"
        srt = tmp_path / "s.srt"
        make_test_video(video)
        srt.write_text("1\n00:00:00,100 --> 00:00:01,000\nX\n",
                       encoding="utf-8")
        rc = main(["burn", str(video), str(srt),
                   "-o", str(tmp_path / "o.mp4"), "--color", "nope"])
        assert rc == 2
