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


_ASS_TEMPLATE = """[Script Info]
PlayResX: 320
PlayResY: 240
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.10,0:00:01.00,Default,,0,0,0,,ASS {\\b1}bold{\\r} text
"""


class TestAssCli:
    def _write_ass(self, tmp_path, name="s.ass"):
        p = tmp_path / name
        p.write_text(_ASS_TEMPLATE, encoding="utf-8")
        return p

    def test_ass_extension_routes_to_ass(self, tmp_path):
        from subburn.cli import _is_ass_file
        assert _is_ass_file(str(self._write_ass(tmp_path))) is True
        assert _is_ass_file(str(tmp_path / "x.srt")) is False

    def test_content_detection_with_other_extension(self, tmp_path):
        from subburn.cli import _is_ass_file
        p = self._write_ass(tmp_path, "renamed.txt")
        assert _is_ass_file(str(p)) is True

    @pytest.mark.parametrize("extra", [
        ["--color", "red"], ["--font-size", "30"], ["--box"],
        ["--halign", "left"], ["--margin", "12"], ["--font-weight", "bold"],
    ])
    def test_srt_only_args_rejected_for_ass(self, tmp_path, extra):
        video = tmp_path / "v.mp4"
        ass = self._write_ass(tmp_path)
        rc = main(["burn", str(video), str(ass), "-o",
                   str(tmp_path / "o.mp4"), *extra])
        assert rc == 2

    def test_crf_and_preset_allowed_for_ass(self, tmp_path):
        # 通用编码参数不应被误判为 SRT 专属
        video = tmp_path / "v.mp4"
        make_test_video(video)
        ass = self._write_ass(tmp_path)
        out = tmp_path / "o.mp4"
        rc = main(["burn", str(video), str(ass), "-o", str(out),
                   "--crf", "20", "--preset", "fast", "-y", "--json"])
        assert rc == 0 and out.exists()

    @pytest.mark.usefixtures("ffmpeg")
    def test_burn_ass_via_main(self, tmp_path):
        video = tmp_path / "v.mp4"
        ass = self._write_ass(tmp_path)
        out = tmp_path / "o.mp4"
        make_test_video(video)
        rc = main(["burn", str(video), str(ass), "-o", str(out), "-y"])
        assert rc == 0 and out.exists()

    def test_ass_empty_dialogues_returns_rc_2(self, tmp_path):
        video = tmp_path / "v.mp4"
        ass = tmp_path / "empty.ass"
        ass.write_text("[Script Info]\n[Events]\n", encoding="utf-8")
        rc = main(["burn", str(video), str(ass),
                   "-o", str(tmp_path / "o.mp4")])
        assert rc == 2

    def test_ass_strict_raises_rc_2(self, tmp_path):
        video = tmp_path / "v.mp4"
        ass = tmp_path / "bad.ass"
        ass.write_text(
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
            "Dialogue: 0,bad,0:00:02.00,D,,0,0,0,,x\n",
            encoding="utf-8")
        rc = main(["burn", str(video), str(ass),
                   "-o", str(tmp_path / "o.mp4"), "--strict"])
        assert rc == 2


class TestTransformCli:
    """shift / retime 子命令端到端测试（纯文件操作，不依赖 ffmpeg）。"""

    def _write_srt(self, tmp_path, name="in.srt"):
        p = tmp_path / name
        p.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nworld\n",
            encoding="utf-8",
        )
        return p

    def test_shift_writes_offset_srt(self, tmp_path):
        src = self._write_srt(tmp_path)
        out = tmp_path / "o.srt"
        rc = main(["shift", str(src), "-o", str(out),
                   "--offset", "1500", "-y", "--json"])
        assert rc == 0 and out.exists()
        text = out.read_text(encoding="utf-8")
        assert "00:00:02,500 --> 00:00:03,500" in text
        # 序号连续
        assert text.startswith("1\n") and "\n2\n" in text

    def test_negative_shift_clips_to_lower_bound(self, tmp_path):
        src = self._write_srt(tmp_path)
        out = tmp_path / "o.srt"
        rc = main(["shift", str(src), "-o", str(out),
                   "--offset", "-1500", "-y"])
        assert rc == 0
        # 第一条 [1000,2000)-1500 -> 裁为 [0,500)；第二条 [3000,4000) -> [1500,2500)
        text = out.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:00,500" in text
        assert "00:00:01,500 --> 00:00:02,500" in text

    def test_shift_default_output_name(self, tmp_path):
        src = self._write_srt(tmp_path)
        rc = main(["shift", str(src), "--offset", "100"])
        assert rc == 0
        assert (tmp_path / "in_shifted.srt").exists()

    def test_shift_refuses_overwrite_without_flag(self, tmp_path):
        src = self._write_srt(tmp_path)
        out = tmp_path / "o.srt"
        out.write_text("x", encoding="utf-8")
        rc = main(["shift", str(src), "-o", str(out), "--offset", "100"])
        assert rc == 1

    def test_shift_all_dropped_writes_empty_srt(self, tmp_path):
        src = self._write_srt(tmp_path)
        out = tmp_path / "o.srt"
        rc = main(["shift", str(src), "-o", str(out),
                   "--offset", "-99999", "-y"])
        assert rc == 0
        assert out.read_text(encoding="utf-8") == ""

    def test_retime_speed_curve(self, tmp_path):
        src = self._write_srt(tmp_path)
        out = tmp_path / "o.srt"
        curve = json.dumps([
            {"t": 0, "speed": 1.0},
            {"t": 5000, "speed": 1.0},
        ])
        rc = main(["retime", str(src), "-o", str(out),
                   "--curve", curve, "-y"])
        assert rc == 0
        text = out.read_text(encoding="utf-8")
        # 单位斜率恒等映射，时间码不变
        assert "00:00:01,000 --> 00:00:02,000" in text

    def test_retime_bad_curve_json_rc_2(self, tmp_path):
        src = self._write_srt(tmp_path)
        rc = main(["retime", str(src), "-o", str(tmp_path / "o.srt"),
                   "--curve", "{not json", "-y"])
        assert rc == 2

    def test_retime_unknown_easing_rc_2(self, tmp_path):
        src = self._write_srt(tmp_path)
        curve = json.dumps([
            {"t": 0, "speed": 1, "easing": "nope"},
            {"t": 1000, "speed": 1},
        ])
        rc = main(["retime", str(src), "-o", str(tmp_path / "o.srt"),
                   "--curve", curve, "-y"])
        assert rc == 2

    def test_retime_eased_curve_runs_and_is_nonoverlapping(self, tmp_path):
        src = self._write_srt(tmp_path)
        out = tmp_path / "o.srt"
        curve = json.dumps([
            {"t": 0, "speed": 0.2, "easing": "ease_in_out"},
            {"t": 3000, "speed": 3.0, "easing": "smoothstep"},
            {"t": 6000, "speed": 0.0},
            {"t": 9000, "speed": 1.0},
        ])
        rc = main(["retime", str(src), "-o", str(out),
                   "--curve", curve, "-y", "--json"])
        assert rc == 0 and out.exists()
        # 重新解析产物，钉住有序、正宽度不变量
        from subburn.srt import parse_srt
        cues, _ = parse_srt(out.read_text(encoding="utf-8"))
        for c in cues:
            assert c.end_ms > c.start_ms
        for prev, nxt in zip(cues, cues[1:]):
            assert prev.end_ms <= nxt.start_ms
