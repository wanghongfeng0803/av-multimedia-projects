"""ASS/SSA 解析层测试：颜色/时间码、样式表、事件标签状态机与容错。

纯逻辑用例全部不依赖 ffmpeg 与系统字体，任何环境都能跑；ASS 文档由
:func:`make_ass` 用标准 V4+ 列模板拼装，避免每个用例重复整段 Format。
"""

from __future__ import annotations

import pytest

from subburn.ass import (
    ASSColor,
    ASSParseError,
    ASSStyle,
    parse_ass,
    parse_ass_text,
    parse_ass_time,
)

#: 标准 V4+ 样式/事件列，测试只改 Style 行与 Dialogue 的 Text
_STYLE_FMT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FMT = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
    "MarginV, Effect, Text"
)


def make_ass(styles: str = "", dialogues: list[str] | None = None,
             *, play_x: int = 0, play_y: int = 0) -> str:
    """拼装最小可解析 ASS 文本。styles 为若干 Style 行，dialogues 为 Text。"""
    lines = ["[Script Info]"]
    if play_x:
        lines.append(f"PlayResX: {play_x}")
    if play_y:
        lines.append(f"PlayResY: {play_y}")
    lines += ["[V4+ Styles]", _STYLE_FMT]
    lines.append(styles or (
        "Style: D,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"))
    lines += ["[Events]", _EVENT_FMT]
    for text in (dialogues or []):
        lines.append(
            f"Dialogue: 0,0:00:00.00,0:00:02.00,D,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


class TestTimecode:
    def test_basic_centiseconds(self):
        assert parse_ass_time("0:00:01.50") == 1500
        assert parse_ass_time("1:02:03.25") == 3723250

    def test_one_digit_fraction_right_padded(self):
        # 前导/尾随零有位值意义：.5 -> 500ms，.05 -> 50ms
        assert parse_ass_time("0:00:00.5") == 500
        assert parse_ass_time("0:00:00.05") == 50

    def test_three_digit_fraction_treated_as_ms(self):
        assert parse_ass_time("0:00:00.025") == 25

    def test_out_of_range_raises(self):
        with pytest.raises(ASSParseError):
            parse_ass_time("0:00:60.00")
        with pytest.raises(ASSParseError):
            parse_ass_time("garbage")


class TestColor:
    def test_opaque_and_transparent(self):
        assert ASSColor.parse("&H00FFFFFF").rgba == (255, 255, 255, 255)
        assert ASSColor.parse("&HFF000000").rgba == (0, 0, 0, 0)

    def test_bgr_channel_swap(self):
        # BGR 整数布局：RR=FF 是红，BB=FF 是蓝
        assert ASSColor.parse("&H000000FF&").rgba == (255, 0, 0, 255)
        assert ASSColor.parse("&H00FF0000&").rgba == (0, 0, 255, 255)
        assert ASSColor.parse("&H0000FF00&").rgba == (0, 255, 0, 255)

    def test_short_hex_right_aligned(self):
        # 短色值按整数右对齐到 6 位 BGR：&HFF->0000FF(红)、&HFF00->00FF00(绿)
        assert ASSColor.parse("&HFF&").rgba == (255, 0, 0, 255)
        assert ASSColor.parse("&HFF00&").rgba == (0, 255, 0, 255)
        assert ASSColor.parse("&HFF0000&").rgba == (0, 0, 255, 255)

    def test_partial_alpha_inverted(self):
        # ASS alpha 是透明度：&H80 = 128 透明 -> 127 不透明
        assert ASSColor.parse("&H80000000").rgba == (0, 0, 0, 127)

    def test_invalid_raises(self):
        with pytest.raises(ASSParseError):
            ASSColor.parse("red")
        with pytest.raises(ASSParseError):
            ASSColor.parse("&HZZZZZZ")


class TestStyles:
    def test_v4_plus_style_fields(self):
        style = (
            "Style: D,微软雅黑,28,&H0000FFFF,&H000000FF,&H00000000,"
            "&H00000000,-1,-1,1,1,100,100,0,0,3,1,0,8,20,20,20,1")
        doc, _ = parse_ass_text(make_ass(styles=style, dialogues=["x"]))
        st = doc.styles["D"]
        assert st.font_name == "微软雅黑" and st.font_size == 28
        assert st.bold and st.italic and st.underline and st.strikeout
        assert st.border_style == 3 and st.alignment == 8
        assert (st.margin_l, st.margin_v) == (20, 20)
        assert st.primary_color.rgb == (255, 255, 0)

    def test_bad_number_keeps_default(self):
        # 坏数值不应让整条样式解析失败（样式名保持 D 以匹配对话引用）
        style = (
            "Style: D,Arial,NaN,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1")
        doc, warns = parse_ass_text(make_ass(styles=style, dialogues=["x"]))
        assert doc.styles["D"].font_size == 20.0  # 回退默认
        assert not warns

    def test_ssa_v4_alignment_mapping(self):
        v4 = (
            "[V4 Styles]\n"
            f"Format: Name, Fontname, Fontsize, PrimaryColour, "
            "SecondaryColour, TertiaryColour, BackColour, Bold, Italic, "
            "BorderStyle, Outline, Shadow, AlphaLevel, Encoding, MarginL, "
            "MarginR, MarginV, Alignment\n"
            "Style: O,Arial,18,16777215,255,16777215,0,0,0,1,2,2,0,1,"
            "10,10,10,4\n"
            "[Events]\n"
            "Format: Marked, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
            "Dialogue: Marked=0,0:00:00.00,0:00:01.00,O,,0,0,0,,hi\n")
        doc, warns = parse_ass_text(v4)
        assert not warns
        assert doc.styles["O"].alignment == 8  # v4 顶部中 -> numpad 8


class TestEventsAndTags:
    def test_comment_skipped_and_index_assigned(self):
        text = make_ass(dialogues=["visible"])
        text += "Comment: 0,0:00:00.00,0:00:01.00,D,,0,0,0,,hidden\n"
        doc, _ = parse_ass_text(text)
        assert len(doc.cues) == 1
        assert doc.cues[0].index == 1
        assert doc.cues[0].lines[0].runs[0].text == "visible"

    def test_bold_italic_underline_strike(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\b1}B{\b0} {\i1}I{\r} {\u1}U {\s1}S"]))
        runs = doc.cues[0].lines[0].runs
        flags = [(r.text, r.bold, r.italic, r.underline, r.strikeout)
                 for r in runs]
        assert ("B", True, False, False, False) in flags
        assert ("I", False, True, False, False) in flags
        # \r 之后斜体复位
        assert all(not r.italic for r in runs if r.text.startswith("U"))
        assert any(r.underline for r in runs)
        assert any(r.strikeout for r in runs)

    def test_reset_with_named_style(self):
        styles = (
            "Style: D,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n"
            "Style: B,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1")
        doc, _ = parse_ass_text(
            make_ass(styles=styles, dialogues=[r"a{\b1}b{\rB}c"]))
        runs = doc.cues[0].lines[0].runs
        assert [(r.text, r.bold) for r in runs] == [
            ("a", False), ("b", True), ("c", True)]  # B 样式本身 bold

    def test_color_and_alpha_tags(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\c&H000000FF&}red {\alpha&H80&}half"]))
        runs = doc.cues[0].lines[0].runs
        assert runs[0].primary.rgb == (255, 0, 0)
        assert runs[1].primary.a == 127

    def test_font_name_and_size(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\fn微软雅黑\fs40}big"]))
        run = doc.cues[0].lines[0].runs[0]
        assert run.font_name == "微软雅黑" and run.font_size == 40

    def test_scale_and_spacing(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\fscx150\fscy80\fsp4}x"]))
        run = doc.cues[0].lines[0].runs[0]
        assert (run.scale_x, run.scale_y) == (150, 80)
        assert run.spacing == 4

    def test_alignment_and_position(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\an7\pos(12.5, 30)}x", r"{\a4}y"]))
        assert doc.cues[0].alignment == 7
        assert doc.cues[0].pos == (12.5, 30.0)
        assert doc.cues[1].alignment == 8  # 旧式 \a4 -> numpad 8

    def test_move_and_fade(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\move(1,2,3,4)\fad(100,200)}x",
            r"{\fade(255,0,255,10,20,80,90)}y"]))
        c0, c1 = doc.cues
        assert c0.move == (1.0, 2.0, 3.0, 4.0)
        assert c0.fade_in_ms == 100 and c0.fade_out_ms == 200
        assert c1.fade_in_ms == 10 and c1.fade_out_ms == 10

    def test_karaoke_syllables(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"{\k40}la {\kf60}la {\K50}la"]))
        kr = [r for r in doc.cues[0].lines[0].runs if r.k_dur_ms]
        assert [(r.k_dur_ms, r.k_fill, r.k_start_ms) for r in kr] == [
            (400, "k", 0), (600, "kf", 400), (500, "kf", 1000)]

    def test_hard_line_break(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[r"line1\Nline2"]))
        cue = doc.cues[0]
        assert len(cue.lines) == 2
        assert cue.lines[0].runs[0].text == "line1"
        assert cue.lines[1].runs[0].text == "line2"

    def test_nbsp_and_comment_block(self):
        # 无反斜杠的 {...} 是 ASS 注释，整块文本都不产出 run
        doc, _ = parse_ass_text(make_ass(dialogues=[r"a{this is note}c"]))
        text = "".join(r.text for r in doc.cues[0].lines[0].runs)
        assert "this is note" not in text
        assert text.replace(" ", "") == "ac"

        # \h 转义为不换行空格（U+00A0），仍是可见文本
        doc2, _ = parse_ass_text(make_ass(dialogues=[r"a\hb"]))
        text2 = "".join(r.text for r in doc2.cues[0].lines[0].runs)
        assert text2 == "a b"  # a 与 b 之间是 U+00A0 硬空格

    def test_unknown_tag_ignored(self):
        # 未实现标签安静跳过，不影响后续文本与标签
        doc, warns = parse_ass_text(make_ass(dialogues=[
            r"{\frz45\blur2\t(0,500,\fscx200)\b1}x"]))
        assert not warns
        assert doc.cues[0].lines[0].runs[0].bold is True

    def test_drawing_mode_text_dropped(self):
        doc, _ = parse_ass_text(make_ass(dialogues=[
            r"keep {\p1}m 0 0 l{\p0} shown"]))
        text = "".join(r.text for r in doc.cues[0].lines[0].runs)
        assert "keep" in text and "shown" in text
        assert "m 0 0" not in text


class TestDocumentAndRecovery:
    def test_play_res_and_wrap_style(self):
        text = make_ass(play_x=1280, play_y=720, dialogues=["x"])
        text = text.replace("[Script Info]",
                            "[Script Info]\nWrapStyle: 2", 1)
        doc, _ = parse_ass_text(text)
        assert doc.play_res_x == 1280 and doc.play_res_y == 720
        assert doc.wrap_style == 2

    def test_missing_style_falls_back(self):
        text = make_ass(dialogues=[])
        text += ("Dialogue: 0,0:00:00.00,0:00:02.00,Ghost,,0,0,0,,x\n")
        doc, warns = parse_ass_text(text)
        assert len(doc.cues) == 1
        assert any("未定义样式" in w for w in warns)
        assert isinstance(doc.style_of(doc.cues[0]), ASSStyle)

    def test_bad_timecode_event_dropped(self):
        text = make_ass(dialogues=[])
        text += ("Dialogue: 0,bad,0:00:02.00,D,,0,0,0,,x\n"
                 "Dialogue: 0,0:00:00.00,0:00:02.00,D,,0,0,0,,y\n")
        doc, warns = parse_ass_text(text)
        assert len(doc.cues) == 1 and doc.cues[0].lines[0].runs[0].text == "y"
        assert any("时间码" in w for w in warns)

    def test_strict_mode_raises(self):
        text = make_ass(dialogues=[])
        text += "Dialogue: 0,bad,0:00:02.00,D,,0,0,0,,x\n"
        with pytest.raises(ASSParseError):
            parse_ass_text(text, strict=True)

    def test_empty_events_returns_no_cues(self, tmp_path):
        p = tmp_path / "empty.ass"
        p.write_text("[Script Info]\n[Events]\n" + _EVENT_FMT + "\n",
                     encoding="utf-8")
        doc, _ = parse_ass(str(p))
        assert doc.cues == ()

    def test_utf8_bom_decoding(self, tmp_path):
        p = tmp_path / "bom.ass"
        p.write_bytes("﻿".encode("utf-8") + make_ass(
            dialogues=["中文"]).encode("utf-8"))
        doc, _ = parse_ass(str(p))
        assert doc.cues[0].lines[0].runs[0].text == "中文"
