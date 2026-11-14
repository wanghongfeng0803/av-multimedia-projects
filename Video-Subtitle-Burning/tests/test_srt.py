"""时间码换算与 SRT 状态机解析的单元测试。"""

from __future__ import annotations

import pytest

from subburn.srt import (
    SRTParseError,
    decode_srt,
    parse_srt,
    parse_timecode,
)


# --------------------------------------------------------------------------
# parse_timecode
# --------------------------------------------------------------------------

class TestTimecode:
    def test_basic_comma(self):
        assert parse_timecode("00:00:01,500") == 1500

    def test_dot_separator_equivalent(self):
        assert parse_timecode("00:00:01.500") == 1500

    def test_full_range(self):
        # 1:02:03.004 = 3723004ms
        assert parse_timecode("1:02:03,004") == 3723004

    def test_missing_hours(self):
        assert parse_timecode("01:02,500") == 62500

    def test_fraction_padding(self):
        # 小数位不足 3 位时右侧补零
        assert parse_timecode("00:00:01.5") == 1500
        assert parse_timecode("00:00:01.05") == 1050
        assert parse_timecode("00:00:01.005") == 1005

    def test_whitespace_tolerated(self):
        assert parse_timecode("  00:00:01,500  ") == 1500

    @pytest.mark.parametrize("bad", [
        "", "garbage", "00:00:01", "1,500",
        "00:60:00,000",   # 分钟越界
        "00:00:60,000",   # 秒越界
        "00:00:01,1234",  # 小数位过多
    ])
    def test_reject_bad(self, bad):
        with pytest.raises(SRTParseError):
            parse_timecode(bad)


# --------------------------------------------------------------------------
# decode_srt：编码与换行
# --------------------------------------------------------------------------

class TestDecode:
    @pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig"])
    def test_utf8_variants(self, enc):
        raw = "1\n00:00:01,000 --> 00:00:02,000\n你好\n".encode(enc)
        text = decode_srt(raw)
        assert "你好" in text and not text.startswith("﻿")

    def test_utf16_le_bom(self):
        base = "1\n00:00:01,000 --> 00:00:02,000\nhello\n"
        text = decode_srt(b"\xff\xfe" + base.encode("utf-16-le"))
        assert "hello" in text and "\r" not in text

    def test_utf16_be_bom(self):
        base = "1\n00:00:01,000 --> 00:00:02,000\nhello\n"
        text = decode_srt(b"\xfe\xff" + base.encode("utf-16-be"))
        assert "hello" in text

    def test_gb18030(self):
        base = "1\n00:00:01,000 --> 00:00:02,000\n中文测试\n"
        text = decode_srt(base.encode("gb18030"))
        assert "中文测试" in text

    def test_newline_normalization(self):
        text = decode_srt(b"a\r\nb\rc\nd")
        assert text == "a\nb\nc\nd"

    def test_undecodable_never_raises(self):
        # 任意字节序列都不应让解码抛异常（latin-1 兜底）
        text = decode_srt(bytes(range(256)))
        assert len(text) == 256


# --------------------------------------------------------------------------
# parse_srt：状态机
# --------------------------------------------------------------------------

class TestParse:
    def test_canonical(self):
        srt = "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n" \
              "2\n00:00:02,000 --> 00:00:03,000\nworld\n"
        cues, warns = parse_srt(srt)
        assert len(cues) == 2 and not warns
        assert cues[0].index == 1 and cues[0].raw_index == "1"
        assert cues[0].start_ms == 1000 and cues[0].end_ms == 2000
        assert cues[0].text == "hello"
        # 解析器分配的 index 连续，不依赖文件序号
        assert cues[1].index == 2

    def test_multiline_text(self):
        cues, _ = parse_srt("1\n00:00:01,000 --> 00:00:02,000\na\nb\nc\n")
        assert cues[0].text == "a\nb\nc"

    def test_missing_index(self):
        # 序号缺失也能开块
        cues, warns = parse_srt("00:00:01,000 --> 00:00:02,000\nno index\n")
        assert len(cues) == 1 and cues[0].text == "no index"
        assert cues[0].raw_index == ""

    def test_extra_blank_lines(self):
        srt = "\n\n1\n\n00:00:01,000 --> 00:00:02,000\nx\n\n\n\n"
        cues, _ = parse_srt(srt)
        assert len(cues) == 1 and cues[0].text == "x"

    def test_crlf_and_cr(self):
        cues, _ = parse_srt(
            "1\r00:00:01,000 --> 00:00:02,000\rCR only\r"
        )
        assert len(cues) == 1 and cues[0].text == "CR only"

    def test_glued_blocks_without_blank_line(self):
        # 块间缺空行，时间码行直接跟上：应拆成两条并告警
        srt = "1\n00:00:01,000 --> 00:00:02,000\nfirst\n" \
              "2\n00:00:02,000 --> 00:00:03,000\nsecond\n"
        cues, warns = parse_srt(srt)
        assert [c.text for c in cues] == ["first", "second"]
        assert warns  # 有诊断但不抛异常

    def test_position_info_stripped(self):
        cues, _ = parse_srt(
            "1\n00:00:01,000 --> 00:00:02,000 X1:10 X2:20 Y1:30 Y2:40\nx\n"
        )
        assert cues[0].start_ms == 1000 and cues[0].end_ms == 2000

    def test_reversed_interval_dropped(self):
        srt = "1\n00:00:03,000 --> 00:00:02,000\nbackwards\n"
        cues, warns = parse_srt(srt)
        assert cues == [] and any("结束时间早于开始" in w for w in warns)

    def test_zero_length_cue_warns(self):
        cues, warns = parse_srt("1\n00:00:05,000 --> 00:00:05,000\nx\n")
        assert len(cues) == 1 and any("起止时间相同" in w for w in warns)

    def test_garbage_line_skipped(self):
        cues, warns = parse_srt("not a valid line at all\n\n"
                                "1\n00:00:01,000 --> 00:00:02,000\nok\n")
        assert len(cues) == 1 and cues[0].text == "ok" and warns

    def test_strict_mode_raises(self):
        with pytest.raises(SRTParseError):
            parse_srt("garbage\n", strict=True)

    def test_empty_input(self):
        assert parse_srt("") == ([], [])
        assert parse_srt("\n\n  \n") == ([], [])

    def test_unsorted_input_is_sorted_later(self):
        # 解析保持文件顺序，排序职责在调度器；这里只确认全部解析成功
        srt = "2\n00:00:05,000 --> 00:00:06,000\nlate\n\n" \
              "1\n00:00:01,000 --> 00:00:02,000\nearly\n"
        cues, _ = parse_srt(srt)
        assert [c.text for c in cues] == ["late", "early"]
