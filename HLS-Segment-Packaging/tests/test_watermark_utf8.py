"""自研 UTF-8 解码器测试：合法多字节序列与全部异常输入类别。"""

from __future__ import annotations

import pytest

from hls_packager.watermark.utf8 import decode_utf8


def test_empty_input():
    assert decode_utf8(b"") == []


def test_ascii_single_byte():
    assert decode_utf8(b"Hello!") == [72, 101, 108, 108, 111, 33]


def test_two_byte_sequences():
    assert decode_utf8("é".encode("utf-8")) == [0xE9]


def test_three_byte_cjk():
    assert decode_utf8("水".encode("utf-8")) == [0x6C34]
    assert decode_utf8("水印".encode("utf-8")) == [0x6C34, 0x5370]


def test_four_byte_emoji():
    assert decode_utf8("😀".encode("utf-8")) == [0x1F600]


def test_mixed_width_text():
    assert decode_utf8("A 水!😀".encode("utf-8")) == [
        ord("A"), ord(" "), ord("水"), ord("!"), 0x1F600
    ]


def test_agrees_with_python_decoder_on_valid_input():
    samples = [
        "plain ASCII 123",
        "café résumé naïve",
        "中文水印测试",
        "日本語テスト",
        "한국어 시험",
        "😀🎉混合 emoji 🚀",
        "tab\tnewline\nreturn\r",
    ]
    for sample in samples:
        assert decode_utf8(sample.encode("utf-8")) == [ord(c) for c in sample]


def test_truncated_two_byte_sequence():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xC3")


def test_truncated_three_byte_sequence():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xE6\xB0")


def test_truncated_four_byte_sequence():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xF0\x9F\x98")


def test_continuation_byte_at_start_is_rejected():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\x80")
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xBF")


def test_missing_continuation_byte():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xC3A")
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xE6\xB0A")


def test_overlong_encodings_are_rejected():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xC0\xAF")
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xE0\x80\x80")
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xF0\x80\x80\xBF")


def test_surrogate_codepoints_are_rejected():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xED\xA0\x80")
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xED\xBF\xBF")


def test_codepoint_above_unicode_range_rejected():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xF4\x90\x80\x80")


def test_illegal_leading_byte_0xFF():
    with pytest.raises(UnicodeDecodeError):
        decode_utf8(b"\xFF")


def test_bytearray_input_works():
    assert decode_utf8(bytearray("水".encode("utf-8"))) == [0x6C34]


def test_non_bytes_input_raises_type_error():
    with pytest.raises(TypeError):
        decode_utf8("not bytes")  # type: ignore[arg-type]
