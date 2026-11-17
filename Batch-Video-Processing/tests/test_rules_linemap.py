"""JSON 行号定位器与重复键检测测试（纯文本，不依赖文件系统/ffmpeg）。"""

from __future__ import annotations

from bvp.rules.linemap import build_line_map


def test_maps_object_keys_to_starting_lines():
    text = "{\n  \"version\": 1,\n  \"rules\": [\n    {\n      \"name\": \"r1\"\n    }\n  ]\n}"
    line_map = build_line_map(text)
    assert line_map.line_of("version") == 2
    assert line_map.line_of("rules") == 3
    assert line_map.line_of("rules", 0, "name") == 5


def test_maps_scalar_values_and_nested_params():
    text = (
        "{\n"
        "  \"rules\": [\n"
        "    {\n"
        "      \"name\": \"r1\",\n"
        "      \"params\": {\n"
        "        \"codec\": \"h265\",\n"
        "        \"crf\": 28\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    line_map = build_line_map(text)
    assert line_map.line_of("rules", 0, "params", "codec") == 6
    assert line_map.line_of("rules", 0, "params", "crf") == 7
    # 标量值本身也有路径行
    assert line_map.line_of("version") is None  # 本例无 version 键


def test_detects_duplicate_keys_with_second_line():
    text = (
        "{\n"
        "  \"rules\": [],\n"
        "  \"rules\": []\n"
        "}"
    )
    line_map = build_line_map(text)
    assert len(line_map.duplicates) == 1
    dup = line_map.duplicates[0]
    assert dup.key == "rules"
    assert dup.line == 3


def test_escaped_strings_do_not_break_scanning():
    text = '{\n  "pattern": "a\\"quote",\n  "name": "x"\n}'
    line_map = build_line_map(text)
    assert line_map.line_of("pattern") == 2
    assert line_map.line_of("name") == 3
    assert line_map.duplicates == ()


def test_array_indexes_advance_on_commas_not_on_closing_brackets():
    text = '{\n  "ext": [\n    ".mp4",\n    ".mov"\n  ]\n}'
    line_map = build_line_map(text)
    assert line_map.line_of("ext", 0) == 3
    assert line_map.line_of("ext", 1) == 4
