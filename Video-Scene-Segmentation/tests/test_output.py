"""output.py 确定性序列化的纯逻辑测试。"""

from __future__ import annotations

import json

from scene_seg_tool.detector import detect_scenes_in_frames
from scene_seg_tool.output import (
    FLOAT_DIGITS,
    format_float,
    render_boundaries_csv,
    render_json_report,
    result_to_dict,
)

WIDTH, HEIGHT, FPS = 4, 4, 25.0
PIXELS = WIDTH * HEIGHT


def _result():
    frames = [bytes([30] * PIXELS)] * 12 + [bytes([220] * PIXELS)] * 12
    return detect_scenes_in_frames(frames, WIDTH, HEIGHT, FPS, input_path="x.mp4")


def test_format_float_uses_fixed_digits_and_dot_decimal_point():
    text = format_float(1 / 3.0)
    assert text == f"%.{FLOAT_DIGITS}f" % (1 / 3.0)
    assert "." in text and "," not in text


def test_json_report_is_valid_json_with_fixed_key_order():
    report = render_json_report(_result())
    data = json.loads(report)
    assert data["input_path"] == "x.mp4"
    assert data["cut_count"] == 1
    # 顶层键顺序固定
    assert list(data.keys()) == [
        "input_path", "frame_count", "width", "height", "fps", "fps_source",
        "pixel_count", "threshold", "threshold_source", "flash_boundaries",
        "suppressed_boundaries", "cut_count", "cuts", "boundaries",
    ]


def test_json_report_contains_full_boundary_series():
    data = json.loads(render_json_report(_result()))
    assert len(data["boundaries"]) == 23
    assert list(data["boundaries"][0].keys()) == [
        "frame_index", "time_seconds", "raw_distance", "score",
        "return_distance", "decision",
    ]


def test_csv_has_header_and_one_row_per_boundary():
    csv_text = render_boundaries_csv(_result())
    rows = csv_text.split("\r\n")
    assert rows[0] == (
        "frame_index,time_seconds,raw_distance,score,return_distance,decision"
    )
    # 末尾 \r\n 后还有一个空串
    assert rows[-1] == ""
    assert len(rows) - 1 == 23 + 1  # 表头 + 23 条边界


def test_csv_single_frame_video_has_header_only():
    single = detect_scenes_in_frames(
        [bytes([10] * PIXELS)], WIDTH, HEIGHT, FPS
    )
    csv_text = render_boundaries_csv(single)
    rows = [r for r in csv_text.split("\r\n") if r]
    assert len(rows) == 1
    assert rows[0].startswith("frame_index,")


def test_serialization_is_deterministic():
    result = _result()
    payload = result_to_dict(result)
    assert render_json_report(result) == render_json_report(
        detect_scenes_in_frames(
            [bytes([30] * PIXELS)] * 12 + [bytes([220] * PIXELS)] * 12,
            WIDTH, HEIGHT, FPS, input_path="x.mp4",
        )
    )
    assert payload == result_to_dict(_result())
