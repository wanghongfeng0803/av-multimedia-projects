"""probe.py 中帧率解析等纯函数的测试（不启动 ffprobe）。"""

from __future__ import annotations

import pytest

from scene_seg_tool.probe import parse_frame_rate, DEFAULT_FPS


@pytest.mark.parametrize(
    "text,expected",
    [
        ("25/1", 25.0),
        ("30000/1001", pytest.approx(30000 / 1001)),
        ("24000/1001", pytest.approx(24000 / 1001)),
        ("23.976", 23.976),
        ("0/0", 0.0),
        ("", 0.0),
        (None, 0.0),
        ("1/0", 0.0),
        ("garbage", 0.0),
    ],
)
def test_parse_frame_rate(text, expected):
    assert parse_frame_rate(text) == expected


def test_default_fps_is_positive():
    assert DEFAULT_FPS > 0
