"""probe.py 测试：帧率有理数解析为纯逻辑，真实探针走 ffmpeg 夹具。"""

from __future__ import annotations

import pytest

from video_transition.errors import ProbeError
from video_transition.probe import parse_rate, probe_clip

from .conftest import ffmpeg_available, make_clip


class TestParseRate:
    @pytest.mark.parametrize(
        "literal,expected",
        [
            ("25/1", (25, 1)),
            ("30000/1001", (30000, 1001)),
            ("30", (30, 1)),
            ("50/2", (25, 1)),       # 自动约分
            ("24000/1001", (24000, 1001)),
        ],
    )
    def test_valid_rates(self, literal, expected):
        assert parse_rate(literal) == expected

    @pytest.mark.parametrize("literal", [None, "", "N/A", "0/0", "10/0", "-5/1"])
    def test_invalid_rates(self, literal):
        with pytest.raises(ProbeError):
            parse_rate(literal)


@pytest.mark.skipif(not ffmpeg_available(), reason="无 ffmpeg")
class TestProbeClip:
    def test_probe_basic(self, tmp_path):
        src = tmp_path / "clip.mp4"
        frames = make_clip(str(src), duration=2, rate=25, with_audio=True)
        meta = probe_clip(str(src))
        assert (meta.width, meta.height) == (320, 240)
        assert (meta.fps_num, meta.fps_den) == (25, 1)
        assert meta.frame_count == frames
        assert meta.has_audio is True

    def test_probe_clip_without_audio(self, tmp_path):
        src = tmp_path / "silent.mp4"
        make_clip(str(src), duration=1, with_audio=False)
        meta = probe_clip(str(src))
        assert meta.has_audio is False

    def test_missing_file(self):
        with pytest.raises(ProbeError):
            probe_clip("definitely-not-here.mp4")
