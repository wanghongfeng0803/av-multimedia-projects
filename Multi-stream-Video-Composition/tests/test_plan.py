"""对齐规划的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

from fractions import Fraction

import pytest

from compose_tool.errors import InvalidInputError
from compose_tool.models import MediaInfo, StreamInfo
from compose_tool.plan import (
    AUDIO_LEFT,
    AUDIO_MIX,
    AUDIO_NONE,
    AUDIO_RIGHT,
    DEFAULT_FALLBACK_FPS,
    DURATION_PAD,
    DURATION_TRIM,
    LAYOUT_FIT,
    even_ceil,
    parse_fps,
    parse_sar,
    plan_composition,
    round_even,
    scaled_width_for_height,
)

from conftest import make_media


class TestParseFps:
    def test_rational(self):
        assert parse_fps("30000/1001") == Fraction(30000, 1001)

    def test_integer(self):
        assert parse_fps("25") == Fraction(25, 1)

    def test_float(self):
        assert parse_fps("29.97") == Fraction(2997, 100)

    def test_zero_and_garbage(self):
        assert parse_fps("0/0") is None
        assert parse_fps("abc") is None
        assert parse_fps("") is None
        assert parse_fps(None) is None
        assert parse_fps("-25") is None

    def test_passthrough_fraction_and_number(self):
        assert parse_fps(Fraction(24, 1)) == Fraction(24, 1)
        assert parse_fps(30) == Fraction(30, 1)
        assert parse_fps(0) is None


class TestParseSar:
    def test_normal(self):
        assert parse_sar("4:3") == pytest.approx(4 / 3)

    def test_square_and_invalid(self):
        assert parse_sar("1:1") == 1.0
        assert parse_sar(None) == 1.0
        assert parse_sar("0:0") == 1.0
        assert parse_sar("bad") == 1.0


class TestEvenRounding:
    def test_even_ceil(self):
        assert even_ceil(361) == 362
        assert even_ceil(360) == 360
        assert even_ceil(0.5) == 2

    def test_round_even(self):
        assert round_even(640.0) == 640
        assert round_even(641.0) == 640  # 奇数向下取偶
        assert round_even(1.2) == 2  # 不小于最小维度

    def test_scaled_width(self):
        # 1920x1080 缩到高 720 → 1280
        assert scaled_width_for_height(1920, 1080, 720) == 1280
        # 641x363 缩到高 360 → 约 635.5 → 取偶
        assert scaled_width_for_height(641, 363, 360) % 2 == 0


class TestResolutionPlan:
    def test_canvas_height_is_max_even(self):
        plan = plan_composition(
            make_media(width=320, height=240),
            make_media(width=640, height=360),
        )
        assert plan.height == 360
        # 左路 320x240 等比缩放到高 360 → 480；右路保持 640
        assert plan.left.scaled_width == 480
        assert plan.right.scaled_width == 640
        assert plan.width == 480 + 640

    def test_odd_height_rounded_up(self):
        plan = plan_composition(
            make_media(width=100, height=361),
            make_media(width=100, height=100),
        )
        assert plan.height == 362

    def test_height_override(self):
        plan = plan_composition(
            make_media(width=320, height=240),
            make_media(width=640, height=360),
            height=720,
        )
        assert plan.height == 720
        assert plan.left.scaled_width == 960
        assert plan.right.scaled_width == 1280

    def test_fit_layout_pads_to_equal_cells(self):
        plan = plan_composition(
            make_media(width=320, height=240),
            make_media(width=640, height=360),
            layout=LAYOUT_FIT,
        )
        # 单元格取较宽者 640，左路居中，左右各留 (640-480)/2=80 黑边
        assert plan.left.cell_width == plan.right.cell_width == 640
        assert plan.left.pad_x == 80
        assert plan.right.pad_x == 0
        assert plan.width == 1280

    def test_sar_correction(self):
        # SAR 4:3 的 720x576 显示宽度为 720*4/3=960，缩放到高 576 → 宽 960
        plan = plan_composition(
            make_media(width=720, height=576, sar="4:3"),
            make_media(width=720, height=576, sar="1:1"),
        )
        assert plan.left.scaled_width == 960
        assert plan.right.scaled_width == 720


class TestFpsPlan:
    def test_takes_max(self):
        plan = plan_composition(
            make_media(fps="15/1"),
            make_media(fps="30000/1001"),
        )
        assert plan.fps == Fraction(30000, 1001)
        assert plan.fps_text == "30000/1001"

    def test_override(self):
        plan = plan_composition(make_media(), make_media(), fps="24")
        assert plan.fps == Fraction(24, 1)

    def test_invalid_override_rejected(self):
        with pytest.raises(InvalidInputError):
            plan_composition(make_media(), make_media(), fps="not-a-fps")

    def test_fallback_when_unknown(self):
        plan = plan_composition(
            make_media(fps="0/0"),
            make_media(fps="0/0"),
        )
        assert plan.fps == DEFAULT_FALLBACK_FPS
        assert any("帧率" in w for w in plan.warnings)


class TestDurationPlan:
    def test_pad_takes_longer_and_freezes_shorter(self):
        plan = plan_composition(
            make_media(duration=2.0),
            make_media(duration=3.5),
            duration_mode=DURATION_PAD,
        )
        assert plan.duration == 3.5
        assert plan.left.freeze_seconds == pytest.approx(1.5)
        assert plan.right.freeze_seconds == 0.0

    def test_trim_takes_shorter(self):
        plan = plan_composition(
            make_media(duration=2.0),
            make_media(duration=3.5),
            duration_mode=DURATION_TRIM,
        )
        assert plan.duration == 2.0
        assert plan.left.freeze_seconds == 0.0
        assert plan.right.freeze_seconds == 0.0

    def test_unknown_duration_fails(self):
        bad = make_media(duration=0.0, format_duration=None)
        with pytest.raises(InvalidInputError, match="时长"):
            plan_composition(bad, make_media())

    def test_stream_duration_missing_falls_back_to_format(self):
        media = make_media(duration=0.0, format_duration=4.0)
        plan = plan_composition(media, make_media(duration=2.0))
        assert plan.duration == 4.0


class TestAudioPlan:
    def test_both_have_audio_mixes(self):
        plan = plan_composition(make_media(), make_media())
        assert plan.audio_mode == AUDIO_MIX
        assert plan.audio_sources == (0, 1)

    def test_single_side_used(self):
        plan = plan_composition(make_media(has_audio=True), make_media(has_audio=False))
        assert plan.audio_mode == AUDIO_LEFT
        assert plan.audio_sources == (0,)
        plan = plan_composition(make_media(has_audio=False), make_media(has_audio=True))
        assert plan.audio_mode == AUDIO_RIGHT
        assert plan.audio_sources == (1,)

    def test_no_audio_at_all(self):
        plan = plan_composition(
            make_media(has_audio=False), make_media(has_audio=False)
        )
        assert plan.audio_mode == AUDIO_NONE
        assert plan.audio_sources == ()

    def test_explicit_missing_side_is_honored_with_warning(self):
        plan = plan_composition(
            make_media(has_audio=False),
            make_media(has_audio=True),
            audio_mode=AUDIO_LEFT,
        )
        assert plan.audio_mode == AUDIO_NONE
        assert plan.audio_sources == ()
        assert any("左路" in w for w in plan.warnings)

    def test_mix_request_degrades_with_warning(self):
        plan = plan_composition(
            make_media(has_audio=True),
            make_media(has_audio=False),
            audio_mode=AUDIO_MIX,
        )
        assert plan.audio_mode == AUDIO_LEFT
        assert any("混音" in w for w in plan.warnings)


class TestInvalidParams:
    def test_bad_duration_mode(self):
        with pytest.raises(InvalidInputError):
            plan_composition(make_media(), make_media(), duration_mode="bogus")

    def test_bad_layout(self):
        with pytest.raises(InvalidInputError):
            plan_composition(make_media(), make_media(), layout="bogus")

    def test_bad_audio_mode(self):
        with pytest.raises(InvalidInputError):
            plan_composition(make_media(), make_media(), audio_mode="bogus")

    def test_missing_video_stream(self):
        no_video = MediaInfo(
            file_path="audio_only.mp3",
            format_name="mp3",
            duration_seconds=2.0,
            streams=[StreamInfo(index=0, codec_type="audio")],
        )
        with pytest.raises(InvalidInputError, match="视频流"):
            plan_composition(no_video, make_media())

    def test_missing_dimensions(self):
        bad = make_media()
        bad.streams[0].width = None
        with pytest.raises(InvalidInputError, match="尺寸"):
            plan_composition(bad, make_media())


class TestDeterminism:
    def test_same_inputs_same_plan(self):
        left = make_media(width=320, height=240, duration=2.0, fps="15/1")
        right = make_media(width=640, height=360, duration=3.5, fps="25/1")
        plan1 = plan_composition(left, right)
        plan2 = plan_composition(left, right)
        assert plan1.to_dict() == plan2.to_dict()
