"""管线纯逻辑测试：帧数/尺寸推导与参数校验（不依赖 ffmpeg）。"""

import pytest

from vgif.extract import FrameInfo, Manifest
from vgif.pipeline import (
    DEFAULT_COLORS,
    DEFAULT_FPS,
    DEFAULT_MAX_WIDTH,
    PipelineError,
    derive_output_frame_count,
    derive_target_size,
)


def make_manifest(n=24, fps=24.0, w=320, h=240):
    frames = tuple(
        FrameInfo(index=i, pts_time=i / fps, duration_time=1.0 / fps)
        for i in range(n)
    )
    return Manifest(width=w, height=h, fps=fps, frames=frames)


class TestDeriveFrameCount:
    def test_integer_fps_ratio(self):
        m = make_manifest(n=24, fps=24)
        # 24fps × 1s → 12fps 目标 = 12 帧。
        assert derive_output_frame_count(m, 12) == 12

    def test_half_up_rounding(self):
        m = make_manifest(n=10, fps=10)  # 时长 1.0s
        # 10 × 0.333 = 3.33 → 3。
        assert derive_output_frame_count(m, 3.33) == 3

    def test_capped_at_source_count(self):
        m = make_manifest(n=5, fps=5)
        # 请求 100fps 也不能凭空造帧。
        assert derive_output_frame_count(m, 100) == 5

    def test_at_least_one(self):
        m = make_manifest(n=24, fps=24)
        assert derive_output_frame_count(m, 0.001) == 1

    def test_rejects_nonpositive_fps(self):
        with pytest.raises(PipelineError):
            derive_output_frame_count(make_manifest(), 0)


class TestDeriveTargetSize:
    def test_no_downscale_when_smaller(self):
        assert derive_target_size(320, 240, 480) == (320, 240)

    def test_keeps_aspect_ratio(self):
        w, h = derive_target_size(1920, 1080, 480)
        assert w == 480
        assert h == 270
        assert abs(h / w - 1080 / 1920) < 0.01

    def test_none_disables_scaling(self):
        assert derive_target_size(1920, 1080, None) == (1920, 1080)

    def test_min_height_one(self):
        w, h = derive_target_size(1000, 1, 480)
        assert (w, h) == (480, 1)


def test_defaults_documented():
    # 锁定默认参数，改动须同步 README 与 docstring 的折中依据。
    assert DEFAULT_FPS == 12
    assert DEFAULT_MAX_WIDTH == 480
    assert DEFAULT_COLORS == 256


def test_manifest_duration_includes_last_frame():
    m = make_manifest(n=10, fps=10)
    # 末帧 PTS 0.9 + 持续 0.1 = 1.0，而非 0.9。
    assert m.duration == pytest.approx(1.0)
