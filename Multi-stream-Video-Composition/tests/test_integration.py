"""端到端集成测试：需要真实 ffmpeg/ffprobe（无 ffmpeg 时自动 skip）。

样本特意制造「全不一致」场景：左 320x240@15fps/2s/有音频，
右 640x360@25fps/3.5s/无音频，验证确定性对齐策略在真实管线中生效。
"""

from __future__ import annotations

import hashlib
import os

import pytest

from compose_tool.composer import compose
from compose_tool.errors import OutputExistsError
from compose_tool.probe import probe_media

from conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestComposeEndToEnd:
    def test_mismatched_inputs(self, left_clip, right_clip, work_dir):
        out = os.path.join(work_dir, "out.mp4")
        result = compose(left_clip, right_clip, out)
        assert result.success

        info = probe_media(out)
        video = info.primary_video
        assert video is not None
        # 画布高 = max(240, 360) = 360；左 320x240→480 宽，右保持 640 → 总宽 1120
        assert (video.width, video.height) == (1120, 360)
        # 帧率取较大者 25，输出为 CFR
        assert video.avg_frame_rate == "25/1"
        assert video.pix_fmt == "yuv420p"
        # 时长取较长者 3.5s（容器时长允许音频 priming 的微小误差）
        assert info.duration_seconds == pytest.approx(3.5, abs=0.15)
        # 仅左路有音频 → 输出有 AAC 48kHz 立体声音轨
        audio = info.primary_audio
        assert audio is not None
        assert audio.codec_name == "aac"
        assert audio.sample_rate == 48000
        assert audio.channels == 2
        assert result.audio_mode == "left"

    def test_trim_mode(self, left_clip, right_clip, work_dir):
        out = os.path.join(work_dir, "trim.mp4")
        result = compose(left_clip, right_clip, out, duration_mode="trim")
        info = probe_media(out)
        assert info.duration_seconds == pytest.approx(2.0, abs=0.15)
        assert result.duration_mode == "trim"

    def test_audio_mix(self, left_clip, right_audio_clip, work_dir):
        out = os.path.join(work_dir, "mix.mp4")
        result = compose(left_clip, right_audio_clip, out)
        assert result.audio_mode == "mix"
        info = probe_media(out)
        assert info.primary_audio is not None
        assert info.primary_audio.codec_name == "aac"

    def test_fit_layout_symmetric(self, left_clip, right_clip, work_dir):
        out = os.path.join(work_dir, "fit.mp4")
        result = compose(left_clip, right_clip, out, layout="fit")
        info = probe_media(out)
        video = info.primary_video
        assert video is not None
        # 等宽单元格 640 → 总宽 1280，左右严格对称
        assert (video.width, video.height) == (1280, 360)

    def test_height_and_fps_override(self, left_clip, right_clip, work_dir):
        out = os.path.join(work_dir, "override.mp4")
        compose(left_clip, right_clip, out, height=720, fps="30")
        info = probe_media(out)
        video = info.primary_video
        assert video is not None
        assert video.height == 720
        assert video.avg_frame_rate == "30/1"

    def test_reproducible_byte_identical(self, left_clip, right_clip, work_dir):
        """同输入同参数跑两次，输出必须逐字节一致。"""
        out1 = os.path.join(work_dir, "run1.mp4")
        out2 = os.path.join(work_dir, "run2.mp4")
        compose(left_clip, right_clip, out1)
        compose(left_clip, right_clip, out2)
        assert _sha256(out1) == _sha256(out2)

    def test_refuses_overwrite(self, left_clip, right_clip, work_dir):
        out = os.path.join(work_dir, "exists.mp4")
        compose(left_clip, right_clip, out)
        with pytest.raises(OutputExistsError):
            compose(left_clip, right_clip, out)
        # 允许覆盖后成功
        compose(left_clip, right_clip, out, overwrite=True)

    def test_progress_callback_reaches_100(self, left_clip, right_clip, work_dir):
        ratios: list[float] = []
        out = os.path.join(work_dir, "progress.mp4")
        compose(left_clip, right_clip, out, progress_callback=ratios.append)
        assert ratios
        assert ratios[-1] == pytest.approx(1.0)
        assert ratios == sorted(ratios)  # 单调不减
