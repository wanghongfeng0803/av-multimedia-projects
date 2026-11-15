"""filter_complex 生成的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

from compose_tool.filtergraph import build_filter_complex
from compose_tool.plan import DURATION_TRIM, LAYOUT_FIT, plan_composition

from conftest import make_media


def _plan(**kwargs):
    left = kwargs.pop("left", None) or make_media(
        width=320, height=240, duration=2.0, fps="15/1"
    )
    right = kwargs.pop("right", None) or make_media(
        width=640, height=360, duration=3.5, fps="25/1", has_audio=False
    )
    return plan_composition(left, right, **kwargs)


class TestVideoChain:
    def test_basic_structure(self):
        fc = build_filter_complex(_plan())
        assert "[0:v]" in fc and "[1:v]" in fc
        assert "hstack=inputs=2[vout]" in fc
        # 输出帧率取较大者 25
        assert "fps=25" in fc
        # 画布高 360，左路 320x240 → 480x360
        assert "scale=480:360:flags=lanczos" in fc
        assert "scale=640:360:flags=lanczos" in fc
        assert "setsar=1" in fc
        assert "format=yuv420p" in fc

    def test_pad_mode_freezes_shorter_side_only(self):
        fc = build_filter_complex(_plan())
        # 左路 2s < 输出 3.5s → 冻结 1.5s；右路不冻结
        assert "tpad=stop_mode=clone:stop_duration=1.500000" in fc
        assert fc.count("tpad") == 1
        # 两路视频都按输出时长截齐（",trim=" 前缀避免误匹配音频的 atrim）
        assert fc.count(",trim=duration=3.500000") == 2

    def test_trim_mode_has_no_tpad(self):
        fc = build_filter_complex(_plan(duration_mode=DURATION_TRIM))
        assert "tpad" not in fc
        assert fc.count(",trim=duration=2.000000") == 2

    def test_fit_layout_adds_pad_filter(self):
        fc = build_filter_complex(_plan(layout=LAYOUT_FIT))
        # 单元格等宽 640，左路缩放宽 480 → 居中 x=80
        assert "pad=640:360:80:0:color=black" in fc

    def test_auto_layout_has_no_pad_filter(self):
        fc = build_filter_complex(_plan())
        assert "pad=" not in fc.replace("tpad", "").replace("apad", "")

    def test_fps_override_in_graph(self):
        fc = build_filter_complex(_plan(fps="30000/1001"))
        assert "fps=30000/1001" in fc


class TestAudioChain:
    def test_single_audio_side(self):
        fc = build_filter_complex(_plan())
        # 仅左路有音频：重采样 + 补静音 + 截齐，无 amix
        assert "[0:a]" in fc
        assert "aresample=48000" in fc
        assert "aformat=sample_fmts=fltp:channel_layouts=stereo" in fc
        assert "apad" in fc
        assert "atrim=duration=3.500000" in fc
        assert "amix" not in fc
        assert "[a0]acopy[aout]" in fc

    def test_both_audio_mixes(self):
        fc = build_filter_complex(
            _plan(right=make_media(width=640, height=360, duration=3.5, has_audio=True))
        )
        assert "[0:a]" in fc and "[1:a]" in fc
        assert "amix=inputs=2:duration=longest:normalize=1[aout]" in fc

    def test_no_audio_has_no_audio_chain(self):
        fc = build_filter_complex(
            _plan(
                left=make_media(has_audio=False),
                right=make_media(has_audio=False),
            )
        )
        assert ":a]" not in fc
        assert "[aout]" not in fc

    def test_determinism(self):
        assert build_filter_complex(_plan()) == build_filter_complex(_plan())
