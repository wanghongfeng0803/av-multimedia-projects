"""数据模型与命令组装的纯逻辑测试。"""

from __future__ import annotations

from fractions import Fraction

from compose_tool.composer import build_command, default_output_path
from compose_tool.filtergraph import build_filter_complex
from compose_tool.models import StreamInfo
from compose_tool.plan import plan_composition

from conftest import make_media


class TestStreamInfo:
    def test_fps_prefers_avg_then_r(self):
        s = StreamInfo(
            index=0, codec_type="video",
            avg_frame_rate="30000/1001", r_frame_rate="30/1",
        )
        assert s.fps == Fraction(30000, 1001)
        s = StreamInfo(index=0, codec_type="video", avg_frame_rate="0/0", r_frame_rate="30/1")
        assert s.fps == Fraction(30, 1)
        s = StreamInfo(index=0, codec_type="video", avg_frame_rate=None, r_frame_rate=None)
        assert s.fps is None

    def test_display_width_applies_sar(self):
        s = StreamInfo(index=0, codec_type="video", width=720, sample_aspect_ratio="4:3")
        assert s.display_width == 960.0
        s = StreamInfo(index=0, codec_type="video", width=720, sample_aspect_ratio=None)
        assert s.display_width == 720.0
        s = StreamInfo(index=0, codec_type="video", width=None)
        assert s.display_width is None


class TestDefaultOutputPath:
    def test_naming(self):
        out = default_output_path("/a/b/left.mp4", "/c/right.mkv")
        assert out.endswith("left_x_right.mp4")

    def test_same_directory_as_left(self):
        import os
        out = default_output_path(os.path.join("somewhere", "l.mp4"), "r.mp4")
        assert os.path.dirname(out) == os.path.abspath("somewhere")


class TestBuildCommand:
    def _cmd(self, **plan_kwargs):
        plan = plan_composition(make_media(), make_media(has_audio=False), **plan_kwargs)
        fc = build_filter_complex(plan)
        return plan, build_command(
            "ffmpeg", "l.mp4", "r.mp4", plan, fc, "out.mp4",
            overwrite=True, crf=20, preset="veryfast",
        )

    def test_reproducibility_flags(self):
        _, cmd = self._cmd()
        assert "-bitexact" in cmd
        assert cmd[cmd.index("-map_metadata") + 1] == "-1"
        assert cmd[cmd.index("-video_track_timescale") + 1] == "90000"
        assert "+faststart" in cmd

    def test_video_spec(self):
        _, cmd = self._cmd()
        assert cmd[cmd.index("-c:v") + 1] == "libx264"
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert "[vout]" in cmd

    def test_audio_mapped_only_when_present(self):
        plan, cmd = self._cmd()
        assert "[aout]" in cmd
        assert cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[cmd.index("-ar") + 1] == "48000"

        plan2 = plan_composition(
            make_media(has_audio=False), make_media(has_audio=False)
        )
        cmd2 = build_command(
            "ffmpeg", "l.mp4", "r.mp4", plan2,
            build_filter_complex(plan2), "out.mp4",
            overwrite=False, crf=20, preset="veryfast",
        )
        assert "[aout]" not in cmd2
        assert "-c:a" not in cmd2

    def test_overwrite_flag(self):
        plan = plan_composition(make_media(), make_media())
        fc = build_filter_complex(plan)
        cmd_y = build_command("ffmpeg", "l", "r", plan, fc, "o", overwrite=True, crf=20, preset="p")
        cmd_n = build_command("ffmpeg", "l", "r", plan, fc, "o", overwrite=False, crf=20, preset="p")
        assert "-y" in cmd_y and "-n" not in cmd_y
        assert "-n" in cmd_n and "-y" not in cmd_n

    def test_progress_args_present(self):
        _, cmd = self._cmd()
        assert "-progress" in cmd
        assert "pipe:1" in cmd
