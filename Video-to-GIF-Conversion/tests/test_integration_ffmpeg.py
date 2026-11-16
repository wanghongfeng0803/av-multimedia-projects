"""ffmpeg 端到端集成测试：无 ffmpeg 时整体 skip。

验证三件最关键的事：

1. **时序零漂移**：GIF 总时长相对源时长偏差 < 1 厘秒，逐帧延迟与
   :class:`vgif.timing.DelayPlan` 完全一致，调色板/缩放不影响时序；
2. **三步解耦可复用**：同一工作目录二次转换复用抽帧产物，换帧率 /
   尺寸 / 色深不必重新解码，且产物逐字节稳定；
3. **抽帧 1:1**：落盘帧数等于 ffprobe 帧数（-fps_mode passthrough）。
"""

import json
import subprocess

import pytest

from tests.conftest import FFMPEG, FFPROBE, needs_ffmpeg  # noqa: F401
from vgif.extract import Manifest, extract_rgb_frames, probe_frames
from vgif.gif_writer import write_gif
from vgif.pipeline import convert_to_gif
from vgif.quantize import quantize_frames
from vgif.sampling import sample_frames
from vgif.scale import bilinear_resize
from vgif.timing import plan_gif_delays

from .test_gif_writer import parse_gif


def _gif_total_duration(path):
    _, _, _, frames = parse_gif(path)
    return sum(f["delay"] for f in frames) / 100.0


@needs_ffmpeg
class TestEndToEnd:
    def test_probe_and_extract_frame_match(self, video_factory, tmp_path):
        video = video_factory(frames=24, fps=24)
        work = tmp_path / "work"
        manifest = extract_rgb_frames(video, work)
        assert manifest.frame_count == 24
        # 每帧字节数 = 64*48*3。
        assert (work / "frame_000001.rgb").stat().st_size == 64 * 48 * 3
        # manifest 已落盘且可重建。
        rebuilt = Manifest.from_dict(
            json.loads((work / "manifest.json").read_text("utf-8"))
        )
        assert rebuilt.frame_count == 24

    def test_full_convert_timing_no_drift(self, video_factory, tmp_path):
        video = video_factory(frames=30, fps=30)  # 时长 1.0s
        out = tmp_path / "out.gif"
        result = convert_to_gif(video, out, target_fps=12, max_width=48)
        assert out.is_file() and out.stat().st_size > 0
        # 源时长 1.0s → 100 厘秒，误差 < 1 厘秒。
        assert abs(result.gif_duration - 1.0) < 0.01
        assert abs(result.duration_error) < 0.01
        # 文件里实际延迟总和与规划一致。
        assert _gif_total_duration(out) == result.gif_duration
        assert result.frame_count_out == 12

    def test_delay_independent_of_palette_and_scale(
        self, video_factory, tmp_path
    ):
        video = video_factory(frames=25, fps=25)  # 1.0s
        outs = []
        for colors, width in ((256, 48), (64, 32), (16, None)):
            o = tmp_path / f"c{colors}_{width}.gif"
            r = convert_to_gif(
                video, o, target_fps=10, max_width=width, colors=colors
            )
            outs.append((o, r))
        # 同帧率 → 延迟规划逐帧相同，与色深/尺寸无关。
        base = outs[0][1].delays_cs
        for _, r in outs[1:]:
            assert r.delays_cs == base
        for o, _ in outs:
            assert abs(_gif_total_duration(o) - 1.0) < 0.01

    def test_workdir_reuse_skips_reextraction(
        self, video_factory, tmp_path
    ):
        video = video_factory(frames=24, fps=24)
        work = tmp_path / "persist"
        r1 = convert_to_gif(
            video, tmp_path / "a.gif", work_dir=work, target_fps=12
        )
        assert r1.reused_extract is False
        r2 = convert_to_gif(
            video, tmp_path / "b.gif", work_dir=work, target_fps=8
        )
        assert r2.reused_extract is True
        # 复用后帧数口径仍正确。
        assert r2.frame_count_out == 8

    def test_reuse_deterministic_bytes(self, video_factory, tmp_path):
        video = video_factory(frames=20, fps=20)
        work = tmp_path / "w"
        a = tmp_path / "a.gif"
        b = tmp_path / "b.gif"
        convert_to_gif(video, a, work_dir=work, target_fps=10)
        convert_to_gif(video, b, work_dir=work, target_fps=10)
        assert a.read_bytes() == b.read_bytes()

    def test_three_stage_manual_pipeline(self, video_factory, tmp_path):
        """手动分三步跑，结果与一体转换时序一致。"""
        video = video_factory(frames=24, fps=24)
        work = tmp_path / "stages"

        # 阶段一：抽帧。
        manifest = extract_rgb_frames(video, work)
        n_out = 12
        plan = plan_gif_delays(manifest.duration, n_out, 12.0)

        # 阶段二：采样 → 缩放 → 量化。
        src = [
            manifest.frame_path(work, i).read_bytes()
            for i in range(manifest.frame_count)
        ]
        picked, _ = sample_frames(src, n_out)
        scaled = [
            bilinear_resize(f, manifest.width, manifest.height, 48, 36)
            for f in picked
        ]
        palette, qframes = quantize_frames(scaled, 48, 36, 256)

        # 阶段三：编码。
        out = tmp_path / "manual.gif"
        write_gif(out, qframes, palette, plan.delays_cs, 48, 36)
        assert abs(_gif_total_duration(out) - manifest.duration) < 0.01

    def test_high_fps_rejected_not_silently_drifted(
        self, video_factory, tmp_path
    ):
        video = video_factory(frames=60, fps=60)  # 1.0s
        with pytest.raises(Exception, match="50fps"):
            convert_to_gif(video, tmp_path / "x.gif", target_fps=100)

    def test_indexed_scale_path(self, video_factory, tmp_path):
        video = video_factory(frames=12, fps=12)
        out = tmp_path / "idx.gif"
        r = convert_to_gif(
            video, out, target_fps=10, max_width=32, indexed_scale=True
        )
        assert out.is_file()
        assert (r.width, r.height) == (32, 24)

    def test_vfr_passthrough_no_frame_invention(self, tmp_path):
        """非整数帧率素材：抽帧数必须等于 ffprobe 实解帧数。"""
        video = tmp_path / "vfr.mp4"
        subprocess.run(
            [
                FFMPEG, "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=32x24:rate=15",
                "-frames:v", "10", "-pix_fmt", "yuv420p",
                str(video),
            ],
            check=True,
        )
        manifest = probe_frames(video)
        work = tmp_path / "w"
        extract_rgb_frames(video, work)
        assert len(list(work.glob("frame_*.rgb"))) == manifest.frame_count


@needs_ffmpeg
class TestCli:
    def test_convert_command(self, video_factory, tmp_path):
        from gif_convert import main

        video = video_factory(frames=24, fps=24)
        out = tmp_path / "cli.gif"
        rc = main([str(video), str(out), "--fps", "12", "--max-width",
                   "48", "--colors", "16"])
        assert rc == 0
        assert out.is_file()
        assert abs(_gif_total_duration(out) - 1.0) < 0.01

    def test_split_subcommands(self, video_factory, tmp_path):
        from gif_convert import main

        video = video_factory(frames=20, fps=20)
        work = tmp_path / "work"
        payload = tmp_path / "q.pkl"
        out = tmp_path / "split.gif"

        assert main(["extract", str(video), str(work)]) == 0
        assert main(["quantize", str(work), str(payload),
                     "--fps", "10", "--max-width", "32"]) == 0
        assert main(["encode", str(payload), str(out)]) == 0
        assert out.is_file()
        assert abs(_gif_total_duration(out) - 1.0) < 0.01

    def test_missing_input_returns_error(self, tmp_path):
        from gif_convert import main

        rc = main([str(tmp_path / "nope.mp4"), str(tmp_path / "o.gif")])
        assert rc == 1
