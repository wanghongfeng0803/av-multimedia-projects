"""擦除转场端到端集成测试：真实 ffmpeg 管道下的掩码几何与账本承诺。

无 ffmpeg 时自动 skip。覆盖：

* 四方向成片帧数恒等式 Na+Nb-No，head/tail 逐帧映射不丢帧；
* 重叠区首尾帧与纯 A/纯 B 连续（扩展行程锚点），任意帧无黑边；
* 软边宽度超过画面时钳制告警且成片仍无黑边、锚点仍精确；
* 硬边擦除与同参数硬滑动成片一致（同一条混合原语）；
* 音频随同一条进度曲线增益，边界能量连续。
"""

from __future__ import annotations

import subprocess

import pytest

from video_transition.plan import build_plan
from video_transition.pipeline import cross_fade
from video_transition.probe import probe_clip
from video_transition.reader import read_rgb24_frames
from video_transition.slide import DIRECTIONS
from video_transition.wipe import WipeSoftWidthWarning

from .conftest import ffmpeg_available, make_clip

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="无 ffmpeg")


def _solid_clip(path, color, *, duration=2, rate=25, size="320x240"):
    return make_clip(
        str(path), duration=duration, rate=rate, size=size,
        with_audio=False, color=color,
    )


def _solid_pair(tmp_path, color_a, color_b, *, size="320x240"):
    a_path = tmp_path / "a.mp4"
    b_path = tmp_path / "b.mp4"
    _solid_clip(a_path, color_a, size=size)
    _solid_clip(b_path, color_b, size=size)
    return a_path, b_path


class TestWipeEndToEnd:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_frame_identity_and_anchor_continuity(self, tmp_path, direction):
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        na = _solid_clip(a_path, "red")
        nb = _solid_clip(b_path, "blue")
        no = 12

        result = cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True, crf=12,
            transition="wipe", direction=direction, soft_width=10,
        )
        assert result.frames_written == na + nb - no
        assert result.transition == "wipe"
        assert result.soft_width == 10

        meta_a = probe_clip(str(a_path))
        plan = build_plan(
            meta_a, probe_clip(str(b_path)),
            overlap_frames=no, transition="wipe",
            direction=direction, soft_width=10,
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        a_frames = read_rgb24_frames(meta_a)
        b_frames = read_rgb24_frames(probe_clip(str(b_path)))
        tolerance = 6

        def mean_diff(f1, f2):
            diffs = [abs(f1[j] - f2[j]) for j in range(0, len(f1), 37)]
            return sum(diffs) / len(diffs)

        # head 纯 A、tail 纯 B（擦除不改重叠区之外的帧）
        for i in range(na - no):
            assert mean_diff(out_frames[i], a_frames[i]) <= tolerance
        for local in range(nb - no):
            assert mean_diff(out_frames[na + local], b_frames[no + local]) <= tolerance

        # 锚点：重叠首帧贴近纯 A、末帧贴近纯 B
        ov = plan.overlap
        assert mean_diff(out_frames[ov.out_start], a_frames[ov.a_start]) <= tolerance
        assert mean_diff(out_frames[ov.out_end - 1], b_frames[no - 1]) <= tolerance

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_no_black_border_at_any_overlap_frame(self, tmp_path, direction):
        # 白/灰两源都足够亮，任何未覆盖黑边都会出现三通道同时接近 0
        a_path, b_path = _solid_pair(tmp_path, "white", "0x808080")
        out_path = tmp_path / "o.mp4"
        no = 16
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True, crf=12,
            transition="wipe", direction=direction, soft_width=24,
        )
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="wipe",
            direction=direction, soft_width=24,
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        ov = plan.overlap

        for f in range(ov.out_start, ov.out_end):
            frame = out_frames[f]
            darkest = min(
                min(frame[j], frame[j + 1], frame[j + 2])
                for j in range(0, len(frame) - 2, 303)
            )
            assert darkest >= 48, f"{direction} 第 {f} 帧疑似出现黑边"

    def test_soft_width_overflow_clamped_and_still_clean(self, tmp_path):
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        _solid_clip(a_path, "white", size="160x120")
        _solid_clip(b_path, "0x808080", size="160x120")
        no = 14
        with pytest.warns(WipeSoftWidthWarning) as record:
            result = cross_fade(
                str(a_path), str(b_path), str(out_path),
                overlap_frames=no, overwrite=True, crf=12,
                transition="wipe", direction="right", soft_width=10_000,
            )
        # 水平轴长 160：钳制后写进结果摘要
        assert result.soft_width == 160
        assert "10000" in str(record[0].message)

        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="wipe",
            direction="right", soft_width=160,
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        ov = plan.overlap
        # 钳制后锚点仍精确：首帧白（A）、末帧灰（B），全程无黑边
        first, last = out_frames[ov.out_start], out_frames[ov.out_end - 1]
        assert min(first[::303]) >= 240
        assert 100 <= min(last[::303]) <= 160
        for f in range(ov.out_start, ov.out_end):
            assert min(out_frames[f][::303]) >= 48

    def test_hard_wipe_matches_hard_slide_output(self, tmp_path):
        # 同帧 A/B、同进度下，硬擦除与硬滑动走同一套分区/混合原语
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        wipe_out = tmp_path / "wipe.mp4"
        slide_out = tmp_path / "slide.mp4"
        _solid_clip(a_path, "red", size="160x120")
        _solid_clip(b_path, "blue", size="160x120")
        common = dict(
            overlap_frames=10, overwrite=True, crf=10, direction="left",
        )
        cross_fade(str(a_path), str(b_path), str(wipe_out),
                   transition="wipe", soft_width=0, **common)
        cross_fade(str(a_path), str(b_path), str(slide_out),
                   transition="slide", **common)
        wipe_frames = read_rgb24_frames(probe_clip(str(wipe_out)))
        slide_frames = read_rgb24_frames(probe_clip(str(slide_out)))
        assert len(wipe_frames) == len(slide_frames)
        for fw, fs in zip(wipe_frames, slide_frames):
            diffs = [abs(fw[j] - fs[j]) for j in range(0, len(fw), 37)]
            assert sum(diffs) / len(diffs) <= 2

    def test_blue_reveal_grows_monotonically_right(self, tmp_path):
        # right 擦除：随掩码推进，顶行蓝（B）像素数单调不减、首尾为 0/满宽
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        _solid_clip(a_path, "red", size="160x120")
        _solid_clip(b_path, "blue", size="160x120")
        no = 10
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True, crf=10,
            transition="wipe", direction="right", soft_width=6,
        )
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="wipe",
            direction="right", soft_width=6,
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        ov = plan.overlap
        width = plan.width

        def blue_column_count(frame):
            return sum(
                1 for x in range(width)
                if frame[x * 3 + 2] > frame[x * 3] + 40
            )

        counts = [
            blue_column_count(out_frames[ov.out_start + k]) for k in range(no)
        ]
        assert counts[0] == 0
        assert counts[-1] == width
        assert counts == sorted(counts)

    def test_wipe_with_audio_keeps_boundary_continuity(self, tmp_path):
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        make_clip(str(a_path), duration=2, rate=25, audio_freq=880)
        make_clip(str(b_path), duration=2, rate=25, audio_freq=220)
        no = 20
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="wipe",
            direction="up", soft_width=8,
        )
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True,
            transition="wipe", direction="up", soft_width=8,
        )
        from video_transition.audio import S16_CHANNELS, frame_sample_bounds
        import array

        pcm_path = tmp_path / "o.pcm"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(out_path),
            "-vn", "-f", "s16le", "-ac", "2", "-ar", "48000", str(pcm_path),
        ], check=True)
        pcm = array.array("h")
        pcm.frombytes(pcm_path.read_bytes())
        bounds = frame_sample_bounds(plan.output_frames, 48000, 25, 1)

        def frame_rms(idx):
            s0, s1 = bounds[idx] * S16_CHANNELS, bounds[idx + 1] * S16_CHANNELS
            block = pcm[s0:s1]
            return (sum(v * v for v in block) / len(block)) ** 0.5

        ov = plan.overlap
        rms_head_end = frame_rms(ov.out_start - 1)
        rms_ov_first = frame_rms(ov.out_start)
        rms_ov_last = frame_rms(ov.out_end - 1)
        rms_tail_first = frame_rms(ov.out_end)
        assert 0.5 < rms_ov_first / rms_head_end < 2.0
        assert 0.5 < rms_tail_first / rms_ov_last < 2.0
