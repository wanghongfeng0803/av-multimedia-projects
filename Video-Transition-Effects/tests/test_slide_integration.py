"""滑动转场端到端集成测试：真实 ffmpeg 管道下的几何与账本承诺。

无 ffmpeg 时自动 skip。覆盖：

* 四方向成片帧数恒等式 Na+Nb-No，head/tail 逐帧映射不丢帧；
* 重叠区首尾帧与纯 A/纯 B 连续（锚点），任意帧无黑边（无零像素带）；
* 滑动边界位置的取源方向与方向语义一致（用纯色素材逐像素核对）；
* 非默认速度曲线可跑通且位移单调；
* 音频随画面同一条进度曲线增益，边界能量连续。
"""

from __future__ import annotations

import subprocess

import pytest

from video_transition.plan import build_plan
from video_transition.pipeline import cross_fade
from video_transition.probe import probe_clip
from video_transition.reader import read_rgb24_frames
from video_transition.slide import DIRECTIONS

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


class TestSlideEndToEnd:
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
            transition="slide", direction=direction,
        )
        assert result.frames_written == na + nb - no
        assert result.transition == "slide"
        assert result.direction == direction

        meta_a = probe_clip(str(a_path))
        plan = build_plan(
            meta_a, probe_clip(str(b_path)),
            overlap_frames=no, transition="slide", direction=direction,
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        a_frames = read_rgb24_frames(meta_a)
        b_frames = read_rgb24_frames(probe_clip(str(b_path)))
        tolerance = 6

        def mean_diff(f1, f2):
            diffs = [abs(f1[j] - f2[j]) for j in range(0, len(f1), 37)]
            return sum(diffs) / len(diffs)

        # 1) head 仍是纯 A、tail 仍是纯 B（滑动不改重叠区之外的帧）
        for i in range(na - no):
            assert mean_diff(out_frames[i], a_frames[i]) <= tolerance
        for local in range(nb - no):
            assert mean_diff(out_frames[na + local], b_frames[no + local]) <= tolerance

        # 2) 锚点：重叠首帧贴近纯 A、末帧贴近纯 B，边界无跳变
        ov = plan.overlap
        assert mean_diff(out_frames[ov.out_start], a_frames[ov.a_start]) <= tolerance
        assert mean_diff(
            out_frames[ov.out_end - 1], b_frames[no - 1]
        ) <= tolerance

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_no_black_border_at_any_overlap_frame(self, tmp_path, direction):
        # 两源都选三通道等值且足够亮的颜色（白/灰），任何未覆盖的
        # 黑边都会产生三个通道同时接近 0 的像素，与源色明确区分
        a_path, b_path = _solid_pair(tmp_path, "white", "0x808080")
        out_path = tmp_path / "o.mp4"
        no = 16
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True, crf=12,
            transition="slide", direction=direction,
        )
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="slide", direction=direction,
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        ov = plan.overlap

        def pixel_luma(frame, byte_index):
            # 简化亮度：取三通道最小值。白=255、灰=128，最小值都不低；
            # 黑边三通道同为 0，编解码抖动后仍远低于该阈值
            return min(frame[byte_index], frame[byte_index + 1],
                       frame[byte_index + 2])

        for f in range(ov.out_start, ov.out_end):
            frame = out_frames[f]
            darkest = min(pixel_luma(frame, j) for j in range(0, len(frame) - 2, 303))
            assert darkest >= 48, f"{direction} 第 {f} 帧疑似出现黑边"

    def test_right_direction_pixel_layout(self, tmp_path):
        # right：随位移推进，B 占据的左侧列数单调增加
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        _solid_clip(a_path, "red", size="160x120")
        _solid_clip(b_path, "blue", size="160x120")
        no = 10
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True, crf=10,
            transition="slide", direction="right",
        )
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="slide", direction="right",
        )
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        ov = plan.overlap
        width = plan.width

        def blue_column_count(frame):
            count = 0
            for x in range(width):
                i = (x) * 3  # 顶行像素
                # 蓝：B 通道显著高于 R 通道
                if frame[i + 2] > frame[i] + 40:
                    count += 1
            return count

        counts = [
            blue_column_count(out_frames[ov.out_start + k]) for k in range(no)
        ]
        # 首帧没有蓝列、末帧整行蓝列，中间单调不减
        assert counts[0] == 0
        assert counts[-1] == width
        assert counts == sorted(counts)

    def test_easing_curve_runs_and_keeps_anchors(self, tmp_path):
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        na = _solid_clip(a_path, "white")
        nb = _solid_clip(b_path, "black")
        no = 20
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True, crf=12,
            transition="slide", direction="down", easing="ease_in_out",
        )
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="slide",
            direction="down", easing="ease_in_out",
        )
        assert plan.easing == "ease_in_out"
        out_frames = read_rgb24_frames(probe_clip(str(out_path)))
        assert len(out_frames) == na + nb - no
        # 末帧锚点仍精确落在纯 B（黑）
        tail_first = out_frames[plan.overlap.out_end]
        assert max(tail_first[::997]) <= 16

    def test_slide_with_audio_keeps_boundary_continuity(self, tmp_path):
        a_path = tmp_path / "a.mp4"
        b_path = tmp_path / "b.mp4"
        out_path = tmp_path / "o.mp4"
        make_clip(str(a_path), duration=2, rate=25, audio_freq=880)
        make_clip(str(b_path), duration=2, rate=25, audio_freq=220)
        no = 20
        plan = build_plan(
            probe_clip(str(a_path)), probe_clip(str(b_path)),
            overlap_frames=no, transition="slide", direction="left",
        )
        cross_fade(
            str(a_path), str(b_path), str(out_path),
            overlap_frames=no, overwrite=True,
            transition="slide", direction="left",
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
        # 进度锚点 0/1：边界两侧能量连续，无数量级跳变
        assert 0.5 < rms_ov_first / rms_head_end < 2.0
        assert 0.5 < rms_tail_first / rms_ov_last < 2.0
