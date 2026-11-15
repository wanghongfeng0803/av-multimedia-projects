"""audio.py 测试：采样边界累积取整、整轨淡化拼装与视频规划同源。"""

from __future__ import annotations

import array
import math

import pytest

from video_transition.audio import (
    S16_CHANNELS,
    assemble_audio,
    choose_sample_rate,
    frame_sample_bounds,
    overlap_weight_fraction,
)
from video_transition.errors import PlanError
from video_transition.plan import build_plan
from video_transition.probe import ClipMeta

_FRAME_BYTES = S16_CHANNELS * 2


def _meta(name: str, *, frames: int, fps=(25, 1)) -> ClipMeta:
    num, den = fps
    return ClipMeta(
        path=name, width=8, height=8, fps_num=num, fps_den=den,
        frame_count=frames, duration=frames * den / num, has_audio=True,
    )


def _pcm_samples(pcm: bytes) -> list[int]:
    arr = array.array("h")
    arr.frombytes(bytes(pcm))
    return list(arr)


class TestSampleBounds:
    def test_starts_at_zero_and_monotonic(self):
        bounds = frame_sample_bounds(10, 48000, 25, 1)
        assert bounds[0] == 0
        assert all(bounds[i + 1] > bounds[i] for i in range(10))
        assert bounds[-1] == 10 * 48000 // 25

    def test_exact_division_gives_constant_span(self):
        # 48000/25 = 1920 整除：每帧严格 1920 采样
        bounds = frame_sample_bounds(7, 48000, 25, 1)
        spans = [bounds[i + 1] - bounds[i] for i in range(7)]
        assert spans == [1920] * 7

    def test_ntsc_spans_alternate_but_boundaries_stay_integer(self):
        # 48000 配 30000/1001：每帧名义 1601.6 采样，1601/1602 交替
        bounds = frame_sample_bounds(10, 48000, 30000, 1001)
        spans = [bounds[i + 1] - bounds[i] for i in range(10)]
        assert set(spans) <= {1601, 1602}
        # 累积边界与名义值的偏差始终小于 1 个采样
        for f, edge in enumerate(bounds):
            nominal = f * 48000 * 1001 / 30000
            assert abs(edge - nominal) < 1.0

    def test_choose_sample_rate_prefers_exact_divisor(self):
        # 25fps 下 48000 可整除（1920），应优先选中
        assert choose_sample_rate(25, 1) == 48000
        # 30000/1001 无精确解，回退 48000
        assert choose_sample_rate(30000, 1001) == 48000


class TestWeightFraction:
    def test_matches_video_weights(self):
        assert overlap_weight_fraction(0, 10) == (0, 9)
        assert overlap_weight_fraction(9, 10) == (9, 9)
        assert overlap_weight_fraction(4, 10) == (4, 9)

    def test_single_frame_half(self):
        assert overlap_weight_fraction(0, 1) == (1, 2)

    def test_invalid_index(self):
        with pytest.raises(PlanError):
            overlap_weight_fraction(10, 10)


class TestAssembleAudio:
    def _full_pcm(self, meta: ClipMeta, sr: int, fill: int) -> bytearray:
        bounds = frame_sample_bounds(meta.frame_count, sr, meta.fps_num, meta.fps_den)
        return bytearray([0 if fill == 0 else (fill & 0xFF)] * (bounds[-1] * _FRAME_BYTES))

    def test_output_length_matches_output_timeline(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        sr = 48000
        pcm = assemble_audio(plan, self._full_pcm(a, sr, 0), self._full_pcm(b, sr, 0), sr)
        bounds_out = frame_sample_bounds(plan.output_frames, sr, 25, 1)
        assert len(pcm) == bounds_out[-1] * _FRAME_BYTES

    def test_head_is_pure_a_tail_is_pure_b(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        sr = 48000
        # A 轨全 +1000，B 轨全 -1000（按采样填值）
        bounds_a = frame_sample_bounds(50, sr, 25, 1)
        bounds_b = frame_sample_bounds(40, sr, 25, 1)
        pcm_a = bytearray(array.array("h", [1000] * (bounds_a[-1] * S16_CHANNELS)).tobytes())
        pcm_b = bytearray(array.array("h", [-1000] * (bounds_b[-1] * S16_CHANNELS)).tobytes())
        out = assemble_audio(plan, pcm_a, pcm_b, sr)
        samples = _pcm_samples(out)

        # head 第一帧（采样 0..1919）全部是 A 的 +1000
        assert all(v == 1000 for v in samples[:1920])
        # tail 最后一帧全部是 B 的 -1000
        assert all(v == -1000 for v in samples[-1920:])

    def test_overlap_fades_frame_by_frame(self):
        a, b = _meta("a", frames=50), _meta("b", frames=40)
        plan = build_plan(a, b, overlap_frames=10)
        sr = 48000
        bounds_a = frame_sample_bounds(50, sr, 25, 1)
        bounds_b = frame_sample_bounds(40, sr, 25, 1)
        pcm_a = bytearray(array.array("h", [800] * (bounds_a[-1] * S16_CHANNELS)).tobytes())
        pcm_b = bytearray(array.array("h", [200] * (bounds_b[-1] * S16_CHANNELS)).tobytes())
        out = assemble_audio(plan, pcm_a, pcm_b, sr)
        samples = _pcm_samples(out)
        bounds_out = frame_sample_bounds(plan.output_frames, sr, 25, 1)

        # 重叠区第 k 帧：800*(9-k)/9 + 200*k/9，远离零四舍五入
        ov_start = plan.overlap.out_start
        for k in range(10):
            s0, s1 = bounds_out[ov_start + k], bounds_out[ov_start + k + 1]
            num = 800 * (9 - k) + 200 * k
            expected = (num + 4) // 9 if num >= 0 else -((-num + 4) // 9)
            block = samples[s0 * S16_CHANNELS:s1 * S16_CHANNELS]
            assert block, "重叠区帧块不能为空"
            assert all(v == expected for v in block), f"第 {k} 帧音频增益错误"

    def test_silence_inputs_produce_silence(self):
        a, b = _meta("a", frames=20), _meta("b", frames=20)
        plan = build_plan(a, b, overlap_frames=5)
        sr = 48000
        pcm = assemble_audio(
            plan, self._full_pcm(a, sr, 0), self._full_pcm(b, sr, 0), sr
        )
        assert pcm == bytearray(len(pcm))

    def test_total_sample_count_identity(self):
        # 采样域同样满足 Nout = Na + Nb - No 的帧映射（按帧边界）
        a, b = _meta("a", frames=33), _meta("b", frames=27)
        plan = build_plan(a, b, overlap_frames=7)
        sr = 48000
        out = assemble_audio(
            plan, self._full_pcm(a, sr, 1), self._full_pcm(b, sr, 2), sr
        )
        bounds = frame_sample_bounds(plan.output_frames, sr, 25, 1)
        assert len(out) == bounds[-1] * _FRAME_BYTES
        assert math.isfinite(plan.output_duration)
