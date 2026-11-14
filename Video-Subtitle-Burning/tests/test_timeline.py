"""帧时间轴与字幕调度器测试：重点钉住半开区间与边界帧语义。"""

from __future__ import annotations

import pytest

from subburn.srt import Cue
from subburn.timeline import SubtitleScheduler, frame_time_ms


def make_cue(start_ms: int, end_ms: int, text: str = "x") -> Cue:
    return Cue(index=1, start_ms=start_ms, end_ms=end_ms, text=text)


class TestFrameTime:
    def test_linear_axis(self):
        assert frame_time_ms(0, 25.0) == 0.0
        assert frame_time_ms(25, 25.0) == pytest.approx(1000.0)
        assert frame_time_ms(1, 1000.0) == pytest.approx(1.0)

    def test_fractional_fps_no_drift(self):
        # 23.976fps 非整数帧率：前 1000 帧的时间应严格线性，无累积偏差
        fps = 24000 / 1001
        assert frame_time_ms(1000, fps) == pytest.approx(1000 * 1000.0 / fps)


class TestScheduler:
    def test_half_open_start_inclusive(self):
        sch = SubtitleScheduler([make_cue(1000, 2000)], fps=25.0)
        # 帧 25 恰好是 1000ms：起点包含，必须显示
        assert sch.active_cues_at(25) != []

    def test_half_open_end_exclusive(self):
        sch = SubtitleScheduler([make_cue(1000, 2000)], fps=25.0)
        sch.active_cues_at(25)  # 推进到区间内
        # 帧 50 恰好是 2000ms：终点排除，必须不显示
        assert sch.active_cues_at(50) == []

    def test_adjacent_cues_never_coincide_on_boundary(self):
        # 两条首尾相接的字幕在边界帧上只能看到后者，不能同帧闪现
        cues = [make_cue(0, 2000, "a"), make_cue(2000, 4000, "b")]
        sch = SubtitleScheduler(cues, fps=25.0)
        sch.active_cues_at(0)
        at_boundary = sch.active_cues_at(50)
        assert [c.text for c in at_boundary] == ["b"]

    def test_overlapping_cues_both_active(self):
        cues = [make_cue(1000, 3000, "a"), make_cue(2000, 4000, "b")]
        sch = SubtitleScheduler(cues, fps=25.0)
        # 2500ms = 帧 62.5 附近，取帧 63（2520ms），两条都在场
        active = sch.active_cues_at(63)
        assert {c.text for c in active} == {"a", "b"}

    def test_enter_and_exit_once(self):
        # 全区间扫描，统计每条字幕的帧数，验证入场/出场各一次
        cues = [make_cue(0, 1000, "a"), make_cue(2000, 3000, "b")]
        sch = SubtitleScheduler(cues, fps=10.0)
        counts = {"a": 0, "b": 0}
        for i in range(50):  # 0..4900ms
            for c in sch.active_cues_at(i):
                counts[c.text] += 1
        # a: [0,1000) -> 帧 0..9 共 10 帧；b: [2000,3000) -> 帧 20..29
        assert counts == {"a": 10, "b": 10}

    def test_unsorted_cues_handled(self):
        cues = [make_cue(3000, 4000, "late"), make_cue(0, 1000, "early")]
        sch = SubtitleScheduler(cues, fps=10.0)
        assert sch.active_cues_at(0)[0].text == "early"
        for _ in range(3):
            sch.active_cues_at(sch._last_frame + 1)
        assert sch.active_cues_at(35)[0].text == "late"

    def test_monotonic_required(self):
        sch = SubtitleScheduler([make_cue(0, 5000)], fps=25.0)
        sch.active_cues_at(10)
        with pytest.raises(ValueError):
            sch.active_cues_at(9)

    def test_negative_frame_rejected(self):
        sch = SubtitleScheduler([make_cue(0, 1000)], fps=25.0)
        with pytest.raises(ValueError):
            sch.active_cues_at(-1)

    def test_bad_fps(self):
        with pytest.raises(ValueError):
            SubtitleScheduler([], fps=0)

    def test_no_cues(self):
        sch = SubtitleScheduler([], fps=25.0)
        assert sch.active_cues_at(0) == []
        assert sch.active_cues_at(100) == []
