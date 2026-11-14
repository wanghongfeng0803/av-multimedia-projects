"""分段规划与 concat 列表生成的单元测试（不启动 ffmpeg）。"""

from __future__ import annotations

import os
import subprocess

import pytest

from transcode_tool.segments import (
    MAX_SEGMENTS,
    Segment,
    cleanup_workdir,
    concat_segments,
    make_temp_workdir,
    plan_segments,
    segment_output_path,
    write_concat_list,
)


def test_single_segment_when_disabled():
    plan = plan_segments(100.0)
    assert plan.count == 1
    assert plan.segments[0] == Segment(0, 0.0, 100.0)
    assert plan.warnings == ()


def test_unknown_duration_falls_back_to_single():
    plan = plan_segments(None, segment_duration=10.0)
    assert plan.count == 1
    assert plan.warnings and "单段" in plan.warnings[0]


@pytest.mark.parametrize("duration", [0, -5.0])
def test_non_positive_duration_falls_back(duration):
    plan = plan_segments(duration, segment_count=3)
    assert plan.count == 1


def test_segment_count_math_covers_full_duration():
    plan = plan_segments(100.0, segment_count=4)
    assert plan.count == 4
    # 首尾相接、无重叠、无缺口
    assert plan.segments[0].start == 0.0
    assert plan.segments[-1].end == pytest.approx(100.0)
    for prev, nxt in zip(plan.segments, plan.segments[1:]):
        assert nxt.start == pytest.approx(prev.end)
    # 时长近似均匀
    durations = [s.duration for s in plan.segments]
    assert max(durations) - min(durations) <= 0.01


def test_segment_duration_rounds_up():
    # 100s / 30s → 4 段（向上取整，不能丢掉尾部）
    plan = plan_segments(100.0, segment_duration=30.0)
    assert plan.count == 4
    assert plan.segments[-1].end == pytest.approx(100.0)


def test_segment_duration_below_minimum_floor():
    plan = plan_segments(10.0, segment_duration=0.01)
    # 最小段长 1s → 10 段，而非 1000 段
    assert plan.count == 10


def test_segment_cap_raises_for_count():
    with pytest.raises(ValueError, match="不能超过"):
        plan_segments(1000.0, segment_count=MAX_SEGMENTS + 1)


def test_segment_duration_overflow_auto_caps():
    # 段长极小导致段数超上限：自动收敛到 MAX_SEGMENTS 并给出警告
    plan = plan_segments(1e9, segment_duration=1.0)
    assert plan.count == MAX_SEGMENTS
    assert plan.warnings and "上限" in plan.warnings[0]


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_segment_count(bad):
    with pytest.raises(ValueError):
        plan_segments(100.0, segment_count=bad)


def test_invalid_segment_duration():
    with pytest.raises(ValueError):
        plan_segments(100.0, segment_duration=-10.0)


def test_segment_count_rejects_bool_and_float():
    with pytest.raises(ValueError):
        plan_segments(100.0, segment_count=True)  # type: ignore[arg-type]


def test_span_ratio_weighted_by_duration():
    plan = plan_segments(100.0, segment_count=4)
    total = sum(
        plan.span_ratio(s, 100.0) for s in plan.segments
    )
    assert total == pytest.approx(1.0)


def test_temp_workdir_lifecycle():
    workdir = make_temp_workdir()
    assert os.path.isdir(workdir)
    try:
        path = segment_output_path(workdir, Segment(2, 10.0, 5.0), ".webm")
        assert path.endswith("segment_0002.webm")
        with open(path, "wb") as fh:
            fh.write(b"x")
    finally:
        cleanup_workdir(workdir)
    assert not os.path.exists(workdir)


def test_cleanup_missing_dir_is_noop():
    cleanup_workdir(os.path.join("nonexistent", "path"))  # 不抛异常


def test_concat_list_uses_relative_names_and_escapes_quotes(tmp_path):
    seg_dir = tmp_path
    names = ["segment_0000.mp4", "segment_0001.mp4", "it's tricky.mp4"]
    paths = [str(seg_dir / n) for n in names]
    list_path = write_concat_list(str(seg_dir), paths)
    content = open(list_path, encoding="utf-8").read()
    lines = [ln for ln in content.splitlines() if ln]
    assert lines[0] == "file 'segment_0000.mp4'"
    assert lines[1] == "file 'segment_0001.mp4'"
    # 单引号按 concat 语法转义
    assert "'\\''" in lines[2]
    # 剔除合法的单引号转义序列后，不得残留任何路径分隔符或盘符冒号
    stripped = content.replace("'\\''", "")
    assert ":" not in stripped
    assert "\\" not in stripped
    assert "/" not in stripped


def test_concat_segments_failure_raises(monkeypatch, tmp_path):
    """拼接命令失败时抛出受控的 FFmpegRunError。"""
    from transcode_tool.errors import FFmpegRunError

    list_path = write_concat_list(str(tmp_path), [str(tmp_path / "a.mp4")])

    class FakeProc:
        returncode = 1
        stderr = "concat failed badly"
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    with pytest.raises(FFmpegRunError, match="拼接失败"):
        concat_segments(list_path, str(tmp_path / "out.mp4"))
