"""detector.py 内存帧编排与流式评分器的纯逻辑测试（无 ffmpeg 依赖）。"""

from __future__ import annotations

import pytest

from scene_seg_tool.detector import (
    StreamingHistogramScorer,
    detect_scenes_in_frames,
)
from scene_seg_tool.errors import InvalidInputError
from scene_seg_tool.output import render_json_report

WIDTH, HEIGHT, FPS = 4, 4, 25.0
PIXELS = WIDTH * HEIGHT


def frame(value: int) -> bytes:
    """整帧统一灰度的常量帧。"""
    return bytes([value] * PIXELS)


# ---------------------------------------------------------------- 评分器

def test_scorer_single_frame_has_no_boundaries():
    scorer = StreamingHistogramScorer()
    scorer.push(frame(100))
    distances, returns, pixel_count = scorer.snapshot()
    assert scorer.frame_count == 1
    assert distances == []
    assert returns == []
    assert pixel_count == PIXELS


def test_scorer_emits_distance_and_three_frame_return():
    scorer = StreamingHistogramScorer()
    scorer.push(frame(0))      # 帧 0：A
    scorer.push(frame(0))      # 帧 1：A
    scorer.push(frame(255))    # 帧 2：B
    distances, returns, _ = scorer.snapshot()
    assert len(distances) == 2
    assert distances[0] == 0
    assert distances[1] == 2 * PIXELS          # 完全不重叠，取 L1 上界
    # 边界 1 的回返距离 |h[0]-h[2]|；最后一条边界无后继帧，保持 -1
    assert returns[0] == 2 * PIXELS
    assert returns[1] == -1


def test_scorer_three_frame_return_zero_when_middle_is_flash():
    scorer = StreamingHistogramScorer()
    scorer.push(frame(30))
    scorer.push(frame(255))   # 闪光帧
    scorer.push(frame(30))
    _, returns, _ = scorer.snapshot()
    # 进入边界 1 跳过闪光帧后的回返距离为 0
    assert returns[0] == 0
    assert returns[1] == -1


# ---------------------------------------------------------------- 单帧 / 静止

def test_pipeline_single_frame_is_valid_empty_result():
    result = detect_scenes_in_frames([frame(50)], WIDTH, HEIGHT, FPS)
    assert result.frame_count == 1
    assert result.boundaries == []
    assert result.cuts == []
    assert result.threshold_source == "fallback"


def test_pipeline_long_static_video_no_cuts():
    frames = [frame(80)] * 50
    result = detect_scenes_in_frames(frames, WIDTH, HEIGHT, FPS)
    assert result.frame_count == 50
    assert len(result.boundaries) == 49
    assert all(b.score == 0.0 for b in result.boundaries)
    assert result.cuts == []
    assert result.threshold_source == "fallback"


# ---------------------------------------------------------------- 硬切 / 闪光

def test_pipeline_detects_hard_cut_at_exact_boundary():
    frames = [frame(30)] * 12 + [frame(220)] * 12
    result = detect_scenes_in_frames(frames, WIDTH, HEIGHT, FPS)
    assert len(result.cuts) == 1
    cut = result.cuts[0]
    assert cut.frame_index == 12
    assert cut.score == 1.0
    assert cut.time_seconds == pytest.approx(12 / FPS)
    # 真实硬切的三帧回返距离很大，不能被当作闪光
    assert result.flash_boundaries == ()
    assert all(b.decision != "flash" for b in result.boundaries)


def test_pipeline_single_flash_frame_suppresses_both_spikes():
    frames = [frame(30)] * 12 + [frame(255)] + [frame(30)] * 12
    result = detect_scenes_in_frames(frames, WIDTH, HEIGHT, FPS)
    assert result.cuts == []
    assert set(result.flash_boundaries) == {12, 13}
    decisions = {b.frame_index: b.decision for b in result.boundaries}
    assert decisions[12] == "flash"
    assert decisions[13] == "flash"


def test_pipeline_flash_suppression_can_be_disabled():
    frames = [frame(30)] * 12 + [frame(255)] + [frame(30)] * 12
    result = detect_scenes_in_frames(
        frames, WIDTH, HEIGHT, FPS, flash_suppression=False
    )
    assert {c.frame_index for c in result.cuts} == {12, 13}
    assert result.flash_boundaries == ()


def test_pipeline_min_scene_length_nms_keeps_earlier_on_tie():
    # 暗→亮（3 帧）→暗：两条等距切点相隔 3 帧，关闭闪光抑制后做 NMS
    frames = [frame(30)] * 10 + [frame(220)] * 3 + [frame(30)] * 10
    result = detect_scenes_in_frames(
        frames, WIDTH, HEIGHT, FPS,
        flash_suppression=False, min_scene_frames=5,
    )
    assert [c.frame_index for c in result.cuts] == [10]
    assert set(result.suppressed_boundaries) == {13}


def test_pipeline_decision_labels_cover_all_boundaries():
    frames = [frame(30)] * 12 + [frame(255)] + [frame(30)] * 12
    result = detect_scenes_in_frames(frames, WIDTH, HEIGHT, FPS)
    labels = {b.decision for b in result.boundaries}
    assert labels <= {"cut", "flash", "suppressed", "below_threshold"}
    assert "flash" in labels and "below_threshold" in labels


# ---------------------------------------------------------------- 错误输入

def test_pipeline_empty_frame_raises():
    with pytest.raises(InvalidInputError):
        detect_scenes_in_frames([], WIDTH, HEIGHT, FPS)


def test_pipeline_wrong_frame_size_raises():
    bad = bytes([0] * (PIXELS - 1))
    with pytest.raises(InvalidInputError):
        detect_scenes_in_frames([frame(0), bad], WIDTH, HEIGHT, FPS)


# ---------------------------------------------------------------- 可复现

def test_pipeline_results_are_byte_for_byte_reproducible():
    frames = [frame(30)] * 12 + [frame(255)] + [frame(30)] * 12
    first = detect_scenes_in_frames(
        frames, WIDTH, HEIGHT, FPS, input_path="clip.mp4"
    )
    second = detect_scenes_in_frames(
        frames, WIDTH, HEIGHT, FPS, input_path="clip.mp4"
    )
    assert render_json_report(first) == render_json_report(second)
