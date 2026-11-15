"""端到端集成测试：真实 ffmpeg 解码 + 完整检测链路。

模块级 pytestmark：环境中缺少 ffmpeg/ffprobe 时整个模块自动 skip，
纯逻辑测试不受影响。
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from scene_seg_tool.cli import main
from scene_seg_tool.detector import detect_scenes
from scene_seg_tool.errors import SceneSegError
from scene_seg_tool.frames import stream_gray_frames
from scene_seg_tool.output import render_boundaries_csv, render_json_report
from scene_seg_tool.probe import probe_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="环境中未安装 ffmpeg / ffprobe",
)


# ---------------------------------------------------------------- 探测与解码

def test_probe_reports_dimensions_fps_and_frame_count(hard_cut_video):
    probe = probe_video(hard_cut_video)
    assert probe.width == 160
    assert probe.height == 120
    assert probe.fps_source in {"avg_frame_rate", "r_frame_rate", "fallback"}
    assert probe.fps == pytest.approx(25.0)
    assert probe.nb_frames == 50


def test_stream_decodes_every_frame_exactly_once(hard_cut_video):
    probe = probe_video(hard_cut_video)
    sizes = []
    for frame in stream_gray_frames(hard_cut_video, probe.width, probe.height):
        assert len(frame) == probe.width * probe.height
        sizes.append(len(frame))
    assert len(sizes) == 50


def test_stream_generator_early_close_kills_process(hard_cut_video):
    probe = probe_video(hard_cut_video)
    stream = stream_gray_frames(hard_cut_video, probe.width, probe.height)
    next(stream)
    stream.close()  # 显式提前关闭，不应遗留进程或抛异常
    # 再跑一次完整检测，证明管道与进程状态干净可重入
    result = detect_scenes(hard_cut_video)
    assert result.frame_count == 50


# ---------------------------------------------------------------- 场景判定

def test_end_to_end_hard_cut(hard_cut_video):
    result = detect_scenes(hard_cut_video)
    assert result.frame_count == 50
    assert len(result.boundaries) == 49
    assert [c.frame_index for c in result.cuts] == [25]
    cut = result.cuts[0]
    assert cut.score > 0.5
    assert cut.time_seconds == pytest.approx(1.0, abs=0.05)


def test_end_to_end_single_frame_video(single_frame_video):
    result = detect_scenes(single_frame_video)
    assert result.frame_count == 1
    assert result.boundaries == []
    assert result.cuts == []
    assert result.threshold_source == "fallback"


def test_end_to_end_long_static_has_no_cuts(static_video):
    result = detect_scenes(static_video)
    assert result.frame_count == 50
    assert result.cuts == []
    assert all(b.decision == "below_threshold" for b in result.boundaries)


def test_end_to_end_flash_is_suppressed_not_cut(flash_video):
    result = detect_scenes(flash_video)
    assert result.cuts == []
    # 闪光在第 25 帧（0 起算），两条边界 25、26 都应被抑制
    assert 25 in result.flash_boundaries
    assert 26 in result.flash_boundaries


def test_end_to_end_flash_disabled_reports_spikes(flash_video):
    result = detect_scenes(flash_video, flash_suppression=False)
    frames = {c.frame_index for c in result.cuts}
    assert {25, 26} <= frames


# ---------------------------------------------------------------- 可复现性

def test_end_to_end_reproducible_outputs(hard_cut_video, work_dir):
    first = detect_scenes(hard_cut_video)
    second = detect_scenes(hard_cut_video)
    assert render_json_report(first) == render_json_report(second)
    assert render_boundaries_csv(first) == render_boundaries_csv(second)

    json_a = os.path.join(work_dir, "a.json")
    json_b = os.path.join(work_dir, "b.json")
    csv_a = os.path.join(work_dir, "a.csv")
    csv_b = os.path.join(work_dir, "b.csv")
    assert main([hard_cut_video, "--json", json_a, "--csv", csv_a, "--quiet"]) == 0
    assert main([hard_cut_video, "--json", json_b, "--csv", csv_b, "--quiet"]) == 0
    with open(json_a, "rb") as fa, open(json_b, "rb") as fb:
        assert fa.read() == fb.read()
    with open(csv_a, "rb") as fa, open(csv_b, "rb") as fb:
        assert fa.read() == fb.read()


def test_end_to_end_json_report_structure_on_disk(hard_cut_video, work_dir):
    out = os.path.join(work_dir, "report.json")
    assert main([hard_cut_video, "--json", out, "--quiet"]) == 0
    with open(out, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["frame_count"] == 50
    assert data["cut_count"] == 1
    assert len(data["boundaries"]) == 49
    assert data["cuts"][0]["frame_index"] == 25


def test_end_to_end_min_scene_length_nms(hard_cut_video):
    # 仅一个切点时 NMS 不改变结果；主要校验参数贯通与结果稳定
    result = detect_scenes(hard_cut_video, min_scene_frames=10)
    assert [c.frame_index for c in result.cuts] == [25]


# ---------------------------------------------------------------- 错误路径

def test_end_to_end_text_disguised_file_fails_cleanly(text_disguised_file):
    with pytest.raises(SceneSegError):
        detect_scenes(text_disguised_file)


def test_end_to_end_truncated_file_does_not_hang(truncated_video):
    # 截断输入必须显式报错（ffmpeg 失败或帧不完整），而不是死锁或静默
    with pytest.raises(SceneSegError):
        detect_scenes(truncated_video)


def test_end_to_end_cli_exit_codes(text_disguised_file, hard_cut_video, work_dir):
    missing = main(["no_such_file_987654.mp4", "--quiet"])
    assert missing == 3
    bad = main([text_disguised_file, "--quiet"])
    assert bad in (4, 6)
    out = os.path.join(work_dir, "r.json")
    ok = main([hard_cut_video, "--json", out, "--quiet"])
    assert ok == 0
