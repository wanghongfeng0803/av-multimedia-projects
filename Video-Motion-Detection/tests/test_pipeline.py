"""端到端流水线测试：真实 ffmpeg 解码/编码，无 ffmpeg 时自动 skip。

覆盖需求主线：运动检出与事件落盘、全局光照突变被抑制、局部光照变化
能检出、标记视频与事件 JSON 逐帧对照、无运动时退出码 2。
"""

import pytest

from vmotion.cli import main
from vmotion.detector import MotionConfig
from vmotion.io_utils import probe
from vmotion.model import load_detection_json
from vmotion.pipeline import detect_video
from vmotion.render import render_video

from conftest import count_frames, make_motion_video, needs_ffmpeg


@needs_ffmpeg
def test_detect_motion_video_produces_events(tmp_path):
    src = make_motion_video(
        tmp_path / "motion.mp4", size="96x64", frames=16,
        motion_range=(3, 11), step=(6, 0),
    )
    out_json = tmp_path / "det.json"
    result = detect_video(src, out_json, MotionConfig(
        block=8, min_area=32, open_radius=0, close_radius=0,
        trigger_frames=2, release_frames=2,
    ))
    assert out_json.exists()
    assert result.stats["events"] >= 1
    assert result.stats["motion_frames"] > 0
    # 逐帧记录数与视频帧数一致
    assert count_frames(src) == len(result.frames)
    # JSON 可回读且帧号连续
    loaded = load_detection_json(out_json)
    assert [f.frame for f in loaded.frames] == list(range(len(loaded.frames)))
    # 事件帧区间内每帧 event_id 一致；至少一帧带区域几何
    # （迟滞保持帧允许为空区域，但状态仍属于该事件）
    ev = loaded.events[0]
    span = [f for f in loaded.frames
            if ev["start_frame"] <= f.frame < ev["end_frame"]]
    assert span and all(f.event_id == ev["event_id"] for f in span)
    assert any(f.regions and "x" in f.regions[0] for f in span)


@needs_ffmpeg
def test_global_illumination_change_suppressed_e2e(tmp_path):
    src = make_motion_video(
        tmp_path / "illum.mp4", size="64x64", frames=14,
        motion_range=(0, 0),          # 全程无物体移动
        illum_frame=6, illum_delta=80,
    )
    result = detect_video(src, tmp_path / "d.json", MotionConfig(
        block=8, min_area=16, illum_change_threshold=25.0,
        illum_hold_frames=2, open_radius=0, close_radius=0,
    ))
    # 全局突变帧被标记 holdoff，且全程无运动事件
    held = [f for f in result.frames if f.stats.get("held_off")]
    assert held, "应至少有一帧被光照突变抑制"
    assert result.stats["events"] == 0
    assert result.stats["motion_frames"] == 0


@needs_ffmpeg
def test_local_illumination_detected_e2e(tmp_path):
    # 静止背景 + 一块逐帧移动的局部亮斑（车灯模型），无全局均值跳变
    src = make_motion_video(
        tmp_path / "local.mp4", size="96x64", frames=12,
        motion_range=(2, 9), block=12, step=(7, 0),
        illum_frame=None,
    )
    result = detect_video(src, tmp_path / "d.json", MotionConfig(
        block=8, local_radius=1, gain=2.0, min_area=24,
        illum_change_threshold=25.0, open_radius=0, close_radius=0,
    ))
    assert result.stats["events"] >= 1
    # 没有任何帧触发全局 holdoff
    assert result.stats["held_off_frames"] == 0


@needs_ffmpeg
def test_mark_video_matches_json_frame_by_frame(tmp_path):
    src = make_motion_video(
        tmp_path / "m.mp4", size="96x64", frames=14,
        motion_range=(3, 10),
    )
    jpath = tmp_path / "d.json"
    detect_video(src, jpath, MotionConfig(block=8, min_area=32,
                                          open_radius=0, close_radius=0))
    out = tmp_path / "marked.mp4"
    info = render_video(src, out, jpath, font_scale=1)
    assert out.exists()
    assert info["frames"] == count_frames(src)
    probed = probe(out)
    assert (probed["width"], probed["height"]) == (96, 64)


@needs_ffmpeg
def test_cli_detect_exit_codes_and_mark(tmp_path):
    # 无运动视频 → 退出码 2，但 JSON 仍落盘
    static = make_motion_video(
        tmp_path / "static.mp4", size="64x64", frames=8,
        motion_range=(0, 0),
    )
    jpath = tmp_path / "static.json"
    rc = main(["detect", str(static), str(jpath)])
    assert rc == 2
    assert jpath.exists()

    moving = make_motion_video(
        tmp_path / "moving.mp4", size="96x64", frames=14,
        motion_range=(2, 10),
    )
    j2 = tmp_path / "moving.json"
    rc = main(["detect", str(moving), str(j2)])
    assert rc == 0
    marked = tmp_path / "marked.mp4"
    rc = main(["mark", str(moving), str(j2), str(marked)])
    assert rc == 0
    assert marked.exists()


@needs_ffmpeg
def test_render_rejects_frame_count_mismatch(tmp_path):
    src = make_motion_video(tmp_path / "a.mp4", size="64x64", frames=8,
                            motion_range=(2, 5))
    jpath = tmp_path / "d.json"
    detect_video(src, jpath)
    other = make_motion_video(tmp_path / "b.mp4", size="64x64", frames=10,
                              motion_range=(2, 5))
    with pytest.raises(ValueError, match="不匹配"):
        render_video(other, tmp_path / "x.mp4", jpath)
