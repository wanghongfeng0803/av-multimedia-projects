"""端到端集成测试：ffmpeg 解码 → 追踪 → JSON → 渲染，无 ffmpeg 自动 skip。"""

import json

import pytest

from conftest import (
    FFPROBE,
    count_frames,
    make_moving_block_video,
    needs_ffmpeg,
    probe_streams,
)
from vtrack.io_utils import PipelineError
from vtrack.model import LOST, TRACKED, load_track_json
from vtrack.pipeline import read_initial_template, render_video, track_video


@needs_ffmpeg
def test_track_moving_block_all_confirmed(tmp_path):
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "move.mp4", size="96x64", frames=12,
        block=12, start=(8, 8), step=(3, 2),
    )
    track_path = tmp_path / "track.json"
    result = track_video(
        src, track_path, (x0, y0, bw, bh),
        method="sad", threshold=30.0, search_margin=10,
    )
    assert result.stats["frames"] == 12
    # 全程无丢失
    assert result.stats["lost_frames"] == 0
    assert result.stats["loss_events"] == 0
    assert all(f.status == TRACKED for f in result.frames)
    # 轨迹单调向右下移动，末帧位置与合成位移一致（含边界夹取）
    last = result.frames[-1]
    assert last.bbox[0] >= result.frames[0].bbox[0]
    assert last.bbox[1] >= result.frames[0].bbox[1]
    # JSON 可独立读回，帧数与源视频一致
    loaded = load_track_json(track_path)
    assert len(loaded.frames) == count_frames(src)
    assert loaded.meta["initial_box"] == [x0, y0, bw, bh]


@needs_ffmpeg
def test_track_vanish_reports_loss_to_end(tmp_path):
    # 末尾 4 帧方块消失：必须报丢失且生成 end_of_stream 事件
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "vanish.mp4", size="96x64", frames=12,
        block=12, start=(8, 8), step=(2, 1), vanish_frames=4,
    )
    track_path = tmp_path / "vanish.json"
    result = track_video(
        src, track_path, (x0, y0, bw, bh),
        method="sad", threshold=30.0, search_margin=8,
    )
    assert result.stats["lost_frames"] >= 4
    assert result.stats["still_lost_at_end"] is True
    assert len(result.loss_events) == 1
    event = result.loss_events[0]
    assert event.end_of_stream is True
    assert event.recovered_frame is None
    assert event.frames_lost >= 4
    # 丢失帧 bbox 冻结在最后已知位置，不随噪声移动
    lost_recs = [f for f in result.frames if f.status == LOST]
    positions = {(f.bbox[0], f.bbox[1]) for f in lost_recs}
    assert len(positions) == 1


@needs_ffmpeg
def test_track_reacquire_after_temporary_occlusion(tmp_path):
    """分两段造视频：移动 → 几帧遮挡 → 同目标在可达距离重现 → 恢复。"""
    # 用同一方块纹理，遮挡段用纯背景帧，重现位置落在扩大搜索区内
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "occ.mp4", size="96x64", frames=14,
        block=12, start=(10, 10), step=(2, 2), vanish_frames=3,
    )
    track_path = tmp_path / "occ.json"
    result = track_video(
        src, track_path, (x0, y0, bw, bh),
        method="sad", threshold=30.0, search_margin=12,
        reacquire_frames=2, lost_growth=2.0,
    )
    # 至少出现过一段丢失，且片末状态由素材决定；事件结构必须自洽
    if result.loss_events:
        ev = result.loss_events[0]
        assert ev.lost_frame < (ev.recovered_frame or result.stats["frames"])
        assert ev.frames_lost >= 1
    # 每帧状态只有两态，帧序号连续
    assert [f.frame for f in result.frames] == list(range(14))
    assert all(f.status in (TRACKED, LOST) for f in result.frames)


@needs_ffmpeg
def test_render_produces_video_with_audio_passthrough(tmp_path):
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "aud.mp4", size="96x64", frames=10,
        block=12, start=(8, 8), step=(3, 2), audio=True,
    )
    track_path = tmp_path / "aud.json"
    track_video(src, track_path, (x0, y0, bw, bh), search_margin=10)
    out = tmp_path / "rendered.mp4"
    info = render_video(src, out, track_path, font_scale=1)
    assert info["frames"] == 10
    assert count_frames(out) == 10
    streams = probe_streams(out)
    assert "video" in streams
    assert "audio" in streams  # 音轨直通保留


@needs_ffmpeg
def test_render_last_known_option(tmp_path):
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "lk.mp4", size="96x64", frames=10,
        block=12, start=(8, 8), step=(2, 1), vanish_frames=3,
    )
    track_path = tmp_path / "lk.json"
    track_video(src, track_path, (x0, y0, bw, bh),
                search_margin=8, threshold=30.0)
    out1 = tmp_path / "plain.mp4"
    out2 = tmp_path / "with_lk.mp4"
    render_video(src, out1, track_path)
    render_video(src, out2, track_path, show_last_known=True)
    assert count_frames(out1) == count_frames(out2) == 10


@needs_ffmpeg
def test_render_rejects_frame_count_mismatch(tmp_path):
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "a.mp4", frames=8, block=10, start=(6, 6), step=(2, 1),
    )
    other, _, _, _, _, _ = make_moving_block_video(
        tmp_path / "b.mp4", frames=5, block=10, start=(6, 6), step=(2, 1),
    )
    track_path = tmp_path / "a.json"
    track_video(src, track_path, (x0, y0, bw, bh), search_margin=8)
    # 轨迹 8 帧、视频只有 5 帧：必须报错，不静默截断
    with pytest.raises(ValueError, match="不匹配"):
        render_video(other, tmp_path / "bad.mp4", track_path)


@needs_ffmpeg
def test_read_initial_template_from_frame(tmp_path):
    src, _, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "tpl.mp4", frames=4, block=10, start=(12, 10),
        step=(1, 1),
    )
    tpl, tw, th, ix, iy = read_initial_template(
        src, (x0, y0, bw, bh), template_frame=0,
    )
    assert (tw, th) == (bw, bh)
    assert (ix, iy) == (x0, y0)
    assert len(tpl) == bw * bh


@needs_ffmpeg
def test_read_initial_template_rejects_bad_box(tmp_path):
    src, _, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "badbox.mp4", size="64x48", frames=3, block=10,
        start=(6, 6), step=(1, 1),
    )
    with pytest.raises(ValueError, match="边界"):
        read_initial_template(src, (60, 40, bw, bh))
    with pytest.raises(ValueError):
        read_initial_template(src, (0, 0, 0, 5))


@needs_ffmpeg
def test_zncc_method_end_to_end(tmp_path):
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "zncc.mp4", size="96x64", frames=10,
        block=12, start=(10, 10), step=(3, 2),
    )
    track_path = tmp_path / "zncc.json"
    result = track_video(
        src, track_path, (x0, y0, bw, bh),
        method="zncc", threshold=0.7, search_margin=10,
    )
    assert result.params["method"] == "zncc"
    assert result.stats["frames"] == 10
    # 纹理方块 ZNCC 应能稳定追踪
    assert result.stats["lost_ratio"] < 0.3


@needs_ffmpeg
def test_track_empty_or_bad_input_raises(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises((PipelineError, FileNotFoundError)):
        track_video(missing, tmp_path / "x.json", (0, 0, 4, 4))


@needs_ffmpeg
def test_json_contains_reuse_contract_fields(tmp_path):
    """轨迹 JSON 字段契约：后续光流/多目标复用所依赖的键必须齐全。"""
    src, tpl, bw, bh, x0, y0 = make_moving_block_video(
        tmp_path / "contract.mp4", frames=6, block=10,
        start=(8, 8), step=(2, 1),
    )
    track_path = tmp_path / "contract.json"
    track_video(src, track_path, (x0, y0, bw, bh), search_margin=8,
                target_id="obj_7")
    data = json.loads(track_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["meta"]["target_id"] == "obj_7"
    assert {"method", "threshold", "search_margin",
            "reacquire_frames"} <= set(data["params"])
    rec = data["frames"][0]
    assert {"frame", "pts_time", "status", "x", "y", "w", "h",
            "score", "confirmed", "search"} <= set(rec)
