"""背景建模与双模式对比的端到端测试：真实 ffmpeg 解码，无则自动 skip。

覆盖需求主线：背景模式产出同构检测 JSON、预热帧落盘、静止物体吸收后
事件闭合、对比命令一次解码产出逐帧不一致清单、标记视频能直接消费
背景模式 JSON。
"""

import json
import subprocess

import pytest

from vmotion.background import BackgroundConfig
from vmotion.cli import main
from vmotion.comparison import compare_video
from vmotion.io_utils import find_tool
from vmotion.model import load_detection_json
from vmotion.pipeline import detect_background_video
from vmotion.render import render_video

from conftest import (
    count_frames,
    make_motion_video,
    make_plane,
    needs_ffmpeg,
    put_rect,
)


def _make_appearing_object_video(path, size="64x64", frames=40,
                                 framerate="10", bg=30,
                                 appear_frame=4, block=12,
                                 rect_pos=(10, 10)):
    """合成「预热后入场并长期停留」的物体序列（rawvideo 管道编码）。

    conftest 的 make_motion_video 让物体从首帧就在画面里，会被预热
    直接吸收为初始背景；本序列前 ``appear_frame`` 帧只有空背景，
    之后物体在固定位置出现并不再移动，专用于检验停留吸收。
    """
    w, h = (int(v) for v in size.split("x"))
    proc = subprocess.Popen(
        [find_tool("ffmpeg"), "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", size,
         "-framerate", framerate, "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    x, y = rect_pos
    for i in range(frames):
        plane = make_plane(w, h, fill=bg)
        if i >= appear_frame:
            put_rect(plane, w, x, y, block, block, min(255, bg + 120))
        proc.stdin.write(bytes(plane))
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg 编码失败（退出码 {rc}）")
    return path


def _bg_cfg(**over):
    base = dict(block=8, min_area=16, open_radius=0, close_radius=0,
                trigger_frames=2, release_frames=2,
                warmup_frames=3, learning_rate=0.2, absorb_frames=6)
    base.update(over)
    return BackgroundConfig(**base)


@needs_ffmpeg
def test_background_detect_proceses_all_frames_and_writes_json(tmp_path):
    src = make_motion_video(
        tmp_path / "motion.mp4", size="96x64", frames=18,
        motion_range=(5, 14), step=(6, 0),
    )
    out_json = tmp_path / "bg.json"
    result = detect_background_video(src, out_json, _bg_cfg())
    assert out_json.exists()
    assert count_frames(src) == len(result.frames)
    # 预热帧数被汇总进整段统计，且 meta 标注了背景模式
    assert result.stats["warmup_frames"] == 4        # 首帧 + warmup_frames=3
    assert result.meta["mode"] == "background"
    loaded = load_detection_json(out_json)
    assert [f.frame for f in loaded.frames] == list(range(len(loaded.frames)))
    # 预热帧全部非运动态
    assert all(not f.motion for f in loaded.frames[:4])
    assert all(f.stats["warmup"] for f in loaded.frames[:4])


@needs_ffmpeg
def test_stationary_object_is_absorbed_in_pipeline(tmp_path):
    """物体预热后入场并停留：先产生事件，吸收后事件闭合，片尾不再报警。"""
    src = _make_appearing_object_video(
        tmp_path / "appear.mp4", frames=40, appear_frame=4)
    result = detect_background_video(
        src, tmp_path / "bg.json",
        _bg_cfg(warmup_frames=3, absorb_frames=6, learning_rate=0.3,
                min_area=8))
    # 发生过块吸收，且片尾若干帧已不再判运动（事件已闭合而非挂到片末）
    assert result.stats["absorbing_frames"] > 0
    tail = result.frames[-5:]
    assert all(not f.motion for f in tail)
    if result.events:
        assert not result.events[-1].end_of_stream


@needs_ffmpeg
def test_compare_video_outputs_disagreement_report(tmp_path):
    src = make_motion_video(
        tmp_path / "cmp.mp4", size="96x64", frames=20,
        motion_range=(2, 15), step=(6, 0),
    )
    out_json = tmp_path / "compare.json"
    payload = compare_video(
        src, out_json,
        frame_config=None,
        bg_config=_bg_cfg(warmup_frames=6),
    )
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["schema_version"] == payload["schema_version"]
    assert data["frames"] == 20
    # 两种模式各自的事件数与逐帧对照表都在
    assert set(data["events"].keys()) == {"frame_diff", "background"}
    assert len(data["per_frame"]) == 20
    assert 0.0 <= data["verdict_agreement"] <= 1.0
    # 背景预热帧必然出现在分歧或双方静默的对照中
    warmup_rows = [row for row in data["per_frame"][:7]
                   if not row["background_motion"]]
    assert warmup_rows


@needs_ffmpeg
def test_cli_background_detect_and_compare_exit_codes(tmp_path):
    moving = make_motion_video(
        tmp_path / "moving.mp4", size="96x64", frames=16,
        motion_range=(3, 11),
    )
    bg_json = tmp_path / "bg.json"
    rc = main(["detect", str(moving), str(bg_json), "--background",
               "--warmup-frames", "2", "--min-area", "16",
               "--open-radius", "0", "--close-radius", "0"])
    assert rc in (0, 2)
    assert bg_json.exists()

    cmp_json = tmp_path / "cmp.json"
    rc = main(["compare", str(moving), str(cmp_json),
               "--warmup-frames", "2"])
    assert rc == 0                    # compare 成功恒为 0
    assert cmp_json.exists()


@needs_ffmpeg
def test_mark_consumes_background_json(tmp_path):
    """标记模块不感知检测模式：背景 JSON 可直接渲染且帧数对齐。"""
    src = make_motion_video(
        tmp_path / "m.mp4", size="96x64", frames=14,
        motion_range=(4, 10),
    )
    jpath = tmp_path / "bg.json"
    detect_background_video(
        src, jpath, _bg_cfg(warmup_frames=2, min_area=16))
    out = tmp_path / "marked.mp4"
    info = render_video(src, out, jpath, font_scale=1)
    assert out.exists()
    assert info["frames"] == count_frames(src)
