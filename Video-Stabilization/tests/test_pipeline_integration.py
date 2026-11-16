"""端到端集成测试：真实 ffmpeg 解码/补偿/裁剪，无 ffmpeg 时自动 skip。"""

import json
import subprocess

from stabilize import (
    MOTION_FORMAT,
    SMOOTH_FORMAT,
    compensate_video,
    crop_video,
    dump_json,
    estimate_motion,
    load_json,
    probe,
    smooth_motion,
)

from conftest import (
    FFPROBE,
    count_frames,
    make_shaky_video,
    make_testsrc,
    needs_ffmpeg,
    probe_dims,
)


# ---------------------------------------------------------- 估计阶段

@needs_ffmpeg
def test_estimate_motion_contract(tmp_path):
    src = make_testsrc(tmp_path / "in.mp4")
    motion_path = tmp_path / "motion.json"
    doc = estimate_motion(src, motion_path)

    # 文件落盘且可回读，契约字段齐全
    on_disk = load_json(motion_path)
    assert on_disk == doc
    assert doc["format"] == MOTION_FORMAT and doc["version"] == 1
    assert doc["width"] == 160 and doc["height"] == 120
    assert doc["frame_count"] == 24
    assert len(doc["frames"]) == 24
    # 第 0 帧恒等基准
    assert doc["frames"][0]["dx"] == 0.0
    assert doc["frames"][0]["dy"] == 0.0
    assert doc["frames"][0]["angle"] == 0.0
    # 每条目字段固定（稳定输出格式）
    expected = {"index", "dx", "dy", "angle", "confidence",
                "valid", "scene_cut"}
    for f in doc["frames"]:
        assert set(f) == expected
    # 原文件未被修改：仍可探测且帧数不变
    assert count_frames(src) == 24


@needs_ffmpeg
def test_estimate_detects_shaky_translation(tmp_path):
    src = make_shaky_video(tmp_path / "shaky.mp4", shake=8)
    doc = estimate_motion(src, tmp_path / "motion.json")
    # 合成抖动有真实位移：至少一半帧对的位移幅度明显非零
    moved = sum(
        1 for f in doc["frames"][1:]
        if abs(f["dx"]) + abs(f["dy"]) > 0.3
    )
    assert moved >= len(doc["frames"]) // 2
    # testsrc2 连续帧无场景切换
    assert not any(f["scene_cut"] for f in doc["frames"])


@needs_ffmpeg
def test_estimate_handles_audio_stream(tmp_path):
    src = make_testsrc(tmp_path / "a.mp4", audio=True)
    info = probe(src)
    assert info["has_audio"] is True
    doc = estimate_motion(src, tmp_path / "m.json")
    assert doc["has_audio"] is True


# ---------------------------------------------------------- 平滑阶段

@needs_ffmpeg
def test_smooth_writes_versioned_json(tmp_path):
    src = make_shaky_video(tmp_path / "s.mp4")
    motion = estimate_motion(src, tmp_path / "motion.json")
    smooth = smooth_motion(motion, radius=4)
    smooth_path = tmp_path / "smooth.json"
    dump_json(smooth, smooth_path)

    loaded = load_json(smooth_path)
    assert loaded["format"] == SMOOTH_FORMAT and loaded["version"] == 1
    assert len(loaded["frames"]) == motion["frame_count"]
    # 裁剪量为正且不超过短边 45%
    w, h = motion["width"], motion["height"]
    assert 0 <= loaded["crop"]["inset"] <= min(w, h) * 0.45 + 0.01


@needs_ffmpeg
def test_smoothed_compensation_reduces_motion_variance(tmp_path):
    src = make_shaky_video(tmp_path / "s.mp4", shake=8, frames=36)
    motion = estimate_motion(src, tmp_path / "m.json")
    smooth = smooth_motion(motion, radius=6)

    def var(seq):
        m = sum(seq) / len(seq)
        return sum((v - m) ** 2 for v in seq) / len(seq)

    raw_x = [f["trajectory"]["x"] for f in smooth["frames"]]
    sm_x = [f["smoothed"]["x"] for f in smooth["frames"]]
    # 平滑轨迹的逐帧抖动方差显著低于原始轨迹
    assert var(sm_x) < var(raw_x) * 0.6


# ---------------------------------------------------------- 补偿阶段

@needs_ffmpeg
def test_compensate_produces_valid_video(tmp_path):
    src = make_shaky_video(tmp_path / "s.mp4")
    motion = estimate_motion(src, tmp_path / "m.json")
    smooth = smooth_motion(motion, radius=4)
    out = tmp_path / "padded.mp4"
    info = compensate_video(smooth, src, out, work_dir=tmp_path / "work")

    assert out.exists()
    assert probe_dims(out) == (160, 120)       # 画幅不变
    assert count_frames(out) == 24             # 帧数不变
    # sendcmd 每帧一条命令
    cmd_text = open(info["sendcmd"], encoding="utf-8").read()
    enter_lines = [ln for ln in cmd_text.splitlines() if "[enter]" in ln]
    assert len(enter_lines) == 24


@needs_ffmpeg
def test_compensate_preserves_audio(tmp_path):
    src = make_testsrc(tmp_path / "a.mp4", audio=True)
    motion = estimate_motion(src, tmp_path / "m.json")
    smooth = smooth_motion(motion, radius=3)
    out = tmp_path / "padded.mp4"
    compensate_video(smooth, src, out, work_dir=tmp_path / "work")

    out_info = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in out_info.stdout


@needs_ffmpeg
def test_crop_step_outputs_original_dimensions(tmp_path):
    src = make_shaky_video(tmp_path / "s.mp4")
    motion = estimate_motion(src, tmp_path / "m.json")
    smooth = smooth_motion(motion, radius=4)
    padded = tmp_path / "padded.mp4"
    final = tmp_path / "final.mp4"
    compensate_video(smooth, src, padded, work_dir=tmp_path / "work")
    info = crop_video(smooth, padded, final)

    assert final.exists()
    assert probe_dims(final) == (160, 120)
    assert count_frames(final) == 24
    # crop 取样框用整数像素，inset 做四舍五入
    assert info["inset"] == round(smooth["crop"]["inset"])


@needs_ffmpeg
def test_crop_reuses_same_smooth_json(tmp_path):
    # 同一份 smooth JSON 支持两个不同 inset 的裁剪产物（复用契约）
    src = make_shaky_video(tmp_path / "s.mp4")
    motion = estimate_motion(src, tmp_path / "m.json")
    smooth = smooth_motion(motion, radius=4)
    padded = tmp_path / "padded.mp4"
    compensate_video(smooth, src, padded, work_dir=tmp_path / "work")

    out1 = tmp_path / "c1.mp4"
    out2 = tmp_path / "c2.mp4"
    crop_video(smooth, padded, out1)
    crop_video(smooth, padded, out2,
               inset=max(1.0, smooth["crop"]["inset"] + 2))
    assert out1.exists() and out2.exists()
    assert probe_dims(out1) == probe_dims(out2) == (160, 120)


# ---------------------------------------------------------- CLI 串联

@needs_ffmpeg
def test_cli_run_end_to_end(tmp_path):
    src = make_shaky_video(tmp_path / "s.mp4")
    out = tmp_path / "stable.mp4"
    # 直接调用 main 入口，等价于命令行 run 子命令
    rc = cli_main(["run", str(src), str(out), "--radius", "4"])
    assert rc == 0
    assert out.exists()
    assert probe_dims(out) == (160, 120)
    assert count_frames(out) == 24


@needs_ffmpeg
def test_cli_step_by_step(tmp_path):
    src = make_shaky_video(tmp_path / "s.mp4")
    motion_json = tmp_path / "motion.json"
    smooth_json = tmp_path / "smooth.json"
    padded = tmp_path / "padded.mp4"
    final = tmp_path / "final.mp4"

    assert cli_main(["estimate", str(src), str(motion_json)]) == 0
    assert cli_main(["smooth", str(motion_json), str(smooth_json),
                     "--radius", "4"]) == 0
    assert cli_main(["compensate", str(src), str(smooth_json), str(padded),
                     "--work-dir", str(tmp_path / "work")]) == 0
    assert cli_main(["crop", str(padded), str(smooth_json),
                     str(final)]) == 0
    assert final.exists() and count_frames(final) == 24

    # 输出 JSON 均为版本化稳定格式
    assert load_json(motion_json)["format"] == MOTION_FORMAT
    assert load_json(smooth_json)["format"] == SMOOTH_FORMAT


@needs_ffmpeg
def test_cli_wrong_format_exits_arg_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    out = tmp_path / "o.json"
    rc = cli_main(["smooth", str(bad), str(out)])
    assert rc == 2


# ---------------------------------------------------------- 辅助

def cli_main(argv):
    """延迟导入 CLI 入口，避免纯逻辑测试依赖命令行模块。"""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from stabilize_cli import main  # noqa: E402
    return main(argv)
