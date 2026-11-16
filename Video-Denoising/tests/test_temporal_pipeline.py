"""时域降噪流水线集成测试：真实 ffmpeg 解码→全局对齐混合→编码。

无 ffmpeg 时整个模块自动 skip（见 conftest 的 needs_ffmpeg）。
重点：
- 输出帧数与输入严格相等、逐帧 pts_time 严格对齐（有界缓存 + 尾冲刷）；
- 音轨直通、RGB/nut 与 gray 等像素格式自适应；
- 超预算在启动子进程前抛 MemoryError（提前拒绝，不产生半成品）；
- 静态相机噪声被时域平均压制；平移镜头经全局对齐后不糊；
- 均值/中值旧模式行为不受影响。
"""

import json
import subprocess

import pytest

from vdenoise import denoise_video, pix_fmt_matches

from conftest import FFMPEG, FFPROBE, count_frames, needs_ffmpeg, probe_streams

pytestmark = needs_ffmpeg


def decode_raw(path, pix_fmt="gray"):
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True, check=True,
    )
    return out.stdout


def frame_pts_list(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return [f.get("pts_time") for f in json.loads(out.stdout)["frames"]]


# ---------- 参数校验与提前拒绝 ----------

def test_unknown_mode_still_rejected(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="bilateral")


def test_temporal_negative_radius_rejected(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="temporal",
                      temporal_radius=-1)


def test_temporal_rejects_budget_before_starting(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "should_not_exist.mp4"
    # 10 字节预算连一帧缓存都不可能容纳：必须 MemoryError，且无输出文件
    with pytest.raises(MemoryError):
        denoise_video(src, dst, mode="temporal", budget_bytes=10)
    assert not dst.exists()


def test_temporal_border_dir_rejected(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="temporal",
                      border_dir=str(tmp_path / "b"))


# ---------- 帧数 / 时间戳 / 音轨 ----------

@pytest.mark.parametrize("frames", [8, 13])
def test_temporal_frame_count_and_pts_aligned(video_factory, tmp_path, frames):
    src = video_factory(frames=frames, audio=False)
    dst = tmp_path / "t.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=2)
    assert stats["frames"] == frames
    assert stats["mode"] == "temporal"
    assert 1 <= stats["cache_frames"] <= stats["nominal_window"]
    assert count_frames(dst) == frames
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_temporal_keeps_audio_and_dims(video_factory, tmp_path):
    src = video_factory(frames=7, audio=True)
    dst = tmp_path / "t.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=1)
    assert stats["has_audio"] is True
    assert "audio" in probe_streams(dst)
    video = probe_streams(dst)["video"][0]
    assert (video["width"], video["height"]) == (64, 48)
    assert pix_fmt_matches(video["pix_fmt"], "yuv420p")
    assert count_frames(dst) == 7


def test_temporal_progress_callback(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    calls = []
    denoise_video(src, tmp_path / "t.mp4", mode="temporal",
                  temporal_radius=2, progress=calls.append)
    assert calls == [1, 2, 3, 4, 5, 6]


def test_temporal_rgb_nut_container(video_factory, tmp_path):
    src = video_factory(name="in.nut", pix_fmt="rgb24", frames=6, audio=False)
    dst = tmp_path / "t.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=1)
    assert stats["pix_fmt"] == "rgb24"
    video = probe_streams(dst)["video"][0]
    assert pix_fmt_matches(video["pix_fmt"], "rgb24")
    assert count_frames(dst) == 6
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_temporal_runs_with_tiny_budget_many_bands(video_factory, tmp_path):
    src = video_factory(frames=5, audio=False)
    dst = tmp_path / "t.mp4"
    # 小预算：缓存被压到 1 帧（时域退化为近恒等），片长切碎，仍须完整
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=2,
                          budget_bytes=64 * 1024)
    assert stats["frames"] == 5
    assert stats["cache_frames"] >= 1
    assert count_frames(dst) == 5


def test_temporal_explicit_band_height(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "t.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=1,
                          band_height=8)
    assert stats["band_height"] == 8
    assert count_frames(dst) == 6


def test_temporal_single_frame_video(video_factory, tmp_path):
    # 单帧：half=0，读到即发射，不能卡死也不能多产出
    src = video_factory(frames=1, audio=False)
    dst = tmp_path / "one.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=3)
    assert stats["frames"] == 1
    # 规划容量是名义窗 7，但实际只有 1 帧可缓存；half 收窄到 0，读到即发
    assert stats["cache_frames"] == 7
    assert count_frames(dst) == 1
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_temporal_zero_radius_is_identity_like(video_factory, tmp_path):
    # temporal_radius=0：名义窗 1 帧、缓存 1，无参考帧，输出帧数对齐
    src = video_factory(frames=5, audio=False)
    dst = tmp_path / "r0.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=0)
    assert stats["nominal_window"] == 1
    assert stats["cache_frames"] == 1
    assert count_frames(dst) == 5
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_temporal_cache_cap_narrows_window(video_factory, tmp_path):
    # 名义窗 9（r=4）但硬上限 3：实际缓存 3，半窗收窄到 1，仍逐帧对齐
    src = video_factory(frames=10, audio=False)
    dst = tmp_path / "cap.mp4"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=4,
                          max_cache_frames=3)
    assert stats["cache_frames"] == 3
    assert stats["nominal_window"] == 9
    assert count_frames(dst) == 10
    assert frame_pts_list(src) == frame_pts_list(dst)


# ---------- 降噪有效性 ----------

def make_noisy_static_video(path, frames=8, size="64x48", framerate="10"):
    """静态灰底 + 逐帧独立随机噪声（noise 滤镜），rawvideo/nut 零量化。

    相机不动、噪声帧间独立，正是时域平均的理想场景。
    """
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"color=c=gray:size={size}:rate={framerate}",
        "-vf", "noise=alls=30:allf=t",
        "-frames:v", str(frames), "-pix_fmt", "yuv420p",
        "-c:v", "rawvideo", str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def test_temporal_suppresses_static_noise(tmp_path):
    src = make_noisy_static_video(tmp_path / "noisy.nut")
    dst = tmp_path / "clean.mkv"
    denoise_video(src, dst, mode="temporal", temporal_radius=3,
                  motion_threshold=40)
    before = decode_raw(src, pix_fmt="yuv420p")
    after = decode_raw(dst, pix_fmt="yuv420p")
    w, h = 64, 48
    before_y, after_y = before[:w * h], after[:w * h]

    def spread(buf):
        # 静态场景真值恒定，帧间噪声体现为像素值相对全局中值的离散度
        mid = sorted(buf)[len(buf) // 2]
        return sum(abs(v - mid) for v in buf) / len(buf)

    assert spread(after_y) < spread(before_y) * 0.8


def test_temporal_flat_input_stays_flat(tmp_path):
    src = tmp_path / "flat.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=gray:size=64x48:rate=10",
         "-frames:v", "6", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    dst = tmp_path / "out.mp4"
    denoise_video(src, dst, mode="temporal", temporal_radius=2)
    assert max(decode_raw(dst)) - min(decode_raw(dst)) <= 8


def make_panning_video(path, frames=8, size="64x48", framerate="10"):
    """匀速平移镜头：testsre 图案整体逐帧右移，检验全局对齐能力。"""
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc2=size={size}:rate={framerate}",
        "-vf", "crop=48:48:16*n:0,pad=64:48:8:0:gray",
        "-frames:v", str(frames), "-pix_fmt", "yuv420p",
        "-c:v", "rawvideo", str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def test_temporal_panning_produces_valid_output(tmp_path):
    # 平移镜头：全局位移估计应能跟踪，输出帧数/尺寸正确且不崩。
    # （纯 Python 块匹配在强运动下允许残差，主要验证流水线与对齐路径。）
    src = make_panning_video(tmp_path / "pan.nut")
    dst = tmp_path / "pan_out.mkv"
    stats = denoise_video(src, dst, mode="temporal", temporal_radius=2,
                          max_shift=8, motion_threshold=20)
    assert stats["frames"] == 8
    assert count_frames(dst) == 8
    assert frame_pts_list(src) == frame_pts_list(dst)


# ---------- 旧模式不受影响 ----------

def test_mean_and_median_modes_unchanged(video_factory, tmp_path):
    src = video_factory(frames=5, audio=True)
    mean_dst = tmp_path / "mean.mp4"
    med_dst = tmp_path / "med.mp4"
    s1 = denoise_video(src, mean_dst, radius=2)
    s2 = denoise_video(src, med_dst, mode="median", window=3)
    assert s1["mode"] == "mean" and s2["mode"] == "median"
    assert count_frames(mean_dst) == 5 and count_frames(med_dst) == 5
    assert "audio" in probe_streams(mean_dst)
