"""空时联合降噪流水线集成测试：真实 ffmpeg 解码→联合处理→编码。

无 ffmpeg 时整个模块自动 skip（见 conftest 的 needs_ffmpeg）。重点：
- 输出帧数与输入严格相等、逐帧 pts_time 严格对齐（有界缓存 + 尾冲刷）；
- 音轨直通、RGB/nut 与 gray 等像素格式自适应（含 4:2:0 色度权重重采样）；
- 超预算在启动子进程前抛 MemoryError（提前拒绝，不产生半成品）；
- 静态噪声下处理后时域/空间 MAD 显著下降，指标字段完整；
- 同输入同参数两次运行输出逐字节一致（可复现）；
- 内存固定：帧数翻倍，缓存/权重图驻留不变（规划结果与片长无关）；
- 均值/中值/时域旧模式行为不受影响。
"""

import json
import subprocess

import pytest

from vdenoise import denoise_video, pix_fmt_matches

from conftest import FFMPEG, FFPROBE, count_frames, needs_ffmpeg, probe_streams

pytestmark = needs_ffmpeg


def decode_raw(path, pix_fmt="yuv420p"):
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

def test_st_negative_radius_rejected(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="spatiotemporal",
                      temporal_radius=-1)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="spatiotemporal",
                      spatial_radius=-1)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="spatiotemporal",
                      mask_radius=-1)


def test_st_rejects_budget_before_starting(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "should_not_exist.mp4"
    with pytest.raises(MemoryError):
        denoise_video(src, dst, mode="spatiotemporal", budget_bytes=10)
    assert not dst.exists()


def test_st_border_dir_rejected(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="spatiotemporal",
                      border_dir=str(tmp_path / "b"))


# ---------- 帧数 / 时间戳 / 音轨 / 像素格式 ----------

@pytest.mark.parametrize("frames", [8, 13])
def test_st_frame_count_and_pts_aligned(video_factory, tmp_path, frames):
    src = video_factory(frames=frames, audio=False)
    dst = tmp_path / "st.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=2)
    assert stats["frames"] == frames
    assert stats["mode"] == "spatiotemporal"
    assert 1 <= stats["cache_frames"] <= stats["nominal_window"]
    assert count_frames(dst) == frames
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_st_keeps_audio_and_dims(video_factory, tmp_path):
    src = video_factory(frames=7, audio=True)
    dst = tmp_path / "st.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=1)
    assert stats["has_audio"] is True
    assert "audio" in probe_streams(dst)
    video = probe_streams(dst)["video"][0]
    assert (video["width"], video["height"]) == (64, 48)
    assert pix_fmt_matches(video["pix_fmt"], "yuv420p")
    assert count_frames(dst) == 7


def test_st_progress_callback(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    calls = []
    denoise_video(src, tmp_path / "st.mp4", mode="spatiotemporal",
                  temporal_radius=2, progress=calls.append)
    assert calls == [1, 2, 3, 4, 5, 6]


def test_st_rgb_nut_container(video_factory, tmp_path):
    src = video_factory(name="in.nut", pix_fmt="rgb24", frames=6, audio=False)
    dst = tmp_path / "st.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=1)
    assert stats["pix_fmt"] == "rgb24"
    video = probe_streams(dst)["video"][0]
    assert pix_fmt_matches(video["pix_fmt"], "rgb24")
    assert count_frames(dst) == 6
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_st_tiny_budget_many_bands(video_factory, tmp_path):
    src = video_factory(frames=5, audio=False)
    dst = tmp_path / "st.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=2,
                          budget_bytes=64 * 1024)
    assert stats["frames"] == 5
    assert stats["cache_frames"] >= 1
    assert count_frames(dst) == 5


def test_st_explicit_band_height(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "st.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=1,
                          band_height=8)
    assert stats["band_height"] == 8
    assert count_frames(dst) == 6


def test_st_single_frame_video(video_factory, tmp_path):
    src = video_factory(frames=1, audio=False)
    dst = tmp_path / "one.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=3)
    assert stats["frames"] == 1
    assert stats["cache_frames"] == 7
    assert count_frames(dst) == 1
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_st_zero_temporal_radius_runs(video_factory, tmp_path):
    # temporal_radius=0：无参考帧，时间级恒等，仅空间均值在工作
    src = video_factory(frames=5, audio=False)
    dst = tmp_path / "r0.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=0)
    assert stats["nominal_window"] == 1
    assert stats["cache_frames"] == 1
    assert count_frames(dst) == 5
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_st_zero_spatial_radius_temporal_only(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst = tmp_path / "nos.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=2,
                          spatial_radius=0)
    assert stats["spatial_radius"] == 0
    assert count_frames(dst) == 6
    assert frame_pts_list(src) == frame_pts_list(dst)


# ---------- 指标与降噪有效性 ----------

def make_noisy_static_video(path, frames=8, size="64x48", framerate="10"):
    """静态灰底 + 逐帧独立随机噪声（noise 滤镜），rawvideo/nut 零量化。"""
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


def test_st_metrics_present_and_improved_on_static_noise(tmp_path):
    src = make_noisy_static_video(tmp_path / "noisy.nut")
    dst = tmp_path / "clean.mkv"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=3,
                          motion_threshold=40)
    m = stats["metrics"]
    # 指标字段完整且取值合理
    for key in ("spatial_mad_before", "spatial_mad_after",
                "temporal_mad_before", "temporal_mad_after",
                "weight_accept_ratio"):
        assert key in m
    assert m["spatial_mad_after"] <= m["spatial_mad_before"]
    assert m["temporal_mad_after"] < m["temporal_mad_before"]
    assert 0.0 <= m["weight_accept_ratio"] <= 1.0
    # 静态相机噪声帧间独立，时域 MAD 应有可观下降
    assert m["temporal_mad_after"] < m["temporal_mad_before"] * 0.8


def test_st_suppresses_static_noise_in_pixels(tmp_path):
    src = make_noisy_static_video(tmp_path / "noisy.nut")
    dst = tmp_path / "clean.mkv"
    denoise_video(src, dst, mode="spatiotemporal", temporal_radius=3,
                  motion_threshold=40)
    before = decode_raw(src, pix_fmt="yuv420p")
    after = decode_raw(dst, pix_fmt="yuv420p")
    w, h = 64, 48
    before_y, after_y = before[:w * h], after[:w * h]

    def spread(buf):
        mid = sorted(buf)[len(buf) // 2]
        return sum(abs(v - mid) for v in buf) / len(buf)

    assert spread(after_y) < spread(before_y) * 0.85


def test_st_flat_input_stays_flat(tmp_path):
    src = tmp_path / "flat.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=gray:size=64x48:rate=10",
         "-frames:v", "6", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    dst = tmp_path / "out.mp4"
    denoise_video(src, dst, mode="spatiotemporal", temporal_radius=2)
    raw = decode_raw(dst)
    assert max(raw) - min(raw) <= 8


def make_panning_video(path, frames=8, size="64x48", framerate="10"):
    """匀速平移镜头：图案整体逐帧右移，检验运动区权重压制与不拖影。"""
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


def test_st_panning_detects_motion_and_stays_aligned(tmp_path):
    src = make_panning_video(tmp_path / "pan.nut")
    dst = tmp_path / "pan_out.mkv"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=2,
                          max_shift=8, motion_threshold=20)
    assert stats["frames"] == 8
    assert count_frames(dst) == 8
    assert frame_pts_list(src) == frame_pts_list(dst)
    # 平移镜头存在大量对齐残差：权重接受率必须明显低于静态画面（<100%），
    # 证明运动区时域贡献被软权重压制，而非无脑平均导致拖影
    assert stats["metrics"]["weight_accept_ratio"] < 0.99


def test_st_gray_single_plane_runs(tmp_path):
    src = tmp_path / "gray.nut"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=gray:size=64x48:rate=10",
         "-vf", "noise=alls=20:allf=t",
         "-frames:v", "6", "-pix_fmt", "gray",
         "-c:v", "rawvideo", str(src)],
        check=True,
    )
    dst = tmp_path / "gray_out.mp4"
    stats = denoise_video(src, dst, mode="spatiotemporal", temporal_radius=2)
    assert stats["frames"] == 6
    assert count_frames(dst) == 6
    assert stats["metrics"]["temporal_mad_after"] \
        <= stats["metrics"]["temporal_mad_before"]


# ---------- 可复现 ----------

def test_st_bit_exact_reproducibility(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    dst1 = tmp_path / "r1.mkv"
    dst2 = tmp_path / "r2.mkv"
    kwargs = dict(mode="spatiotemporal", temporal_radius=2,
                  spatial_radius=1, mask_radius=1, motion_threshold=24)
    s1 = denoise_video(src, dst1, **kwargs)
    s2 = denoise_video(src, dst2, **kwargs)
    assert decode_raw(dst1) == decode_raw(dst2)
    assert s1["metrics"] == s2["metrics"]


# ---------- 内存固定不随片长涨 ----------

def test_st_memory_independent_of_frame_count(video_factory, tmp_path):
    short = video_factory(name="s.mp4", frames=6, audio=False)
    long = video_factory(name="l.mp4", frames=18, audio=False)
    # 显式同参数分别跑，规划口径（缓存帧数/片长/峰值估算）必须完全一致
    r1 = denoise_video(short, tmp_path / "so.mp4", mode="spatiotemporal",
                       temporal_radius=2)
    r2 = denoise_video(long, tmp_path / "lo.mp4", mode="spatiotemporal",
                       temporal_radius=2)
    assert r1["cache_frames"] == r2["cache_frames"]
    assert r1["band_height"] == r2["band_height"]
    assert r1["memory_bytes"] == r2["memory_bytes"]


# ---------- 旧模式不受影响 ----------

def test_legacy_modes_unchanged(video_factory, tmp_path):
    src = video_factory(frames=5, audio=True)
    mean_dst = tmp_path / "mean.mp4"
    med_dst = tmp_path / "med.mp4"
    tmp_dst = tmp_path / "tmp.mp4"
    s1 = denoise_video(src, mean_dst, radius=2)
    s2 = denoise_video(src, med_dst, mode="median", window=3)
    s3 = denoise_video(src, tmp_dst, mode="temporal", temporal_radius=1)
    assert s1["mode"] == "mean" and s2["mode"] == "median"
    assert s3["mode"] == "temporal"
    assert count_frames(mean_dst) == 5
    assert count_frames(med_dst) == 5
    assert count_frames(tmp_dst) == 5
    assert "audio" in probe_streams(mean_dst)
