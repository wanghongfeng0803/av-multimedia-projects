"""中值滤波流水线集成测试：真实 ffmpeg 解码→中值→编码。

无 ffmpeg 时整个模块自动 skip（见 conftest 的 needs_ffmpeg）。
重点：
- 输出帧数与输入严格相等，逐帧 pts 严格对齐；
- 椒盐噪声（黑/白冲激）经中值后被压制；
- 边界转储目录每帧一份 raw + manifest，可单独抽帧复核；
- 均值模式的旧调用方式不受影响。
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
    """ffprobe 逐帧 pts_time。不同容器时间基不同，原始 pts 整数不可
    直接比较，统一用秒级 pts_time 验证时间戳对齐。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return [f.get("pts_time") for f in json.loads(out.stdout)["frames"]]


def make_saltpepper_video(path, frames=6, size="64x48", framerate="10"):
    """灰底上用 geq 注入确定性黑白冲激（椒盐），rawvideo+nut 无损封装，
    避免有损压缩把冲激抹掉导致测试失去意义。输出 yuv420p，亮度平面
    携带冲激，色度保持中性（128）。geq 表达式含逗号，必须整体引号。"""
    luma = (
        "if(gt(mod(X*13+Y*7\\,29)\\,26)\\,255\\,"
        "if(lt(mod(X*11+Y*5\\,23)\\,2)\\,0\\,128))"
    )
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i",
         f"color=c=gray:size={size}:rate={framerate}",
         "-vf", f"format=yuv420p,geq={luma}:cb=128:cr=128",
         "-frames:v", str(frames), "-pix_fmt", "yuv420p",
         "-c:v", "rawvideo", str(path)],
        check=True,
    )
    return path


# ---------- 参数校验 ----------

def test_median_rejects_even_window(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="median", window=4)


def test_unknown_mode_rejected(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="bilateral")


def test_border_dir_requires_median_mode(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    with pytest.raises(ValueError):
        denoise_video(src, tmp_path / "o.mp4", mode="mean",
                      border_dir=str(tmp_path / "b"))


# ---------- 帧数与时间戳严格对齐 ----------

@pytest.mark.parametrize("window", [3, 5])
def test_median_frame_count_and_pts_aligned(video_factory, tmp_path, window):
    src = video_factory(frames=8, audio=False)
    dst = tmp_path / "med.mp4"
    stats = denoise_video(src, dst, mode="median", window=window)
    assert stats["frames"] == 8
    assert stats["mode"] == "median"
    assert stats["window"] == window
    assert count_frames(dst) == 8
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_median_keeps_audio_and_dims(video_factory, tmp_path):
    src = video_factory(frames=6, audio=True)
    dst = tmp_path / "med.mp4"
    stats = denoise_video(src, dst, mode="median", window=3)
    assert stats["has_audio"] is True
    streams = probe_streams(dst)
    assert "audio" in streams
    video = streams["video"][0]
    assert (video["width"], video["height"]) == (64, 48)
    assert pix_fmt_matches(video["pix_fmt"], "yuv420p")
    assert count_frames(dst) == 6


def test_median_rgb_nut_container(video_factory, tmp_path):
    src = video_factory(name="in.nut", pix_fmt="rgb24", frames=5, audio=False)
    dst = tmp_path / "med.mp4"
    stats = denoise_video(src, dst, mode="median", window=3)
    assert stats["pix_fmt"] == "rgb24"
    video = probe_streams(dst)["video"][0]
    assert pix_fmt_matches(video["pix_fmt"], "rgb24")
    assert count_frames(dst) == 5
    assert frame_pts_list(src) == frame_pts_list(dst)


def test_median_tiny_budget_many_bands(video_factory, tmp_path):
    src = video_factory(frames=5, audio=False)
    dst = tmp_path / "med.mp4"
    stats = denoise_video(src, dst, mode="median", window=5,
                          budget_bytes=4096)
    assert stats["frames"] == 5
    assert count_frames(dst) == 5


def test_median_progress_callback(video_factory, tmp_path):
    src = video_factory(frames=6, audio=False)
    calls = []
    denoise_video(src, tmp_path / "med.mp4", mode="median", window=3,
                  progress=calls.append)
    assert calls == [1, 2, 3, 4, 5, 6]


# ---------- 椒盐噪声压制 ----------

def test_median_suppresses_salt_and_pepper(tmp_path):
    src = make_saltpepper_video(tmp_path / "sp.nut")
    # 输入 rawvideo/nut 零量化；输出 ffv1/mkv 无损承载 yuv420p，
    # 只验证中值本身对冲激的压制
    dst = tmp_path / "clean.mkv"
    denoise_video(src, dst, mode="median", window=3)
    before = decode_raw(src, pix_fmt="yuv420p")
    after = decode_raw(dst, pix_fmt="yuv420p")
    assert len(before) == len(after)
    w, h = 64, 48
    before_y, after_y = before[:w * h], after[:w * h]  # 只看亮度平面

    def extreme_ratio(buf):
        return sum(1 for v in buf if v <= 8 or v >= 247) / len(buf)

    before_ratio = extreme_ratio(before_y)
    after_ratio = extreme_ratio(after_y)
    assert before_ratio > 0.01  # 输入确实含足量椒盐冲激
    assert after_ratio < before_ratio * 0.5
    # 灰底区域保持稳定（中值滤波不污染背景）
    background = [v for v in after_y if 100 <= v <= 156]
    assert len(background) / len(after_y) > 0.9


def test_median_flat_input_stays_flat(tmp_path):
    src = tmp_path / "flat.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=gray:size=64x48:rate=10",
         "-frames:v", "5", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    dst = tmp_path / "out.mp4"
    denoise_video(src, dst, mode="median", window=5)
    assert max(decode_raw(dst)) - min(decode_raw(dst)) <= 8


# ---------- 边界抽帧复核 ----------

def test_border_dump_per_frame_and_manifest(video_factory, tmp_path):
    src = video_factory(frames=4, audio=False)
    border_dir = tmp_path / "border"
    stats = denoise_video(src, tmp_path / "med.mp4", mode="median",
                          window=3, border_dir=str(border_dir))
    assert stats["border_dir"] == str(border_dir)
    manifest = json.loads((border_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pix_fmt"] == "yuv420p"
    assert (manifest["width"], manifest["height"]) == (64, 48)
    assert manifest["window"] == 3 and manifest["radius"] == 1
    assert len(manifest["files"]) == 4
    for name in manifest["files"]:
        raw = (border_dir / name).read_bytes()
        # 每帧转储字节数 = 工作格式一帧大小（Y + U + V）
        assert len(raw) == manifest["frame_bytes"]
    # 内点必须被掩码置 0，边界带必须保留滤波后的值（不全为 0）
    first = (border_dir / manifest["files"][0]).read_bytes()
    w, h, r = 64, 48, 1
    interior = first[(r + 1) * w + (r + 1):(h - r - 1) * w]
    assert interior[:w - 2 * (r + 1)].count(0) == len(interior[:w - 2 * (r + 1)])
    assert any(first[:w])  # 顶行边界带非全零


def test_border_dump_rgb_three_planes(video_factory, tmp_path):
    src = video_factory(name="in.nut", pix_fmt="rgb24", frames=3, audio=False)
    border_dir = tmp_path / "border"
    denoise_video(src, tmp_path / "med.mp4", mode="median", window=3,
                  border_dir=str(border_dir))
    manifest = json.loads((border_dir / "manifest.json").read_text(encoding="utf-8"))
    frame_bytes = 64 * 48 * 3
    assert manifest["frame_bytes"] == frame_bytes
    for name in manifest["files"]:
        assert len((border_dir / name).read_bytes()) == frame_bytes


# ---------- 均值模式行为不变 ----------

def test_mean_mode_default_unchanged(video_factory, tmp_path):
    src = video_factory(frames=5, audio=True)
    dst = tmp_path / "mean.mp4"
    stats = denoise_video(src, dst, radius=2)  # 旧调用方式，不带 mode
    assert stats["mode"] == "mean"
    assert stats["window"] is None
    assert count_frames(dst) == 5
    assert "audio" in probe_streams(dst)
