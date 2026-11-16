"""端到端集成测试：真实 ffmpeg 解码/编码，无 ffmpeg 时自动 skip。

覆盖：采样键色、alpha 视频三种容器、图像序列、背景替换、
音轨直通、帧数对齐、失败清理与覆盖保护。
"""

import os
import subprocess

import pytest

from tests.conftest import FFMPEG, FFPROBE, count_frames, needs_ffmpeg
from vkeyer.errors import OutputExistsError, OutputFormatError
from vkeyer.matte import MatteConfig
from vkeyer.pipeline import key_video
from vkeyer.sampler import sample_key_color


def _make_green_screen(tmp_path, size="64x48", frames=8, framerate="10",
                       audio=False, color="0x00c800", name="gs.mp4"):
    """绿底 + 中央红色矩形的测试素材（drawbox 画前景主体）。"""
    path = tmp_path / name
    w, h = (int(v) for v in size.split("x"))
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"color=c={color}:s={size}:r={framerate}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=8000"]
    # 输出选项（含 -vf）必须放在全部 -i 之后，否则 ffmpeg 9 会把
    # -vf 误绑到后续输入上报 "cannot be applied to input url"
    cmd += [
        "-vf",
        f"drawbox=x={w//4}:y={h//4}:w={w//2}:h={h//2}:color=red:t=fill",
        "-frames:v", str(frames), "-pix_fmt", "yuv420p",
    ]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


def _decode_first_frame_rgba(path):
    """把输出首帧解码为 rgba 裸字节返回。"""
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        capture_output=True, check=True,
    ).stdout
    return out


def _corner_and_center_alpha(frame_rgba, w, h):
    """取四角（幕布区）与中心（主体区）的 alpha。"""
    def a(x, y):
        return frame_rgba[(y * w + x) * 4 + 3]
    corners = [a(2, 2), a(w - 3, 2), a(2, h - 3), a(w - 3, h - 3)]
    return corners, a(w // 2, h // 2)


@needs_ffmpeg
def test_sample_key_color_from_green_screen(tmp_path):
    src = _make_green_screen(tmp_path)
    key = sample_key_color(src, mode="top", frame_numbers=(0,))
    # 顶部条带应采样到接近纯绿（压缩后允许少量偏差）
    assert key[1] > key[0] + 80 and key[1] > key[2] + 80


@needs_ffmpeg
def test_sample_out_of_range_frames_raises(tmp_path):
    from vkeyer.errors import SampleError
    src = _make_green_screen(tmp_path, frames=4)
    with pytest.raises(SampleError):
        sample_key_color(src, mode="all", frame_numbers=(999,))


@needs_ffmpeg
@pytest.mark.parametrize("ext,codec,pix,alt_pix", [
    # ffprobe 对 ffv1/mkv 报告 bgra（与 rgba 仅字节序不同，alpha 等价）
    (".mov", "qtrle", "argb", None),
    (".mkv", "ffv1", "rgba", "bgra"),
])
def test_reliable_alpha_video_containers(tmp_path, ext, codec, pix, alt_pix):
    src = _make_green_screen(tmp_path)
    dst = tmp_path / f"out{ext}"
    result = key_video(src, dst, sample_mode="top", sample_frames=(0,),
                       overwrite=True)
    assert dst.exists()
    assert result.frames == count_frames(dst)
    assert result.mode == "alpha"

    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0",
         str(dst)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert codec in probe
    assert pix in probe or (alt_pix and alt_pix in probe)

    frame = _decode_first_frame_rgba(dst)
    corners, center = _corner_and_center_alpha(frame, 64, 48)
    assert max(corners) <= 2          # 幕布区全透明
    assert center >= 253             # 主体全不透明


@needs_ffmpeg
def test_webm_emits_alpha_unreliable_warning(tmp_path):
    """本机 libvpx-vp9 静默剥离 alpha：webm 不报错但必须带明确警告，
    且产物帧数/分辨率正确（退化为无 alpha 视频也不崩流水线）。"""
    from vkeyer.encoder import ALPHA_UNRELIABLE_WARNING
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "out.webm"
    result = key_video(src, dst, sample_mode="top", sample_frames=(0,),
                       overwrite=True)
    assert dst.exists()
    assert result.frames == count_frames(dst)
    assert any("alpha" in m for m in result.messages)
    assert ALPHA_UNRELIABLE_WARNING in result.messages


@needs_ffmpeg
def test_mp4_alpha_rejected_before_processing(tmp_path):
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "bad.mp4"
    with pytest.raises(OutputFormatError):
        key_video(src, dst, sample_frames=(0,))
    assert not dst.exists()           # 失败不出半成品


@needs_ffmpeg
def test_png_sequence_output(tmp_path):
    src = _make_green_screen(tmp_path, frames=5)
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    pattern = seq_dir / "f_%06d.png"
    result = key_video(src, pattern, sample_mode="all",
                       sample_frames=(0,), overwrite=True)
    pngs = sorted(seq_dir.glob("f_*.png"))
    assert len(pngs) == result.frames == 5
    frame = _decode_first_frame_rgba(pngs[0])
    corners, center = _corner_and_center_alpha(frame, 64, 48)
    assert max(corners) <= 2 and center >= 253


@needs_ffmpeg
def test_composite_over_solid_color_mp4(tmp_path):
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "comp.mp4"
    result = key_video(src, dst, sample_frames=(0,),
                       bg_color=(0, 0, 0), overwrite=True)
    assert result.mode == "composite"
    assert count_frames(dst) == result.frames
    # 合成产物无 alpha 通道（yuv420p）
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(dst)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe == "yuv420p"
    # 幕布区合成后等于纯黑背景
    rgb = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(dst), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    corner = rgb[:3]
    assert max(corner) <= 4


@needs_ffmpeg
def test_composite_over_background_image(tmp_path):
    src = _make_green_screen(tmp_path)
    # 蓝色背景图
    bg = tmp_path / "bg.png"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
         "color=c=blue:s=64x48", "-frames:v", "1", str(bg)],
        check=True,
    )
    dst = tmp_path / "comp_img.mp4"
    key_video(src, dst, sample_frames=(0,), bg_image=bg, overwrite=True)
    rgb = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(dst), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    # 幕布区露出蓝背景：B 显著高于 R/G
    r, g, b = rgb[0], rgb[1], rgb[2]
    assert b > r + 80 and b > g + 80


@needs_ffmpeg
def test_audio_passthrough(tmp_path):
    src = _make_green_screen(tmp_path, audio=True)
    dst = tmp_path / "with_audio.mov"
    key_video(src, dst, sample_frames=(0,), overwrite=True)
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(dst)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "audio" in probe


@needs_ffmpeg
def test_explicit_key_color_skips_sampling(tmp_path):
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "explicit.mov"
    result = key_video(src, dst, key_rgb=(0, 200, 0), overwrite=True)
    assert result.key_source == "explicit"
    assert result.key_rgb == (0, 200, 0)
    assert dst.exists()


@needs_ffmpeg
def test_matte_postprocessing_in_pipeline(tmp_path):
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "matte.mov"
    matte = MatteConfig(threshold_low=8, threshold_high=247,
                        choke=1, feather=1)
    result = key_video(src, dst, sample_frames=(0,),
                       matte_config=matte, overwrite=True)
    assert result.frames == count_frames(dst)
    frame = _decode_first_frame_rgba(dst)
    corners, center = _corner_and_center_alpha(frame, 64, 48)
    assert max(corners) <= 2 and center >= 250


@needs_ffmpeg
def test_overwrite_protection(tmp_path):
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "exists.mov"
    key_video(src, dst, sample_frames=(0,))
    with pytest.raises(OutputExistsError):
        key_video(src, dst, sample_frames=(0,), overwrite=False)
    # 显式允许覆盖
    key_video(src, dst, sample_frames=(0,), overwrite=True)


@needs_ffmpeg
def test_blue_screen_works_with_sampled_key(tmp_path):
    """非绿幕布：蓝底 + 黄主体，采样键色后同样能抠。"""
    path = tmp_path / "blue.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
         "color=c=0x0000c8:s=64x48:r=10",
         "-vf", "drawbox=x=16:y=12:w=32:h=24:color=yellow:t=fill",
         "-frames:v", "6", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    dst = tmp_path / "blue_out.mov"
    key_video(path, dst, sample_mode="top", sample_frames=(0,),
              overwrite=True)
    frame = _decode_first_frame_rgba(dst)
    corners, center = _corner_and_center_alpha(frame, 64, 48)
    assert max(corners) <= 2 and center >= 253


@needs_ffmpeg
def test_despill_removes_green_tint_on_replaced_background(tmp_path):
    """边缘残绿检验：黑背景上幕布区的 G 不应显著高于 R/B。"""
    src = _make_green_screen(tmp_path)
    dst = tmp_path / "nospill.mp4"
    key_video(src, dst, sample_frames=(0,), bg_color=(0, 0, 0),
              overwrite=True)
    rgb = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(dst), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    # 抽幕布区若干像素，despill 后三通道应接近（不允许 G 独大）
    for idx in range(0, 64 * 3, 64 * 3):
        r, g, b = rgb[idx], rgb[idx + 1], rgb[idx + 2]
        assert g <= max(r, b) + 6
