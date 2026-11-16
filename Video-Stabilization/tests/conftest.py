"""pytest 公共夹具：ffmpeg 可用性检测与测试视频生成。

纯逻辑测试不依赖 ffmpeg；需要真实 ffmpeg 的集成测试用
``needs_ffmpeg`` 标记，无 ffmpeg 时自动 skip。
"""

import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

needs_ffmpeg = pytest.mark.skipif(
    not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe 不在 PATH"
)


def make_testsrc(path, size="160x120", frames=24, framerate="12",
                 pix_fmt="yuv420p", audio=False):
    """生成 testsrc2 测试视频（可选 sine 音轨）。"""
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={framerate}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000"]
    cmd += ["-frames:v", str(frames), "-pix_fmt", pix_fmt]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


def make_shaky_video(path, size="160x120", frames=24, framerate="12",
                     shake=6, audio=False):
    """用 ffmpeg crop 周期性平移 testsrc2，合成已知抖动视频。

    画面内容随时间做正弦平移（幅度 ``shake`` 像素），无旋转；
    稳定后逐帧平移量应被显著压制。
    """
    w, h = (int(v) for v in size.split("x"))
    cw, ch = w - 2 * shake, h - 2 * shake
    x_expr = f"{shake}+{shake}*sin(2*PI*t*2)"
    y_expr = f"{shake}+{shake}*cos(2*PI*t*1.5)"
    vf = (
        f"crop={cw}:{ch}:x={x_expr}:y={y_expr},"
        f"pad={w}:{h}:{shake}:{shake}:color=black"
    )
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={framerate}",
        "-vf", vf, "-frames:v", str(frames),
        "-pix_fmt", "yuv420p",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
                "-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


def count_frames(path):
    """ffprobe 实际解码数帧数。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def probe_dims(path):
    """ffprobe 读 (宽, 高)。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = out.stdout.strip().split(",")
    return int(w), int(h)
