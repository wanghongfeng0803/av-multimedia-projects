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


@pytest.fixture
def video_factory(tmp_path):
    """生成测试视频：testsrc2 画面 + 可选 sine 音轨。

    RGB 像素格式用 rawvideo 视频流 + nut 容器（rawvideo rgb24
    只有 nut 能封装），其余默认 libx264 + mp4。
    """
    def make(name="in.mp4", pix_fmt="yuv420p", size="64x48",
             frames=10, audio=True, framerate="10"):
        path = tmp_path / name
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={framerate}",
        ]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000"]
        cmd += ["-frames:v", str(frames), "-pix_fmt", pix_fmt]
        if pix_fmt in ("rgb24", "gbrp", "bgr24"):
            cmd += ["-c:v", "rawvideo"]
        if audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(path)]
        subprocess.run(cmd, check=True)
        return path
    return make


def count_frames(path):
    """ffprobe 实际解码数帧数。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def probe_streams(path):
    """返回 {codec_type: [stream dict, ...]}。"""
    import json
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = {}
    for s in json.loads(out.stdout).get("streams", []):
        streams.setdefault(s["codec_type"], []).append(s)
    return streams
