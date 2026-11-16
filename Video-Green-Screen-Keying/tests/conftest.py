"""pytest 公共夹具：ffmpeg 可用性检测与测试素材生成。

纯逻辑测试不依赖 ffmpeg；需要真实 ffmpeg 的集成测试用
``needs_ffmpeg`` 标记，无 ffmpeg 时自动 skip。
"""

import json
import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

needs_ffmpeg = pytest.mark.skipif(
    not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe 不在 PATH"
)


def make_rgb_frame(width, height, painter):
    """生成 rgb24 裸帧字节；``painter(x, y) -> (r, g, b)`` 逐像素定色。"""
    buf = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            r, g, b = painter(x, y)
            buf[i] = r
            buf[i + 1] = g
            buf[i + 2] = b
            i += 3
    return bytes(buf)


def green_screen_frame(width=16, height=16, subject_box=(4, 4, 11, 11),
                       screen=(0, 200, 0), subject=(200, 30, 30)):
    """合成一帧绿幕 rgb24：幕布色打底，矩形区域放前景主体。"""
    x0, y0, x1, y1 = subject_box

    def paint(x, y):
        if x0 <= x < x1 and y0 <= y < y1:
            return subject
        return screen

    return make_rgb_frame(width, height, paint)


@pytest.fixture
def video_factory(tmp_path):
    """生成测试视频：lavfi 纯色/测试源 + 可选 sine 音轨。"""
    def make(name="in.mp4", source=None, size="64x48", frames=10,
             framerate="10", audio=False, pix_fmt="yuv420p"):
        path = tmp_path / name
        cmd = [FFMPEG, "-y", "-v", "error",
               "-f", "lavfi", "-i", source or f"testsrc2=size={size}:rate={framerate}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000"]
        cmd += ["-frames:v", str(frames), "-pix_fmt", pix_fmt]
        if audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(path)]
        subprocess.run(cmd, check=True)
        return path
    return make


@pytest.fixture
def color_video_factory(tmp_path):
    """生成整帧纯色视频（用于确定性的键色采样/抠像集成测试）。"""
    def make(color, name="green.mp4", size="64x48", frames=6,
             framerate="10", audio=False):
        r, g, b = color
        path = tmp_path / name
        source = f"color=c=0x{r:02x}{g:02x}{b:02x}:s={size}:r={framerate}"
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", source]
        if audio:
            cmd += ["-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=8000"]
        cmd += ["-frames:v", str(frames), "-pix_fmt", "yuv420p"]
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
