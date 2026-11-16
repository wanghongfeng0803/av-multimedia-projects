"""pytest 公共夹具：ffmpeg 可用性检测与测试视频生成。

纯逻辑测试（钳制 / 网格 / 打码）不依赖 ffmpeg；需要真实 ffmpeg 的
集成测试用 ``needs_ffmpeg`` 标记，无 ffmpeg 时自动 skip，与各兄弟
项目的 conftest 口径一致。
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
    """生成 CFR 测试视频：testsrc2 画面，指定帧率与帧数。"""
    def make(name="in.mp4", size="64x48", frames=24, fps=24,
             pix_fmt="yuv420p", with_audio=False):
        path = tmp_path / name
        cmd = [FFMPEG, "-y", "-v", "error"]
        if with_audio:
            cmd += [
                "-f", "lavfi",
                "-i", f"testsrc2=size={size}:rate={fps}",
                "-f", "lavfi", "-i", "sine=frequency=440",
            ]
        else:
            cmd += [
                "-f", "lavfi",
                "-i", f"testsrc2=size={size}:rate={fps}",
            ]
        cmd += ["-frames:v", str(frames), "-pix_fmt", pix_fmt, str(path)]
        subprocess.run(cmd, check=True)
        return path
    return make


def decode_rgb_frame(path, index, width, height):
    """用 ffmpeg 抽出第 ``index`` 帧的 rgb24 原始字节。"""
    out = subprocess.run(
        [
            FFMPEG, "-v", "error", "-i", str(path),
            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, check=True,
    )
    assert len(out.stdout) == width * height * 3
    return out.stdout


def probe_video(path):
    """ffprobe 读取首路视频流的尺寸 / 帧率 / 像素格式 / 帧数。"""
    import json
    out = subprocess.run(
        [
            FFPROBE, "-v", "error",
            "-show_entries",
            "stream=codec_type,width,height,pix_fmt,avg_frame_rate,"
            "nb_frames,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return video, has_audio
