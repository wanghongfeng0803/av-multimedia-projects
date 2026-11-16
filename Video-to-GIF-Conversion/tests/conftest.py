"""pytest 公共夹具：ffmpeg 可用性检测与测试视频生成。

纯逻辑测试（时序 / 量化 / 缩放 / LZW / GIF 字节）不依赖 ffmpeg；
需要真实 ffmpeg 的集成测试用 ``needs_ffmpeg`` 标记，无 ffmpeg 时
自动 skip，与各兄弟项目的 conftest 口径一致。
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
             pix_fmt="yuv420p"):
        path = tmp_path / name
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate={fps}",
            "-frames:v", str(frames),
            "-pix_fmt", pix_fmt,
            str(path),
        ]
        subprocess.run(cmd, check=True)
        return path
    return make


def make_plain_video(path, size="64x48", frames=24, fps=24,
                     color="red"):
    """生成纯色测试视频（量化/时序断言用，画面内容确定）。"""
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi",
        "-i", f"color=c={color}:s={size}:r={fps}",
        "-frames:v", str(frames),
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path
