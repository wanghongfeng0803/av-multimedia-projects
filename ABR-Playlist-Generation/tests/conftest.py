"""pytest 公共夹具：ffmpeg 可用性检测与测试媒体生成。

纯逻辑测试（时间轴 / 播放列表）不依赖 ffmpeg；需要真实 ffmpeg 的
集成测试用 ``needs_ffmpeg`` 标记，环境缺 ffmpeg/ffprobe 时自动 skip，
与各兄弟项目的 conftest 口径一致。
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
    """生成 CFR 测试视频：testsrc2 画面，指定帧率/帧数，可选正弦音轨。"""
    def make(name="in.mp4", size="64x48", frames=150, fps=25,
             with_audio=False):
        path = tmp_path / name
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate={fps}",
        ]
        if with_audio:
            cmd += [
                "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=44100",
                "-shortest",
            ]
        cmd += ["-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(cmd, check=True)
        return path
    return make
