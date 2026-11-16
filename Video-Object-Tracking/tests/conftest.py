"""pytest 公共夹具：ffmpeg 可用性检测、合成帧与测试视频生成。

纯逻辑测试（模板匹配、状态机、数据契约、绘图字节）不依赖 ffmpeg；
需要真实 ffmpeg 的端到端测试用 ``needs_ffmpeg`` 标记，无 ffmpeg 时
自动 skip。
"""

import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

needs_ffmpeg = pytest.mark.skipif(
    not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe 不在 PATH"
)


# ---------------------------------------------------------------- 合成帧
def make_plane(width, height, fill=0):
    """生成纯灰填充平面（bytearray，便于测试改写）。"""
    return bytearray([fill]) * (width * height)


def put_rect(plane, width, x, y, rw, rh, value):
    """在平面上画实心矩形（模拟目标块）。"""
    for row in range(y, y + rh):
        base = row * width + x
        for col in range(rw):
            plane[base + col] = value


def put_pattern(plane, width, x, y, rw, rh, seed=1):
    """画带内部纹理的矩形：棋盘 + 梯度，保证 ZNCC 方差非零且可区分。"""
    for row in range(rh):
        for col in range(rw):
            v = (40 + ((row + seed) * 7 + (col + seed) * 13) % 160
                 + (30 if (row + col) % 2 == 0 else 0))
            plane[(y + row) * width + (x + col)] = min(255, v)


def extract(plane, width, x, y, rw, rh):
    """抠取紧凑窗口 bytes。"""
    rows = []
    for r in range(y, y + rh):
        base = r * width + x
        rows.append(bytes(plane[base:base + rw]))
    return b"".join(rows)


@pytest.fixture
def frame_factory():
    """返回一组合成帧工具函数。"""
    return {
        "make_plane": make_plane,
        "put_rect": put_rect,
        "put_pattern": put_pattern,
        "extract": extract,
    }


# ---------------------------------------------------------------- 测试视频
@pytest.fixture
def video_factory(tmp_path):
    """生成测试视频：testsrc2 画面 + 可选 sine 音轨。"""
    def make(name="in.mp4", size="64x48", frames=10,
             audio=False, framerate="10", pix_fmt="yuv420p"):
        path = tmp_path / name
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size={size}:rate={framerate}",
        ]
        if audio:
            cmd += ["-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=8000"]
        cmd += ["-frames:v", str(frames), "-pix_fmt", pix_fmt]
        if audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(path)]
        subprocess.run(cmd, check=True)
        return path
    return make


def make_moving_block_video(path, size="96x64", frames=12, framerate="10",
                            block=12, start=(10, 10), step=(4, 2),
                            vanish_frames=0, audio=False):
    """用 lavfi 拼出移动方块视频较繁琐，改为 rawvideo 管道直接喂帧。

    亮方块在暗背景上逐帧平移；``vanish_frames>0`` 时末尾若干帧把方块
    抹掉（模拟遮挡/跟丢）。返回 (路径, 模板 bytes, 模板宽, 高, 初始 x, y)。
    """
    w, h = (int(v) for v in size.split("x"))
    bw, bh = block, block
    x0, y0 = start
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", size,
        "-framerate", framerate, "-i", "-",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=8000", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    template = None
    for i in range(frames):
        plane = make_plane(w, h, fill=20)
        vanished = vanish_frames > 0 and i >= frames - vanish_frames
        if not vanished:
            x = x0 + step[0] * i
            y = y0 + step[1] * i
            x = min(x, w - bw)
            y = min(y, h - bh)
            put_pattern(plane, w, x, y, bw, bh, seed=3)
            if i == 0:
                template = extract(plane, w, x, y, bw, bh)
        proc.stdin.write(bytes(plane))
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg 编码失败（退出码 {rc}）")
    return path, template, bw, bh, x0, y0


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
