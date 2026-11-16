"""pytest 公共夹具：ffmpeg 可用性检测与合成帧/测试视频生成。

纯逻辑测试（形态学、检测状态机、事件、数据契约、绘图字节）不依赖
ffmpeg；需要真实 ffmpeg 的端到端测试用 ``needs_ffmpeg`` 标记，
无 ffmpeg 时自动 skip。
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
    """在平面上画实心矩形。"""
    for row in range(y, y + rh):
        base = row * width + x
        for col in range(rw):
            plane[base + col] = value


def count_frames(path):
    """ffprobe 实际解码数帧数。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def make_motion_video(path, size="96x64", frames=14, framerate="10",
                      bg=30, block=16, start=(10, 10), step=(6, 0),
                      motion_range=(3, 10), illum_frame=None,
                      illum_delta=80, local_light_frame=None,
                      local_light_rect=None, audio=False):
    """用 rawvideo 管道合成运动测试视频。

    - ``motion_range``：方块移动的帧号区间（含端点），其余帧静止；
    - ``illum_frame``：该帧起整帧亮度叠加 ``illum_delta``（全局突变）；
    - ``local_light_frame``：该帧在 ``local_light_rect`` 区域内提亮
      （模拟车灯/阴影等局部光照变化，区域外不动）。

    返回视频路径。
    """
    w, h = (int(v) for v in size.split("x"))
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
    bx, by = start
    for i in range(frames):
        plane = make_plane(w, h, fill=bg)
        if motion_range[0] <= i <= motion_range[1]:
            k = i - motion_range[0]
            x = min(bx + step[0] * k, w - block)
            y = min(by + step[1] * k, h - block)
            put_rect(plane, w, x, y, block, block, min(255, bg + 120))
        else:
            put_rect(plane, w, bx, by, block, block, min(255, bg + 120))
        if illum_frame is not None and i >= illum_frame:
            for p in range(len(plane)):
                plane[p] = min(255, plane[p] + illum_delta)
        if local_light_frame is not None and i >= local_light_frame \
                and local_light_rect is not None:
            lx, ly, lw, lh = local_light_rect
            put_rect(plane, w, lx, ly, lw, lh,
                     min(255, bg + 90))
        proc.stdin.write(bytes(plane))
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg 编码失败（退出码 {rc}）")
    return path
