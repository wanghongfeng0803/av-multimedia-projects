"""ffmpeg 解码 → 均值滤波 → ffmpeg 编码 流水线。

- 解码：``ffmpeg -i 输入 -f rawvideo -pix_fmt <工作格式> -``，逐帧读原始字节；
- 滤波：每帧按像素格式拆平面，各平面按内存预算分带处理
  （长视频内存恒定 O(单帧)，单帧峰值 O(预算)）；
- 编码：第二路 ``-i 输入`` 供音轨 ``-c:a copy`` 直通，
  无音轨时 ``-map 1:a:0?`` 的可选标记自动忽略；
- 两个子进程的 stderr 一律 DEVNULL：置为 PIPE 而不消费会在
  长视频上撑满管道缓冲区造成死锁。
"""

import json
import shutil
import subprocess

from .bands import box_blur_plane_banded, plan_bands
from .memory import derive_budget
from .pixfmt import (
    encoder_for,
    filter_dims,
    frame_size,
    join_planes,
    split_planes,
    working_format,
)

__all__ = ["probe", "denoise_video", "PipelineError"]


class PipelineError(RuntimeError):
    """流水线子进程失败或数据流异常。"""


def _find(tool):
    path = shutil.which(tool)
    if not path:
        raise PipelineError(f"找不到 {tool}，请确认已安装并在 PATH 中")
    return path


def probe(path):
    """ffprobe 读取首路视频流信息及是否含音轨。

    返回 ``{"width", "height", "pix_fmt", "fps", "has_audio"}``。
    """
    ffprobe = _find("ffprobe")
    out = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries",
            "stream=codec_type,width,height,pix_fmt,avg_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    video = None
    has_audio = False
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and video is None:
            video = s
        elif s.get("codec_type") == "audio":
            has_audio = True
    if video is None:
        raise PipelineError(f"{path} 中没有视频流")
    fps = video.get("avg_frame_rate") or "25"
    if fps in ("0/0", ""):
        fps = "25"
    return {
        "width": video["width"],
        "height": video["height"],
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "fps": fps,
        "has_audio": has_audio,
    }


def _read_exact(stream, size):
    """读满 size 字节；流结束时未读到任何字节返回 None，读了一半报错。"""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if chunks:
                raise PipelineError("解码输出在帧中途截断")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def denoise_video(src, dst, radius=2, budget_bytes=None, progress=None):
    """对 ``src`` 做均值滤波降噪，结果写入 ``dst``，音轨原样保留。

    ``radius`` 为滤波半径（窗口 (2r+1)²）；``budget_bytes`` 为单帧
    分带内存预算，缺省按可用内存推导；``progress`` 若给出，每处理完
    一帧以已处理帧数调用一次。返回处理统计 dict。
    """
    if radius < 0:
        raise ValueError("radius 不能为负")
    ffmpeg = _find("ffmpeg")
    info = probe(src)
    work = working_format(info["pix_fmt"])
    codec, out_pix = encoder_for(work)
    w, h = info["width"], info["height"]
    if budget_bytes is None:
        budget_bytes = derive_budget()

    dims = filter_dims(work, w, h)
    band_plans = [plan_bands(ph, pw, radius, budget_bytes) for pw, ph in dims]
    fsize = frame_size(work, w, h)

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", work, "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    enc_cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", work, "-s", f"{w}x{h}",
        "-framerate", info["fps"], "-i", "-",
    ]
    if info["has_audio"]:
        enc_cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "copy"]
    enc_cmd += ["-c:v", codec]
    if out_pix:
        enc_cmd += ["-pix_fmt", out_pix]
    enc_cmd += [str(dst)]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)

    frames = 0
    try:
        while True:
            buf = _read_exact(dec.stdout, fsize)
            if buf is None:
                break
            planes, _ = split_planes(buf, work, w, h)
            out_planes = [
                box_blur_plane_banded(plane, pw, ph, radius, band_plans[i])
                for i, (plane, (pw, ph)) in enumerate(zip(planes, dims))
            ]
            try:
                enc.stdin.write(join_planes(out_planes, work, w, h))
            except (BrokenPipeError, OSError) as exc:
                raise PipelineError(f"编码器异常退出：{exc}") from exc
            frames += 1
            if progress:
                progress(frames)
    finally:
        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except OSError:
            pass
    dec.stdout.close()
    dec_rc = dec.wait()
    enc_rc = enc.wait()
    if dec_rc != 0:
        raise PipelineError(f"ffmpeg 解码失败（退出码 {dec_rc}）")
    if enc_rc != 0:
        raise PipelineError(f"ffmpeg 编码失败（退出码 {enc_rc}）")
    return {
        "frames": frames,
        "pix_fmt": work,
        "width": w,
        "height": h,
        "has_audio": info["has_audio"],
    }
