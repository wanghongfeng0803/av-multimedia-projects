"""ffmpeg 解码 → 空间滤波（均值/中值）→ ffmpeg 编码 流水线。

- 解码：``ffmpeg -i 输入 -f rawvideo -pix_fmt <工作格式> -``，逐帧读原始字节；
- 滤波：每帧按像素格式拆平面，各平面按内存预算分带处理
  （长视频内存恒定 O(单帧)，单帧峰值 O(预算)）；
- 编码：第二路 ``-i 输入`` 供音轨 ``-c:a copy`` 直通，
  无音轨时 ``-map 1:a:0?`` 的可选标记自动忽略；
- 两个子进程的 stderr 一律 DEVNULL：置为 PIPE 而不消费会在
  长视频上撑满管道缓冲区造成死锁。

两种滤波模式：
- ``mean``（默认）：可分离盒式均值，边界窗口截断——历史行为，
  参数与结果逐字节保持不变；
- ``median``：Huang 直方图加速中值（椒盐噪声），边界镜像填充，
  可选 ``border_dir`` 把每帧边界区域（掩码外置 0）落盘供抽帧复核。
"""

import json
import os
import shutil
import subprocess

from .bands import box_blur_plane_banded, plan_bands
from .median import (
    dump_border_frame,
    median_plane_banded,
    plan_median_bands,
)
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

#: 支持的滤波模式
FILTER_MODES = ("mean", "median")


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


def denoise_video(src, dst, radius=2, budget_bytes=None, progress=None,
                  mode="mean", window=None, border_dir=None):
    """对 ``src`` 做空间滤波降噪，结果写入 ``dst``，音轨原样保留。

    ``radius`` 为滤波半径（窗口 (2r+1)²）；``budget_bytes`` 为单帧
    分带内存预算，缺省按可用内存推导；``progress`` 若给出，每处理完
    一帧以已处理帧数调用一次。返回处理统计 dict。

    ``mode``：
    - ``"mean"``（默认）：可分离盒式均值，行为与历史版本完全一致；
    - ``"median"``：直方图加速中值滤波（椒盐噪声），边界镜像填充。
      ``window`` 可直接给奇数窗口边长（按素材配置），给出时优先于
      ``radius``；不给时窗口为 (2·radius+1)²（默认 radius=2 即 5×5）。
      window=1（radius=0）时中值退化为恒等，与均值模式语义一致。
    ``border_dir`` 仅在中值模式下有效：给出目录后，每帧的边界区域
    （距边缘 <= 半径的像素，掩码外置 0）以工作格式原始字节落盘，
    文件名 ``frame_XXXXXX_border.raw``，另写 ``manifest.json``
    记录工作格式、尺寸、半径与每帧文件名，供单独抽帧复核。

    无论哪种模式，输出帧数与时间戳都由同一套 rawvideo 帧循环保证，
    与输入严格对齐（一帧入、一帧出，framerate 原样透传）。
    """
    if radius < 0:
        raise ValueError("radius 不能为负")
    if mode not in FILTER_MODES:
        raise ValueError(f"未知滤波模式 {mode!r}，可选：{', '.join(FILTER_MODES)}")
    if mode == "median":
        if window is not None:
            if window < 1 or window % 2 == 0:
                raise ValueError("中值窗口边长必须为正奇数")
            radius = (window - 1) // 2
        # window 缺省时沿用 radius；radius=0 即 1x1 中值，退化为恒等
    ffmpeg = _find("ffmpeg")
    info = probe(src)
    work = working_format(info["pix_fmt"])
    codec, out_pix = encoder_for(work)
    w, h = info["width"], info["height"]
    if budget_bytes is None:
        budget_bytes = derive_budget()

    dims = filter_dims(work, w, h)
    if mode == "median":
        band_plans = [
            plan_median_bands(ph, pw, radius, budget_bytes)
            for pw, ph in dims
        ]
    else:
        band_plans = [plan_bands(ph, pw, radius, budget_bytes) for pw, ph in dims]
    fsize = frame_size(work, w, h)

    if border_dir is not None:
        if mode != "median":
            raise ValueError("border_dir 仅在中值模式下可用")
        os.makedirs(border_dir, exist_ok=True)
        border_files = []

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
            if mode == "median":
                out_planes = [
                    median_plane_banded(plane, pw, ph, radius, band_plans[i])
                    for i, (plane, (pw, ph)) in enumerate(zip(planes, dims))
                ]
                if border_dir is not None:
                    name = f"frame_{frames:06d}_border.raw"
                    dump_border_frame(
                        os.path.join(border_dir, name),
                        out_planes, dims, radius,
                    )
                    border_files.append(name)
            else:
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
    if border_dir is not None:
        with open(os.path.join(border_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(
                {
                    "pix_fmt": work,
                    "width": w,
                    "height": h,
                    "fps": info["fps"],
                    "radius": radius,
                    "window": 2 * radius + 1,
                    "frame_bytes": fsize,
                    "files": border_files,
                },
                f, ensure_ascii=False, indent=2,
            )
    return {
        "frames": frames,
        "pix_fmt": work,
        "width": w,
        "height": h,
        "has_audio": info["has_audio"],
        "mode": mode,
        "radius": radius,
        "window": 2 * radius + 1 if mode == "median" else None,
        "border_dir": os.fspath(border_dir) if border_dir is not None else None,
    }
