"""编码模块：ffmpeg 解码 → 逐帧消费块表打码 → ffmpeg 编码。

三模块解耦：本模块只消费 ``frame_index -> BlockTable`` 契约
（由区域管理模块的 :meth:`vblur.regions.RegionManager.as_source`
或未来的自动检测模块产出），自身不理解区域的语义；打码由
:mod:`vblur.mosaic` 完成。契约即 :class:`vblur.mosaic.BlockTable`。

输出与输入一致性的保证：

- **帧率**：``-framerate`` 原样透传 ffprobe 的 ``avg_frame_rate``；
- **帧数 / 时间戳**：一帧入、一帧出，不抽帧、不补帧、不改时间基；
- **尺寸**：rawvideo 尺寸 = 探测尺寸；
- **像素格式**：yuv420p/422/444 原样回写；RGB 输入走 libx264rgb
  （H.264 RGB 产物被 ffprobe 探为 ``gbrp`` 而非 ``rgb24``，属
  ffmpeg 已知口径，:func:`pix_fmt_matches` 视为等价）；gray 及
  其他非常见格式回退 yuv420p（libx264 只收 YUV 系，见 README）；
- **音轨**：第二路 ``-i`` 原样 ``-c:a copy`` 直通，无音轨时
  ``-map 1:a:0?`` 的可选标记自动忽略。

工作格式固定 rgb24（RGB 全局统一，与调色项目口径一致）：打码在
RGB 空间对三通道整体求均值，避免只糊亮度留下清晰色度的隐私泄漏。

两个子进程的 stderr 一律 DEVNULL：置为 PIPE 而不消费会在长视频上
撑满 Windows 管道缓冲区（约 4KB）造成死锁。
"""

from __future__ import annotations

import json
import shutil
import subprocess

from .mosaic import apply_mosaic
from .regions import RegionManager

__all__ = [
    "DEFAULT_BLOCK",
    "PipelineError",
    "blur_video",
    "pix_fmt_matches",
    "probe",
]

#: 默认马赛克块边长（像素）
DEFAULT_BLOCK = 16

# H.264 RGB 产物会被探为 gbrp，与 rgb24 视为等价
_RGB_EQUIV = frozenset(("rgb24", "gbrp", "bgr24"))

# ffprobe 报告名 -> 输出回写名
_PIX_ALIASES = {
    "yuvj420p": "yuv420p",
    "yuvj422p": "yuv422p",
    "yuvj444p": "yuv444p",
}

#: 可原样回写的 YUV 系格式
_YUV_PASSTHROUGH = ("yuv420p", "yuv422p", "yuv444p")


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


def _encoder_for(src_pix_fmt):
    """由输入像素格式推导 ``(视频编码器, 输出像素格式或 None)``。

    yuv420p/422/444 原样回写；RGB 系换 libx264rgb（libx264 只收
    YUV 系）；gray 及其余格式回退 yuv420p。
    """
    fmt = _PIX_ALIASES.get(src_pix_fmt, src_pix_fmt)
    if fmt in _RGB_EQUIV:
        return "libx264rgb", None
    if fmt in _YUV_PASSTHROUGH:
        return "libx264", fmt
    return "libx264", "yuv420p"


def pix_fmt_matches(probed, expected):
    """ffprobe 探测格式与期望格式是否一致（rgb24/gbrp 视为等价）。"""
    a = _PIX_ALIASES.get(probed, probed)
    b = _PIX_ALIASES.get(expected, expected)
    if a == b:
        return True
    return a in _RGB_EQUIV and b in _RGB_EQUIV


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


def blur_video(src, dst, regions=(), block=DEFAULT_BLOCK, source=None,
               progress=None):
    """对 ``src`` 的指定区域打马赛克，结果写入 ``dst``，音轨原样保留。

    :param regions: 静态区域 ``(x, y, w, h)`` 可迭代（像素坐标，允许
        负尺寸 / 越界，由区域管理模块钳制；完全越界 / 零面积被丢弃）。
    :param block: 全局网格块边长（像素，>= 1）；``block=1`` 退化为
        恒等基线。
    :param source: 可选的 ``frame_index -> BlockTable`` 契约可调用
        对象。给出时忽略 ``regions``——运行时增删请自持
        :class:`vblur.regions.RegionManager` 并传入
        ``manager.as_source(block)``；自动检测模块同样实现该契约即可
        接入。逐帧调用，返回的块表必须不可变或不再被修改。
    :param progress: 若给出，每处理完一帧以已处理帧数调用一次。
    :return: 处理统计 dict（帧数、尺寸、像素格式、块边长等）。
    """
    if block < 1:
        raise ValueError("块边长必须 >= 1")
    if source is not None and regions:
        raise ValueError("regions 与 source 互斥，只能给一个")

    ffmpeg = _find("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]
    codec, out_pix = _encoder_for(info["pix_fmt"])

    if source is None:
        manager = RegionManager(w, h)
        for r in regions:
            if len(r) != 4:
                raise ValueError(f"区域必须是 (x, y, w, h) 四元组：{r!r}")
            manager.add(*r)
        source = manager.as_source(block)

    fsize = w * h * 3  # 工作格式 rgb24

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    enc_cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
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
            frame = bytearray(buf)
            table = source(frames)
            if table.width != w or table.height != h:
                raise PipelineError(
                    f"块表尺寸 {table.width}x{table.height} 与帧 "
                    f"{w}x{h} 不符"
                )
            apply_mosaic(frame, table, channels=3)
            try:
                enc.stdin.write(frame)
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
        "width": w,
        "height": h,
        "src_pix_fmt": info["pix_fmt"],
        "out_pix_fmt": out_pix or info["pix_fmt"],
        "fps": info["fps"],
        "block": block,
        "has_audio": info["has_audio"],
    }
