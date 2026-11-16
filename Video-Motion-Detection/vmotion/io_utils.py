"""ffmpeg 子进程共享工具：可执行文件探测、ffprobe、定长帧读取。

检测（:mod:`vmotion.pipeline`）与标记（:mod:`vmotion.render`）共用，
单独成模块避免循环导入。子进程 stderr 一律 DEVNULL：置为 PIPE 而不
消费会在长视频上撑满管道缓冲区造成死锁（与兄弟模块同一约定）。
"""

import json
import shutil
import subprocess

__all__ = ["PipelineError", "find_tool", "probe", "read_exact"]


class PipelineError(RuntimeError):
    """流水线子进程失败或数据流异常。"""


def find_tool(tool):
    """在 PATH 中查找 ffmpeg/ffprobe，缺失抛 :class:`PipelineError`。"""
    path = shutil.which(tool)
    if not path:
        raise PipelineError(f"找不到 {tool}，请确认已安装并在 PATH 中")
    return path


def probe(path):
    """ffprobe 读取首路视频流信息及是否含音轨。

    返回 ``{"width", "height", "pix_fmt", "fps", "fps_float",
    "has_audio"}``。
    """
    ffprobe = find_tool("ffprobe")
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries",
                "stream=codec_type,width,height,pix_fmt,avg_frame_rate,r_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise PipelineError(
            f"ffprobe 无法读取 {path}（退出码 {exc.returncode}），"
            "文件不存在或不是可识别的媒体"
        ) from exc
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
    fps = video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1"
    if fps in ("0/0", ""):
        fps = "25/1"
    num, _, den = fps.partition("/")
    den = den or "1"
    fps_float = int(num) / int(den) if int(den) else 25.0
    return {
        "width": video["width"],
        "height": video["height"],
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "fps": fps,
        "fps_float": fps_float,
        "has_audio": has_audio,
    }


def read_exact(stream, size):
    """读满 size 字节；流结束时未读到任何字节返回 None，读一半报错。"""
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
