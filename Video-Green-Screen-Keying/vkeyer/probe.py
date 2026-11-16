"""ffprobe 封装：读取视频流尺寸、帧率、像素格式与音轨情况。

与兄弟项目一致，所有像素处理统一在 rgb24（R, G, B 字节序）上进行，
本模块只负责读元数据，不做格式回退决策。
"""

from __future__ import annotations

import json
import shutil
import subprocess

from .errors import BinaryNotFoundError, FFmpegRunError, InvalidInputError

__all__ = ["find_tool", "probe_video"]


def find_tool(name: str) -> str:
    """在 PATH 中查找 ffmpeg / ffprobe，找不到抛 :class:`BinaryNotFoundError`。"""
    path = shutil.which(name)
    if not path:
        raise BinaryNotFoundError(f"找不到 {name}，请确认已安装并在 PATH 中")
    return path


def probe_video(path) -> dict:
    """读取首路视频流信息及是否含音轨。

    返回 ``{"width", "height", "pix_fmt", "fps", "has_audio"}``；
    帧率字段保留 ffprobe 的分数字符串（如 ``"30000/1001"``），
    直接回传给编码器 ``-framerate``，不做浮点化，避免时间基漂移。
    """
    ffprobe = find_tool("ffprobe")
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries",
                "stream=codec_type,width,height,pix_fmt,avg_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FFmpegRunError(
            f"ffprobe 无法识别输入：{path}", stderr=exc.stderr or ""
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
        raise InvalidInputError(f"{path} 中没有视频流")
    fps = video.get("avg_frame_rate") or "25"
    if fps in ("0/0", ""):
        fps = "25"
    width = video.get("width")
    height = video.get("height")
    if not width or not height:
        raise InvalidInputError(f"{path} 缺少视频尺寸信息")
    return {
        "width": int(width),
        "height": int(height),
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "fps": fps,
        "has_audio": has_audio,
    }
