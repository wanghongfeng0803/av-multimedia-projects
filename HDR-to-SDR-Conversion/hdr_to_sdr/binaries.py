"""ffmpeg / ffprobe 可执行文件查找（与兄弟项目同一套查找顺序）。"""

from __future__ import annotations

import os
import shutil

from .errors import BinaryNotFoundError

__all__ = ["find_ffmpeg", "find_ffprobe", "find_binary"]

_WINDOWS = os.name == "nt"
_EXE_SUFFIX = ".exe" if _WINDOWS else ""


def _candidate_names(binary: str) -> list[str]:
    """返回在 PATH 中查找时使用的文件名候选。"""
    if binary.endswith(".exe") or not _WINDOWS:
        return [binary]
    return [binary + _EXE_SUFFIX, binary]


def find_binary(binary: str, env_var: str | None = None) -> str:
    """定位 ffmpeg / ffprobe。

    查找顺序：
    1. 同名环境变量显式指定的完整路径（``FFMPEG`` / ``FFPROBE``）；
    2. PATH 中的可执行文件；
    3. Windows 下的同目录伴随安装（ffmpeg.exe 旁边找 ffprobe.exe）。

    :raises BinaryNotFoundError: 全部位置都找不到时抛出。
    """
    explicit = os.environ.get(env_var or "") if env_var else None
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise BinaryNotFoundError(
            f"环境变量 {env_var} 指向的文件不存在: {explicit}"
        )

    for name in _candidate_names(binary):
        found = shutil.which(name)
        if found:
            return found

    # 同目录伴随安装：先找到 ffmpeg，再在其目录里找另一个
    peer = "ffprobe" if binary.startswith("ffmpeg") else "ffmpeg"
    peer_path = shutil.which(peer + _EXE_SUFFIX) or shutil.which(peer)
    if peer_path:
        guess = os.path.join(
            os.path.dirname(peer_path), binary + _EXE_SUFFIX if _WINDOWS else binary
        )
        if os.path.isfile(guess):
            return guess

    raise BinaryNotFoundError(
        f"找不到 {binary}，请将其加入 PATH，或设置环境变量 {env_var or binary.upper()}"
    )


def find_ffmpeg() -> str:
    """返回 ffmpeg 可执行文件的完整路径。"""
    return find_binary("ffmpeg", "FFMPEG")


def find_ffprobe() -> str:
    """返回 ffprobe 可执行文件的完整路径。"""
    return find_binary("ffprobe", "FFPROBE")
