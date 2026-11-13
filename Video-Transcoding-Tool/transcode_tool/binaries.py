"""ffmpeg / ffprobe 可执行文件查找。"""

from __future__ import annotations

import os
import shutil

from .errors import BinaryNotFoundError

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
    1. 同名环境变量显式指定的完整路径（如 ``FFMPEG`` / ``FFPROBE``）；
    2. PATH 中的可执行文件；
    3. Windows 下常见的同目录伴随安装（ffmpeg.exe 与 ffprobe.exe 放在一起）。

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

    # 兜底：如果另一个工具已在 PATH 中找到，尝试同目录下的伴随文件
    sibling = _find_sibling(binary)
    if sibling:
        return sibling

    raise BinaryNotFoundError(
        f"未找到 {binary} 可执行文件。请安装 FFmpeg 并将其加入 PATH，"
        f"或通过环境变量 {env_var or binary.upper()} 指定完整路径。"
    )


def _find_sibling(binary: str) -> str | None:
    """在已知另一个工具的同目录下查找目标工具。"""
    other = "ffprobe" if binary == "ffmpeg" else "ffmpeg"
    try:
        other_path = shutil.which(other + _EXE_SUFFIX) or shutil.which(other)
    except Exception:
        return None
    if not other_path:
        return None
    candidate = os.path.join(os.path.dirname(other_path), binary + _EXE_SUFFIX)
    return candidate if os.path.isfile(candidate) else None


def find_ffmpeg() -> str:
    return find_binary("ffmpeg", env_var="FFMPEG")


def find_ffprobe() -> str:
    return find_binary("ffprobe", env_var="FFPROBE")
