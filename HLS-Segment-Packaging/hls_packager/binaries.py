"""ffmpeg / ffprobe 可执行文件发现。

查找顺序：
1. 环境变量 ``FFMPEG_BIN`` / ``FFPROBE_BIN``（显式指定，优先级最高）；
2. ``PATH`` 中按平台可执行名查找（Windows 自动补 ``.exe``）；
3. 与当前解释器同目录下的可执行文件（虚拟环境 / 便携部署场景）。

零第三方依赖：不使用 shutil.which 之外的任何工具，且对 shutil.which
做了同目录兜底，保证便携目录布局也能工作。
"""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache

from .errors import BinaryNotFoundError

_IS_WINDOWS = sys.platform.startswith("win")


def _candidate_names(binary: str) -> list[str]:
    """返回某二进制在当前平台上的全部候选文件名。"""
    if _IS_WINDOWS and not binary.lower().endswith(".exe"):
        return [f"{binary}.exe", binary]
    return [binary]


def _resolve(binary: str, env_var: str) -> str:
    # 1) 环境变量显式指定
    env_path = os.environ.get(env_var)
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        raise BinaryNotFoundError(
            f"环境变量 {env_var} 指向的文件不存在：{env_path}"
        )

    # 2) PATH 查找
    for name in _candidate_names(binary):
        found = shutil.which(name)
        if found:
            return found

    # 3) 解释器同目录（便携部署：ffmpeg.exe 与 python.exe 放在一起）
    interpreter_dir = os.path.dirname(sys.executable)
    for name in _candidate_names(binary):
        local = os.path.join(interpreter_dir, name)
        if os.path.isfile(local):
            return local

    raise BinaryNotFoundError(
        f"未找到 {binary}，请安装 ffmpeg 并将其加入 PATH，"
        f"或通过环境变量 {env_var} 显式指定可执行文件路径"
    )


@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """返回 ffmpeg 可执行文件的绝对路径（结果缓存）。"""
    return _resolve("ffmpeg", "FFMPEG_BIN")


@lru_cache(maxsize=1)
def ffprobe_path() -> str:
    """返回 ffprobe 可执行文件的绝对路径（结果缓存）。"""
    return _resolve("ffprobe", "FFPROBE_BIN")
