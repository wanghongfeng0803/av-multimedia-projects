"""重封装编排层：读取 MP4 → 解析 → 写出 MKV。"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import RemuxError
from .mkv_writer import write_mkv
from .mp4_parser import parse_mp4

# 基础版本面向小规模数据，整体读入内存；设置一个保护性上限。
MAX_INPUT_BYTES = 512 * 1024 * 1024  # 512 MiB

# 常见容器扩展名（仅用于友好提示，不参与格式判定）
_MP4_SUFFIXES = {".mp4", ".m4v", ".mov", ".m4a"}
_MKV_SUFFIXES = {".mkv", ".mk3d"}


def remux(input_path: str, output_path: str, *, overwrite: bool = False):
    """执行 MP4 → MKV 无损重封装。

    :return: (写出的 MKV 字节数, 样本总数, track 数)。
    :raises RemuxError: 输入无效或包含不支持的结构时抛出（中文消息）。
    """
    src = Path(input_path)
    dst = Path(output_path)

    if not src.exists():
        raise RemuxError(f"输入文件不存在: {input_path}")
    if not src.is_file():
        raise RemuxError(f"输入路径不是普通文件: {input_path}")
    if dst.parent.exists() and src.resolve() == dst.resolve():
        raise RemuxError("输出文件不能与输入文件为同一个路径")
    if dst.exists() and not overwrite:
        raise RemuxError(
            f"输出文件已存在: {output_path}（如需覆盖请使用 --overwrite）"
        )
    if src.suffix.lower() not in _MP4_SUFFIXES:
        # 扩展名只是提示，真正的合法性仍由解析器判定
        raise RemuxError(
            f"输入文件扩展名 {src.suffix or '(无)'} 不是常见 MP4 容器"
            f"（支持: {', '.join(sorted(_MP4_SUFFIXES))}）"
        )
    if dst.suffix.lower() not in _MKV_SUFFIXES:
        raise RemuxError(
            f"输出文件扩展名 {dst.suffix or '(无)'} 不是 Matroska 容器"
            f"（支持: {', '.join(sorted(_MKV_SUFFIXES))}）"
        )

    size = src.stat().st_size
    if size == 0:
        raise RemuxError("输入文件为空（0 字节），不可能包含媒体数据")
    if size > MAX_INPUT_BYTES:
        raise RemuxError(
            f"输入文件 {size / 1024 / 1024:.1f} MiB 超过基础版本 "
            f"{MAX_INPUT_BYTES // 1024 // 1024} MiB 的内存处理上限"
        )

    data = src.read_bytes()
    mp4 = parse_mp4(data)
    if not mp4.tracks:
        raise RemuxError("MP4 解析成功但未发现任何 track")

    mkv = write_mkv(mp4)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        tmp.write_bytes(mkv)
        os.replace(tmp, dst)
    except OSError as exc:  # 写入失败时清理半成品
        if tmp.exists():
            tmp.unlink()
        raise RemuxError(f"写出 MKV 失败: {exc}") from exc

    total_samples = sum(len(t.samples) for t in mp4.tracks)
    return len(mkv), total_samples, len(mp4.tracks)
