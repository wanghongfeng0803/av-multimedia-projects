"""重封装编排层：读取 MP4 → 解析 → 写出 MKV。

默认走**流式直写**：输出文件直接以二进制流方式逐簇写入目标路径，
全程不创建 ``.part`` 临时文件、不在内存里拼整份 MKV（同一时刻只保留
一个簇）。写失败时清理半成品目标文件，不会留下半个 MKV。

若需要“目标文件在成功前完全不可见、且覆盖过程绝不出现半成品”的
原子语义，可传 ``use_temp=True`` 退回“临时文件 + ``os.replace``”模式。
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import RemuxError
from .mkv_writer import write_mkv, write_mkv_stream
from .mp4_parser import parse_mp4

# 基础版本面向小规模数据，整体读入内存；设置一个保护性上限。
MAX_INPUT_BYTES = 512 * 1024 * 1024  # 512 MiB

# 常见容器扩展名（仅用于友好提示，不参与格式判定）
_MP4_SUFFIXES = {".mp4", ".m4v", ".mov", ".m4a"}
_MKV_SUFFIXES = {".mkv", ".mk3d"}


def _validate_paths(src: Path, dst: Path, overwrite: bool) -> None:
    """集中处理输入/输出路径的确定性前置校验（失败均为 RemuxError）。"""
    if not src.exists():
        raise RemuxError(f"输入文件不存在: {src}")
    if not src.is_file():
        raise RemuxError(f"输入路径不是普通文件: {src}")
    if dst.parent.exists() and src.resolve() == dst.resolve():
        raise RemuxError("输出文件不能与输入文件为同一个路径")
    if dst.exists() and not overwrite:
        raise RemuxError(
            f"输出文件已存在: {dst}（如需覆盖请使用 --overwrite）"
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


def remux(
    input_path: str,
    output_path: str,
    *,
    overwrite: bool = False,
    use_temp: bool = False,
):
    """执行 MP4 → MKV 无损重封装。

    :param overwrite: 输出已存在时是否允许覆盖。
    :param use_temp: 写出策略。``False``（默认）为流式直写目标路径，
        不产生临时文件，失败清理半成品；``True`` 退回“写 ``.part``
        临时文件 + ``os.replace`` 原子替换”，覆盖过程中目标始终是
        完整的旧文件或完整的新文件，不会出现半成品。
    :return: (写出的 MKV 字节数, 样本总数, track 数)。
    :raises RemuxError: 输入无效或包含不支持的结构时抛出（中文消息）。
    """
    src = Path(input_path)
    dst = Path(output_path)
    _validate_paths(src, dst, overwrite)

    size = src.stat().st_size
    if size == 0:
        raise RemuxError("输入文件为空（0 字节），不可能包含媒体数据")
    if size > MAX_INPUT_BYTES:
        raise RemuxError(
            f"输入文件 {size / 1024 / 1024:.1f} MiB 超过基础版本 "
            f"{MAX_INPUT_BYTES // 1024 // 1024} MiB 的内存处理上限"
        )

    # MP4 元数据需要完整的盒树（moov/stbl），源仍整体读入；
    # 输出侧才是流式——不再持有整份 MKV 字节串。
    data = src.read_bytes()
    mp4 = parse_mp4(data)
    if not mp4.tracks:
        raise RemuxError("MP4 解析成功但未发现任何 track")
    total_samples = sum(len(t.samples) for t in mp4.tracks)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if use_temp:
        byte_count = _write_via_temp(mp4, dst)
    else:
        byte_count = _write_streaming(mp4, dst)

    return byte_count, total_samples, len(mp4.tracks)


def _write_streaming(mp4, dst: Path) -> int:
    """流式直写目标路径：无临时文件，任何失败都清理半成品后原样上抛。"""
    try:
        with open(dst, "wb") as out:
            return write_mkv_stream(mp4, out)
    except BaseException:
        # 无论确定性错误（RemuxError）还是瞬时 I/O 错误（OSError），
        # 都不能在目标路径留下半个 MKV；清理后保留原异常类型上抛，
        # 以便上层重试逻辑区分“可重试”与“确定性失败”。
        try:
            if dst.exists():
                dst.unlink()
        except OSError:
            pass
        raise


def _write_via_temp(mp4, dst: Path) -> int:
    """原子模式：写 .part 临时文件再 os.replace，失败清理半成品。"""
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        # 内存路径输出完整字节，沿用历史的临时文件 + 原子替换语义
        mkv = write_mkv(mp4)
        tmp.write_bytes(mkv)
        os.replace(tmp, dst)
    except OSError as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RemuxError(f"写出 MKV 失败: {exc}") from exc
    return len(mkv)
