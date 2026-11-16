"""解码 → 分块节点链映射 → 编码 的 ffmpeg 管道编排。

数据流：

    输入文件 ──ffmpeg 解码──> rgb24 裸流 ──分块并行节点链──> rgb24 裸流 ──ffmpeg 编码──> 输出文件
      │                                                                        ▲
      └────────────── 音轨不经过像素管道，-c:a copy 原样直通 ────────────────────┘

设计要点：

- **保留音轨**：音轨不进像素管道，编码器以原文件为第二输入，
  ``-map 1:a? -c:a copy`` 无重编码直通，没有音画同步漂移；
  无音轨的输入也能正常工作（``?`` 后缀使映射可选）。
- **处理参数只构建一次**：1D LUT 建表、``.cube`` 解析与节点链组装都在
  启动 ffmpeg 之前完成一次，之后全部帧、全部块、worker 复用。
- **失败即保留原图、不出半成品**：解析 / 域校验 / 强度钳制等所有可能
  失败的前置环节都在任何像素处理之前；像素管道或 ffmpeg 一旦失败，
  本次运行新建的输出文件会被删除，预先存在的输出不会被改动。
- **stderr 排空**：两条 ffmpeg 子进程的 stderr 都由独立线程持续读取，
  避免管道缓冲区写满后父子进程互等死锁。
- **内存有界**：任意时刻只有少量行带在途（见
  :func:`color_grading.chunks.map_stream`），整段视频不会载入内存。
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .binaries import find_ffmpeg
from .chunks import DEFAULT_CHUNK_ROWS, map_stream
from .colorspace import clamp_color_params
from .errors import (
    FFmpegRunError,
    InputNotFoundError,
    OutputExistsError,
)
from .lut import build_lut, clamp_params
from .lut3d import CubeLUT, clamp_strength, load_cube
from .nodes import (
    BrightnessContrastNode,
    GradeNode,
    HueSaturationNode,
    LUT3DNode,
)
from .probe import probe_video

# stderr 只保留尾部若干行用于错误报告，避免长视频日志占内存。
_STDERR_TAIL_LINES = 50

#: 3D LUT 节点在默认节点链中的可插入位置。
LUT3D_POSITIONS = ("first", "last")


@dataclass
class GradeResult:
    """一次调色执行的汇总。"""

    input_path: str
    output_path: str
    brightness: float
    contrast: float
    hue: float
    saturation: float
    width: int
    height: int
    fps: float
    frames: int
    has_audio: bool
    elapsed_seconds: float
    messages: list[str] = field(default_factory=list)
    # 3D LUT 相关（未使用时为默认值，保持旧调用的结果结构可读）。
    lut3d_path: str | None = None
    lut3d_strength: float = 0.0
    lut3d_position: str | None = None


def _drain_stderr(stream, sink: deque) -> None:
    """持续读空子进程 stderr，防止管道写满死锁；只保留尾部若干行。"""
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", "replace"))
    finally:
        stream.close()


def _tail(sink: deque) -> str:
    return "".join(sink).strip()


def build_legacy_nodes(
    brightness: float, contrast: float, hue: float, saturation: float
) -> list[GradeNode]:
    """把四个传统参数构建为固定顺序节点链（1D LUT → 色相 → 饱和度）。"""
    table = build_lut(brightness, contrast)
    return [
        BrightnessContrastNode(table),
        HueSaturationNode(float(hue), float(saturation)),
    ]


def _insert_lut3d(
    nodes: list[GradeNode], lut_node: LUT3DNode, position: str
) -> list[GradeNode]:
    """按 ``first`` / ``last`` 把 3D LUT 节点插入节点链。

    位置语义显式且唯一：``first`` 在所有其它节点之前（最先作用于原图），
    ``last`` 在所有其它节点之后（作用于它们的输出）。顺序变化的结果
    因此完全可预期——它就是函数复合顺序的改变。
    """
    if position == "first":
        return [lut_node, *nodes]
    if position == "last":
        return [*nodes, lut_node]
    raise ValueError(
        f"lut3d_position 只支持 {LUT3D_POSITIONS}，得到 {position!r}"
    )


def adjust_nodes(
    input_path: str,
    output_path: str,
    nodes: list[GradeNode],
    *,
    workers: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
    on_frame: Callable[[int], None] | None = None,
    messages: list[str] | None = None,
) -> GradeResult:
    """用一条显式有序节点链调色并保留音轨（节点化的底层入口）。

    节点链的顺序即像素处理顺序。本函数不再做任何参数钳制或 LUT 解析——
    这些应在组装节点前完成；因此进入本函数后不会再因解析 / 域问题失败，
    像素管道阶段的失败只会来自 ffmpeg 或流截断，且都会清理半成品输出。

    :raises InputNotFoundError: 输入不存在
    :raises OutputExistsError: 输出已存在且未允许覆盖
    :raises InvalidInputError: 输入不是可处理的视频
    :raises FFmpegRunError: ffmpeg 执行失败（已清理本次新建的输出）
    """
    if not os.path.isfile(input_path):
        raise InputNotFoundError(f"输入文件不存在: {input_path}")
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(f"输出文件已存在（未指定覆盖）: {output_path}")

    messages = list(messages or [])
    info = probe_video(input_path)

    if info.width % 2 or info.height % 2:
        # yuv420p 要求偶数尺寸；向下取偶并明确告知调用方。
        messages.append("输入宽/高为奇数，yuv420p 要求偶数尺寸，已向下缩放为偶数。")

    return _run_ffmpeg_pipeline(
        input_path=input_path,
        output_path=output_path,
        info=info,
        nodes=list(nodes),
        workers=workers,
        chunk_rows=chunk_rows,
        crf=crf,
        preset=preset,
        overwrite=overwrite,
        on_frame=on_frame,
        messages=messages,
    )


def adjust(
    input_path: str,
    output_path: str,
    brightness: float = 0.0,
    contrast: float = 0.0,
    hue: float = 0.0,
    saturation: float = 0.0,
    *,
    lut3d_path: str | None = None,
    lut3d_strength: float = 1.0,
    lut3d_position: str = "last",
    workers: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
    on_frame: Callable[[int], None] | None = None,
) -> GradeResult:
    """调整视频亮度 / 对比度 / 色相 / 饱和度，并可选叠加一张 3D LUT。

    默认参数（四项均为 0、不给 3D LUT）为恒等映射，输出像素与输入
    逐字节一致（裸流级别），这是旧无参数调用的兼容基线。

    全工具像素统一为 **RGB** 字节序（ffmpeg ``rgb24``），不存在 BGR
    分支。传统四项的处理顺序固定为：

    1. 亮度 / 对比度 1D LUT（逐通道）；
    2. 色相旋转（``hue`` 度，逆时针为正）；
    3. 饱和度缩放（``saturation`` 百分比）。

    3D LUT 作为**独立节点**按 ``lut3d_position`` 插入：``last``（默认）
    在上述三步之后，``first`` 在它们之前。节点位置改变即改变函数复合
    顺序，结果确定可预期。需要更自由的多节点组合时用
    :func:`adjust_nodes`。

    强度 / 解析 / 域的失败处理：

    - ``lut3d_strength`` 超 [0,1] 自动钳制并提示；``0`` 精确等于原图、
      ``1`` 精确等于全量 LUT（逐字节）。
    - ``.cube`` 解析错误带文件名与行号（:class:`LUTParseError`），
      域 / 尺寸不匹配分别抛 :class:`LUTDomainError` / :class:`LUTParseError`，
      都在启动 ffmpeg 前抛出，不会产生任何输出文件。

    :param lut3d_path: ``.cube`` 文件路径，``None`` 表示不启用 3D LUT。
    :param lut3d_strength: 3D LUT 强度 [0,1]，默认 1（全量）。
    :param lut3d_position: ``first`` 或 ``last``，默认 ``last``。
    :raises LUTParseError: .cube 无法解析或尺寸不匹配（带行号）
    :raises LUTDomainError: .cube 域不是单位 [0,1] 域
    """
    if not os.path.isfile(input_path):
        raise InputNotFoundError(f"输入文件不存在: {input_path}")
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(f"输出文件已存在（未指定覆盖）: {output_path}")

    # --- 所有可能失败 / 钳制的前置环节都在启动 ffmpeg 之前完成 ---
    brightness, contrast, messages = clamp_params(brightness, contrast)
    hue, saturation, hsv_messages = clamp_color_params(hue, saturation)
    messages.extend(hsv_messages)

    nodes = build_legacy_nodes(brightness, contrast, hue, saturation)

    applied_lut3d: CubeLUT | None = None
    applied_strength = 0.0
    applied_position: str | None = None
    if lut3d_path is not None:
        if lut3d_position not in LUT3D_POSITIONS:
            raise ValueError(
                f"lut3d_position 只支持 {LUT3D_POSITIONS}，"
                f"得到 {lut3d_position!r}"
            )
        # 解析 / 域 / 尺寸错误在此抛出，ffmpeg 尚未启动，无半成品。
        cube = load_cube(lut3d_path)
        strength, strength_messages = clamp_strength(lut3d_strength)
        messages.extend(strength_messages)
        if strength != 0.0:
            lut_node = LUT3DNode(cube, strength)
            nodes = _insert_lut3d(nodes, lut_node, lut3d_position)
        # 强度被钳到/取到 0 时节点恒等，不插入链，但结果字段仍如实记录。
        applied_lut3d = cube
        applied_strength = strength
        applied_position = lut3d_position

    info = probe_video(input_path)

    if info.width % 2 or info.height % 2:
        messages.append("输入宽/高为奇数，yuv420p 要求偶数尺寸，已向下缩放为偶数。")

    result = _run_ffmpeg_pipeline(
        input_path=input_path,
        output_path=output_path,
        info=info,
        nodes=nodes,
        workers=workers,
        chunk_rows=chunk_rows,
        crf=crf,
        preset=preset,
        overwrite=overwrite,
        on_frame=on_frame,
        messages=messages,
    )
    result.brightness = brightness
    result.contrast = contrast
    result.hue = hue
    result.saturation = saturation
    result.lut3d_path = applied_lut3d.file_path if applied_lut3d is not None else None
    result.lut3d_strength = applied_strength
    result.lut3d_position = applied_position
    return result


def _run_ffmpeg_pipeline(
    *,
    input_path: str,
    output_path: str,
    info,
    nodes: list[GradeNode],
    workers: int | None,
    chunk_rows: int,
    crf: int,
    preset: str,
    overwrite: bool,
    on_frame: Callable[[int], None] | None,
    messages: list[str],
) -> GradeResult:
    """启动解码/编码两条 ffmpeg 并跑节点链；失败时清理本次新建的输出。"""
    ffmpeg = find_ffmpeg()
    # 记录输出是否早已存在：失败时只删除本次运行新建的文件，
    # 绝不动用户预先存在的文件（OutputExistsError 已在更上层拦截）。
    pre_existed = os.path.exists(output_path)

    decoder = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y" if overwrite else "-n",
            # 第一路输入：处理后的 rgb24 裸流（stdin）
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{info.width}x{info.height}",
            "-r", info.fps_fraction,
            "-i", "-",
            # 第二路输入：原文件，只为直通音轨
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            *(_odd_size_filter(info)),
            "-c:a", "copy",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    dec_err: deque = deque(maxlen=_STDERR_TAIL_LINES)
    enc_err: deque = deque(maxlen=_STDERR_TAIL_LINES)
    drainers = [
        threading.Thread(target=_drain_stderr, args=(decoder.stderr, dec_err), daemon=True),
        threading.Thread(target=_drain_stderr, args=(encoder.stderr, enc_err), daemon=True),
    ]
    for t in drainers:
        t.start()

    started = time.monotonic()
    frames = 0
    broken_pipe = False
    pipeline_error: Exception | None = None
    try:
        try:
            frames = map_stream(
                decoder.stdout, encoder.stdin,
                info.width, info.height,
                nodes=nodes,
                workers=workers, chunk_rows=chunk_rows, on_frame=on_frame,
            )
        except (BrokenPipeError, ConnectionResetError):
            # 编码器提前退出（如参数不被支持），其 stderr 有真正原因。
            broken_pipe = True
    except Exception as exc:  # 流截断等映射阶段错误：同样走失败清理
        pipeline_error = exc
    finally:
        try:
            encoder.stdin.close()
        except OSError:
            pass
        if (broken_pipe or pipeline_error is not None) and decoder.poll() is None:
            # 不再消费解码输出，主动终止避免它阻塞在管道写上。
            decoder.terminate()
        enc_rc = encoder.wait()
        dec_rc = decoder.wait()
        for t in drainers:
            t.join(timeout=5)
    elapsed = time.monotonic() - started

    def _cleanup_partial() -> None:
        if not pre_existed:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass

    if pipeline_error is not None:
        _cleanup_partial()
        raise pipeline_error
    if broken_pipe:
        _cleanup_partial()
        raise FFmpegRunError("ffmpeg 编码器提前退出，像素管道中断。", stderr=_tail(enc_err))
    if dec_rc != 0:
        _cleanup_partial()
        raise FFmpegRunError(f"ffmpeg 解码失败（退出码 {dec_rc}）。", stderr=_tail(dec_err))
    if enc_rc != 0:
        _cleanup_partial()
        raise FFmpegRunError(f"ffmpeg 编码失败（退出码 {enc_rc}）。", stderr=_tail(enc_err))

    return GradeResult(
        input_path=input_path,
        output_path=output_path,
        brightness=0.0,
        contrast=0.0,
        hue=0.0,
        saturation=0.0,
        width=info.width,
        height=info.height,
        fps=info.fps,
        frames=frames,
        has_audio=info.has_audio,
        elapsed_seconds=elapsed,
        messages=messages,
    )


def _odd_size_filter(info) -> list[str]:
    """奇数宽高时给编码器加偶数化缩放滤镜（与原管道行为一致）。"""
    if info.width % 2 or info.height % 2:
        return ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    return []
