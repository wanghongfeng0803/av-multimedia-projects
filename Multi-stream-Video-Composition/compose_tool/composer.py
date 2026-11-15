"""合成编排：探测 → 规划 → 生成命令 → 执行。

输出格式规范（固定，不随输入变化）：

- 容器：MP4（``+faststart``，moov 前置便于流式播放）
- 视频：H.264（libx264）/ yuv420p / 偶数尺寸 / CFR
- 音频：AAC / 48kHz / 立体声（无音频输入时输出无音轨）
- 元数据：剥离输入元数据（``-map_metadata -1``）

可复现性：同版本 ffmpeg + 同输入 + 同参数 ⇒ 输出逐字节一致。
为此命令中固定了所有会引入不确定性的开关：``-bitexact``（不写
muxer/编码器版本串）、``-map_metadata -1``、固定
``-video_track_timescale``、规划层全部取确定值（见 plan 模块）。
"""

from __future__ import annotations

import os

from .binaries import find_ffmpeg
from .errors import FFmpegRunError, InputNotFoundError, OutputExistsError
from .filtergraph import build_filter_complex
from .models import ComposeResult
from .plan import (
    AUDIO_AUTO,
    DURATION_PAD,
    LAYOUT_AUTO,
    ComposePlan,
    plan_composition,
)
from .probe import probe_media
from .runner import ProgressCallback, build_progress_args, run_ffmpeg

#: 视频质量与速度档位的默认值（CRF 越小越清晰）
DEFAULT_CRF = 20
DEFAULT_PRESET = "veryfast"
#: 音频码率（AAC）
AUDIO_BITRATE = "128k"
#: 固定的视频时间基，避免不同输入时基导致输出时间戳不可复现
VIDEO_TIMESCALE = 90000
#: 输出视频编码（规范的一部分，见模块 docstring）
VIDEO_ENCODER = "libx264"
AUDIO_ENCODER = "aac"


def default_output_path(left_path: str, right_path: str) -> str:
    """省略输出路径时的默认命名：``<左名>_x_<右名>.mp4``，放在左路同目录。"""
    left_stem = os.path.splitext(os.path.basename(left_path))[0]
    right_stem = os.path.splitext(os.path.basename(right_path))[0]
    directory = os.path.dirname(os.path.abspath(left_path))
    return os.path.join(directory, f"{left_stem}_x_{right_stem}.mp4")


def compose(
    left_path: str,
    right_path: str,
    output_path: str | None = None,
    *,
    overwrite: bool = False,
    duration_mode: str = DURATION_PAD,
    layout: str = LAYOUT_AUTO,
    height: int | None = None,
    fps: object = None,
    audio_mode: str = AUDIO_AUTO,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    progress_callback: ProgressCallback | None = None,
    log_path: str | None = None,
) -> ComposeResult:
    """把两路视频左右并排合成为单个 MP4。

    所有不一致（分辨率/时长/帧率/音频）都由规划层按确定性策略对齐，
    详见 :mod:`compose_tool.plan`。

    :raises InputNotFoundError: 任一输入不存在
    :raises OutputExistsError: 输出已存在且未指定 ``overwrite``
    :raises InvalidInputError: 输入无法作为视频处理或参数非法
    :raises FFmpegRunError: ffmpeg 执行失败
    """
    for path, side in ((left_path, "左"), (right_path, "右")):
        if not os.path.isfile(path):
            raise InputNotFoundError(f"{side}路输入文件不存在: {path}")

    if output_path is None:
        output_path = default_output_path(left_path, right_path)
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(
            f"输出文件已存在: {output_path}（使用 --overwrite 允许覆盖）"
        )

    left_info = probe_media(left_path)
    right_info = probe_media(right_path)
    plan = plan_composition(
        left_info,
        right_info,
        duration_mode=duration_mode,
        layout=layout,
        height=height,
        fps=fps,
        audio_mode=audio_mode,
    )

    filter_complex = build_filter_complex(plan)
    cmd = build_command(
        find_ffmpeg(),
        left_path,
        right_path,
        plan,
        filter_complex,
        output_path,
        overwrite=overwrite,
        crf=crf,
        preset=preset,
    )

    returncode, stderr_tail = run_ffmpeg(
        cmd,
        log_path=log_path,
        total_duration=plan.duration,
        progress_callback=progress_callback,
    )
    if returncode != 0:
        raise FFmpegRunError(
            f"ffmpeg 合成失败（退出码 {returncode}）。",
            stderr=stderr_tail,
            cmd=cmd,
        )
    if not os.path.isfile(output_path):
        raise FFmpegRunError(
            "ffmpeg 报告成功但输出文件未生成。", stderr=stderr_tail, cmd=cmd
        )

    return ComposeResult(
        left_path=os.path.abspath(left_path),
        right_path=os.path.abspath(right_path),
        output_path=output_path,
        success=True,
        width=plan.width,
        height=plan.height,
        fps=plan.fps_text,
        duration_seconds=plan.duration,
        duration_mode=plan.duration_mode,
        layout=plan.layout,
        audio_mode=plan.audio_mode,
        video_codec="h264",
        audio_codec="aac" if plan.audio_sources else None,
        warnings=plan.warnings,
        ffmpeg_log_path=log_path,
    )


def build_command(
    ffmpeg: str,
    left_path: str,
    right_path: str,
    plan: ComposePlan,
    filter_complex: str,
    output_path: str,
    *,
    overwrite: bool,
    crf: int,
    preset: str,
) -> list[str]:
    """组装完整的 ffmpeg 命令（独立成函数便于测试与复现问题）。"""
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", left_path,
        "-i", right_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
    ]
    if plan.audio_sources:
        cmd += ["-map", "[aout]"]
    cmd += [
        "-c:v", VIDEO_ENCODER,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    if plan.audio_sources:
        cmd += [
            "-c:a", AUDIO_ENCODER,
            "-b:a", AUDIO_BITRATE,
            "-ar", str(plan.sample_rate),
            "-ac", str(plan.channels),
        ]
    cmd += [
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        "-bitexact",
        "-video_track_timescale", str(VIDEO_TIMESCALE),
    ]
    cmd += build_progress_args()
    cmd.append(output_path)
    return cmd
