"""转码核心逻辑：把任意可解码输入转成 H.264(yuv420p) + MP4。"""

from __future__ import annotations

import os
import subprocess

from .binaries import find_ffmpeg
from .errors import (
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
)
from .models import MediaInfo, StreamInfo, TranscodeResult
from .probe import probe_media

# yuv420p 要求宽高均为偶数；不规范的奇数尺寸用此滤镜向下取整并保持宽高比
_EVEN_DIM_FILTER = (
    "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,setsar=1"
)
# 非 yuv420p 像素格式（含 10bit、RGB 等）统一转换，保证最大播放器兼容性
_PIX_FMT_FILTER = (
    "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,"
    "format=yuv420p,setsar=1"
)

# MP4 容器中可直接保留、无需重新编码的音频编码
_MP4_PASSTHROUGH_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}


def transcode(
    input_path: str,
    output_path: str | None = None,
    *,
    crf: int = 23,
    preset: str = "medium",
    overwrite: bool = False,
    audio_bitrate: str = "128k",
) -> TranscodeResult:
    """将输入视频转码为 H.264 编码的 MP4 文件。

    :param input_path: 输入文件路径（扩展名不要求是 .mp4，按实际内容探测）
    :param output_path: 输出路径；为 None 时在输入旁生成 ``<名>_h264.mp4``
    :param crf: x264 质量参数（0-51，越小质量越高，常用 18-28）
    :param preset: x264 速度档位（如 ultrafast / medium / slow）
    :param overwrite: 输出已存在时是否覆盖
    :param audio_bitrate: 音频需要重编码时使用的码率
    :returns: :class:`TranscodeResult` 结果摘要
    :raises: :mod:`transcode_tool.errors` 中定义的各类 TranscodeError
    """
    _validate_crf(crf)
    _validate_preset(preset)

    if not os.path.isfile(input_path):
        raise InputNotFoundError(f"输入文件不存在或不是普通文件: {input_path}")
    input_path = os.path.abspath(input_path)

    output_path = _resolve_output_path(input_path, output_path)
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(
            f"输出文件已存在: {output_path}（传入 overwrite=True 可覆盖）"
        )

    info = probe_media(input_path)
    video = info.primary_video
    if video is None:  # probe_media 已拦截，这里仅做类型收窄
        raise InvalidInputError("输入文件不含视频流。")

    warnings = list(info.warnings)
    cmd = _build_command(
        input_path=input_path,
        output_path=output_path,
        info=info,
        video=video,
        crf=crf,
        preset=preset,
        audio_bitrate=audio_bitrate,
        overwrite=overwrite,
        warnings=warnings,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log_path = output_path + ".ffmpeg.log"

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        _write_log(log_path, proc.stderr)
        # 失败时清理 ffmpeg 可能留下的半成品文件
        _safe_remove(output_path)
        raise FFmpegRunError(
            f"ffmpeg 转码失败（退出码 {proc.returncode}）: {_tail(proc.stderr)}",
            stderr=proc.stderr,
            cmd=cmd,
        )

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        _write_log(log_path, proc.stderr)
        _safe_remove(output_path)
        raise FFmpegRunError(
            "ffmpeg 报告成功但未生成有效的输出文件。", stderr=proc.stderr, cmd=cmd
        )

    # 用 ffprobe 复核输出确实是 H.264
    result_info = _verify_output(input_path, output_path, log_path, proc.stderr)
    result_info.warnings = warnings + result_info.warnings
    # 日志仅在失败时落盘，成功时不报告一个不存在的路径
    if not os.path.exists(log_path):
        result_info.ffmpeg_log_path = None
    return result_info


def _build_command(
    *,
    input_path: str,
    output_path: str,
    info: MediaInfo,
    video: StreamInfo,
    crf: int,
    preset: str,
    audio_bitrate: str,
    overwrite: bool,
    warnings: list[str],
) -> list[str]:
    ffmpeg = find_ffmpeg()
    cmd: list[str] = [ffmpeg, "-hide_banner", "-nostdin", "-y" if overwrite else "-n"]

    # 对可能截断/损坏的输入尽量挽救，而不是直接失败
    if info.warnings:
        cmd += ["-err_detect", "ignore_err", "-fflags", "+genpts+ignidx"]

    cmd += ["-i", input_path]

    # 只保留主视频流与主音频流，丢弃字幕/数据流等 MP4 兼容性差的流
    cmd += ["-map", "0:v:0"]
    audio = info.primary_audio
    if audio is not None:
        cmd += ["-map", "0:a:0"]
    else:
        warnings.append("输入无音频流，输出将为纯视频文件。")

    dropped = len(info.streams) - len(info.video_streams[:1]) - (
        1 if audio is not None else 0
    )
    if dropped > 0:
        warnings.append(f"已丢弃 {dropped} 条额外流（多余音轨/字幕/数据流等）。")

    vf, video_warnings = _video_filter(video)
    warnings.extend(video_warnings)
    if vf:
        cmd += ["-vf", vf]

    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        # 快速启动：moov atom 前置，便于网络播放
        "-movflags", "+faststart",
    ]

    if audio is None:
        cmd += ["-an"]
    elif audio.codec_name in _MP4_PASSTHROUGH_AUDIO:
        cmd += ["-c:a", "copy"]
    else:
        warnings.append(
            f"音频编码 {audio.codec_name or '未知'} 与 MP4 兼容性不佳，已重编码为 AAC。"
        )
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]

    cmd += [output_path]
    return cmd


def _video_filter(video: StreamInfo) -> tuple[str, list[str]]:
    """根据源视频像素格式与尺寸决定视频滤镜。"""
    warnings: list[str] = []
    width, height = video.width or 0, video.height or 0
    odd = bool(width and height and (width % 2 or height % 2))

    if odd:
        warnings.append(
            f"源视频尺寸 {width}x{height} 含奇数像素，已向下取整为偶数尺寸以适配 H.264。"
        )

    # yuv420p 仅需保证偶数尺寸；其它像素格式（10bit、RGB、yuv444 等）统一转换
    if video.pix_fmt == "yuv420p":
        return (_EVEN_DIM_FILTER if odd else "", warnings)

    if video.pix_fmt and video.pix_fmt not in {"yuv420p", "yuvj420p"}:
        warnings.append(
            f"源像素格式 {video.pix_fmt} 已转换为 yuv420p（H.264 通用兼容格式）。"
        )
    return _PIX_FMT_FILTER, warnings


def _verify_output(
    input_path: str, output_path: str, log_path: str, ffmpeg_stderr: str
) -> TranscodeResult:
    verify_warnings: list[str] = []
    try:
        verified = probe_media(output_path)
    except Exception as exc:  # 输出探测失败不应掩盖产物，作为警告记录
        verify_warnings.append(f"输出复核探测失败: {exc}")
        return TranscodeResult(
            input_path=input_path,
            output_path=output_path,
            success=True,
            video_codec=None,
            audio_codec=None,
            width=None,
            height=None,
            duration_seconds=None,
            warnings=verify_warnings,
            ffmpeg_log_path=log_path,
        )

    video = verified.primary_video
    if video is not None and video.codec_name != "h264":
        raise FFmpegRunError(
            f"输出视频编码为 {video.codec_name}，预期为 h264。", stderr=ffmpeg_stderr
        )

    return TranscodeResult(
        input_path=input_path,
        output_path=output_path,
        success=True,
        video_codec=video.codec_name if video else None,
        audio_codec=verified.primary_audio.codec_name if verified.primary_audio else None,
        width=video.width if video else None,
        height=video.height if video else None,
        duration_seconds=verified.duration_seconds,
        warnings=verify_warnings,
        ffmpeg_log_path=log_path,
    )


def _resolve_output_path(input_path: str, output_path: str | None) -> str:
    if output_path:
        # 输出被写成目录路径时，放入该目录并沿用输入文件名
        if output_path.endswith(("\\", "/")) or os.path.isdir(output_path):
            base = os.path.splitext(os.path.basename(input_path))[0] + "_h264.mp4"
            output_path = os.path.join(output_path, base)
        elif os.path.splitext(output_path)[1].lower() != ".mp4":
            # 扩展名缺失或不规范：补成 .mp4 而不是让 ffmpeg 按扩展名猜容器
            output_path = output_path + ".mp4"
        return os.path.abspath(output_path)

    stem = os.path.splitext(input_path)[0]
    default = stem + "_h264.mp4"
    # 输入与默认输出同名（如本身就叫 xxx_h264.mp4）时避免自覆盖
    if os.path.abspath(default) == os.path.abspath(input_path):
        default = stem + "_transcoded.mp4"
    return os.path.abspath(default)


def _validate_crf(crf: int) -> None:
    if not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf 必须是 0-51 之间的整数。")


def _validate_preset(preset: str) -> None:
    allowed = {
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow", "placebo",
    }
    if preset not in allowed:
        raise ValueError(f"preset 必须是以下之一: {', '.join(sorted(allowed))}")


def _write_log(log_path: str, content: str) -> None:
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(content or "")
    except OSError:
        pass


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _tail(text: str, max_chars: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return "…" + text[-max_chars:]
