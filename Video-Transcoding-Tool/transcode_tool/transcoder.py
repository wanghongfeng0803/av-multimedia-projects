"""转码核心编排：探测 → 规划 → 编码（可分段）→ 拼接 → 复核。

本模块是各底层模块的组装层：
- :mod:`codecs` 决定视频编码器与质量/速度参数；
- :mod:`containers` 决定容器、扩展名与音频直通/重编码策略；
- :mod:`capabilities` 在本机 ffmpeg 上挑选真实可用的编码器；
- :mod:`runner` 以低内存、带进度的方式运行 ffmpeg；
- :mod:`segments` 在大文件场景下把任务切片并无损拼接。

默认行为（H.264 + yuv420p + MP4、CRF 23、medium）与历史版本完全一致，
旧调用方无需任何修改。
"""

from __future__ import annotations

import os
from typing import Callable

from . import capabilities, segments
from .binaries import find_ffmpeg
from .bitrate import (
    BitratePlan,
    MODE_VBR,
    RATE_MODES,
    build_bitrate_encoder_args,
    plan_bitrate,
)
from .codecs import (
    CodecSpec,
    EncoderSpec,
    build_video_encoder_args,
    resolve_codec,
    validate_crf,
    validate_preset,
)
from .containers import (
    ContainerSpec,
    audio_strategy,
    container_from_extension,
    default_container_for,
    ensure_video_compatible,
    normalize_output_extension,
    resolve_container,
)
from .errors import (
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
)
from .models import MediaInfo, StreamInfo, TranscodeResult
from .probe import probe_media
from .runner import build_progress_args, run_ffmpeg
from .scale import ResolutionPlan, parse_resolution

# 进度回调：参数为 0.0-1.0 的完成比例
ProgressCallback = Callable[[float], None]

# 以下两个滤镜为历史实现逐字保留，无分辨率请求时仍走这条路径：
# yuv420p 要求宽高均为偶数；奇数尺寸用此滤镜向下取整并保持宽高比
_EVEN_DIM_FILTER = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,setsar=1"
# 非 yuv420p 像素格式（含 10bit、RGB 等）统一转换，保证最大播放器兼容性
_PIX_FMT_FILTER = (
    "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,"
    "format=yuv420p,setsar=1"
)
_TARGET_PIX_FMT = "yuv420p"


def transcode(
    input_path: str,
    output_path: str | None = None,
    *,
    crf: int | None = None,
    preset: str | None = None,
    overwrite: bool = False,
    audio_bitrate: str = "128k",
    codec: str | CodecSpec = "h264",
    container: str | ContainerSpec | None = None,
    segment_duration: float | None = None,
    segment_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
    target_bitrate: str | int | float | None = None,
    rate_mode: str = MODE_VBR,
    resolution: str | None = None,
    allow_upscale: bool = False,
) -> TranscodeResult:
    """将输入视频转码为指定编码/容器的媒体文件。

    :param input_path: 输入文件路径（扩展名不要求规范，按实际内容探测）
    :param output_path: 输出路径；为 None 时在输入旁生成
        ``<名>_<编码>.<容器扩展名>``
    :param crf: 质量参数；None 时使用该编码的默认值。范围随编码不同
        （H.264/H.265: 0-51，VP9/AV1: 0-63，VP8: 4-63）。
        与 ``target_bitrate`` 互斥
    :param preset: 速度档位，支持十档命名或编码器原生数字档位
    :param overwrite: 输出已存在时是否覆盖
    :param audio_bitrate: 音频需要重编码时使用的码率
    :param codec: 目标视频编码（规范名或别名），默认 h264
    :param container: 目标容器；None 时按输出扩展名推断，再退回到编码首选容器
    :param segment_duration: 大文件分段时长（秒），与 segment_count 二选一
    :param segment_count: 大文件分段数量，segment_duration 优先
    :param progress_callback: 进度回调，接收 0.0-1.0 且单调不减的完成比例
    :param target_bitrate: 目标视频码率（如 ``"2M"``/``"800k"``/整数 bps）。
        给定后改走码率控制路线，忽略 ``crf``；为 None 时保持 CRF 常量质量
    :param rate_mode: 码率控制模式 ``vbr``/``abr``/``cbr``，仅在指定
        ``target_bitrate`` 时生效
    :param resolution: 输出分辨率表达式（``"1920x1080"``/``"1280"``/
        ``"x720"``/``"50%"``/``"1080p"`` 等）；None 时保持源分辨率
    :param allow_upscale: 是否允许把源放大到更大分辨率，默认 False
    :returns: :class:`TranscodeResult` 结果摘要
    :raises: :mod:`transcode_tool.errors` 中定义的各类 TranscodeError
    """
    codec_spec = _coerce_codec(codec)
    if target_bitrate is not None:
        # 码率路线与 CRF 路线互斥：两者同给属于调用方意图冲突，明确报错，
        # 而不是静默二选一
        if crf is not None:
            raise ValueError("crf 与 target_bitrate 互斥，请只指定一种质量控制方式。")
        if rate_mode not in RATE_MODES:
            raise ValueError(
                f"rate_mode 必须是 {', '.join(RATE_MODES)} 之一，收到 {rate_mode!r}。"
            )
        bitrate_plan: BitratePlan | None = plan_bitrate(target_bitrate, rate_mode)
    else:
        bitrate_plan = None
        if crf is None:
            crf = codec_spec.default_crf
        validate_crf(codec_spec, crf)
    if preset is None:
        preset = codec_spec.default_preset
    validate_preset(codec_spec, preset)
    if not isinstance(audio_bitrate, str) or not audio_bitrate.strip():
        raise ValueError("audio_bitrate 必须是非空字符串，如 '128k'。")

    if not os.path.isfile(input_path):
        raise InputNotFoundError(f"输入文件不存在或不是普通文件: {input_path}")
    if os.path.getsize(input_path) == 0:
        raise InvalidInputError("输入文件大小为 0 字节，不包含任何媒体数据。")
    input_path = os.path.abspath(input_path)

    info = probe_media(input_path)
    video = info.primary_video
    if video is None:  # probe_media 已拦截，这里仅做类型收窄
        raise InvalidInputError("输入文件不含视频流。")

    container_spec = _select_container(container, output_path, codec_spec)
    ensure_video_compatible(container_spec, codec_spec)

    output_path = _resolve_output_path(
        input_path, output_path, codec_spec, container_spec
    )
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(
            f"输出文件已存在: {output_path}（传入 overwrite=True 可覆盖）"
        )

    # 在本机 ffmpeg 上确定真实可用的视频/音频编码器
    ffmpeg_path = find_ffmpeg()
    video_encoder = capabilities.select_video_encoder(codec_spec, ffmpeg_path)

    warnings = list(info.warnings)

    # 分辨率规划（可选）：无请求时为 None，滤镜走历史逻辑，行为完全不变
    resolution_plan = _plan_resolution(
        resolution, video, allow_upscale, warnings
    )

    plan = segments.plan_segments(
        info.duration_seconds,
        segment_duration=segment_duration,
        segment_count=segment_count,
    )
    warnings.extend(plan.warnings)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log_path = output_path + ".ffmpeg.log"

    try:
        if plan.count == 1:
            _run_single(
                input_path, output_path, info, video, codec_spec, video_encoder,
                container_spec, crf, preset, audio_bitrate, overwrite,
                log_path, warnings, progress_callback, ffmpeg_path,
                bitrate_plan, resolution_plan,
            )
        else:
            _run_segmented(
                input_path, output_path, info, video, codec_spec, video_encoder,
                container_spec, crf, preset, audio_bitrate, overwrite,
                log_path, warnings, progress_callback, ffmpeg_path, plan,
                bitrate_plan, resolution_plan,
            )
    except FFmpegRunError as exc:
        # runner 已把 stderr 流式写入 log_path；异常未携带日志内容时补写，
        # 保证失败现场一定有完整 ffmpeg 日志可查
        if not os.path.exists(log_path) and exc.stderr:
            _write_log(log_path, exc.stderr)
        _safe_remove(output_path)
        raise

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        _safe_remove(output_path)
        raise FFmpegRunError(
            "ffmpeg 报告成功但未生成有效的输出文件。", cmd=[ffmpeg_path]
        )

    # 成功时清理 runner 预先创建的日志文件，保持“成功不留日志”的历史语义
    _safe_remove(log_path)

    result = _verify_output(
        input_path, output_path, codec_spec, container_spec, log_path
    )
    result.warnings = warnings + result.warnings
    result.container = container_spec.canonical
    result.encoder = video_encoder.name
    if bitrate_plan is not None:
        result.target_bitrate_bps = bitrate_plan.target_bps
        result.rate_mode = bitrate_plan.mode
    if not os.path.exists(log_path):
        result.ffmpeg_log_path = None
    return result


# ---------------------------------------------------------------------------
# 单次整体转码
# ---------------------------------------------------------------------------

def _run_single(
    input_path, output_path, info, video, codec_spec, video_encoder,
    container_spec, crf, preset, audio_bitrate, overwrite,
    log_path, warnings, progress_callback, ffmpeg_path,
    bitrate_plan, resolution_plan,
) -> None:
    cmd = _build_command(
        input_path=input_path,
        output_path=output_path,
        info=info,
        video=video,
        codec=codec_spec,
        video_encoder=video_encoder,
        container=container_spec,
        crf=crf,
        preset=preset,
        audio_bitrate=audio_bitrate,
        overwrite=overwrite,
        warnings=warnings,
        ffmpeg_path=ffmpeg_path,
        seek=None,
        duration=None,
        bitrate_plan=bitrate_plan,
        resolution_plan=resolution_plan,
    )
    returncode, stderr_tail = run_ffmpeg(
        cmd,
        log_path=log_path,
        total_duration=info.duration_seconds,
        progress_callback=progress_callback,
    )
    if returncode != 0:
        raise FFmpegRunError(
            f"ffmpeg 转码失败（退出码 {returncode}）: {_tail(stderr_tail)}",
            stderr=_read_log_tail(log_path, stderr_tail),
            cmd=cmd,
        )


# ---------------------------------------------------------------------------
# 分段转码
# ---------------------------------------------------------------------------

def _run_segmented(
    input_path, output_path, info, video, codec_spec, video_encoder,
    container_spec, crf, preset, audio_bitrate, overwrite,
    log_path, warnings, progress_callback, ffmpeg_path, plan,
    bitrate_plan, resolution_plan,
) -> None:
    workdir = segments.make_temp_workdir()
    segment_paths: list[str] = []
    total_duration = info.duration_seconds or 0.0
    try:
        last_ratio = 0.0
        for segment in plan.segments:
            seg_path = segments.segment_output_path(
                workdir, segment, container_spec.extension
            )
            segment_paths.append(seg_path)
            span = plan.span_ratio(segment, total_duration)
            cmd = _build_command(
                input_path=input_path,
                output_path=seg_path,
                info=info,
                video=video,
                codec=codec_spec,
                video_encoder=video_encoder,
                container=container_spec,
                crf=crf,
                preset=preset,
                audio_bitrate=audio_bitrate,
                overwrite=True,  # 临时目录内的文件始终可覆盖
                warnings=warnings,
                ffmpeg_path=ffmpeg_path,
                seek=segment.start,
                duration=segment.duration,
                bitrate_plan=bitrate_plan,
                resolution_plan=resolution_plan,
            )
            returncode, stderr_tail = run_ffmpeg(
                cmd,
                log_path=log_path,
                total_duration=segment.duration,
                progress_callback=progress_callback,
                last_ratio=last_ratio,
                ratio_span=span,
            )
            if returncode != 0:
                raise FFmpegRunError(
                    f"第 {segment.index + 1}/{plan.count} 段转码失败"
                    f"（退出码 {returncode}）: {_tail(stderr_tail)}",
                    stderr=_read_log_tail(log_path, stderr_tail),
                    cmd=cmd,
                )
            if not os.path.isfile(seg_path) or os.path.getsize(seg_path) == 0:
                raise FFmpegRunError(
                    f"第 {segment.index + 1} 段未生成有效文件。", cmd=cmd
                )
            last_ratio = min(last_ratio + span, 1.0)

        list_path = segments.write_concat_list(workdir, segment_paths)
        segments.concat_segments(
            list_path, output_path,
            ffmpeg_path=ffmpeg_path, log_path=log_path,
        )
    finally:
        segments.cleanup_workdir(workdir)


# ---------------------------------------------------------------------------
# 命令构建
# ---------------------------------------------------------------------------

def _build_command(
    *,
    input_path: str,
    output_path: str,
    info: MediaInfo,
    video: StreamInfo,
    codec: CodecSpec,
    video_encoder: EncoderSpec,
    container: ContainerSpec,
    crf: int | None,
    preset: str,
    audio_bitrate: str,
    overwrite: bool,
    warnings: list[str],
    ffmpeg_path: str,
    seek: float | None,
    duration: float | None,
    bitrate_plan: BitratePlan | None,
    resolution_plan: ResolutionPlan | None,
) -> list[str]:
    cmd: list[str] = [
        ffmpeg_path, "-hide_banner", "-nostdin",
        "-y" if overwrite else "-n",
    ]

    # 对可能截断/损坏的输入尽量挽救，而不是直接失败
    if info.warnings:
        cmd += ["-err_detect", "ignore_err", "-fflags", "+genpts+ignidx"]

    cmd += ["-i", input_path]

    # 分段切割采用 ffmpeg 官方推荐的“双 -ss”模式：
    # 输入侧 -ss 先快速寻址到目标点之前的关键帧（避免从头解码，保证大文件性能）；
    # 输出侧 -ss 0 把时间戳原点重置到寻址点，使各流（含 -c:a copy 的音频）
    # 都从 0 开始。否则在 MKV 等不归一化时间戳的容器中，视频流带着关键帧回退
    # 的偏移、copy 的音频从 0 开始，两者错位会把容器时长拉长一个关键帧间隔，
    # 多段累积误差很大。输出侧 -t 精确截断本段时长。
    is_segment = seek is not None
    if is_segment and seek > 0:
        cmd += ["-ss", f"{seek:.3f}"]

    cmd += ["-i", input_path]

    if is_segment:
        # 输出寻址：定位到本段精确起点（输入已落到该点附近，二次偏移接近 0）
        cmd += ["-ss", "0"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]

    # 只保留主视频流与主音频流，丢弃字幕/数据流等兼容性差的流
    cmd += ["-map", "0:v:0"]
    audio = info.primary_audio
    if audio is not None:
        cmd += ["-map", "0:a:0"]
    elif seek is None:  # 分段模式下无需对每段重复同一条警告
        warnings.append("输入无音频流，输出将为纯视频文件。")

    if seek is None:
        dropped = len(info.streams) - len(info.video_streams[:1]) - (
            1 if audio is not None else 0
        )
        if dropped > 0:
            warnings.append(
                f"已丢弃 {dropped} 条额外流（多余音轨/字幕/数据流等）。"
            )

    vf, video_warnings = _video_filter(video, codec, resolution_plan)
    if seek is None:
        warnings.extend(video_warnings)
    if vf:
        cmd += ["-vf", vf]

    cmd += ["-c:v", video_encoder.name]
    if bitrate_plan is None:
        # CRF 常量质量路线（历史默认行为）
        cmd += build_video_encoder_args(codec, video_encoder, crf, preset)
    else:
        # 目标码率路线：按编码器分别生成真实可用的码控参数，不再传 -crf
        cmd += build_bitrate_encoder_args(
            codec, video_encoder, bitrate_plan, preset,
            container_canonical=container.canonical,
        )
    cmd += ["-pix_fmt", _TARGET_PIX_FMT]

    # MP4 系容器前置 moov atom 便于网络播放；分段临时文件也保持一致
    if container.faststart:
        cmd += ["-movflags", "+faststart"]

    cmd += _audio_args(container, audio, audio_bitrate, warnings, seek is None)
    cmd += build_progress_args()
    cmd += [output_path]
    return cmd


def _audio_args(
    container: ContainerSpec,
    audio: StreamInfo | None,
    audio_bitrate: str,
    warnings: list[str],
    announce: bool,
) -> list[str]:
    mode, encoder = audio_strategy(
        container, audio.codec_name if audio is not None else None
    )
    if mode == "drop":
        return ["-an"]
    if mode == "copy":
        return ["-c:a", "copy"]

    # 需要重编码：在本机能力范围内挑选真实可用的音频编码器
    chosen = capabilities.select_audio_encoder(
        encoder, container.audio_encoder_fallbacks
    )
    if announce:
        warnings.append(
            f"音频编码 {audio.codec_name or '未知'} 与 {container.canonical} "
            f"容器不兼容，已重编码为 {chosen}。"
        )
    return ["-c:a", chosen, "-b:a", audio_bitrate]


def _plan_resolution(
    resolution: str | None,
    video: StreamInfo,
    allow_upscale: bool,
    warnings: list[str],
) -> ResolutionPlan | None:
    """把可选的分辨率表达式解析为具体尺寸规划，并把告警并入主告警列表。"""
    if resolution is None:
        return None
    width, height = video.width or 0, video.height or 0
    if width <= 0 or height <= 0:
        # 探测不到尺寸时无法做等比规划：明确报错而非产生不可控输出
        raise InvalidInputError(
            "无法从输入探测到视频尺寸，不能应用目标分辨率参数。"
        )
    plan = parse_resolution(resolution, width, height, allow_upscale=allow_upscale)
    warnings.extend(plan.warnings)
    return plan


def _video_filter(
    video: StreamInfo,
    codec: CodecSpec,
    resolution_plan: ResolutionPlan | None,
) -> tuple[str, list[str]]:
    """根据源像素格式/尺寸与可选分辨率规划决定视频滤镜。

    两条分支：
      - 无分辨率请求：**完全复刻历史滤镜字符串与告警**，保证旧行为不退化；
      - 有分辨率请求：三段式管线「显式缩放 →（可选）像素格式归一化 →
        setsar=1」，目标尺寸已在 Python 侧保证偶数并保持宽高比。
    """
    warnings: list[str] = []
    width, height = video.width or 0, video.height or 0
    odd = bool(width and height and (width % 2 or height % 2))

    if odd and resolution_plan is None:
        warnings.append(
            f"源视频尺寸 {width}x{height} 含奇数像素，已向下取整为偶数尺寸"
            f"以适配 {codec.canonical}（yuv420p 要求偶数宽高）。"
        )
    if video.pix_fmt and video.pix_fmt not in {"yuv420p", "yuvj420p"} \
            and resolution_plan is None:
        warnings.append(
            f"源像素格式 {video.pix_fmt} 已转换为 {_TARGET_PIX_FMT}（通用兼容格式）。"
        )

    # --- 分支一：历史路径，滤镜字符串逐字保持 ---
    if resolution_plan is None:
        if video.pix_fmt == "yuv420p":
            return (_EVEN_DIM_FILTER if odd else "", warnings)
        return _PIX_FMT_FILTER, warnings

    # --- 分支二：目标分辨率路径 ---
    if video.pix_fmt and video.pix_fmt not in {"yuv420p", "yuvj420p"}:
        warnings.append(
            f"源像素格式 {video.pix_fmt} 已转换为 {_TARGET_PIX_FMT}（通用兼容格式）。"
        )
    chain = [resolution_plan.scale_filter]
    if video.pix_fmt not in {"yuv420p", "yuvj420p"}:
        chain.append(f"format={_TARGET_PIX_FMT}")
    chain.append("setsar=1")
    return ",".join(chain), warnings


# ---------------------------------------------------------------------------
# 容器选择与输出路径
# ---------------------------------------------------------------------------

def _coerce_codec(codec: str | CodecSpec) -> CodecSpec:
    if isinstance(codec, CodecSpec):
        return codec
    return resolve_codec(codec)


def _coerce_container(container: str | ContainerSpec | None) -> ContainerSpec | None:
    if container is None or isinstance(container, ContainerSpec):
        return container
    return resolve_container(container)


def _select_container(
    container: str | ContainerSpec | None,
    output_path: str | None,
    codec: CodecSpec,
) -> ContainerSpec:
    explicit = _coerce_container(container)
    if explicit is not None:
        return explicit
    if output_path:
        inferred = container_from_extension(output_path)
        if inferred is not None:
            return inferred
    return default_container_for(codec)


def _resolve_output_path(
    input_path: str,
    output_path: str | None,
    codec: CodecSpec,
    container: ContainerSpec,
) -> str:
    if output_path:
        # 输出被写成目录路径时，放入该目录并沿用输入文件名
        if output_path.endswith(("\\", "/")) or os.path.isdir(output_path):
            base = (
                os.path.splitext(os.path.basename(input_path))[0]
                + "_" + codec.canonical + container.extension
            )
            output_path = os.path.join(output_path, base)
        else:
            # 扩展名缺失/与容器不一致：规范化为容器扩展名，
            # 不让 ffmpeg 按扩展名猜测容器
            output_path = normalize_output_extension(output_path, container)
        return os.path.abspath(output_path)

    stem = os.path.splitext(input_path)[0]
    default = f"{stem}_{codec.canonical}{container.extension}"
    # 输入与默认输出同名时避免自覆盖
    if os.path.abspath(default) == os.path.abspath(input_path):
        default = f"{stem}_transcoded{container.extension}"
    return os.path.abspath(default)


# ---------------------------------------------------------------------------
# 产物复核
# ---------------------------------------------------------------------------

def _verify_output(
    input_path: str,
    output_path: str,
    codec: CodecSpec,
    container: ContainerSpec,
    log_path: str,
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
            container=container.canonical,
            encoder=None,
        )

    video = verified.primary_video
    if video is not None and video.codec_name != codec.ffprobe_name:
        raise FFmpegRunError(
            f"输出视频编码为 {video.codec_name}，预期为 {codec.ffprobe_name}。"
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
        container=container.canonical,
        encoder=None,
    )


# ---------------------------------------------------------------------------
# 日志与清理辅助
# ---------------------------------------------------------------------------

def _read_log_tail(log_path: str, fallback: str) -> str:
    """失败时尽量取回完整日志的末尾片段用于异常信息。"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        return content or fallback
    except OSError:
        return fallback


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
