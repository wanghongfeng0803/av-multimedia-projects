"""单文件转码核心：探测 → 规划 → 编码 → 复核。

批量层（:mod:`bvp.batch`）**只允许通过本模块的 :func:`transcode` 复用单文件
能力**，不得在批量层复制任何 ffmpeg 命令拼装或产物复核逻辑。

刻意保持小而完整（零三方依赖、仅 ffmpeg/ffprobe 子进程）：

- 支持 H.264 / H.265 两种视频编码与 mp4 / mkv 两种容器，足以覆盖批量场景；
- CRF 质量控制、可选分辨率缩放、音频直通/重编码、yuv420p 归一化；
- 失败时清理半成品输出，成功时用 ffprobe 复核产物确实含视频流；
- 所有可预期错误都抛 :mod:`bvp.errors` 中带稳定 ``category`` 的异常，
  批量层据此分类汇总，不解析消息文本（见 :mod:`bvp.contract`）。
"""

from __future__ import annotations

import os
from typing import Callable

from .binaries import find_ffmpeg
from .errors import (
    EncoderUnavailableError,
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
)
from .models import MediaInfo, TranscodeResult
from .probe import probe_media
from .runner import build_progress_args, run_ffmpeg

__all__ = [
    "transcode",
    "supported_codecs",
    "supported_containers",
    "parse_resolution_spec",
    "ProgressCallback",
]

#: 进度回调：0.0-1.0 的完成比例（单调不减）
ProgressCallback = Callable[[float], None]

#: 支持的视频编码注册表：规范名 → (ffmpeg 编码器候选, 默认 CRF)
_CODECS: dict[str, dict] = {
    "h264": {"encoders": ("libx264",), "default_crf": 23, "ffprobe": "h264"},
    "h265": {"encoders": ("libx265",), "default_crf": 28, "ffprobe": "hevc"},
}

#: 支持的容器注册表：规范名 → (扩展名, 是否 faststart)
_CONTAINERS: dict[str, dict] = {
    "mp4": {"extension": ".mp4", "faststart": True},
    "mkv": {"extension": ".mkv", "faststart": False},
}

_TARGET_PIX_FMT = "yuv420p"


def supported_codecs() -> tuple[str, ...]:
    """返回支持的视频编码规范名。"""
    return tuple(_CODECS)


def supported_containers() -> tuple[str, ...]:
    """返回支持的容器规范名。"""
    return tuple(_CONTAINERS)


def transcode(
    input_path: str,
    output_path: str | None = None,
    *,
    codec: str = "h264",
    container: str | None = None,
    crf: int | None = None,
    preset: str = "medium",
    audio_bitrate: str = "128k",
    resolution: str | None = None,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TranscodeResult:
    """转码单个视频文件。

    :param input_path: 输入视频路径，按实际内容探测而非扩展名猜测
    :param output_path: 输出路径；None 时在输入旁生成
        ``<名>_<编码><容器扩展名>``
    :param codec: 目标视频编码，``h264`` / ``h265``
    :param container: 目标容器，``mp4`` / ``mkv``；None 时按输出扩展名推断，
        再退回 mp4
    :param crf: 质量参数；None 用编码默认值
    :param preset: x264/x265 速度档位
    :param audio_bitrate: 音频需重编码时的码率
    :param resolution: 目标分辨率（``"1280x720"`` / ``"720p"`` / ``"50%"``）；
        None 保持源分辨率
    :param overwrite: 输出已存在时是否覆盖
    :param progress_callback: 接收 0.0-1.0 完成比例的回调
    :returns: :class:`TranscodeResult` 结果摘要
    :raises TranscodeError: 各类带稳定 ``category`` 的核心错误
    """
    codec_spec = _resolve_codec(codec)
    if crf is None:
        crf = codec_spec["default_crf"]
    _validate_crf(crf)

    # --- 输入侧校验：不存在/不可访问与非视频严格区分 ---------------------
    if not os.path.isfile(input_path):
        raise InputNotFoundError(
            f"输入文件不存在或不是普通文件: {input_path}"
        )
    try:
        if os.path.getsize(input_path) == 0:
            raise InvalidInputError("输入文件大小为 0 字节，不包含任何媒体数据。")
    except OSError as exc:
        raise InputNotFoundError(
            f"输入文件无法访问: {input_path}", os_error=exc
        ) from exc
    input_path = os.path.abspath(input_path)

    # 探测失败会抛 InvalidInputError（NOT_A_VIDEO → 批量层记跳过）
    info = probe_media(input_path)
    video = info.primary_video
    if video is None:  # probe 已拦截，这里仅做类型收窄
        raise InvalidInputError("输入文件不含视频流。")

    container_spec = _resolve_container(container, output_path)
    output_path = _resolve_output_path(
        input_path, output_path, codec, container_spec
    )
    if os.path.exists(output_path) and not overwrite:
        raise OutputExistsError(
            f"输出文件已存在: {output_path}（传入 overwrite=True 可覆盖）"
        )

    ffmpeg_path = find_ffmpeg()
    encoder = _select_encoder(codec_spec, ffmpeg_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log_path = output_path + ".ffmpeg.log"

    try:
        cmd = _build_command(
            input_path=input_path,
            output_path=output_path,
            info=info,
            video=video,
            codec_name=codec,
            codec_spec=codec_spec,
            encoder=encoder,
            container_spec=container_spec,
            crf=crf,
            preset=preset,
            audio_bitrate=audio_bitrate,
            resolution=resolution,
            overwrite=overwrite,
            ffmpeg_path=ffmpeg_path,
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
                stderr=stderr_tail,
                cmd=cmd,
            )
    except FFmpegRunError:
        # 失败现场：半成品输出绝不能留在磁盘上冒充成功产物，
        # 否则批量复核会把不存在的成功条目对不平
        _safe_remove(output_path)
        raise

    # --- 产物复核：ffmpeg 报告成功不等于产物真的可用 ---------------------
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        _safe_remove(output_path)
        raise FFmpegRunError(
            "ffmpeg 报告成功但未生成有效的输出文件。", cmd=[ffmpeg_path]
        )

    result = _verify_output(input_path, output_path, codec_spec, container_spec)
    # 成功不留 ffmpeg 日志，保持输出目录干净
    _safe_remove(log_path)
    result.ffmpeg_log_path = None
    return result


# ---------------------------------------------------------------------------
# 命令构建
# ---------------------------------------------------------------------------

def _build_command(
    *,
    input_path: str,
    output_path: str,
    info: MediaInfo,
    video,
    codec_name: str,
    codec_spec: dict,
    encoder: str,
    container_spec: dict,
    crf: int,
    preset: str,
    audio_bitrate: str,
    resolution: str | None,
    overwrite: bool,
    ffmpeg_path: str,
) -> list[str]:
    cmd: list[str] = [
        ffmpeg_path, "-hide_banner", "-nostdin",
        "-y" if overwrite else "-n",
    ]
    if info.warnings:
        # 对可能截断/损坏的输入尽量挽救而非直接失败
        cmd += ["-err_detect", "ignore_err", "-fflags", "+genpts+ignidx"]

    cmd += ["-i", input_path]
    cmd += ["-map", "0:v:0"]
    audio = info.primary_audio
    if audio is not None:
        cmd += ["-map", "0:a:0"]

    vf = _video_filter(video, resolution)
    if vf:
        cmd += ["-vf", vf]

    cmd += ["-c:v", encoder, "-crf", str(crf), "-preset", preset]
    cmd += ["-pix_fmt", _TARGET_PIX_FMT]
    if container_spec["faststart"]:
        cmd += ["-movflags", "+faststart"]

    if audio is None:
        cmd += ["-an"]
    elif container_spec["extension"] == ".mkv" and audio.codec_name in {
        "aac", "mp3", "opus", "vorbis", "flac", "ac3"
    }:
        # Matroska 兼容常见音频编码，直通省时
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]

    cmd += build_progress_args()
    cmd += [output_path]
    return cmd


def _video_filter(video, resolution: str | None) -> str:
    """构造视频滤镜：偶数尺寸对齐 + 可选缩放 + 像素格式归一化。"""
    parts: list[str] = []
    if resolution is not None:
        width = video.width or 0
        height = video.height or 0
        if width <= 0 or height <= 0:
            raise InvalidInputError(
                "无法从输入探测到视频尺寸，不能应用目标分辨率参数。"
            )
        parts.append(_scale_filter(resolution, width, height))
    # yuv420p 要求偶数宽高；trunc 向下取整保证兼容
    parts.append("scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos")
    parts.append(f"format={_TARGET_PIX_FMT}")
    parts.append("setsar=1")
    return ",".join(parts)


def _scale_filter(resolution: str, src_w: int, src_h: int) -> str:
    """把分辨率表达式解析为 ffmpeg scale 片段（仅支持少量明确写法）。"""
    kind, value = parse_resolution_spec(resolution)
    if kind == "height":
        return f"scale=-2:{value}"
    if kind == "ratio":
        ratio = max(float(value) / 100.0, 0.01)
        return (
            f"scale=trunc(iw*{ratio:.4f}/2)*2:"
            f"trunc(ih*{ratio:.4f}/2)*2"
        )
    width, height = value
    return f"scale={width}:{height}"


def parse_resolution_spec(resolution: str) -> tuple[str, object]:
    """静态解析分辨率表达式，不依赖任何具体文件的属性。

    规则引擎在启动校验时复用本函数，确保配置里的分辨率写法语法成立，
    而不是等到第一个任务运行到转码阶段才报错。

    :returns: ``("height", 整数高)`` / ``("ratio", 百分比浮点)`` /
        ``("explicit", (宽, 高))``
    :raises InvalidInputError: 写法无法识别（与转码期同一异常类型、同一口径）
    """
    text = resolution.strip().lower()
    if text.endswith("p") and text[:-1].isdigit():
        return "height", int(text[:-1])
    if text.endswith("%"):
        try:
            ratio = float(text[:-1])
        except ValueError as exc:
            raise InvalidInputError(
                f"无法解析分辨率表达式: {resolution}"
            ) from exc
        return "ratio", ratio
    if "x" in text:
        w_text, _, h_text = text.partition("x")
        if w_text.isdigit() and h_text.isdigit():
            return "explicit", (int(w_text), int(h_text))
    raise InvalidInputError(
        f"无法解析分辨率表达式: {resolution}（支持 1280x720 / 720p / 50%）"
    )


# ---------------------------------------------------------------------------
# 规划与复核
# ---------------------------------------------------------------------------

def _resolve_codec(codec: str) -> dict:
    key = (codec or "").strip().lower()
    if key not in _CODECS:
        raise InvalidInputError(
            f"不支持的视频编码 {codec!r}，可选: {', '.join(_CODECS)}。"
        )
    return _CODECS[key]


def _validate_crf(crf: int) -> None:
    if not isinstance(crf, int) or not (0 <= crf <= 51):
        raise InvalidInputError(f"crf 必须是 0-51 的整数，收到 {crf!r}。")


def _resolve_container(container: str | None, output_path: str | None) -> dict:
    if container is not None:
        key = container.strip().lower().lstrip(".")
        if key not in _CONTAINERS:
            raise InvalidInputError(
                f"不支持的容器 {container!r}，可选: {', '.join(_CONTAINERS)}。"
            )
        return _CONTAINERS[key]
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        for spec in _CONTAINERS.values():
            if spec["extension"] == ext:
                return spec
    return _CONTAINERS["mp4"]


def _resolve_output_path(
    input_path: str,
    output_path: str | None,
    codec: str,
    container_spec: dict,
) -> str:
    if output_path:
        if output_path.endswith(("\\", "/")) or os.path.isdir(output_path):
            base = (
                os.path.splitext(os.path.basename(input_path))[0]
                + "_" + codec + container_spec["extension"]
            )
            output_path = os.path.join(output_path, base)
        else:
            # 扩展名缺失/不规范时统一替换为容器扩展名，不让 ffmpeg 猜
            stem = os.path.splitext(output_path)[0]
            output_path = stem + container_spec["extension"]
        return os.path.abspath(output_path)

    stem = os.path.splitext(input_path)[0]
    default = f"{stem}_{codec}{container_spec['extension']}"
    if os.path.abspath(default) == os.path.abspath(input_path):
        default = f"{stem}_transcoded{container_spec['extension']}"
    return os.path.abspath(default)


def _select_encoder(codec_spec: dict, ffmpeg_path: str) -> str:
    """在本机 ffmpeg 上挑选真实可用的编码器。"""
    import subprocess

    proc = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    available = proc.stdout or ""
    for candidate in codec_spec["encoders"]:
        if candidate in available:
            return candidate
    raise EncoderUnavailableError(
        f"本机 ffmpeg 不支持编码 {codec_spec['encoders']} 中的任何一个。",
        candidates=list(codec_spec["encoders"]),
    )


def _verify_output(
    input_path: str,
    output_path: str,
    codec_spec: dict,
    container_spec: dict,
) -> TranscodeResult:
    """用 ffprobe 复核产物确实含视频流且编码符合预期。"""
    try:
        verified = probe_media(output_path)
    except InvalidInputError as exc:
        # 复核不过 = 转码失败，分类保持 TRANSCODE_FAILED
        raise FFmpegRunError(
            f"输出产物复核失败（无法识别为有效视频）: {exc}"
        ) from exc

    video = verified.primary_video
    if video is not None and video.codec_name != codec_spec["ffprobe"]:
        raise FFmpegRunError(
            f"输出视频编码为 {video.codec_name}，"
            f"预期为 {codec_spec['ffprobe']}。"
        )

    return TranscodeResult(
        input_path=input_path,
        output_path=output_path,
        success=True,
        video_codec=video.codec_name if video else None,
        audio_codec=(
            verified.primary_audio.codec_name
            if verified.primary_audio else None
        ),
        width=video.width if video else None,
        height=video.height if video else None,
        duration_seconds=verified.duration_seconds,
        warnings=list(verified.warnings),
        container=next(
            name for name, spec in _CONTAINERS.items()
            if spec is container_spec
        ),
        encoder=None,
    )


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
