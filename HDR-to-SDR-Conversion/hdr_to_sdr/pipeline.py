"""转换流水线：ffmpeg 解码 → Python 逐帧引擎 → ffmpeg 编码。

元数据处理（对应需求「转换后清 HDR 元数据」）：

* 编码侧**显式**打上 BT.709 SDR 标签（primaries/transfer/matrix=bt709），
  不依赖编码器默认值；
* ``-map_metadata -1`` 丢弃容器级全局元数据，避免源容器里的
  ``HDR_FORMAT`` 之类标签残留；
* 输出走全新编码进程，HDR 专用 side_data（MDCV/CTA-861.3）不会被复制。

音频策略：优先 ``-map 0:a? -c:a copy`` 原样直通（不重编码、不碰响度）；
若目标容器不支持源音频编码导致 ffmpeg 失败，回退 AAC 192k 重编一次。
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass

from .binaries import find_ffmpeg, find_ffprobe
from .colorscience import RANGE_LIMITED, RANGE_FULL
from .encoders import (
    build_quality_args,
    build_vui_args,
    encoder_spec,
)
from .errors import FFmpegRunError, InputNotFoundError
from .frames import FrameProcessor
from .probe import HDRInfo, probe_hdr
from .tonemap import (
    DEFAULT_TONE_METHOD,
    DEFAULT_REINHARD_WHITE,
    DEFAULT_HABLE_SHARPNESS,
)
from .transfer import TRANSFER_PQ, TRANSFER_HLG

__all__ = ["ConversionResult", "convert_hdr_to_sdr", "build_decode_cmd", "build_encode_cmd"]

# 中间格式与每像素字节数（444 三平面）
_YUV_RAW_FMT = "yuv444p16le"
_RGB_RAW_FMT = "gbrpf32le"


@dataclass
class ConversionResult:
    """一次转换的产物描述。"""

    output_path: str
    transfer: str
    width: int
    height: int
    frames_processed: int
    tone_method: str
    audio_mode: str  # copy / aac / none
    assumed: bool

    def __repr__(self) -> str:
        kind = "PQ" if self.transfer == TRANSFER_PQ else "HLG"
        return (
            f"<ConversionResult {kind}→SDR {self.width}x{self.height} "
            f"frames={self.frames_processed} tone={self.tone_method} "
            f"audio={self.audio_mode} -> {self.output_path}>"
        )


def build_decode_cmd(
    ffmpeg: str, src_path: str, info: HDRInfo
) -> tuple[list[str], str, int]:
    """构建解码命令，返回 (cmd, raw_pix_fmt, bytes_per_pixel)。"""
    raw_fmt = _RGB_RAW_FMT if info.is_rgb else _YUV_RAW_FMT
    bpp = 12 if info.is_rgb else 6
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-map", "0:v:0",
        "-an",
        "-sn", "-dn",
        "-pix_fmt", raw_fmt,
        "-f", "rawvideo",
        "pipe:1",
    ]
    return cmd, raw_fmt, bpp


def build_encode_cmd(
    ffmpeg: str,
    output_path: str,
    width: int,
    height: int,
    frame_rate: float | None,
    raw_fmt: str,
    codec: str,
    dst_bits: int,
    dst_range: str,
    audio_mode: str,
    audio_src: str | None,
    crf: int,
    preset: str,
) -> list[str]:
    """构建编码命令。

    色彩标签全部显式写死为 BT.709 SDR；``-map_metadata -1`` 清全局元数据。
    """
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo",
        "-pix_fmt", raw_fmt,
        "-s", f"{width}x{height}",
    ]
    if frame_rate:
        cmd += ["-r", f"{frame_rate:.6f}"]
    cmd += ["-i", "pipe:0"]

    if audio_mode in ("copy", "aac") and audio_src:
        cmd += ["-i", audio_src]
        if audio_mode == "copy":
            cmd += ["-map", "1:a?", "-c:a", "copy"]
        else:
            cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]

    # 输出像素格式：RGB 中间态交由 ffmpeg 转 YUV；YUV 中间态选目标码深 444
    if raw_fmt == _RGB_RAW_FMT:
        out_pix = "yuv420p" if dst_bits == 8 else "yuv420p10le"
    else:
        out_pix = "yuv444p" if dst_bits == 8 else "yuv444p10le"

    # 软编（libx264/libx265）实测外层 -color_* 不进码流，必须内联 VUI；
    # 硬编（nvenc/qsv/amf）外层标签会写进码流。质量控制各厂参数不同，
    # 统一在 encoders 模块按编码器族翻译，画面数学在 Python 引擎侧完全一致。
    inline_vui, outer_vui = build_vui_args(codec)
    quality = build_quality_args(codec, crf, preset)

    cmd += [
        "-map", "0:v:0",
        "-c:v", codec,
        *quality,
        "-pix_fmt", out_pix,
        *inline_vui,
        # 外层标签保留：部分容器/版本仍读外层，内外双写互为兜底
        *outer_vui,
        "-color_range", "tv" if dst_range == RANGE_LIMITED else "pc",
        # 丢弃容器级全局元数据，避免源 HDR_FORMAT 等标签残留
        "-map_metadata", "-1",
        output_path,
    ]
    return cmd


def _drain_stderr(proc: subprocess.Popen, sink: list[str]) -> None:
    """后台线程抽干 stderr，防止管道写满导致 ffmpeg 阻塞（Windows 尤其敏感）。"""
    try:
        text = proc.stderr.read()
        if text:
            sink.append(text)
    except (ValueError, OSError):
        pass


def convert_hdr_to_sdr(
    src_path: str,
    output_path: str,
    *,
    assume: str | None = None,
    tone_method: str = DEFAULT_TONE_METHOD,
    codec: str = "libx264",
    dst_bits: int = 8,
    dst_range: str = RANGE_LIMITED,
    crf: int = 18,
    preset: str = "medium",
    workers: int = 1,
    progress_every: int = 0,
    reinhard_white: float = DEFAULT_REINHARD_WHITE,
    hable_sharpness: float = DEFAULT_HABLE_SHARPNESS,
) -> ConversionResult:
    """执行完整 HDR→SDR 转换。

    :param assume: 元数据无法确证类型时的显式兜底（"pq"/"hlg"）。
    :param tone_method: reinhard / reinhard_classic / hable / bt2390 / clip。
    :param codec: 视频编码器，软编 libx264/libx265 或硬编
        h264_nvenc/hevc_nvenc/h264_qsv/hevc_qsv/h264_amf/hevc_amf。
        软/硬编共用同一 Python 引擎产出的原始帧，画面一致、仅末端编码不同。
    :param dst_bits: 输出码深 8 或 10。
    :param dst_range: limited / full。
    :param workers: 帧内分带并行进程数，1 为串行（结果逐像素一致）。
    :param progress_every: 每处理 N 帧打印一次进度，0 表示不打印。
    :param reinhard_white: 经典 Reinhard（reinhard_classic）白点，相对
        peak 的倍数；越界钳制并 warning，其它色调方法忽略。
    :param hable_sharpness: Hable/Filmic（hable）肩部锐度，越界钳制并
        warning，其它色调方法忽略。
    :raises InputNotFoundError: 输入不存在。
    :raises FFmpegRunError: ffmpeg 解码/编码失败。
    """
    if not os.path.isfile(src_path):
        raise InputNotFoundError(f"输入文件不存在: {src_path}")
    if dst_bits not in (8, 10):
        raise ValueError("输出码深仅支持 8 或 10")
    if dst_range not in (RANGE_LIMITED, RANGE_FULL):
        raise ValueError("dst_range 仅支持 limited / full")
    # encoder_spec 对未知编码器抛 ValueError（白名单校验，拒绝注入任意 -c:v）
    encoder_spec(codec)

    info = probe_hdr(src_path, assume=assume)
    ffmpeg = find_ffmpeg()
    has_audio = _probe_has_audio(src_path)

    processor = FrameProcessor(
        transfer=info.transfer,
        src_bits=info.bits_per_raw_sample,
        pixel_range=info.pixel_range,
        is_rgb=info.is_rgb,
        tone_method=tone_method,
        peak_nits=info.peak_nits,
        dst_bits=dst_bits,
        dst_range=dst_range,
        reinhard_white=reinhard_white,
        hable_sharpness=hable_sharpness,
    )

    decode_cmd, raw_fmt, bpp = build_decode_cmd(ffmpeg, src_path, info)
    frame_size = info.width * info.height * bpp

    # 第一次：音频直通；若编码端因容器不兼容失败，第二次回退 AAC 重编。
    # 无音频流时直接 none，避免 ffmpeg 对空 map 报错。
    # 注意：每次尝试都要新建解码进程——上一轮的 stdout 已读到 EOF 无法复用。
    attempts = ["copy", "aac"] if has_audio else ["none"]
    frames = 0
    used_mode = attempts[0]

    for audio_mode in attempts:
        encode_cmd = build_encode_cmd(
            ffmpeg, output_path, info.width, info.height, info.frame_rate,
            raw_fmt, codec, dst_bits, dst_range,
            audio_mode, src_path if has_audio else None, crf, preset,
        )
        outcome = _run_pipeline(
            decode_cmd, encode_cmd, processor, info, frame_size,
            workers, progress_every,
        )
        if outcome is not None:
            frames = outcome
            used_mode = audio_mode
            break
        # 编码端失败：仅 copy 模式允许回退到 aac 重试，其余直接抛
        if audio_mode != "copy":
            raise FFmpegRunError(encode_cmd, 1, "编码失败且无回退手段")

    return ConversionResult(
        output_path=output_path,
        transfer=info.transfer,
        width=info.width,
        height=info.height,
        frames_processed=frames,
        tone_method=processor.tone_method,
        audio_mode=used_mode,
        assumed=info.assumed,
    )


def _run_pipeline(
    decode_cmd: list[str],
    encode_cmd: list[str],
    processor: FrameProcessor,
    info: HDRInfo,
    frame_size: int,
    workers: int,
    progress_every: int,
) -> int | None:
    """跑一次双进程管线。成功返回帧数；编码端失败返回 None（触发上层回退）。"""
    dec = subprocess.Popen(
        decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=frame_size,
    )
    enc = subprocess.Popen(
        encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=frame_size,
    )
    dec_err: list[str] = []
    enc_err: list[str] = []
    threading.Thread(target=_drain_stderr, args=(dec, dec_err), daemon=True).start()
    threading.Thread(target=_drain_stderr, args=(enc, enc_err), daemon=True).start()

    frames = 0
    try:
        while True:
            chunk = _read_exact(dec.stdout, frame_size)
            if len(chunk) < frame_size:
                break
            if workers > 1:
                out = processor.process_bands(
                    chunk, info.width, info.height, workers=workers
                )
            else:
                out = processor.process_frame(chunk, info.width, info.height)
            enc.stdin.write(out)
            frames += 1
            if progress_every and frames % progress_every == 0:
                print(f"[hdr-to-sdr] 已处理 {frames} 帧")
        enc.stdin.close()
    except BrokenPipeError:
        pass  # 编码端先死，错误信息从其 stderr 取
    finally:
        dec.wait()
        enc_rc = enc.wait()

    if dec.returncode != 0:
        raise FFmpegRunError(decode_cmd, dec.returncode, "".join(dec_err)[-1200:])
    if enc_rc != 0:
        if frames == 0:
            raise FFmpegRunError(encode_cmd, enc_rc, "".join(enc_err)[-1200:])
        return None  # 编码失败（典型：音频 codec 容器不兼容），交由上层回退
    if frames == 0:
        raise FFmpegRunError(decode_cmd, dec.returncode or 1, "未解码出任何视频帧")
    return frames


def _probe_has_audio(src_path: str) -> bool:
    """用 ffprobe 判断是否存在音频流，避免对无音频输入做无意义的 map。"""
    ffprobe = find_ffprobe()
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", src_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _read_exact(pipe, size: int) -> bytes:
    """从管道读满 size 字节；EOF 时返回残余片段（可能为空）。"""
    buf = bytearray()
    while len(buf) < size:
        part = pipe.read(size - len(buf))
        if not part:
            break
        buf.extend(part)
    return bytes(buf)
