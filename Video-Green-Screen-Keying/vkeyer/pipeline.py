"""ffmpeg 解码 → 三层处理 → ffmpeg 编码 的流水线编排。

数据流::

    输入 ──解码 rgb24 裸流──> [层1 键控] RGBA
                              [层2 alpha 后处理 + despill 刷新]
                              [层3 背景替换?] ──编码──> 输出
      │
      └──────── 音轨不进像素管道，-c:a copy 直通（可选第二输入）──────┐
      背景素材（图/视频）──第二路解码 rgb24，尺寸 scale 对齐 ─────────┘

设计要点（与兄弟项目保持一致）：

- **三层解耦**：键控（:mod:`vkeyer.keyer`）、alpha 后处理
  （:mod:`vkeyer.matte`）、输出编码/合成（:mod:`vkeyer.encoder`）
  各自可单测、可单独复用；本模块只做帧循环与子进程编排。
- **内存有界**：任意时刻在途帧为常数（前景帧、背景帧、输出帧），
  不随视频时长增长；背景视频比前景短时最后一帧冻结续用。
- **失败不出半成品**：所有参数校验在启动子进程前完成；运行中
  ffmpeg 失败时删除本次新建的输出，预先存在的输出不会被动到。
- **stderr DEVNULL**：两路 ffmpeg 的 stderr 不接 PIPE，避免长
  视频上管道缓冲区写满导致父子进程互等死锁（vdenoise 同款坑）。
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field

from .encoder import (
    ALPHA_UNRELIABLE_WARNING,
    IMAGE_SEQUENCE_EXTS,
    composite_over_frame,
    composite_over_solid,
    is_image_sequence,
    resolve_video_spec,
)
from .errors import (
    FFmpegRunError,
    InputNotFoundError,
    OutputExistsError,
    OutputFormatError,
    StreamTruncatedError,
)
from .keyer import Keyer, KeyerConfig
from .matte import MatteConfig, apply_matte_to_frame, postprocess_alpha
from .probe import find_tool, probe_video
from .sampler import DEFAULT_REGION, DEFAULT_SAMPLE_FRAMES, sample_key_color

__all__ = [
    "KeyingResult",
    "key_video",
]


@dataclass
class KeyingResult:
    """一次色度键执行的汇总信息。"""

    input_path: str
    output_path: str
    frames: int
    width: int
    height: int
    fps: str
    key_rgb: tuple[int, int, int]
    key_source: str
    mode: str                      # alpha | composite
    background: str | None
    has_audio: bool
    elapsed_seconds: float
    messages: list[str] = field(default_factory=list)


def _read_exact(stream, size):
    """读满 size 字节；流结束时未读到字节返回 None，读了一半报错。"""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if chunks:
                raise StreamTruncatedError("解码输出在帧中途截断")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _check_output_path(output, overwrite):
    """输出存在性校验（与 color_grading 同策略）。"""
    if os.path.exists(output) and not overwrite:
        raise OutputExistsError(
            f"输出 {output} 已存在；显式传 overwrite=True 才允许覆盖"
        )


def _background_decoder(ffmpeg, bg_path, width, height):
    """启动背景素材解码子进程：统一缩放到前景尺寸、rgb24 裸流输出。

    图片与视频走同一条命令（ffmpeg 对单帧图片自然只产一帧），
    ``scale=w:h:force_original_aspect_ratio=decrease + pad`` 保证
    不变形、不留透明：不足部分补黑，最终仍是 width×height。
    """
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    return subprocess.Popen(
        [
            ffmpeg, "-v", "error", "-i", str(bg_path),
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )


def key_video(src, dst, *, key_rgb=None, sample_mode=DEFAULT_REGION,
              sample_rect=None, sample_frames=DEFAULT_SAMPLE_FRAMES,
              similarity=32.0, softness=64.0, despill=True,
              tolerance=0.0, despill_strength=1.0,
              matte_config=None, bg_color=None, bg_image=None,
              overwrite=False, progress=None):
    """对 ``src`` 做色度键抠像，结果写入 ``dst``，音轨原样保留。

    参数
    ----
    key_rgb:
        显式键色 ``(R, G, B)``；不给则从画面采样
        （``sample_mode`` / ``sample_rect`` / ``sample_frames``）。
    similarity, softness:
        双阈值键控参数，含义见 :class:`vkeyer.keyer.KeyerConfig`。
    despill:
        是否启用溢色抑制（层1 与层2 刷新各一次，同权重）。
    tolerance:
        容差 [0, 1]，按比例放宽距离判定；默认 0 与旧版一致。
        连续映射，增大时半透明边缘平滑变透而非硬切。
    despill_strength:
        溢色抑制强度 [0, 1]，默认 1 与旧版一致；层1 与层2 刷新
        共用同一强度。
    matte_config:
        层2 参数 :class:`MatteConfig`；None 取恒等默认（不后处理）。
    bg_color / bg_image:
        二选一给出时走**背景替换模式**，输出无 alpha 的普通视频；
        都不给时走 **alpha 模式**，输出透明通道视频/图像序列。
    overwrite:
        允许覆盖已存在的输出。
    progress:
        可选回调，每完成一帧以已处理帧数调用。

    返回 :class:`KeyingResult`。
    """
    if bg_color is not None and bg_image is not None:
        raise ValueError("bg_color 与 bg_image 互斥，只能给一种背景")
    if bg_color is not None and (
        len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color)
    ):
        raise ValueError("bg_color 必须是 (R, G, B) 且各通道在 [0, 255]")
    if not os.path.isfile(src):
        raise InputNotFoundError(f"输入不存在或不是普通文件：{src}")
    if bg_image is not None and not os.path.isfile(bg_image):
        raise InputNotFoundError(f"背景素材不存在：{bg_image}")

    dst = os.fspath(dst)
    _check_output_path(dst, overwrite)

    # 输出形态在任何 ffmpeg 调用前定下来，格式不支持直接快速失败
    messages = []
    composite_mode = bg_color is not None or bg_image is not None
    sequence = is_image_sequence(dst)
    if composite_mode and sequence:
        raise OutputFormatError("背景替换模式不支持图像序列输出")
    if composite_mode:
        codec, out_pix, extra = resolve_video_spec(dst, alpha=False)
    elif sequence:
        codec, out_pix, extra = _sequence_codec(dst)
    else:
        codec, out_pix, extra = resolve_video_spec(dst, alpha=True)
        if os.path.splitext(dst)[1].lower() == ".webm":
            # 本机 libvpx-vp9 静默剥离 alpha（见 encoder 模块说明）
            messages.append(ALPHA_UNRELIABLE_WARNING)

    ffmpeg = find_tool("ffmpeg")
    info = probe_video(src)
    w, h, fps = info["width"], info["height"], info["fps"]
    rgb_frame_size = w * h * 3

    # ---- 键色：显式指定优先，否则画面采样（绝不写死绿色）----
    if key_rgb is not None:
        if len(key_rgb) != 3 or not all(
            isinstance(c, int) and 0 <= c <= 255 for c in key_rgb
        ):
            raise ValueError("key_rgb 必须是三个 [0,255] 整数")
        key_rgb = tuple(key_rgb)
        key_source = "explicit"
    else:
        key_rgb = sample_key_color(
            src, mode=sample_mode, rect=sample_rect,
            frame_numbers=sample_frames,
        )
        key_source = f"sample:{sample_mode}"
        messages.append(f"采样键色 RGB{key_rgb}")

    keyer = Keyer(KeyerConfig(
        key_rgb=key_rgb, similarity=similarity,
        softness=softness, despill=despill,
        tolerance=tolerance, despill_strength=despill_strength,
    ))
    if matte_config is None:
        matte_config = MatteConfig()

    # 喂给编码器的裸流格式：背景替换产出 rgb24，alpha 模式产出 rgba
    raw_in_pix = "rgb24" if composite_mode else "rgba"

    started_outputs = []
    dec = bg_dec = enc = None
    t0 = time.monotonic()
    try:
        dec = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(src),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if bg_image is not None:
            bg_dec = _background_decoder(ffmpeg, bg_image, w, h)
            bg_frame = _read_exact(bg_dec.stdout, rgb_frame_size)
            if bg_frame is None:
                raise FFmpegRunError("背景素材解码不出任何帧")
        else:
            bg_frame = None

        enc_cmd = [
            ffmpeg, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", raw_in_pix,
            "-s", f"{w}x{h}", "-framerate", fps, "-i", "-",
        ]
        # 图像序列由 ffmpeg 自行按通配生成多文件；视频需音轨直通
        if not sequence and info["has_audio"]:
            enc_cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:a", "copy"]
        enc_cmd += ["-c:v", codec]
        if out_pix:
            enc_cmd += ["-pix_fmt", out_pix]
        enc_cmd += extra
        enc_cmd += [dst]
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
        if not sequence:
            started_outputs.append(dst)

        frames = 0
        while True:
            rgb = _read_exact(dec.stdout, rgb_frame_size)
            if rgb is None:
                break
            # 层1：键控（含首轮 despill）
            rgba = keyer.key_frame(rgb, w, h)
            # 层2：alpha 后处理，并按最终 alpha 刷新 despill
            if not matte_config.is_identity:
                alpha = postprocess_alpha(rgba[3::4], w, h, matte_config)
                rgba = apply_matte_to_frame(
                    rgba, w, h, alpha, key_rgb, despill=despill,
                    despill_strength=despill_strength,
                )
            # 层3：背景替换或直出 alpha
            if composite_mode:
                if bg_color is not None:
                    out_frame = composite_over_solid(rgba, w, h, bg_color)
                else:
                    nxt = _read_exact(bg_dec.stdout, rgb_frame_size)
                    if nxt is not None:
                        bg_frame = nxt
                    # 背景视频更短时冻结最后一帧（图片天然一直冻结）
                    out_frame = composite_over_frame(rgba, bg_frame, w, h)
                enc.stdin.write(out_frame)
            else:
                enc.stdin.write(bytes(rgba))
            frames += 1
            if progress:
                progress(frames)
    except (BrokenPipeError, OSError) as exc:
        raise FFmpegRunError(f"编码器异常退出：{exc}") from exc
    finally:
        if enc is not None and enc.stdin and not enc.stdin.closed:
            try:
                enc.stdin.close()
            except OSError:
                pass
        if dec is not None:
            dec.stdout.close()
        if bg_dec is not None:
            bg_dec.stdout.close()

    dec_rc = dec.wait() if dec is not None else 0
    bg_rc = bg_dec.wait() if bg_dec is not None else 0
    enc_rc = enc.wait() if enc is not None else 0
    if dec_rc != 0:
        _cleanup(started_outputs)
        raise FFmpegRunError(f"ffmpeg 解码失败（退出码 {dec_rc}）")
    if bg_rc != 0:
        _cleanup(started_outputs)
        raise FFmpegRunError(f"背景素材解码失败（退出码 {bg_rc}）")
    if enc_rc != 0:
        _cleanup(started_outputs)
        raise FFmpegRunError(f"ffmpeg 编码失败（退出码 {enc_rc}）")

    return KeyingResult(
        input_path=os.fspath(src), output_path=dst, frames=frames,
        width=w, height=h, fps=fps, key_rgb=key_rgb,
        key_source=key_source,
        mode="composite" if composite_mode else "alpha",
        background=(
            f"color{tuple(bg_color)}" if bg_color is not None
            else (os.fspath(bg_image) if bg_image is not None else None)
        ),
        has_audio=info["has_audio"],
        elapsed_seconds=time.monotonic() - t0,
        messages=messages,
    )


def _sequence_codec(dst):
    """图像序列输出的 (编码器, 像素格式, 额外参数)。"""
    ext = os.path.splitext(dst)[1].lower()
    if ext == ".png":
        return "png", "rgba", []
    if ext == ".webp":
        # 无损 webp 才能完整保留 alpha
        return "libwebp", "rgba", ["-lossless", "1"]
    raise OutputFormatError(
        f"图像序列仅支持 {', '.join(IMAGE_SEQUENCE_EXTS)}，得到 {ext}"
    )


def _cleanup(paths):
    """运行失败时删除本次新建的输出，不触碰预先存在的文件。"""
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
