"""水印叠加定位与 ffmpeg segment 命令构造。

无水印全程 -c copy；叠水印必须重编码视频轨（音频仍流拷贝）。
为保住"切点严格对齐 GOP、每片首帧即 IDR"的承诺：
* 规划切点同时喂给 -force_key_frames，重编码器在这些时刻产 IDR；
* 不启用 -reset_timestamps，保留全局连续 PTS/DTS；
* 水印 PNG 经第二个 -i + 流标签进过滤器图，路径永不拼进
  filtergraph 字符串，从根上杜绝过滤器注入。
"""

from __future__ import annotations

import os

from ..binaries import ffmpeg_path
from ..errors import UnsupportedInputError, WatermarkError
from ..models import MediaInfo, SegmentInfo
from .config import WatermarkConfig

_ARGV_SOFT_LIMIT = 30000

_VIDEO_ENCODERS = {
    "h264": "libx264",
    "hevc": "libx265",
    "mpeg2video": "mpeg2video",
    "mpeg1video": "mpeg2video",
    "mp4": "libx264",
}

_ANCHOR_EXPR = {
    "top-left":      ("margin_x",     "margin_y"),
    "top-center":    ("(W-w)/2",      "margin_y"),
    "top-right":     ("W-w-margin_x", "margin_y"),
    "middle-left":   ("margin_x",     "(H-h)/2"),
    "center":        ("(W-w)/2",      "(H-h)/2"),
    "middle-right":  ("W-w-margin_x", "(H-h)/2"),
    "bottom-left":   ("margin_x",     "H-h-margin_y"),
    "bottom-center": ("(W-w)/2",      "H-h-margin_y"),
    "bottom-right":  ("W-w-margin_x", "H-h-margin_y"),
}


def overlay_position(anchor, margin_x, margin_y):
    """返回 overlay 的 (x, y) 表达式，margin 以字面量代入。"""
    if anchor not in _ANCHOR_EXPR:
        raise WatermarkError(f"未知锚点：{anchor!r}")
    x_expr, y_expr = _ANCHOR_EXPR[anchor]
    return x_expr.replace("margin_x", str(int(margin_x))), \
        y_expr.replace("margin_y", str(int(margin_y)))


def build_filter_graph(config: WatermarkConfig) -> str:
    """双输入 overlay 过滤器图；只含数字与 W/H/w/h 内建变量。"""
    x_expr, y_expr = overlay_position(
        config.anchor, config.margin_x, config.margin_y
    )
    return f"[1:v]format=rgba[wm];[0:v][wm]overlay={x_expr}:{y_expr}:shortest=1[v]"


def _select_video_encoder(media):
    videos = media.video_streams
    if not videos:
        raise UnsupportedInputError("纯音频输入无法叠加文字水印（没有视频轨道）")
    encoder = _VIDEO_ENCODERS.get(videos[0].codec_name)
    if encoder is None:
        raise WatermarkError(
            f"暂不支持为视频编码 {videos[0].codec_name} 重编码叠加水印"
            f"（当前支持：{', '.join(sorted(_VIDEO_ENCODERS))}）"
        )
    return encoder


def _encoder_arguments(encoder):
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p"]
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "mpeg2video", "-q:v", "4", "-pix_fmt", "yuv420p"]


def build_watermarked_segment_command(
    media: MediaInfo,
    planned: list[SegmentInfo],
    output_dir: str,
    filename_prefix: str,
    maps: list[str],
    watermark_png: str,
    config: WatermarkConfig,
) -> list[str]:
    """构造"重编码视频 + overlay 水印 + segment 切割"的 argv。

    全部路径以独立 argv 元素传递（列表形式，不走 shell），
    含空格/引号/分号/中文的路径都原样到达 ffmpeg。
    """
    if not planned:
        raise WatermarkError("分片规划为空，无法构造水印切片命令")
    if not os.path.isfile(watermark_png):
        raise WatermarkError(f"水印 PNG 不存在：{watermark_png}")

    encoder = _select_video_encoder(media)
    cut_times = ",".join(f"{seg.source_start:.6f}" for seg in planned[1:])
    output_pattern = os.path.join(output_dir, f"{filename_prefix}_%04d.ts")
    filter_graph = build_filter_graph(config)

    argv = [
        ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", media.path,
        # 单张 PNG 默认只解出 1 帧，overlay 的 shortest=1 会让输出在该帧
        # 结束、整段视频被截断；-loop 1 把图片变成无限持续的静态流，
        # 再由 overlay shortest=1 跟随视频流时长收尾。
        "-loop", "1",
        "-i", watermark_png,
        "-filter_complex", filter_graph,
        "-map", "[v]",
    ]
    for extra_map in maps[1:]:
        argv.extend(["-map", extra_map])

    argv.extend(_encoder_arguments(encoder))
    argv.extend(["-c:a", "copy"])

    if cut_times:
        argv.extend(["-force_key_frames", cut_times])
    argv.extend(["-f", "segment", "-segment_format", "mpegts",
                 "-segment_start_number", str(planned[0].sequence_number)])
    if cut_times:
        argv.extend(["-segment_times", cut_times])
    argv.append(output_pattern)

    if sum(len(part) for part in argv) > _ARGV_SOFT_LIMIT:
        raise WatermarkError(
            "分片数量过多导致命令行超长，请增大 --segment-duration 以减少分片数量"
        )
    return argv
