"""命令行入口：``python -m video_transition a.mp4 b.mp4 -o out.mp4 --overlap 0.5``。

重叠时长两种配法互斥：``--overlap 秒数``（按帧率换算到帧）或
``--frames 帧数``（语义最精确）。结束后打印逐段账目，可用
``--json`` 输出机器可读结果。
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .errors import TransitionError
from .pipeline import cross_fade


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器（独立函数便于 CLI 单测）。"""
    parser = argparse.ArgumentParser(
        prog="video_transition",
        description="在两段素材之间做交叉淡化转场（音视频同帧淡化，成片不丢帧）",
    )
    parser.add_argument("clip_a", help="第一段素材路径（淡出方）")
    parser.add_argument("clip_b", help="第二段素材路径（淡入方）")
    parser.add_argument("-o", "--output", required=True, help="成片输出路径（.mp4/.mkv）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--overlap", type=float, metavar="SECONDS",
        help="重叠时长（秒），按素材帧率换算为整数帧",
    )
    group.add_argument(
        "--frames", type=int, metavar="N",
        help="重叠帧数（整数，与 --overlap 互斥）",
    )
    parser.add_argument("--crf", type=int, default=23, help="x264 质量（默认 23）")
    parser.add_argument("--preset", default="medium", help="x264 速度档（默认 medium）")
    parser.add_argument("-y", "--overwrite", action="store_true", help="覆盖已存在的输出")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印结果摘要")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码（0 成功，2 参数/业务错误）。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = cross_fade(
            args.clip_a,
            args.clip_b,
            args.output,
            overlap_seconds=args.overlap,
            overlap_frames=args.frames,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
    except TransitionError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误: 文件不存在: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "output": result.output,
            "width": result.width,
            "height": result.height,
            "fps": result.fps_num / result.fps_den,
            "frames_written": result.frames_written,
            "head_frames": result.head_frames,
            "overlap_frames": result.overlap_frames,
            "tail_frames": result.tail_frames,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            "had_audio": list(result.had_audio),
            "encoder": result.encoder,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"完成: {result.output}")
        print(f"  帧率 {result.fps_num}/{result.fps_den}，"
              f"{result.width}x{result.height}，采样率 {result.sample_rate}Hz")
        print(f"  逐段账目: {result.describe_segments()}")
        print(f"  音轨: A={'有' if result.had_audio[0] else '静音'}，"
              f"B={'有' if result.had_audio[1] else '静音'}")
    return 0
