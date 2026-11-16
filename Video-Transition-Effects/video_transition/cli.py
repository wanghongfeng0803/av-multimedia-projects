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
from .easing import EASINGS
from .errors import TransitionError
from .pipeline import cross_fade
from .slide import DIRECTIONS


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器（独立函数便于 CLI 单测）。"""
    parser = argparse.ArgumentParser(
        prog="video_transition",
        description=(
            "在两段素材之间做转场（默认交叉淡化，可选四方向滑动）；"
            "音视频同帧推进，成片不丢帧"
        ),
    )
    parser.add_argument("clip_a", help="第一段素材路径（移出方）")
    parser.add_argument("clip_b", help="第二段素材路径（推入方）")
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
    parser.add_argument(
        "--transition", choices=("fade", "slide", "wipe"), default="fade",
        help="转场类型：fade=交叉淡化（默认，旧行为），slide=方向滑动，"
             "wipe=擦除（固定画面、移动掩码）",
    )
    parser.add_argument(
        "--direction", choices=DIRECTIONS, default="right",
        help="运动方向（slide/wipe 生效）：%(choices)s，默认 %(default)s",
    )
    parser.add_argument(
        "--easing", choices=EASINGS, default="linear",
        help="速度曲线：%(choices)s，默认 %(default)s",
    )
    parser.add_argument(
        "--soft-width", type=int, metavar="PIXELS", default=0,
        help="擦除软边宽度（像素，仅 --transition wipe 生效），0=硬边；"
             "超过画面运动轴长度时自动钳制并告警，默认 %(default)s",
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
            transition=args.transition,
            direction=args.direction,
            easing=args.easing,
            soft_width=args.soft_width,
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
            "transition": result.transition,
            "direction": result.direction,
            "easing": result.easing,
            "soft_width": result.soft_width,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"完成: {result.output}")
        print(f"  转场: {result.transition_label}")
        print(f"  帧率 {result.fps_num}/{result.fps_den}，"
              f"{result.width}x{result.height}，采样率 {result.sample_rate}Hz")
        print(f"  逐段账目: {result.describe_segments()}")
        print(f"  音轨: A={'有' if result.had_audio[0] else '静音'}，"
              f"B={'有' if result.had_audio[1] else '静音'}")
    return 0
