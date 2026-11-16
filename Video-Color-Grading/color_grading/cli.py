"""命令行入口：python -m color_grading input.mp4 -o out.mp4 -b 20 -c 10"""

from __future__ import annotations

import argparse
import sys

from .chunks import DEFAULT_CHUNK_ROWS
from .errors import ColorGradingError
from .lut import BRIGHTNESS_MAX, BRIGHTNESS_MIN, CONTRAST_MAX, CONTRAST_MIN
from .pipeline import adjust


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color_grading",
        description="视频亮度/对比度调整：按帧处理、保留音轨、分块并行。",
    )
    parser.add_argument("input", help="输入视频文件")
    parser.add_argument("-o", "--output", required=True, help="输出视频文件")
    parser.add_argument(
        "-b", "--brightness", type=float, default=0.0,
        help=f"亮度 [{BRIGHTNESS_MIN:g}, {BRIGHTNESS_MAX:g}]，超范围自动钳制，默认 0",
    )
    parser.add_argument(
        "-c", "--contrast", type=float, default=0.0,
        help=f"对比度 [{CONTRAST_MIN:g}, {CONTRAST_MAX:g}]，超范围自动钳制，默认 0",
    )
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"分块行数（默认 {DEFAULT_CHUNK_ROWS}）")
    parser.add_argument("--crf", type=int, default=18, help="x264 质量因子（默认 18）")
    parser.add_argument("--preset", default="medium", help="x264 预设（默认 medium）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = adjust(
            args.input,
            args.output,
            brightness=args.brightness,
            contrast=args.contrast,
            workers=args.workers,
            chunk_rows=args.chunk_rows,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
    except ColorGradingError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        stderr = getattr(exc, "stderr", "")
        if stderr:
            print(f"ffmpeg 输出: {stderr[-500:]}", file=sys.stderr)
        return exc.exit_code

    for msg in result.messages:
        print(f"提示: {msg}", file=sys.stderr)
    print(
        f"完成: {result.frames} 帧, {result.width}x{result.height}@{result.fps:g}, "
        f"亮度 {result.brightness:g} 对比度 {result.contrast:g}, "
        f"{'含音轨' if result.has_audio else '无音轨'}, "
        f"耗时 {result.elapsed_seconds:.1f}s -> {result.output_path}"
    )
    return 0
