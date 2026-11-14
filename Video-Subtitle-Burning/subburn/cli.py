"""命令行入口：``python -m subburn burn video.mp4 subs.srt -o out.mp4``。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .fonts import load_font
from .pipeline import burn_subtitles
from .render import Style
from .srt import parse_srt_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subburn",
        description="SRT 字幕解析与视频烧录（状态机解析 + 帧级时间对齐 + 像素叠加）",
    )
    parser.add_argument("--version", action="version", version=f"subburn {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("burn", help="把 SRT 字幕烧录进视频")
    p.add_argument("video", help="输入视频路径")
    p.add_argument("srt", help="SRT 字幕文件路径")
    p.add_argument("-o", "--output", help="输出路径（默认 <视频名>_subbed.mp4）")
    p.add_argument("--font", help="TTF/TTC 字体路径；默认自动探测系统字体")
    p.add_argument("--font-size", type=int, default=0,
                   help="字号（TTF 模式）；0 表示按视频高度自动选择")
    p.add_argument("--crf", type=int, default=23, help="x264 CRF 质量（默认 23）")
    p.add_argument("--preset", default="medium",
                   choices=("ultrafast", "superfast", "veryfast", "faster",
                            "fast", "medium", "slow", "slower", "veryslow"))
    p.add_argument("--strict", action="store_true",
                   help="SRT 格式异常时直接报错而非容忍恢复")
    p.add_argument("--no-box", action="store_true", help="关闭半透明底盒")
    p.add_argument("--box", action="store_true", help="强制开启半透明底盒")
    p.add_argument("-y", "--overwrite", action="store_true", help="覆盖已存在的输出")
    p.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cues, warnings = parse_srt_file(args.srt, strict=args.strict)
    for msg in warnings:
        print(f"[warn] {msg}", file=sys.stderr)
    if not cues:
        print("错误：SRT 中没有解析出任何有效字幕", file=sys.stderr)
        return 2

    output = args.output or str(
        Path(args.video).with_name(Path(args.video).stem + "_subbed.mp4")
    )
    font = load_font(args.font, args.font_size or 28)
    style = Style(box=args.box or not args.no_box)  # 默认开启底盒，提升可读性

    last_pct = -1

    def on_progress(ratio: float) -> None:
        nonlocal last_pct
        pct = int(ratio * 100)
        if pct != last_pct and not args.json:
            last_pct = pct
            print(f"\r烧录进度: {pct:3d}%", end="", file=sys.stderr, flush=True)

    try:
        result = burn_subtitles(
            args.video, output, cues,
            font=font, style=style, crf=args.crf, preset=args.preset,
            overwrite=args.overwrite, progress_callback=on_progress,
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "output": result.output, "width": result.width,
            "height": result.height, "fps": result.fps,
            "frames_written": result.frames_written, "cues": result.cues,
            "encoder": result.encoder, "had_audio": result.had_audio,
            "warnings": warnings,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\n完成: {result.output} "
              f"({result.width}x{result.height} @ {result.fps:g}fps, "
              f"{result.frames_written} 帧, {result.cues} 条字幕"
              f"{', 含音频' if result.had_audio else ', 无音频'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
