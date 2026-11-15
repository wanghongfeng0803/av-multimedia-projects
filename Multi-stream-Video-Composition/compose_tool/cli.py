"""命令行入口。

用法::

    python -m compose_tool 左视频 右视频 [-o 输出.mp4]
        [--duration-mode pad|trim] [--layout auto|fit]
        [--height 高度] [--fps 帧率] [--audio-mode auto|mix|left|right|none]
        [--crf N] [--preset 档位] [--overwrite]
        [--dry-run] [--progress] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from .composer import DEFAULT_CRF, DEFAULT_PRESET, compose
from .errors import ComposeError
from .plan import (
    AUDIO_AUTO,
    AUDIO_MODES,
    DURATION_MODES,
    DURATION_PAD,
    LAYOUT_AUTO,
    LAYOUTS,
    plan_composition,
)
from .probe import probe_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose-tool",
        description=(
            "双路视频左右并排合成工具：对分辨率/时长/帧率/音频不一致的输入"
            "按确定性策略对齐，输出规范的 H.264(yuv420p) + AAC MP4。"
        ),
    )
    parser.add_argument("left", help="左画面视频文件路径")
    parser.add_argument("right", help="右画面视频文件路径")
    parser.add_argument(
        "-o", "--output",
        help="输出路径；省略时在左路同目录生成 <左名>_x_<右名>.mp4",
    )
    parser.add_argument(
        "--duration-mode", choices=DURATION_MODES, default=DURATION_PAD,
        help="时长对齐：pad 短路冻结末帧补齐到较长者(默认) / trim 截断到较短者",
    )
    parser.add_argument(
        "--layout", choices=LAYOUTS, default=LAYOUT_AUTO,
        help="布局：auto 单元格即缩放尺寸(默认) / fit 等宽单元格+居中黑边",
    )
    parser.add_argument(
        "--height", type=int, default=None, metavar="PX",
        help="画布高度覆盖（向上取偶）；省略时取两路较大高度",
    )
    parser.add_argument(
        "--fps", default=None, metavar="FPS",
        help="输出帧率覆盖，如 30 或 30000/1001；省略时取两路较大帧率",
    )
    parser.add_argument(
        "--audio-mode", choices=AUDIO_MODES, default=AUDIO_AUTO,
        help="音频：auto 自动(默认) / mix 混音 / left 仅左 / right 仅右 / none 无音轨",
    )
    parser.add_argument(
        "--crf", type=int, default=DEFAULT_CRF,
        help=f"H.264 质量参数，越小越清晰（默认 {DEFAULT_CRF}）",
    )
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET,
        help=f"x264 速度档位 ultrafast…placebo（默认 {DEFAULT_PRESET}）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="输出文件已存在时直接覆盖（默认拒绝覆盖）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只探测并打印对齐规划，不实际执行合成",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="在 stderr 实时显示合成进度百分比",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出结果/规划，便于脚本调用",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.dry_run:
            return _dry_run(args)
        result = compose(
            args.left,
            args.right,
            args.output,
            overwrite=args.overwrite,
            duration_mode=args.duration_mode,
            layout=args.layout,
            height=args.height,
            fps=args.fps,
            audio_mode=args.audio_mode,
            crf=args.crf,
            preset=args.preset,
            progress_callback=_make_progress_printer() if args.progress else None,
        )
    except ComposeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_result(result)
    return 0


def _dry_run(args) -> int:
    """只跑探测与规划，输出确定性对齐方案，不调用 ffmpeg。"""
    try:
        left_info = probe_media(args.left)
        right_info = probe_media(args.right)
        plan = plan_composition(
            left_info,
            right_info,
            duration_mode=args.duration_mode,
            layout=args.layout,
            height=args.height,
            fps=args.fps,
            audio_mode=args.audio_mode,
        )
    except ComposeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"画布: {plan.width}x{plan.height} @ {plan.fps_text}fps，"
              f"时长 {plan.duration:.3f}s（{plan.duration_mode}），布局 {plan.layout}")
        for name, side in (("左", plan.left), ("右", plan.right)):
            print(
                f"{name}路: {side.source_width}x{side.source_height}"
                f" -> 缩放 {side.scaled_width}x{side.scaled_height}"
                f"，单元格 {side.cell_width}x{side.cell_height}"
                f"（偏移 {side.pad_x},{side.pad_y}）"
                f"，冻结补帧 {side.freeze_seconds:.3f}s"
                f"，{'有' if side.has_audio else '无'}音频"
            )
        print(f"音频: {plan.audio_mode}（{plan.sample_rate}Hz/{plan.channels}ch）")
    for warning in plan.warnings:
        print(f"警告: {warning}", file=sys.stderr)
    return 0


def _print_result(result) -> None:
    print(f"输出: {result.output_path}")
    print(f"画布: {result.width}x{result.height} @ {result.fps}fps，"
          f"时长 {result.duration_seconds:.3f}s（{result.duration_mode}）")
    print(f"音频: {result.audio_mode}")
    for warning in result.warnings:
        print(f"警告: {warning}", file=sys.stderr)


def _make_progress_printer():
    last = {"shown": -1}

    def _print(ratio: float) -> None:
        percent = int(ratio * 100)
        if percent > last["shown"]:
            last["shown"] = percent
            print(f"\r进度: {percent:3d}%", end="", file=sys.stderr, flush=True)
            if percent >= 100:
                print(file=sys.stderr)

    return _print
