"""命令行入口。

典型用法::

    # 终端摘要（默认）
    python -m scene_seg_tool input.mp4

    # 完整差异序列 CSV + 完整 JSON 报告
    python -m scene_seg_tool input.mp4 --csv diff.csv --json report.json

    # 调参：更严格的稳健阈值、最小场景 12 帧、关闭闪光抑制
    python -m scene_seg_tool input.mp4 --mad-k 12 --min-scene-frames 12 \\
        --no-flash-suppression

退出码：0 正常（含「未检出切点」）；2 找不到 ffmpeg/ffprobe；
3 输入不存在；4 输入无视频流/零帧；6 ffmpeg 执行失败。
"""

from __future__ import annotations

import argparse
import os
import sys

from .detector import detect_scenes
from .errors import SceneSegError
from .output import (
    render_summary,
    write_boundaries_csv,
    write_json_report,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（独立函数便于 CLI 测试）。"""
    parser = argparse.ArgumentParser(
        prog="scene-seg-tool",
        description=(
            "基于相邻帧灰度直方图差异的场景切换检测工具：单遍流式解码，"
            "纯标准库自研直方图统计与 L1 差异度量，输出完整差异序列与"
            "自适应判定阈值，结果可复现。"
        ),
    )
    parser.add_argument("input", help="输入视频文件路径（格式不限，按内容探测）")
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径，按扩展名决定格式：.json 完整报告 / .csv 差异序列",
    )
    parser.add_argument("--json", dest="json_path", help="写出完整 JSON 报告的路径")
    parser.add_argument(
        "--csv", dest="csv_path", help="写出完整差异序列 CSV 的路径"
    )
    parser.add_argument(
        "--mad-k", type=float, default=9.0,
        help="稳健阈值系数：阈值 = 中位数 + K×1.4826×MAD（默认 9.0）",
    )
    parser.add_argument(
        "--fallback-threshold", type=float, default=0.35,
        help="全静止/单帧等退化场景使用的兜底归一化阈值（默认 0.35）",
    )
    parser.add_argument(
        "--min-scene-frames", type=int, default=1,
        help="最小场景长度（帧），冲突切点按距离做确定性 NMS（默认 1=关闭）",
    )
    parser.add_argument(
        "--no-flash-suppression", action="store_true",
        help="关闭曝光突变（闪光灯/白闪黑闪）抑制",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="不向标准输出打印摘要（仍会正常写文件）",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """校验参数取值范围，非法时走 argparse 的标准错误路径（退出码 2）。"""
    if args.mad_k <= 0:
        parser.error("--mad-k 必须为正数")
    if not 0.0 <= args.fallback_threshold <= 1.0:
        parser.error("--fallback-threshold 必须落在 [0, 1]")
    if args.min_scene_frames < 1:
        parser.error("--min-scene-frames 必须 >= 1")
    if args.output and not args.output.lower().endswith((".json", ".csv")):
        parser.error("-o/--output 仅支持 .json 或 .csv 扩展名")


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""
    # Windows 控制台默认代码页可能是 GBK，中文摘要会乱码；
    # 输出文件本身始终是 UTF-8，这里只修正终端流
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    try:
        result = detect_scenes(
            args.input,
            mad_k=args.mad_k,
            fallback_threshold=args.fallback_threshold,
            min_scene_frames=args.min_scene_frames,
            flash_suppression=not args.no_flash_suppression,
        )

        json_path = args.json_path
        csv_path = args.csv_path
        if args.output:
            if args.output.lower().endswith(".json"):
                json_path = args.output
            else:
                csv_path = args.output

        if json_path:
            _ensure_parent_dir(json_path)
            write_json_report(result, json_path)
        if csv_path:
            _ensure_parent_dir(csv_path)
            write_boundaries_csv(result, csv_path)

        if not args.quiet:
            print(render_summary(result))
        return 0
    except SceneSegError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return exc.exit_code


def _ensure_parent_dir(path: str) -> None:
    """输出文件的父目录不存在时提前创建（与转码工具的显式失败风格一致）。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
