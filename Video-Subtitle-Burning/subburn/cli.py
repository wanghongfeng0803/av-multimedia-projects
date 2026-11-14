"""命令行入口：``python -m subburn burn video.mp4 subs.srt -o out.mp4``。

样式有两条等价配置路径，可叠加使用（命令行单项参数优先级高于 JSON 文件）：

* ``--style style.json``：完整 :class:`~subburn.style.SubtitleStyle` 的 JSON
  描述，字段名与 :meth:`SubtitleStyle.from_dict` 一致；
* 单项参数：``--font-family``（可重复构成回退链）、``--font-weight``、
  ``--color`` / ``--alpha``、``--outline-color``、``--halign`` /
  ``--valign``、``--margin`` 等，适合快速调整。
"""

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
from .style import (
    Color,
    FontSpec,
    FontWeight,
    HAlign,
    Margins,
    Position,
    SubtitleStyle,
    VAlign,
)


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

    # -- 字体 ------------------------------------------------------------
    p.add_argument("--font", help="TTF/TTC 字体文件路径；默认自动探测系统字体")
    p.add_argument("--font-family", action="append", metavar="FAMILY",
                   help="字体族名（如 Arial、微软雅黑）；可重复给出形成回退链")
    p.add_argument("--font-size", type=int, default=0,
                   help="字号（像素）；0 表示按视频高度自动选择")
    p.add_argument("--font-weight",
                   help="字重：100~900 整数或 thin/regular/medium/bold/black 等")

    # -- 颜色与透明度 ----------------------------------------------------
    p.add_argument("--color",
                   help="文字颜色：#rgb/#rrggbb/#rrggbbaa、rgb()/hsl() 或命名色")
    p.add_argument("--alpha", type=int,
                   help="文字不透明度 0~255（覆盖颜色自带的 alpha）")
    p.add_argument("--outline-color", help="描边颜色（格式同 --color）")
    p.add_argument("--shadow-color", help="阴影颜色（格式同 --color，alpha 生效）")

    # -- 位置 ------------------------------------------------------------
    p.add_argument("--halign", choices=("left", "center", "right"),
                   help="水平对齐方式")
    p.add_argument("--valign", choices=("top", "middle", "bottom"),
                   help="垂直对齐方式")
    p.add_argument("--margin", type=int,
                   help="四边统一边距（像素）；更细粒度请在样式 JSON 中配置")

    # -- 既有参数 --------------------------------------------------------
    p.add_argument("--crf", type=int, default=23, help="x264 CRF 质量（默认 23）")
    p.add_argument("--preset", default="medium",
                   choices=("ultrafast", "superfast", "veryfast", "faster",
                            "fast", "medium", "slow", "slower", "veryslow"))
    p.add_argument("--strict", action="store_true",
                   help="SRT 格式异常时直接报错而非容忍恢复")
    p.add_argument("--no-box", action="store_true", help="关闭半透明底盒")
    p.add_argument("--box", action="store_true", help="强制开启半透明底盒")
    p.add_argument("--style", dest="style_file",
                   help="新版样式 JSON 文件（SubtitleStyle 字典）")
    p.add_argument("-y", "--overwrite", action="store_true", help="覆盖已存在的输出")
    p.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")
    return parser


def _build_style(args: argparse.Namespace) -> Style | SubtitleStyle:
    """把样式 JSON 与命令行单项参数合并为样式对象。

    任何新版样式参数出现时走 :class:`SubtitleStyle`；否则保持旧版
    :class:`Style` 路径，默认渲染与历史版本逐像素一致。
    """
    new_style_args = (
        args.style_file, args.font_family, args.font_weight, args.color,
        args.alpha, args.outline_color, args.shadow_color,
        args.halign, args.valign, args.margin,
    )
    using_new_style = any(v is not None for v in new_style_args)
    if not using_new_style:
        return Style(box=args.box or not args.no_box)

    data: dict = {}
    if args.style_file:
        with open(args.style_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("样式 JSON 顶层必须是对象（字典）")
    style = SubtitleStyle.from_dict(data, clamp=True)

    # -- 字体覆盖 --------------------------------------------------------
    families = tuple(args.font_family or ())
    if args.font:  # --font 给出的文件路径作为回退链最高优先级
        families = (args.font, *families)
    weight = (FontWeight.parse(args.font_weight)
              if args.font_weight is not None else style.font.weight)
    if families or args.font_size or args.font_weight is not None:
        object.__setattr__(
            style, "font",
            FontSpec(families=families or style.font.families,
                     size=args.font_size or style.font.size, weight=weight),
        )

    # -- 颜色覆盖（越界输入经 parse(clamp=True) 消化）--------------------
    if args.color is not None:
        object.__setattr__(style, "color", Color.parse(args.color, clamp=True))
    if args.alpha is not None:
        object.__setattr__(
            style, "color",
            style.color.with_alpha(max(0, min(255, args.alpha))),
        )
    if args.outline_color is not None:
        object.__setattr__(
            style, "outline_color",
            Color.parse(args.outline_color, clamp=True),
        )
    if args.shadow_color is not None:
        object.__setattr__(
            style, "shadow_color",
            Color.parse(args.shadow_color, clamp=True),
        )

    # -- 位置覆盖 --------------------------------------------------------
    if args.halign is not None or args.valign is not None or args.margin is not None:
        position = Position(
            halign=HAlign(args.halign) if args.halign else style.position.halign,
            valign=VAlign(args.valign) if args.valign else style.position.valign,
            margins=(Margins.uniform(args.margin) if args.margin is not None
                     else style.position.margins),
        )
        object.__setattr__(style, "position", position)

    # -- 底盒开关（新旧语义一致）----------------------------------------
    if args.box:
        object.__setattr__(style, "box", True)
    elif args.no_box:
        object.__setattr__(style, "box", False)
    return style


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
    try:
        style = _build_style(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"\n样式配置错误: {exc}", file=sys.stderr)
        return 2

    # 显式 --font 且未走新样式时沿用旧入口；新样式由 pipeline 内部解析
    font = None
    if args.font and not isinstance(style, SubtitleStyle):
        font = load_font(args.font, args.font_size or 28)

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
