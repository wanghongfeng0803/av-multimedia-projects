"""命令行入口：``python -m subburn burn video.mp4 subs.srt -o out.mp4``。

支持两种字幕格式，按扩展名（``.ass``/``.ssa``）或文件内容自动识别：

* **SRT**：样式有两条等价配置路径，可叠加使用（命令行单项参数优先级
  高于 JSON 文件）——``--style style.json`` 完整
  :class:`~subburn.style.SubtitleStyle` 描述；以及 ``--font-family``、
  ``--font-weight``、``--color`` / ``--alpha``、``--outline-color``、
  ``--halign`` / ``--valign``、``--margin`` 等单项参数；
* **ASS/SSA**：样式与特效完全由字幕文件内的 ``[V4+ Styles]`` 与行内
  覆盖标签决定，SRT 的样式覆盖参数在该模式下会被拒绝（避免「参数看似
  生效却被忽略」的歧义）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .ass import ASSParseError, parse_ass
from .fonts import load_font
from .pipeline import burn_ass, burn_subtitles, burn_track
from .render import Style
from .srt import parse_srt_file
from .tracks import TrackError, probe_subtitle_tracks
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

#: 这些单项样式参数只对 SRT 有意义；ASS 模式下出现即报错（--font 除外，
#: ASS 渲染器走自己的字体回退链，外部字体文件无对应入口）
_SRT_ONLY_ARGS = (
    "style_file", "font", "font_family", "font_weight", "font_size",
    "color", "alpha", "outline_color", "shadow_color",
    "halign", "valign", "margin", "box", "no_box",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subburn",
        description="SRT/ASS 字幕解析与视频烧录（状态机解析 + 帧级时间对齐 + 像素叠加）",
    )
    parser.add_argument("--version", action="version", version=f"subburn {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("burn", help="把 SRT/ASS 字幕烧录进视频")
    p.add_argument("video", help="输入视频路径")
    # dest 沿用 "srt" 以兼容旧调用方/测试对命名空间属性名的依赖；
    # 实际接受 .srt/.ass/.ssa，格式按内容自动识别
    p.add_argument("srt", metavar="subtitle",
                   help="字幕文件路径（.srt/.ass/.ssa，按内容自动识别）")
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
                   help="字幕格式异常时直接报错而非容忍恢复（SRT/ASS 均适用）")
    p.add_argument("--no-box", action="store_true", help="关闭半透明底盒")
    p.add_argument("--box", action="store_true", help="强制开启半透明底盒")
    p.add_argument("--style", dest="style_file",
                   help="新版样式 JSON 文件（SubtitleStyle 字典）")
    p.add_argument("-y", "--overwrite", action="store_true", help="覆盖已存在的输出")
    p.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")

    # -- 内嵌多轨：枚举与指定烧录 ----------------------------------------
    t = sub.add_parser("tracks", help="列出容器内嵌的全部字幕轨")
    t.add_argument("video", help="输入视频/容器路径")
    t.add_argument("--json", action="store_true", help="以 JSON 输出轨道列表")

    bt = sub.add_parser("burn-track", help="烧录容器内嵌的指定字幕轨")
    bt.add_argument("video", help="输入视频/容器路径")
    bt.add_argument("-o", "--output", help="输出路径（默认 <视频名>_subbed.mp4）")
    sel = bt.add_argument_group("轨道选择（条件取交集；不给出时用默认轨）")
    sel.add_argument("--stream", type=int, metavar="N",
                     help="容器内绝对流索引（ffprobe 的 index）")
    sel.add_argument("--sub-index", type=int, metavar="N",
                     help="字幕轨相对序号（第几条字幕轨，0 起）")
    sel.add_argument("--lang", metavar="TAG",
                     help="按语言标签选择（如 chi、eng，大小写不敏感）")
    sel.add_argument("--title", metavar="TEXT",
                     help="按轨标题选择（大小写不敏感精确匹配）")
    bt.add_argument("--crf", type=int, default=23, help="x264 CRF 质量（默认 23）")
    bt.add_argument("--preset", default="medium",
                    choices=("ultrafast", "superfast", "veryfast", "faster",
                             "fast", "medium", "slow", "slower", "veryslow"))
    bt.add_argument("--strict", action="store_true",
                    help="抽出的字幕格式异常时直接报错而非容忍恢复")
    bt.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")
    bt.add_argument("-y", "--overwrite", action="store_true",
                    help="覆盖已存在的输出")
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


def _is_ass_file(path: str) -> bool:
    """按扩展名优先、内容兜底判断是否为 ASS/SSA。

    扩展名命中 ``.ass``/``.ssa`` 直接认定；否则读取文件头检查
    ``[Script Info]`` 段标记，使改了扩展名的 ASS 文件也能正确路由。
    读取失败时退回 False，交给 SRT 解析器报出更贴切的错误。
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".ass", ".ssa"):
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(4096).decode("utf-8", "ignore").lower()
    except OSError:
        return False
    return "[script info]" in head


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "tracks":
        return _run_tracks(args)
    if args.command == "burn-track":
        return _run_burn_track(args)
    is_ass = _is_ass_file(args.srt)

    if is_ass:
        # ASS 的样式由文件自身描述，SRT 样式覆盖参数没有作用点；明确拒绝
        # 而不是静默忽略，避免用户误以为 --color 等参数生效
        given = []
        for name in _SRT_ONLY_ARGS:
            value = getattr(args, name)
            if name == "font_size":
                used = value and value > 0  # 默认 0 表示自动，不算显式给出
            else:
                used = value not in (None, False)
            if used:
                given.append(name)
        if given:
            print(
                "错误：ASS/SSA 字幕的样式由文件内 [V4+ Styles] 与行内标签"
                f"决定，以下参数不适用: {', '.join('--' + name.replace('_', '-') for name in given)}",
                file=sys.stderr,
            )
            return 2
        return _run_ass(args)

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
            "format": "srt",
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


def _run_ass(args: argparse.Namespace) -> int:
    """ASS/SSA 路径：解析文档 -> burn_ass，错误码与 SRT 路径保持一致。"""
    try:
        document, warnings = parse_ass(args.srt, strict=args.strict)
    except ASSParseError as exc:
        print(f"错误: ASS 解析失败: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"错误: 无法读取字幕文件: {exc}", file=sys.stderr)
        return 1
    for msg in warnings:
        print(f"[warn] {msg}", file=sys.stderr)
    if not document.cues:
        print("错误：ASS 中没有解析出任何有效对话", file=sys.stderr)
        return 2

    output = args.output or str(
        Path(args.video).with_name(Path(args.video).stem + "_subbed.mp4")
    )
    last_pct = -1

    def on_progress(ratio: float) -> None:
        nonlocal last_pct
        pct = int(ratio * 100)
        if pct != last_pct and not args.json:
            last_pct = pct
            print(f"\r烧录进度: {pct:3d}%", end="", file=sys.stderr, flush=True)

    try:
        result = burn_ass(
            args.video, output, document,
            crf=args.crf, preset=args.preset,
            overwrite=args.overwrite, progress_callback=on_progress,
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "format": "ass",
            "output": result.output, "width": result.width,
            "height": result.height, "fps": result.fps,
            "frames_written": result.frames_written, "cues": result.cues,
            "encoder": result.encoder, "had_audio": result.had_audio,
            "styles": len(document.styles),
            "play_res": [document.play_res_x, document.play_res_y],
            "warnings": warnings,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\n完成: {result.output} "
              f"({result.width}x{result.height} @ {result.fps:g}fps, "
              f"{result.frames_written} 帧, {result.cues} 条对话, "
              f"{len(document.styles)} 个样式"
              f"{', 含音频' if result.had_audio else ', 无音频'})")
    return 0


def _run_tracks(args: argparse.Namespace) -> int:
    """``tracks`` 子命令：枚举容器内嵌字幕轨，人读表格或 JSON。"""
    try:
        tracks = probe_subtitle_tracks(args.video)
    except (FileNotFoundError, TrackError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({
            "input": args.video,
            "tracks": [
                {
                    "sub_index": t.sub_index,
                    "stream_index": t.stream_index,
                    "codec_name": t.codec_name,
                    "language": t.language,
                    "title": t.title,
                    "default": t.default,
                    "forced": t.forced,
                    "format": t.format,
                    "burnable": t.burnable,
                }
                for t in tracks
            ],
        }, ensure_ascii=False, indent=2))
    elif tracks:
        print(f"{args.video}: {len(tracks)} 条字幕轨")
        for t in tracks:
            print(f"  {t.describe()}")
    else:
        print(f"{args.video}: 未找到内嵌字幕轨")
    return 0


def _run_burn_track(args: argparse.Namespace) -> int:
    """``burn-track`` 子命令：选择 -> 抽取 -> 复用既有渲染管道。"""
    output = args.output or str(
        Path(args.video).with_name(Path(args.video).stem + "_subbed.mp4")
    )
    last_pct = -1

    def on_progress(ratio: float) -> None:
        nonlocal last_pct
        pct = int(ratio * 100)
        if pct != last_pct and not args.json:
            last_pct = pct
            print(f"\r烧录进度: {pct:3d}%", end="", file=sys.stderr, flush=True)

    try:
        result = burn_track(
            args.video, output,
            stream_index=args.stream, sub_index=args.sub_index,
            language=args.lang, title=args.title,
            strict=args.strict, crf=args.crf, preset=args.preset,
            overwrite=args.overwrite, progress_callback=on_progress,
        )
    except (FileNotFoundError, TrackError) as exc:
        # 轨道枚举/选择/抽取问题属于用法错误，与字幕解析失败同码
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except (FileExistsError, ValueError, RuntimeError) as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "format": "embedded-track",
            "track": result.track,
            "output": result.output, "width": result.width,
            "height": result.height, "fps": result.fps,
            "frames_written": result.frames_written, "cues": result.cues,
            "encoder": result.encoder, "had_audio": result.had_audio,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\n完成: {result.output} (轨 [{result.track}], "
              f"{result.width}x{result.height} @ {result.fps:g}fps, "
              f"{result.frames_written} 帧, {result.cues} 条字幕"
              f"{', 含音频' if result.had_audio else ', 无音频'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
