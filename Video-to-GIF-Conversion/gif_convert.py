#!/usr/bin/env python3
"""视频转 GIF 命令行入口（抽帧 / 量化 / 编码三步解耦）。

完整转换（一条命令跑完三步，抽帧产物落在临时目录）::

    python gif_convert.py 输入视频 输出.gif [--fps 12] [--max-width 480]
                          [--width W --height H] [--align-odd]
                          [--colors 256] [--work-dir DIR] [--no-reuse]
                          [--indexed-scale]

分步执行（三步严格解耦，中间产物可复用）::

    python gif_convert.py extract  输入视频 工作目录
    python gif_convert.py quantize 工作目录 索引产物.pkl --fps 12 ...
    python gif_convert.py encode   索引产物.pkl 输出.gif

缩放规则：缩放严格接在抽帧之后、量化之前，调色板基于最终尺寸生成；
缩小走面积平均抗锯齿、放大走双线性（RGB 域）。显式 --width/--height
按源宽高比等比缩放（只给一个维度时另一个自动推导，同时给出且比例
不符直接报错，不拉伸画面）；--align-odd 把宽高各自规整到最近奇数，
宽高比与奇数对齐冲突时宽高比优先（先等比取整再 ±1px，不拉伸）。

时序保证：GIF 帧延迟以厘秒为单位，按源时长与输出帧数 Bresenham 均摊，
总时长误差 < 1 厘秒；目标帧率高于 50fps（延迟低于 2 厘秒）时直接
报错，避免历史查看器钳制延迟造成播放漂移。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vgif import (  # noqa: E402
    DEFAULT_COLORS,
    DEFAULT_FPS,
    DEFAULT_MAX_WIDTH,
    ExtractError,
    GifEncodeError,
    QuantizeError,
    TimingError,
    extract_rgb_frames,
    frame_pick_indices,
    plan_gif_delays,
    probe_frames,
    quantize_frames,
)
from vgif.pipeline import (  # noqa: E402
    PipelineError,
    convert_to_gif,
    derive_output_frame_count,
    resolve_target_size,
)
from vgif.scale import (  # noqa: E402
    nearest_resize_indexed,
    resize_rgb,
)
from vgif.quantize import QuantizedFrame  # noqa: E402
from vgif.gif_writer import write_gif  # noqa: E402


def _add_common_quality_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="目标帧率，默认 %(default)s（厘秒均摊 8/9，体积与流畅度折中）",
    )
    p.add_argument(
        "--max-width", type=int, default=DEFAULT_MAX_WIDTH,
        help="缩放后最大宽度（像素），默认 %(default)s；传 0 表示不缩放；"
             "给出 --width/--height 任一显式维度后本参数失效",
    )
    p.add_argument(
        "--width", type=int, default=None,
        help="显式目标宽（像素，正整数）；高度按源宽高比自动推导，"
             "与 --max-width 互斥（显式维度优先）",
    )
    p.add_argument(
        "--height", type=int, default=None,
        help="显式目标高（像素，正整数）；宽度按源宽高比自动推导",
    )
    p.add_argument(
        "--align-odd", action="store_true",
        help="把最终宽高各自规整到最近奇数（偶数向下取奇）；宽高比与"
             "奇数对齐冲突时宽高比优先，只产生 ±1px 偏差，不拉伸画面",
    )
    p.add_argument(
        "--colors", type=int, default=DEFAULT_COLORS,
        help="调色板色数 2..256，默认 %(default)s（用满全局颜色表）",
    )
    p.add_argument(
        "--indexed-scale", action="store_true",
        help="先量化再在索引域最近邻缩放（像素风素材），默认 RGB 域"
             "缩放（缩小面积平均抗锯齿、放大双线性）",
    )


def _cmd_convert(args: argparse.Namespace) -> int:
    max_width = None if args.max_width == 0 else args.max_width
    try:
        result = convert_to_gif(
            args.input,
            args.output,
            target_fps=args.fps,
            max_width=max_width,
            colors=args.colors,
            work_dir=args.work_dir,
            reuse=not args.no_reuse,
            indexed_scale=args.indexed_scale,
            width=args.width,
            height=args.height,
            align_odd=args.align_odd,
        )
    except (PipelineError, TimingError, ExtractError, QuantizeError,
            GifEncodeError, OSError) as exc:
        print(f"转换失败：{exc}", file=sys.stderr)
        return 1

    size_kb = result.output.stat().st_size / 1024
    print(
        f"完成：{result.output}\n"
        f"  源 {result.frame_count_in} 帧 → GIF {result.frame_count_out} 帧，"
        f"{result.width}x{result.height}，{result.colors} 色\n"
        f"  源时长 {result.source_duration:.3f}s → GIF "
        f"{result.gif_duration:.3f}s（偏差 {result.duration_error*1000:+.1f}ms）\n"
        f"  抽帧复用：{'是' if result.reused_extract else '否'}；"
        f"文件体积 {size_kb:.1f} KiB"
    )
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    try:
        manifest = extract_rgb_frames(args.input, args.workdir)
    except ExtractError as exc:
        print(f"抽帧失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"抽帧完成：{manifest.frame_count} 帧 {manifest.width}x"
        f"{manifest.height} @ {manifest.fps:.3f}fps，"
        f"时长 {manifest.duration:.3f}s，产物目录 {args.workdir}"
    )
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    """阶段二：读工作目录抽帧产物 → 采样/缩放/量化 → pickle 索引产物。"""
    import json
    work_dir = Path(args.workdir)
    try:
        from vgif.extract import Manifest as _M
        manifest = _M.from_dict(
            json.loads((work_dir / "manifest.json").read_text("utf-8"))
        )
        n_src = manifest.frame_count
        n_out = derive_output_frame_count(manifest, args.fps)
        dst_w, dst_h = resolve_target_size(
            manifest.width, manifest.height,
            max_width=None if args.max_width == 0 else args.max_width,
            width=args.width, height=args.height,
            align_odd=args.align_odd,
        )
        plan = plan_gif_delays(manifest.duration, n_out, args.fps)
        indices = frame_pick_indices(n_src, n_out)
        picked = [
            manifest.frame_path(work_dir, i).read_bytes() for i in indices
        ]
        if args.indexed_scale:
            palette, qframes = quantize_frames(
                picked, manifest.width, manifest.height, args.colors
            )
            qframes = [
                QuantizedFrame(
                    indices=nearest_resize_indexed(
                        q.indices, q.width, q.height, dst_w, dst_h
                    ),
                    width=dst_w, height=dst_h,
                )
                for q in qframes
            ]
        else:
            # 缩放严格在量化前：调色板基于最终尺寸生成。
            scaled = [
                resize_rgb(
                    f, manifest.width, manifest.height, dst_w, dst_h
                )
                for f in picked
            ]
            palette, qframes = quantize_frames(scaled, dst_w, dst_h,
                                               args.colors)
        payload = {
            "palette": palette,
            "qframes": qframes,
            "delays_cs": plan.delays_cs,
            "width": dst_w,
            "height": dst_h,
            "source_duration": manifest.duration,
        }
        with open(args.output, "wb") as fh:
            pickle.dump(payload, fh)
    except (PipelineError, TimingError, QuantizeError, ExtractError,
            OSError, KeyError, FileNotFoundError) as exc:
        print(f"量化失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"量化完成：{n_out} 帧 {dst_w}x{dst_h}，调色板 {len(palette)} 色，"
        f"延迟 {plan.delays_cs[0]}..{plan.delays_cs[-1]} 厘秒 → {args.output}"
    )
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    """阶段三：读 pickle 索引产物 → 纯 Python LZW 写 GIF。"""
    try:
        with open(args.payload, "rb") as fh:
            payload = pickle.load(fh)
        write_gif(
            args.output,
            payload["qframes"],
            payload["palette"],
            payload["delays_cs"],
            payload["width"],
            payload["height"],
        )
    except (GifEncodeError, OSError, KeyError, pickle.UnpicklingError) as exc:
        print(f"编码失败：{exc}", file=sys.stderr)
        return 1
    print(f"编码完成：{args.output}")
    return 0


def _build_convert_parser() -> argparse.ArgumentParser:
    """完整转换参数解析器（同时服务默认路径与 ``convert`` 子命令）。"""
    parser = argparse.ArgumentParser(
        prog="gif_convert",
        description="视频转 GIF：抽帧/量化/编码三步解耦，厘秒级时序零漂移",
    )
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出 GIF 路径")
    _add_common_quality_args(parser)
    parser.add_argument("--work-dir", default=None,
                        help="抽帧工作目录，给出后可跨调用复用抽帧产物")
    parser.add_argument("--no-reuse", action="store_true",
                        help="忽略工作目录已有抽帧产物，强制重新解码")
    return parser


def main(argv: list | None = None) -> int:
    # argparse 的位置参数与子命令无法干净共存，这里手动识别三个阶段
    # 关键字；首个参数不是关键字时整体按「完整转换」解析。
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _build_convert_parser().print_help()
        return 2

    command = argv[0]
    if command in ("extract", "quantize", "encode"):
        rest = argv[1:]
        if command == "extract":
            p = argparse.ArgumentParser(prog="gif_convert extract")
            p.add_argument("input")
            p.add_argument("workdir")
            return _cmd_extract(p.parse_args(rest))
        if command == "quantize":
            p = argparse.ArgumentParser(prog="gif_convert quantize")
            p.add_argument("workdir")
            p.add_argument("output")
            _add_common_quality_args(p)
            return _cmd_quantize(p.parse_args(rest))
        p = argparse.ArgumentParser(prog="gif_convert encode")
        p.add_argument("payload")
        p.add_argument("output")
        return _cmd_encode(p.parse_args(rest))

    # 默认完整转换（兼容 ``convert`` 显式前缀）。
    if command == "convert":
        argv = argv[1:]
    args = _build_convert_parser().parse_args(argv)
    return _cmd_convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
