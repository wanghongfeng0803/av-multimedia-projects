"""命令行入口：python -m vkeyer input.mp4 -o out.mov [选项]

示例
====
    # 绿幕：自动从顶部条带采样键色，输出带 alpha 的 mov
    python key.py green.mp4 out.mov

    # 蓝幕：显式给键色，收紧软化带，羽化 2 像素
    python key.py blue.mp4 out.mov --key 0,70,255 --softness 40 --feather 2

    # 背景替换：抠像后合成到纯黑背景，直接出 mp4
    python key.py green.mp4 out.mp4 --bg-color 0,0,0

    # 背景替换：合成到背景图片/视频
    python key.py green.mp4 out.mp4 --bg-image backdrop.png
"""

from __future__ import annotations

import argparse
import sys

from .errors import ChromaKeyError
from .matte import MatteConfig
from .pipeline import key_video
from .sampler import DEFAULT_SAMPLE_FRAMES, REGION_MODES


def _parse_rgb(text, name):
    """解析 ``R,G,B`` 命令行颜色参数。"""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} 格式应为 R,G,B，得到 {text!r}")
    try:
        vals = tuple(int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} 必须是整数：{text!r}") from exc
    if not all(0 <= v <= 255 for v in vals):
        raise argparse.ArgumentTypeError(f"{name} 各通道须在 [0, 255]：{text!r}")
    return vals


def _parse_rect(text):
    """解析 ``x0,y0,x1,y1`` 归一化矩形。"""
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--sample-rect 格式应为 x0,y0,x1,y1，得到 {text!r}"
        )
    try:
        vals = [float(p.strip()) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"矩形坐标必须是数字：{text!r}") from exc
    if not all(0.0 <= v <= 1.0 for v in vals):
        raise argparse.ArgumentTypeError(f"矩形坐标须在 [0, 1]：{text!r}")
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError(
            f"矩形宽高必须为正（x1>x0 且 y1>y0）：{text!r}"
        )
    return vals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vkeyer",
        description="色度键抠像：键色画面采样（不写死绿色），三层处理"
                    "（键控 / alpha 后处理 / 输出编码），支持透明通道"
                    "输出与背景替换。",
    )
    parser.add_argument("input", help="输入视频文件")
    parser.add_argument("-o", "--output", required=True,
                        help="输出文件：alpha 视频用 .mov/.webm/.mkv，"
                             "图像序列用含 %%06d 的 .png/.webp 路径，"
                             "背景替换可用 .mp4")
    parser.add_argument("--key", dest="key_rgb", type=lambda s: _parse_rgb(s, "--key"),
                        default=None,
                        help="显式键色 R,G,B（0-255）；不给则从画面采样")
    parser.add_argument("--sample-mode", choices=REGION_MODES, default="top",
                        help="键色采样区域（默认 top 顶部条带）")
    parser.add_argument("--sample-rect", type=_parse_rect, default=None,
                        help="rect 模式的归一化矩形 x0,y0,x1,y1（0-1）")
    parser.add_argument("--sample-frames", default="0,30,60",
                        help="采样帧号（0 基，逗号分隔，默认 0,30,60，越界跳过）")
    parser.add_argument("--similarity", type=float, default=32.0,
                        help="内阈值距离，<= 它完全透明，默认 32")
    parser.add_argument("--softness", type=float, default=64.0,
                        help="软化带宽度，默认 64；0 为硬键")
    parser.add_argument("--no-despill", action="store_true",
                        help="关闭溢色抑制（默认开启，去边缘残绿/残蓝）")
    parser.add_argument("--threshold-low", type=int, default=0,
                        help="alpha 硬化下阈值，默认 0（关闭）；建议 8-16")
    parser.add_argument("--threshold-high", type=int, default=255,
                        help="alpha 硬化上阈值，默认 255（关闭）；建议 240-247")
    parser.add_argument("--choke", type=int, default=0,
                        help="遮罩收缩半径（像素），吃掉边缘一圈幕布色，默认 0")
    parser.add_argument("--feather", type=int, default=0,
                        help="盒式羽化半径（像素），默认 0 不羽化")
    bg = parser.add_mutually_exclusive_group()
    bg.add_argument("--bg-color", type=lambda s: _parse_rgb(s, "--bg-color"),
                    default=None, help="背景替换：纯色 R,G,B")
    bg.add_argument("--bg-image", default=None,
                    help="背景替换：背景图片或视频路径（自动缩放对齐）")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已存在的输出文件")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sample_frames = tuple(
            int(x.strip()) for x in args.sample_frames.split(",") if x.strip()
        )
        matte = MatteConfig(
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            choke=args.choke,
            feather=args.feather,
        )

        def progress(n):
            print(f"\r已处理 {n} 帧", end="", file=sys.stderr)

        result = key_video(
            args.input, args.output,
            key_rgb=args.key_rgb,
            sample_mode=args.sample_mode,
            sample_rect=args.sample_rect,
            sample_frames=sample_frames or DEFAULT_SAMPLE_FRAMES,
            similarity=args.similarity,
            softness=args.softness,
            despill=not args.no_despill,
            matte_config=matte,
            bg_color=args.bg_color,
            bg_image=args.bg_image,
            overwrite=args.overwrite,
            progress=progress,
        )
    except ChromaKeyError as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        stderr = getattr(exc, "stderr", "")
        if stderr:
            print(f"ffmpeg 输出: {stderr[-500:]}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        return 1

    for msg in result.messages:
        print(f"提示: {msg}", file=sys.stderr)
    print(
        f"\n完成: {result.frames} 帧, {result.width}x{result.height}, "
        f"键色 RGB{result.key_rgb}（{result.key_source}）, "
        f"模式 {result.mode}"
        + (f", 背景 {result.background}" if result.background else ""),
        file=sys.stderr,
    )
    print(f"输出: {result.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
