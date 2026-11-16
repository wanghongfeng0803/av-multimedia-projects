#!/usr/bin/env python3
"""视频降噪命令行入口（均值 / 中值两种模式）。

用法：
    python denoise.py 输入 输出 [-r 半径] [--budget-mb 预算]
    python denoise.py 输入 输出 --mode median [--window 奇数边长]
                                [--dump-border 目录]

处理接在 ffmpeg 解码与编码之间：解码为原始帧 → 分带空间滤波 →
重新编码，音轨原样直通。分带预算缺省按可用内存推导。

默认均值模式的参数与行为与历史版本完全一致；中值模式采用镜像
边界与直方图加速，专门处理椒盐噪声。
"""

import argparse
import sys

from vdenoise import FILTER_MODES, denoise_video


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="denoise",
        description="视频降噪（解码→滤波→编码，音轨直通）：均值 / 中值",
    )
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="滤波半径，窗口 (2r+1)^2，默认 2（均值模式）")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="单帧分带内存预算（MB），默认按可用内存推导")
    parser.add_argument("--mode", choices=FILTER_MODES, default="mean",
                        help="滤波模式：mean 均值（默认）/ median 中值（椒盐噪声）")
    parser.add_argument("--window", type=int, default=None,
                        help="中值模式窗口边长（正奇数，按素材配置），"
                             "给出时优先于 --radius；缺省沿用 -r（默认 5x5）")
    parser.add_argument("--dump-border", default=None, metavar="DIR",
                        help="中值模式：把每帧边界区域原始字节落盘到该目录，"
                             "供单独抽帧复核（附 manifest.json）")
    args = parser.parse_args(argv)

    budget = int(args.budget_mb * 1024 * 1024) if args.budget_mb else None

    def progress(n):
        print(f"\r已处理 {n} 帧", end="", file=sys.stderr)

    try:
        info = denoise_video(
            args.input, args.output,
            radius=args.radius, budget_bytes=budget, progress=progress,
            mode=args.mode, window=args.window, border_dir=args.dump_border,
        )
    except Exception as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1
    if info["mode"] == "median":
        detail = f"中值窗口 {info['window']}x{info['window']}"
        if info["border_dir"]:
            detail += f"，边界复核目录 {info['border_dir']}"
    else:
        detail = f"半径 {info['radius']}"
    print(f"\n完成：{info['frames']} 帧（{info['mode']}，{detail}，"
          f"工作格式 {info['pix_fmt']}） -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
