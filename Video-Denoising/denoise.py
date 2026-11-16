#!/usr/bin/env python3
"""均值滤波视频降噪命令行入口。

用法：
    python denoise.py 输入 输出 [-r 半径] [--budget-mb 预算]

处理接在 ffmpeg 解码与编码之间：解码为原始帧 → 分带均值滤波 →
重新编码，音轨原样直通。分带预算缺省按可用内存推导。
"""

import argparse
import sys

from vdenoise import denoise_video


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="denoise",
        description="均值滤波视频降噪（解码→滤波→编码，音轨直通）",
    )
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="滤波半径，窗口 (2r+1)^2，默认 2")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="单帧分带内存预算（MB），默认按可用内存推导")
    args = parser.parse_args(argv)

    budget = int(args.budget_mb * 1024 * 1024) if args.budget_mb else None

    def progress(n):
        print(f"\r已处理 {n} 帧", end="", file=sys.stderr)

    try:
        info = denoise_video(args.input, args.output,
                             radius=args.radius, budget_bytes=budget,
                             progress=progress)
    except Exception as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1
    print(f"\n完成：{info['frames']} 帧（工作格式 {info['pix_fmt']}）"
          f" -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
