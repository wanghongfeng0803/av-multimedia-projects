#!/usr/bin/env python3
"""视频降噪命令行入口（均值 / 中值 / 时域三种模式）。

用法：
    python denoise.py 输入 输出 [-r 半径] [--budget-mb 预算]
    python denoise.py 输入 输出 --mode median [--window 奇数边长]
                                [--dump-border 目录]
    python denoise.py 输入 输出 --mode temporal [--temporal-radius 帧数半径]
                                [--max-shift 搜索半径] [--motion-threshold 差]
                                [--max-cache 缓存帧上限] [--band-height 片长]

处理接在 ffmpeg 解码与编码之间：解码为原始帧 → 滤波 → 重新编码，
音轨原样直通。内存预算缺省按可用内存推导；时域模式下缓存帧数、
片长与预算三者互推，超预算在开工前直接报错。

默认均值模式的参数与行为与历史版本完全一致；中值模式采用镜像
边界与直方图加速，专门处理椒盐噪声；时域模式先做全局位移对齐
再混合时间窗，运动区掩码避免拖影。
"""

import argparse
import sys

from vdenoise import FILTER_MODES, denoise_video


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="denoise",
        description="视频降噪（解码→滤波→编码，音轨直通）：均值 / 中值 / 时域",
    )
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="空间滤波半径，窗口 (2r+1)^2，默认 2（均值模式）")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="内存预算（MB），默认按可用内存推导")
    parser.add_argument("--mode", choices=FILTER_MODES, default="mean",
                        help="滤波模式：mean 均值（默认）/ median 中值（椒盐）"
                             "/ temporal 时域（帧间对齐混合）")
    parser.add_argument("--window", type=int, default=None,
                        help="中值模式窗口边长（正奇数，按素材配置），"
                             "给出时优先于 --radius；缺省沿用 -r（默认 5x5）")
    parser.add_argument("--dump-border", default=None, metavar="DIR",
                        help="中值模式：把每帧边界区域原始字节落盘到该目录，"
                             "供单独抽帧复核（附 manifest.json）")
    parser.add_argument("--temporal-radius", type=int, default=2,
                        help="时域模式：时间窗半径，窗 [t-r, t+r]，默认 2")
    parser.add_argument("--max-shift", type=int, default=6,
                        help="时域模式：全局位移搜索半径（像素），默认 6")
    parser.add_argument("--motion-threshold", type=int, default=24,
                        help="时域模式：对齐残差超过该灰度差判为运动区，"
                             "该参考帧不参与混合，默认 24")
    parser.add_argument("--max-cache", type=int, default=None,
                        help="时域模式：缓存帧数硬上限，默认取模块内置上限")
    parser.add_argument("--band-height", type=int, default=None,
                        help="时域模式：显式片长（每带核心行数），"
                             "缺省在预算内取最大片长")
    args = parser.parse_args(argv)

    budget = int(args.budget_mb * 1024 * 1024) if args.budget_mb else None

    def progress(n):
        print(f"\r已处理 {n} 帧", end="", file=sys.stderr)

    try:
        kwargs = dict(
            radius=args.radius, budget_bytes=budget, progress=progress,
            mode=args.mode, window=args.window, border_dir=args.dump_border,
        )
        if args.mode == "temporal":
            kwargs.update(
                temporal_radius=args.temporal_radius,
                max_shift=args.max_shift,
                motion_threshold=args.motion_threshold,
                band_height=args.band_height,
            )
            if args.max_cache is not None:
                kwargs["max_cache_frames"] = args.max_cache
        info = denoise_video(args.input, args.output, **kwargs)
    except Exception as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1
    if info["mode"] == "median":
        detail = f"中值窗口 {info['window']}x{info['window']}"
        if info["border_dir"]:
            detail += f"，边界复核目录 {info['border_dir']}"
    elif info["mode"] == "temporal":
        detail = (f"时域半径 {info['temporal_radius']}，实际缓存 "
                  f"{info['cache_frames']}/{info['nominal_window']} 帧，"
                  f"片长 {info['band_height']} 行，"
                  f"峰值估算 {info['memory_bytes'] // 1024} KiB")
    else:
        detail = f"半径 {info['radius']}"
    print(f"\n完成：{info['frames']} 帧（{info['mode']}，{detail}，"
          f"工作格式 {info['pix_fmt']}） -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
