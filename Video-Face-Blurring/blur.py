#!/usr/bin/env python3
"""视频区域马赛克命令行入口（打码 / 区域管理 / 编码三模块解耦）。

用法::

    python blur.py 输入视频 输出视频 --region X,Y,W,H [--region ...]
                   [--block 16]

区域以像素坐标给出，可重复多次；支持负尺寸（``W,H`` 为负时视为终点
坐标写法）与越界（部分越界自动裁剪，完全越界/零面积丢弃并告警）。
所有区域共用同一套原点 (0,0)、边长 ``--block`` 的全局块网格，重叠
处只打一次、无拼接缝。

输出帧率 / 尺寸 / 像素格式 / 时间戳与输入一致，音轨原样直通。

退出码：0 成功；2 参数错误（argparse）；5 探测失败；6 处理失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vblur import DEFAULT_BLOCK, PipelineError, blur_video, probe  # noqa: E402

EXIT_PROBE_ERROR = 5
EXIT_PIPELINE_ERROR = 6


def _parse_region(text):
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"区域必须是 X,Y,W,H 四个整数：{text!r}"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"区域坐标必须是整数：{text!r}"
        ) from None


def main(argv=None):
    p = argparse.ArgumentParser(
        description="视频区域马赛克：多区域共用全局块网格，重叠只打一次",
    )
    p.add_argument("src", help="输入视频")
    p.add_argument("dst", help="输出视频")
    p.add_argument(
        "--region", action="append", type=_parse_region, default=[],
        metavar="X,Y,W,H",
        help="打码区域（像素），可重复；W/H 可为负（终点坐标写法），"
             "越界自动钳制",
    )
    p.add_argument(
        "--block", type=int, default=DEFAULT_BLOCK,
        help="全局网格块边长（像素），默认 %(default)s；1 为恒等",
    )
    args = p.parse_args(argv)

    if not args.region:
        p.error("至少需要一条 --region")
    if args.block < 1:
        p.error("--block 必须 >= 1")

    try:
        probe(args.src)
    except Exception as exc:
        print(f"探测失败：{exc}", file=sys.stderr)
        return EXIT_PROBE_ERROR

    try:
        stats = blur_video(
            args.src, args.dst,
            regions=args.region, block=args.block,
            progress=lambda n: print(f"\r已处理 {n} 帧", end="", flush=True),
        )
    except (PipelineError, ValueError) as exc:
        print(f"\n处理失败：{exc}", file=sys.stderr)
        return EXIT_PIPELINE_ERROR
    print(f"\n完成：{stats['frames']} 帧 -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
