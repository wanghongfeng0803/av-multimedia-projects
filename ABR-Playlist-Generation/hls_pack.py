#!/usr/bin/env python3
"""单码率 HLS 打包命令行入口（探测 / 规划 / 切割 / 列表四步解耦）。

用法::

    python hls_pack.py 输入视频 输出目录 [--target 6] [--playlist index.m3u8]
                          [--prefix seg] [--tolerance 20]

时序保证：切割与播放列表共用同一份整数毫秒时间轴计划，EXTINF 写各片
ffprobe 实测时长（3 位小数），TARGETDURATION 取实测上限向上取整；
末片不足目标时长时在汇总里单独标出。任一片实测偏差超过容差则整体
失败，不会落盘对不上的播放列表。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vhls import (  # noqa: E402
    DEFAULT_DURATION_TOLERANCE_MS,
    DEFAULT_TARGET_SECONDS,
    PackError,
    pack_hls,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hls_pack",
        description="单码率 HLS 打包：统一整数毫秒时间轴，切片与 m3u8 零漂移",
    )
    parser.add_argument("input", help="输入视频路径（含视频流，音频可选）")
    parser.add_argument("output_dir", help="产物目录（分片与 m3u8 均写入此目录）")
    parser.add_argument(
        "--target", type=float, default=DEFAULT_TARGET_SECONDS,
        help="目标分片时长（秒），默认 %(default)s",
    )
    parser.add_argument(
        "--playlist", default="index.m3u8",
        help="播放列表文件名，默认 %(default)s",
    )
    parser.add_argument(
        "--prefix", default="seg",
        help="分片文件名前缀，默认 %(default)s（产物为 <prefix>_00000.ts …）",
    )
    parser.add_argument(
        "--tolerance", type=int, default=DEFAULT_DURATION_TOLERANCE_MS,
        help="分片实测时长与计划值容差（毫秒），默认 %(default)s",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = pack_hls(
            args.input,
            args.output_dir,
            playlist_name=args.playlist,
            target_seconds=args.target,
            segment_prefix=args.prefix,
            tolerance_ms=args.tolerance,
        )
    except PackError as exc:
        print(f"打包失败：{exc}", file=sys.stderr)
        return 1

    pl = result.playlist
    lines = [
        f"完成：{pl.path}",
        f"  分片 {pl.segment_count} 个，TARGETDURATION={pl.target_duration}s，"
        f"媒体序列号 {pl.media_sequence} 起连续",
        f"  各片实测时长之和 {pl.total_ms / 1000:.3f}s，"
        f"源时长 {result.plan.source_duration_ms / 1000:.3f}s",
    ]
    for seg in result.short_segments:
        lines.append(
            f"  注意：末片 {seg.spec.filename} 为不足目标时长的短尾片，"
            f"实测 {seg.actual_duration_ms / 1000:.3f}s（目标 "
            f"{result.plan.target_ms / 1000:.3f}s），EXTINF 已按真实时长写出"
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
