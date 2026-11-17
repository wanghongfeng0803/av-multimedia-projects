#!/usr/bin/env python3
"""多码率（ABR）HLS 打包命令行入口。

用法::

    python abr_pack.py 输入视频 输出目录 [--target 6] [--tolerance 20]
                        [--ladder 360p:400k,480p:800k,720p:2500k]

不指定 ``--ladder`` 时按源高度自动生成阶梯（不放大，至少一路源分辨率）。
主列表的 BANDWIDTH/RESOLUTION 均以切割产物实测为准，可用 ffprobe 与
分片文件独立核对；各路码率共享同一份时间轴，分片数与切点严格一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vhls import (  # noqa: E402
    DEFAULT_DURATION_TOLERANCE_MS,
    DEFAULT_TARGET_SECONDS,
    AbrError,
    Rendition,
    pack_abr,
)


def _parse_ladder(text: str) -> tuple:
    """解析 ``360p:400k,480p:800k`` 形式的码率阶梯参数。"""
    renditions = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, br_text = item.partition(":")
        if not name or not br_text:
            raise ValueError(f"阶梯项格式应为 <名称>:<码率>，实际为 {item!r}")
        height = int(name.rstrip("p"))
        br_text = br_text.strip().lower()
        if br_text.endswith("k"):
            bitrate = int(float(br_text[:-1]) * 1000)
        elif br_text.endswith("m"):
            bitrate = int(float(br_text[:-1]) * 1_000_000)
        else:
            bitrate = int(br_text)
        renditions.append(
            Rendition(name=name.strip(), height=height, video_bitrate=bitrate)
        )
    if not renditions:
        raise ValueError("码率阶梯不能为空")
    return tuple(renditions)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abr_pack",
        description="多码率 HLS 打包：共享时间轴，主列表标注可独立核对",
    )
    parser.add_argument("input", help="输入视频路径（含视频流，音频可选）")
    parser.add_argument("output_dir", help="产物根目录（主列表与各路子目录）")
    parser.add_argument(
        "--target", type=float, default=DEFAULT_TARGET_SECONDS,
        help="目标分片时长（秒），默认 %(default)s",
    )
    parser.add_argument(
        "--tolerance", type=int, default=DEFAULT_DURATION_TOLERANCE_MS,
        help="分片实测时长与计划值容差（毫秒），默认 %(default)s",
    )
    parser.add_argument(
        "--ladder", default=None,
        help="码率阶梯，如 360p:400k,480p:800k,720p:2500k；缺省按源高度自动",
    )
    parser.add_argument(
        "--master", default="master.m3u8",
        help="主列表文件名，默认 %(default)s",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        renditions = _parse_ladder(args.ladder) if args.ladder else None
    except ValueError as exc:
        print(f"阶梯参数错误：{exc}", file=sys.stderr)
        return 2

    try:
        result = pack_abr(
            args.input,
            args.output_dir,
            renditions=renditions,
            target_seconds=args.target,
            tolerance_ms=args.tolerance,
            master_name=args.master,
        )
    except AbrError as exc:
        print(f"打包失败：{exc}", file=sys.stderr)
        return 1

    lines = [
        f"完成：{result.master_path}",
        f"  共享时间轴 {result.plan.segment_count} 片，"
        f"源时长 {result.plan.source_duration_ms / 1000:.3f}s",
    ]
    for v in result.variants:
        lines.append(
            f"  {v.rendition.name}: {v.width}x{v.height} "
            f"BANDWIDTH={v.bandwidth}bps "
            f"({v.playlist.segment_count} 片, "
            f"TARGETDURATION={v.playlist.target_duration}s)"
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
