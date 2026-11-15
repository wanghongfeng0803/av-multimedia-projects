"""命令行入口。

示例
====
::

    python -m hdr_to_sdr input.mkv output_sdr.mp4
    python -m hdr_to_sdr input.mkv out.mp4 --tone bt2390 --codec libx265 --10bit
    python -m hdr_to_sdr input.mkv out.mp4 --assume hlg --workers 4
"""

from __future__ import annotations

import argparse
import sys

from .colorscience import RANGE_FULL, RANGE_LIMITED
from .errors import HDRConversionError
from .pipeline import convert_hdr_to_sdr
from .probe import probe_hdr
from .tonemap import TONE_BT2390, TONE_CLIP, TONE_REINHARD
from .transfer import TRANSFER_HLG, TRANSFER_PQ

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器（独立成函数便于 CLI 单测）。"""
    parser = argparse.ArgumentParser(
        prog="hdr_to_sdr",
        description="HDR(PQ/HLG) → SDR(BT.709) 转换：自动识别传递函数，"
        "亮度压缩 + 色域映射，输出清除全部 HDR 元数据。",
    )
    parser.add_argument("input", help="输入 HDR 视频文件")
    parser.add_argument("output", help="输出 SDR 视频文件")
    parser.add_argument(
        "--tone",
        choices=[TONE_REINHARD, TONE_BT2390, TONE_CLIP],
        default=TONE_REINHARD,
        help="色调映射曲线（默认 reinhard，高光软压缩不过曝）",
    )
    parser.add_argument(
        "--codec",
        choices=["libx264", "libx265"],
        default="libx264",
        help="输出编码器（默认 libx264）",
    )
    parser.add_argument(
        "--10bit",
        dest="ten_bit",
        action="store_true",
        help="输出 10bit（默认 8bit）",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="输出 full range 码值（默认 limited/TV range）",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="编码器 CRF（默认 18）",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="编码器 preset（默认 medium）",
    )
    parser.add_argument(
        "--assume",
        choices=["pq", "hlg"],
        default=None,
        help="元数据缺失/不可信时显式指定 HDR 类型；正常情况下无需使用，"
        "工具不会猜测类型",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="帧内分带并行进程数（默认 1 串行，结果逐像素一致）",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="只探测并打印 HDR 信息，不执行转换",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=0,
        metavar="N",
        help="每 N 帧打印一次进度（默认不打印）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""
    args = build_parser().parse_args(argv)

    try:
        if args.probe_only:
            info = probe_hdr(args.input, assume=args.assume)
            kind = "PQ (SMPTE ST 2084)" if info.transfer == TRANSFER_PQ \
                else "HLG (ARIB STD-B67)"
            print(f"文件: {args.input}")
            print(f"HDR 类型: {kind}{'（assume 指定）' if info.assumed else ''}")
            print(f"尺寸: {info.width}x{info.height}")
            print(f"码深/范围: {info.bits_per_raw_sample}bit / {info.pixel_range}")
            print(f"原色: {info.color_primaries}  RGB 像素族: {info.is_rgb}")
            print(f"帧率: {info.frame_rate}")
            print(f"母版峰值: {info.peak_nits or '未声明'} nit")
            return 0

        result = convert_hdr_to_sdr(
            args.input,
            args.output,
            assume=args.assume,
            tone_method=args.tone,
            codec=args.codec,
            dst_bits=10 if args.ten_bit else 8,
            dst_range=RANGE_FULL if args.full_range else RANGE_LIMITED,
            crf=args.crf,
            preset=args.preset,
            workers=max(1, args.workers),
            progress_every=max(0, args.progress),
        )
    except HDRConversionError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    print(f"转换完成: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
