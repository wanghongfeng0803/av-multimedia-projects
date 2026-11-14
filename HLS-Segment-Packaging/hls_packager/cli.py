"""命令行入口：python -m hls_packager input.mp4 output_dir/"""

from __future__ import annotations

import argparse
import sys

from .errors import PackagerError
from .packager import PLAYLIST_FILENAME, package

EXIT_OK = 0
EXIT_PACKAGER_ERROR = 2
EXIT_UNEXPECTED = 3


def _force_utf8_console() -> None:
    """Windows 控制台默认代码页可能不是 UTF-8，显式切换避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hls-packager",
        description=(
            "把视频按关键帧边界封装为 MPEG-TS 分片并生成 M3U8 播放列表"
            "（不重编码，切点严格对齐 GOP）"
        ),
    )
    parser.add_argument("input", help="输入媒体文件路径（mp4/mkv/mov/ts 等）")
    parser.add_argument("output_dir", help="HLS 产物输出目录（不存在会自动创建）")
    parser.add_argument(
        "-d", "--segment-duration",
        type=float,
        default=6.0,
        help="目标分片时长（秒），实际切点对齐最近关键帧，默认 6",
    )
    parser.add_argument(
        "-w", "--window-size",
        type=int,
        default=None,
        help="直播滑动窗口保留的分片数；不指定则输出 VOD 全量列表并带 ENDLIST",
    )
    parser.add_argument(
        "-s", "--media-sequence",
        type=int,
        default=0,
        help="首个分片的 EXT-X-MEDIA-SEQUENCE 序号，默认 0",
    )
    parser.add_argument(
        "-p", "--filename-prefix",
        default="seg",
        help="分片文件名前缀，默认 seg（产出 seg_0000.ts ...）",
    )
    parser.add_argument(
        "-y", "--overwrite",
        action="store_true",
        help="允许写入非空输出目录并覆盖同名产物",
    )
    return parser


def _print_summary(result) -> None:
    segments = result.segments
    max_dur = max((s.duration for s in segments), default=0.0)
    total = sum(s.duration for s in segments)
    print(f"完成：{result.media.path}")
    print(f"  输出目录   : {result.output_dir}")
    print(f"  播放列表   : {PLAYLIST_FILENAME}")
    print(
        f"  分片数量   : {len(segments)}（TARGETDURATION="
        f"{result.target_duration}s，最大实际片长 {max_dur:.3f}s，"
        f"总时长 {total:.3f}s）"
    )
    print(
        "  列表模式   : "
        + (
            f"滑动窗口 {result.window_size} 片"
            if result.window_size is not None
            else "VOD（含 ENDLIST）"
        )
    )
    print("  分片明细（前 5 片）:")
    for seg in segments[:5]:
        print(
            f"    #{seg.sequence_number:04d} {seg.filename}  "
            f"src@{seg.source_start:7.3f}s  轴@{seg.start_time:7.3f}s  "
            f"时长 {seg.duration:6.3f}s"
        )
    if len(segments) > 5:
        print(f"    ... 其余 {len(segments) - 5} 片见 segments.json")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        result = package(
            input_path=args.input,
            output_dir=args.output_dir,
            target_duration=args.segment_duration,
            window_size=args.window_size,
            media_sequence=args.media_sequence,
            filename_prefix=args.filename_prefix,
            overwrite=args.overwrite,
        )
    except PackagerError as exc:
        # 可预期的输入/环境/切片问题：明确中文提示 + 非零退出码
        detail = f"：{exc.stderr_tail}" if getattr(exc, "stderr_tail", "") else ""
        print(f"错误：{exc}{detail}", file=sys.stderr)
        return EXIT_PACKAGER_ERROR
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_PACKAGER_ERROR

    _print_summary(result)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
