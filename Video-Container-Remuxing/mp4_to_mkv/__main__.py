"""命令行入口：python -m mp4_to_mkv input.mp4 output.mkv"""

from __future__ import annotations

import argparse
import sys

from .errors import RemuxError
from .remuxer import remux

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_UNEXPECTED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mp4-to-mkv",
        description="把 MP4 容器无损重封装为 MKV（不重编码，帧字节原样拷贝）",
    )
    parser.add_argument("input", help="输入 MP4 文件路径（.mp4/.m4v/.mov/.m4a）")
    parser.add_argument("output", help="输出 MKV 文件路径（.mkv）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        byte_count, sample_count, track_count = remux(
            args.input, args.output, overwrite=args.overwrite
        )
    except RemuxError as exc:
        # 可预期的输入/格式问题：明确的中文提示 + 非零退出码
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except MemoryError:
        print("错误：内存不足，输入文件超出基础版本的处理能力", file=sys.stderr)
        return EXIT_UNEXPECTED

    print(
        f"完成：{args.input} -> {args.output}\n"
        f"  轨道数 {track_count}，样本数 {sample_count}，"
        f"输出 {byte_count} 字节（全程未重编码）"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
