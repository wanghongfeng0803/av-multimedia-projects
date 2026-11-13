"""命令行入口。

用法::

    python -m transcode_tool input.mp4 [-o output.mp4] [--crf 23]
                                        [--preset medium] [--overwrite] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import TranscodeError
from .transcoder import transcode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcode-tool",
        description="将输入视频转码为 H.264 (yuv420p) 编码的 MP4 文件。",
    )
    parser.add_argument("input", help="输入视频文件路径（格式不限，按内容探测）")
    parser.add_argument(
        "-o", "--output",
        help="输出 MP4 路径；省略时在输入旁生成 <文件名>_h264.mp4",
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="x264 质量参数 0-51，越小质量越高（默认 23）",
    )
    parser.add_argument(
        "--preset", default="medium",
        help="x264 速度档位，如 ultrafast/medium/slow（默认 medium）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="输出文件已存在时直接覆盖（默认拒绝覆盖）",
    )
    parser.add_argument(
        "--audio-bitrate", default="128k",
        help="音频需重编码为 AAC 时的码率（默认 128k）",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出结果摘要，便于脚本调用",
    )
    return parser


def _make_stdio_lenient() -> None:
    """统一标准流编码，避免 Windows 控制台/管道下打印崩溃或乱码。

    优先重配置为 UTF-8（真实控制台、重定向与管道均适用）；不支持时
    退化为保留原编码、仅把无法编码的字符替换为 ?。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _make_stdio_lenient()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = transcode(
            input_path=args.input,
            output_path=args.output,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
            audio_bitrate=args.audio_bitrate,
        )
    except TranscodeError as exc:
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)},
                             ensure_ascii=False, indent=2))
        else:
            print(f"错误: {exc}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        # 参数本身不合法（crf/preset）
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human_readable(result)
    return 0


def _print_human_readable(result) -> None:
    print("转码完成 [OK]")
    print(f"  输出文件: {result.output_path}")
    print(f"  视频编码: {result.video_codec} ({result.width}x{result.height})")
    print(f"  音频编码: {result.audio_codec or '无音频'}")
    if result.duration_seconds is not None:
        print(f"  时长: {result.duration_seconds:.2f} 秒")
    for warning in result.warnings:
        print(f"  注意: {warning}")
    if result.ffmpeg_log_path:
        print(f"  ffmpeg 日志: {result.ffmpeg_log_path}")


if __name__ == "__main__":
    sys.exit(main())
