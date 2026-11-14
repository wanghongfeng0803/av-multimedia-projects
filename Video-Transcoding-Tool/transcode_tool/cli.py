"""命令行入口。

用法::

    python -m transcode_tool input.mp4 [-o out.mp4] [--codec h265]
                                        [--container mp4] [--crf 28]
                                        [--preset medium] [--overwrite]
                                        [--segment-duration 300] [--progress]
                                        [--json] [--list-codecs]
"""

from __future__ import annotations

import argparse
import json
import sys

from .codecs import list_codecs
from .containers import ContainerError, list_containers
from .errors import TranscodeError
from .transcoder import transcode


def build_parser() -> argparse.ArgumentParser:
    codec_choices = _codec_choice_help()
    parser = argparse.ArgumentParser(
        prog="transcode-tool",
        description=(
            "多编码视频转码工具：支持 H.264 / H.265 / VP9 / AV1 / VP8，"
            "输出 MP4 / WebM / MKV，默认 H.264(yuv420p) + MP4。"
        ),
    )
    parser.add_argument("input", nargs="?", help="输入视频文件路径（格式不限，按内容探测）")
    parser.add_argument(
        "-o", "--output",
        help="输出路径；省略时在输入旁生成 <文件名>_<编码>.<容器扩展名>",
    )
    parser.add_argument(
        "--codec", default="h264", metavar="CODEC",
        help=f"目标视频编码（默认 h264）。{codec_choices}",
    )
    parser.add_argument(
        "--container",
        help="目标容器 mp4/webm/mkv；省略时按输出扩展名推断，再取编码首选容器",
    )
    parser.add_argument(
        "--crf", type=int, default=None,
        help="质量参数，范围随编码而定；省略时使用该编码的默认值",
    )
    parser.add_argument(
        "--preset", default=None,
        help="速度档位：十档命名（ultrafast…placebo）或编码器原生数字档位",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="输出文件已存在时直接覆盖（默认拒绝覆盖）",
    )
    parser.add_argument(
        "--audio-bitrate", default="128k",
        help="音频需重编码时的码率（默认 128k）",
    )
    segment = parser.add_mutually_exclusive_group()
    segment.add_argument(
        "--segment-duration", type=float, default=None, metavar="SECONDS",
        help="大文件分段转码的单段时长（秒），分段无损拼接",
    )
    segment.add_argument(
        "--segment-count", type=int, default=None, metavar="N",
        help="大文件分段转码的段数（与 --segment-duration 二选一）",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="在 stderr 实时显示转码进度百分比",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出结果摘要，便于脚本调用",
    )
    parser.add_argument(
        "--list-codecs", action="store_true",
        help="列出本机 ffmpeg 支持的编码/编码器/容器后退出",
    )
    return parser


def _codec_choice_help() -> str:
    parts = []
    for spec in list_codecs():
        aliases = "/".join(spec.aliases)
        parts.append(f"{spec.canonical}（别名 {aliases}）" if aliases else spec.canonical)
    return "可选: " + "、".join(parts)


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

    if args.list_codecs:
        return _print_codec_catalog(args.json)

    if not args.input:
        parser.error("缺少输入文件参数")

    progress_callback = _stderr_progress if args.progress else None

    try:
        result = transcode(
            input_path=args.input,
            output_path=args.output,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
            audio_bitrate=args.audio_bitrate,
            codec=args.codec,
            container=args.container,
            segment_duration=args.segment_duration,
            segment_count=args.segment_count,
            progress_callback=progress_callback,
        )
    except ContainerError as exc:
        # 容器/编码组合不兼容属于参数层面的错误，退出码 1
        return _fail(args.json, str(exc))
    except TranscodeError as exc:
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)},
                             ensure_ascii=False, indent=2))
        else:
            print(f"错误: {exc}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        # 参数本身不合法（crf/preset/codec/分段参数）
        return _fail(args.json, str(exc))

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human_readable(result)
    return 0


def _fail(as_json: bool, message: str) -> int:
    if as_json:
        print(json.dumps({"success": False, "error": message},
                         ensure_ascii=False, indent=2))
    else:
        print(f"错误: {message}", file=sys.stderr)
    return 1


def _stderr_progress(ratio: float) -> None:
    """进度回调：在 stderr 同一行刷新百分比，不污染 stdout 的 JSON 结果。"""
    try:
        print(f"\r转码进度: {ratio * 100:6.2f}%", end="", file=sys.stderr,
              flush=True)
        if ratio >= 1.0:
            print(file=sys.stderr)
    except (ValueError, OSError):
        pass


def _print_codec_catalog(as_json: bool) -> int:
    from .capabilities import available_encoders

    available = available_encoders()
    if as_json:
        payload = {
            "codecs": [
                {
                    "codec": spec.canonical,
                    "label": spec.label,
                    "aliases": list(spec.aliases),
                    "default_crf": spec.default_crf,
                    "crf_range": [spec.crf_min, spec.crf_max],
                    "encoders": [enc.name for enc in spec.encoder_chain],
                    "preferred_containers": list(spec.preferred_containers),
                }
                for spec in list_codecs()
            ],
            "containers": [c.canonical for c in list_containers()],
            "ffmpeg_encoders": sorted(available) if available is not None else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("可用视频编码：")
    for spec in list_codecs():
        enc_status = []
        for enc in spec.encoder_chain:
            if available is None:
                state = "未探测"
            else:
                state = "可用" if enc.name in available else "缺失"
            enc_status.append(f"{enc.name}[{state}]")
        print(
            f"  {spec.canonical:<5} {spec.label:<24} "
            f"CRF {spec.crf_min}-{spec.crf_max}（默认 {spec.default_crf}）"
            f"  编码器: {', '.join(enc_status)}"
        )
    print("可用容器：", ", ".join(c.canonical for c in list_containers()))
    if available is None:
        print("提示：未能执行编码器探测，实际可用性以转码时 ffmpeg 输出为准。")
    return 0


def _print_human_readable(result) -> None:
    print("转码完成 [OK]")
    print(f"  输出文件: {result.output_path}")
    container = f" / {result.container}" if result.container else ""
    encoder = f"（编码器 {result.encoder}）" if result.encoder else ""
    print(
        f"  视频编码: {result.video_codec}{container}{encoder} "
        f"({result.width}x{result.height})"
    )
    print(f"  音频编码: {result.audio_codec or '无音频'}")
    if result.duration_seconds is not None:
        print(f"  时长: {result.duration_seconds:.2f} 秒")
    for warning in result.warnings:
        print(f"  注意: {warning}")
    if result.ffmpeg_log_path:
        print(f"  ffmpeg 日志: {result.ffmpeg_log_path}")


if __name__ == "__main__":
    sys.exit(main())
