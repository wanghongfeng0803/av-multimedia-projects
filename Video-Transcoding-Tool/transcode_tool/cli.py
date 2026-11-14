"""命令行入口。

单文件用法（历史接口，参数与行为完全保持不变）::

    python -m transcode_tool input.mp4 [-o out.mp4] [--codec h265]
                                        [--container mp4] [--crf 28]
                                        [--preset medium] [--overwrite]
                                        [--segment-duration 300] [--progress]
                                        [--json] [--list-codecs]

批量用法（首参数为 ``batch``，路由到独立解析器，不影响单文件接口）::

    python -m transcode_tool batch 视频目录 [...] [-o 输出目录] [--recursive]
                                        [-j 并发数] [--on-fail continue|stop]
                                        [--on-exist skip|overwrite|suffix]
                                        [--log 错误日志] [--json] [--no-progress]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading

from .codecs import list_codecs
from .containers import ContainerError, list_containers
from .errors import TranscodeError
from .transcoder import transcode

#: batch 子命令名，也是新旧两套参数解析的路由标记
BATCH_COMMAND = "batch"
#: 批量任务被用户中断时的退出码（单文件退出码 1-7 已被 errors 占用）
_BATCH_INTERRUPTED_EXIT_CODE = 8
#: 批量任务存在失败文件时的退出码（对齐 FFmpegRunError.exit_code）
_BATCH_FAILED_EXIT_CODE = 6


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
    parser.add_argument(
        "--bitrate", default=None, metavar="RATE",
        help="目标视频码率，如 800k / 2M / 1500000；给定后按码率编码并忽略 --crf",
    )
    parser.add_argument(
        "--rate-mode", choices=["vbr", "abr", "cbr"], default="vbr",
        help="码率控制模式（仅配合 --bitrate）：vbr 可变(默认) / abr 平均 / cbr 恒定",
    )
    parser.add_argument(
        "--resolution", default=None, metavar="SPEC",
        help="输出分辨率：1920x1080 / 1280(定宽) / x720(定高) / 50%% / 720p、1080p、4k 等",
    )
    parser.add_argument(
        "--allow-upscale", action="store_true",
        help="允许把源放大到更大分辨率（默认禁止放大，超过源的请求会收敛回源尺寸）",
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # 首参数路由：batch 走独立解析器；其余一切走历史单文件解析路径，
    # 旧 CLI 的位置参数、可选参数与退出码语义完全不变
    if raw_argv and raw_argv[0] == BATCH_COMMAND:
        return _run_batch(raw_argv[1:])

    parser = build_parser()
    args = parser.parse_args(raw_argv)

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
            target_bitrate=args.bitrate,
            rate_mode=args.rate_mode,
            resolution=args.resolution,
            allow_upscale=args.allow_upscale,
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
    if result.target_bitrate_bps:
        from .bitrate import format_bitrate

        print(
            f"  目标码率: {format_bitrate(result.target_bitrate_bps)}"
            f"（{result.rate_mode or 'vbr'} 模式）"
        )
    print(f"  音频编码: {result.audio_codec or '无音频'}")
    if result.duration_seconds is not None:
        print(f"  时长: {result.duration_seconds:.2f} 秒")
    for warning in result.warnings:
        print(f"  注意: {warning}")
    if result.ffmpeg_log_path:
        print(f"  ffmpeg 日志: {result.ffmpeg_log_path}")


# ---------------------------------------------------------------------------
# batch 子命令：文件夹批量转码
# ---------------------------------------------------------------------------

def build_batch_parser() -> argparse.ArgumentParser:
    """构建批量子命令的独立参数解析器（与单文件解析器互不影响）。"""
    parser = argparse.ArgumentParser(
        prog="transcode-tool batch",
        description=(
            "文件夹批量转码：扫描一个或多个目录/文件，按统一参数有界并发"
            "转码，实时显示聚合进度，单文件失败记入错误日志而不拖垮整批。"
        ),
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="输入文件或目录路径（可给多个；目录按视频扩展名筛选）",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="统一输出目录，按输入根镜像子目录结构；省略时产物生成在各输入旁",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归扫描输入目录的子目录（输出目录位于输入目录内时自动排除）",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="并发 ffmpeg 数（默认 1，顺序执行内存峰值最低；建议不超过物理核数）",
    )
    parser.add_argument(
        "--on-fail", choices=["continue", "stop"], default="continue",
        help="单文件失败后的策略：continue 继续其余文件(默认) / stop 停止派发",
    )
    parser.add_argument(
        "--on-exist", choices=["skip", "overwrite", "suffix"], default="skip",
        help="输出已存在时：skip 跳过(默认) / overwrite 覆盖 / suffix 自动加序号",
    )
    parser.add_argument(
        "--extensions", default=None,
        help="自定义扫描扩展名，逗号分隔，如 '.mp4,.mkv'；省略用内置视频扩展名集合",
    )
    parser.add_argument(
        "--log", default=None,
        help="错误日志路径；省略时写入 <输出目录>/_batch_errors.log",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="关闭 stderr 实时聚合进度（默认开启）",
    )
    # 以下与单文件 CLI 同名同义，原样透传给每个任务
    parser.add_argument("--codec", default="h264", help="目标视频编码（默认 h264）")
    parser.add_argument("--container", default=None,
                        help="目标容器 mp4/webm/mkv；省略时取编码首选容器")
    parser.add_argument("--crf", type=int, default=None, help="质量参数")
    parser.add_argument("--preset", default=None, help="速度档位")
    parser.add_argument("--audio-bitrate", default="128k", help="音频重编码码率")
    parser.add_argument("--bitrate", default=None, help="目标视频码率（与 --crf 互斥）")
    parser.add_argument("--rate-mode", choices=["vbr", "abr", "cbr"], default="vbr")
    parser.add_argument("--resolution", default=None, help="输出分辨率表达式")
    parser.add_argument("--allow-upscale", action="store_true", help="允许放大")
    parser.add_argument(
        "--segment-duration", type=float, default=None, help="分段单段时长（秒）"
    )
    parser.add_argument("--segment-count", type=int, default=None, help="分段数量")
    parser.add_argument(
        "--json", action="store_true", help="以 JSON 格式输出批量汇总结果"
    )
    return parser


class _BatchProgressDisplay:
    """stderr 聚合进度显示器：单行动态刷新，多任务并发时输出不交错。"""

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._last_line = ""

    def __call__(self, snapshot) -> None:
        if not self._enabled:
            return
        running = "、".join(snapshot.current_files) or "-"
        line = (
            f"\r批量进度: {snapshot.overall_ratio * 100:6.2f}% "
            f"[{snapshot.finished}/{snapshot.total}]"
            f" 成功 {snapshot.done} 失败 {snapshot.failed}"
            f" 跳过 {snapshot.skipped}"
            f" | 转码中: {running[:40]}"
        )
        with self._lock:
            try:
                print(line, end="", file=sys.stderr, flush=True)
                self._last_line = line
            except (ValueError, OSError):
                pass

    def finish(self) -> None:
        if self._enabled and self._last_line:
            try:
                print(file=sys.stderr)
            except (ValueError, OSError):
                pass


def _run_batch(argv: list[str]) -> int:
    """batch 子命令入口：解析参数 → 调用 batch_transcode → 输出汇总。"""
    from .batch import batch_transcode

    parser = build_batch_parser()
    args = parser.parse_args(argv)

    extensions = None
    if args.extensions:
        extensions = frozenset(
            part.strip().lower() for part in args.extensions.split(",") if part.strip()
        )

    display = _BatchProgressDisplay(enabled=not args.no_progress)

    try:
        report = batch_transcode(
            args.inputs,
            output_dir=args.output_dir,
            recursive=args.recursive,
            jobs=args.jobs,
            on_fail=args.on_fail,
            on_exist=args.on_exist,
            extensions=extensions,
            log_path=args.log,
            progress_callback=display,
            codec=args.codec,
            container=args.container,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            target_bitrate=args.bitrate,
            rate_mode=args.rate_mode,
            resolution=args.resolution,
            allow_upscale=args.allow_upscale,
            segment_duration=args.segment_duration,
            segment_count=args.segment_count,
        )
    except ValueError as exc:
        # 参数不合法（jobs/策略/扩展名等），退出码 1
        return _fail(args.json, str(exc))
    except TranscodeError as exc:
        # 批量启动前的整批级错误（如找不到 ffmpeg）
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)},
                             ensure_ascii=False, indent=2))
        else:
            print(f"错误: {exc}", file=sys.stderr)
        return exc.exit_code

    display.finish()

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_batch_summary(report)

    if report.interrupted:
        return _BATCH_INTERRUPTED_EXIT_CODE
    if report.failed_count > 0:
        return _BATCH_FAILED_EXIT_CODE
    return 0


def _print_batch_summary(report) -> None:
    print("批量转码结束")
    print(
        f"  总计 {report.total} | 成功 {report.done_count} | "
        f"失败 {report.failed_count} | 跳过 {report.skipped_count} | "
        f"取消 {report.cancelled_count}"
    )
    if report.log_path:
        print(f"  错误日志: {report.log_path}")
    for outcome in report.outcomes:
        if outcome.status == "failed":
            print(f"  [失败] {outcome.input_path}")
            print(f"         {outcome.error_type}: {outcome.error}")
            if outcome.ffmpeg_log_path:
                print(f"         ffmpeg 日志: {outcome.ffmpeg_log_path}")
        elif outcome.status == "skipped":
            print(f"  [跳过] {outcome.input_path}")
        elif outcome.status == "cancelled":
            print(f"  [取消] {outcome.input_path}")


if __name__ == "__main__":
    sys.exit(main())
