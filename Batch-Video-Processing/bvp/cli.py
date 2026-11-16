"""命令行入口：``python -m bvp batch <输入...> [选项]``。

退出码约定（与错误分类契约对齐，便于脚本编排）：

- 0：全部成功或跳过，且清单对账通过；
- 8：批量跑完但存在失败条目，或对账发现清单与磁盘不符；
- 2：启动前参数错误（未扫描、未转码即快速失败）。

CLI 只做参数解析与结果打印，所有编排逻辑都在 :mod:`bvp.batch`。
"""

from __future__ import annotations

import argparse
import json
import sys

from .batch import batch_transcode
from .planning import EXIST_OVERWRITE, EXIST_SKIP, EXIST_SUFFIX
from .verify import reconcile_manifest_files

__all__ = ["main", "build_parser"]

_EXIT_OK = 0
_EXIT_BATCH_FAILURE = 8
_EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bvp",
        description="可复核的目录批量视频转码（成功/失败/跳过清单落盘）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser(
        "batch", help="对一个或多个文件/目录执行批量转码"
    )
    p_batch.add_argument(
        "inputs", nargs="+", help="输入视频文件或目录（目录可嵌套递归）"
    )
    p_batch.add_argument(
        "-o", "--output-dir", default=None,
        help="统一输出目录，镜像输入子目录；缺省产物生成在输入旁",
    )
    p_batch.add_argument(
        "--manifest-dir", default=None,
        help="清单目录；缺省取输出目录，再退化为当前工作目录",
    )
    p_batch.add_argument(
        "--codec", default="h264", choices=("h264", "h265"),
        help="目标视频编码（默认 h264）",
    )
    p_batch.add_argument(
        "--container", default="mp4", choices=("mp4", "mkv"),
        help="目标容器（默认 mp4）",
    )
    p_batch.add_argument("--crf", type=int, default=None, help="质量参数 CRF")
    p_batch.add_argument("--preset", default="medium", help="x264/x265 速度档位")
    p_batch.add_argument(
        "--audio-bitrate", default="128k", help="音频重编码码率（默认 128k）"
    )
    p_batch.add_argument(
        "--resolution", default=None,
        help="目标分辨率，如 1280x720 / 720p / 50%%",
    )
    p_batch.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="并发工作线程数（默认 1，建议不超过物理核数）",
    )
    p_batch.add_argument(
        "--on-exist", default=EXIST_SKIP,
        choices=(EXIST_SKIP, EXIST_OVERWRITE, EXIST_SUFFIX),
        help="输出已存在策略（默认 skip）",
    )
    p_batch.add_argument(
        "--no-recursive", action="store_true",
        help="目录只扫描一层，不递归子目录",
    )
    p_batch.add_argument(
        "--ext", action="append", default=None,
        help="自定义扫描扩展名，可重复给出，如 --ext .mp4 --ext .mov",
    )

    p_verify = sub.add_parser(
        "verify", help="对清单目录执行独立对账（不运行转码）"
    )
    p_verify.add_argument("manifest_dir", help="存放三份清单的目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        return _run_verify(args.manifest_dir)
    return _run_batch(args)


def _run_batch(args: argparse.Namespace) -> int:
    extensions = (
        frozenset(args.ext) if args.ext else None
    )
    try:
        result = batch_transcode(
            list(args.inputs),
            output_dir=args.output_dir,
            recursive=not args.no_recursive,
            jobs=args.jobs,
            on_exist=args.on_exist,
            codec=args.codec,
            container=args.container,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            resolution=args.resolution,
            extensions=extensions,
            manifest_dir=args.manifest_dir,
        )
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    summary = result.summary
    print("=" * 60)
    print("批量转码结束")
    print(
        f"  总计 {summary['total']} | 成功 {summary['success']} | "
        f"失败 {summary['failed']} | 跳过 {summary['skipped']}"
    )
    for label, key in (("全量清单", "full"), ("失败清单", "failed"),
                       ("跳过清单", "skipped")):
        print(f"  {label}: {result.manifest_paths[key]}")
    print(f"  对账结果: {'通过' if result.reconciliation.ok else '不符'}")
    if not result.reconciliation.ok:
        for mismatch in result.reconciliation.mismatches:
            print(f"    - [{mismatch.kind}] {mismatch.message}", file=sys.stderr)

    if result.success:
        return _EXIT_OK
    return _EXIT_BATCH_FAILURE


def _run_verify(manifest_dir: str) -> int:
    result = reconcile_manifest_files(manifest_dir)
    payload = {
        "ok": result.ok,
        "summary": result.summary,
        "produced_outputs": result.produced_outputs,
        "existing_outputs": result.existing_outputs,
        "mismatches": [
            {
                "index": m.index,
                "input_path": m.input_path,
                "kind": m.kind,
                "message": m.message,
            }
            for m in result.mismatches
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return _EXIT_OK if result.ok else _EXIT_BATCH_FAILURE
