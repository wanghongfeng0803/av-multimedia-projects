"""命令行入口.

用法:
    python -m audio_loudness input_dir output_dir [选项]
    python -m audio_loudness input_dir output_dir --retry-failed report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .batch import failed_files_from_report, run_batch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-loudness",
        description="批量音频响度处理: 测量 -> 增益 -> 真峰值限幅, 一次成片",
    )
    p.add_argument("input_dir", help="输入目录 (递归扫描 wav/mp3/flac/aac/mp4/mov/mkv 等)")
    p.add_argument("output_dir", help="输出目录 (保持相对目录结构, 输出 24-bit WAV)")
    p.add_argument("--target", type=float, default=-23.0, help="目标综合响度 LUFS (默认 -23)")
    p.add_argument("--true-peak", type=float, default=-1.0, help="真峰值天花板 dBTP (默认 -1)")
    p.add_argument("--concurrency", type=int, default=8, help="并发上限 (默认 8)")
    p.add_argument("--retries", type=int, default=2, help="失败重试次数, 指数退避 (默认 2)")
    p.add_argument("--include-lfe", action="store_true", help="5.1 下混时混入 LFE (默认丢弃)")
    p.add_argument("--report", default=None, help="报告路径前缀 (默认 <output_dir>/report)")
    p.add_argument(
        "--retry-failed", metavar="REPORT_JSON", default=None,
        help="只重跑指定报告中失败的条目",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    only_files = None
    prev_report = None
    if args.retry_failed:
        only_files = failed_files_from_report(args.retry_failed)
        if not only_files:
            print("报告中没有失败条目, 无需重跑")
            return 0
        print(f"重跑 {len(only_files)} 条失败任务")
        prev_report = Path(args.retry_failed)

    if args.report:
        report = args.report
    elif prev_report is not None:
        report = str(prev_report.with_suffix(""))  # 默认回写原报告
    else:
        report = str(Path(args.output_dir) / "report")
    results = run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target=args.target,
        true_peak=args.true_peak,
        concurrency=args.concurrency,
        retries=args.retries,
        include_lfe=args.include_lfe,
        report_path=None if prev_report else report,
        only_files=only_files,
    )

    if prev_report is not None:
        # 把重跑结果合并回上次报告 (按 input_file 替换), 再整体写出
        import json

        from .batch import write_report

        old = json.loads(prev_report.read_text(encoding="utf-8"))["results"]
        merged = {e["input_file"]: e for e in old}
        for e in results:
            merged[e["input_file"]] = e
        results = [merged[k] for k in sorted(merged)]
        write_report(results, report)

    n_ok = sum(1 for e in results if e["status"] == "success")
    n_fail = sum(1 for e in results if e["status"] == "failed")
    n_skip = sum(1 for e in results if e["status"] == "skipped")
    print(f"\n完成: 成功 {n_ok}, 失败 {n_fail}, 跳过 {n_skip}, 共 {len(results)} 条")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
