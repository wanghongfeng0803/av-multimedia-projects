"""批量处理: 并发上限、指数退避重试、进度显示、报告输出."""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .io import SUPPORTED_EXTS
from .processor import process_one

REPORT_FIELDS = [
    "input_file", "output_file", "status",
    "in_duration_s", "in_sample_rate", "in_channels",
    "in_lufs", "in_lra", "in_true_peak_db",
    "out_lufs", "out_lra", "out_true_peak_db",
    "applied_gain_db", "limiter_max_reduction_db",
    "attempts", "warning", "error",
]


def scan_inputs(input_dir: Path) -> list[Path]:
    files = [
        p for p in sorted(Path(input_dir).rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return files


def _process_with_retry(
    input_path: str, output_path: str, target: float, tp: float,
    include_lfe: bool, max_retries: int,
) -> dict:
    """指数退避重试包装 (在子进程中运行)。"""
    entry: dict = {}
    for attempt in range(1, max_retries + 2):
        entry = process_one(input_path, output_path, target, tp, include_lfe)
        if entry["status"] != "failed":
            break
        if attempt <= max_retries:
            time.sleep(min(2 ** attempt, 30))  # 2s, 4s, 8s ... 封顶 30s
    entry["attempts"] = attempt
    return entry


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    target: float = -23.0,
    true_peak: float = -1.0,
    concurrency: int = 8,
    retries: int = 2,
    include_lfe: bool = False,
    report_path: str | Path | None = None,
    only_files: list[str] | None = None,
    progress: bool = True,
) -> list[dict]:
    """批量处理一个目录, 返回报告条目列表。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if only_files is not None:
        files = [Path(f) for f in only_files]
    else:
        files = scan_inputs(input_dir)
    if not files:
        raise SystemExit(f"未找到可处理文件: {input_dir}")

    def out_path_for(p: Path) -> Path:
        try:
            rel = p.relative_to(input_dir)
        except ValueError:
            rel = Path(p.name)
        return (output_dir / rel).with_suffix(".wav")

    jobs = {p: out_path_for(p) for p in files}
    total = len(jobs)
    results: list[dict] = []
    done = ok = fail = skip = 0

    with ProcessPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _process_with_retry, str(src), str(dst),
                target, true_peak, include_lfe, retries,
            ): src
            for src, dst in jobs.items()
        }
        for fut in as_completed(futures):
            entry = fut.result()
            results.append(entry)
            done += 1
            if entry["status"] == "success":
                ok += 1
            elif entry["status"] == "skipped":
                skip += 1
            else:
                fail += 1
            if progress:
                name = Path(entry["input_file"]).name
                out = entry.get("output") or {}
                detail = ""
                if entry["status"] == "success":
                    detail = (
                        f" {entry['input']['lufs']} -> {out.get('lufs')} LUFS,"
                        f" TP {out.get('true_peak_db')} dBTP"
                    )
                elif entry["status"] == "failed":
                    detail = f" {entry.get('error', '')}"
                print(
                    f"[{done}/{total}] {entry['status'].upper():7s} {name}{detail}"
                    f"  (ok={ok} fail={fail} skip={skip})",
                    flush=True,
                )

    results.sort(key=lambda e: e["input_file"])
    if report_path:
        write_report(results, report_path)
    return results


def write_report(results: list[dict], report_path: str | Path) -> None:
    """同时写 JSON 与 CSV 报告。"""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = report_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        w.writeheader()
        for e in results:
            w.writerow(_flatten(e))
    print(f"报告已写出: {json_path} / {csv_path}")


def _flatten(e: dict) -> dict:
    i = e.get("input") or {}
    o = e.get("output") or {}
    return {
        "input_file": e.get("input_file"),
        "output_file": e.get("output_file"),
        "status": e.get("status"),
        "in_duration_s": i.get("duration_s"),
        "in_sample_rate": i.get("sample_rate"),
        "in_channels": i.get("channels"),
        "in_lufs": i.get("lufs"),
        "in_lra": i.get("lra"),
        "in_true_peak_db": i.get("true_peak_db"),
        "out_lufs": o.get("lufs"),
        "out_lra": o.get("lra"),
        "out_true_peak_db": o.get("true_peak_db"),
        "applied_gain_db": o.get("applied_gain_db"),
        "limiter_max_reduction_db": o.get("limiter_max_reduction_db"),
        "attempts": e.get("attempts"),
        "warning": e.get("warning", ""),
        "error": e.get("error", ""),
    }


def failed_files_from_report(report_path: str | Path) -> list[str]:
    """从上次报告中取出失败条目, 用于单独重跑。"""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return [e["input_file"] for e in data["results"] if e.get("status") == "failed"]
