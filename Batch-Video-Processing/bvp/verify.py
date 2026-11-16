"""清单与磁盘产出的对账复核。

复核回答一个问题：**清单里写的成功/失败/跳过数字，和磁盘上真实存在的
产物文件是否一一对得上？** 复核不调用 ffmpeg，只做文件系统事实检查，
因此可以在任意时刻、任意机器上对同一份清单重复执行。

对账规则（每条规则产生一条 :class:`ReconciliationMismatch`）：

- ``success`` 条目：输出必须存在、是普通文件、非空，且字节数与清单记录的
  ``output_size`` 一致；
- ``failed`` 条目：输出必须**不存在**——核心失败时会清理半成品，若文件
  仍在，说明失败条目与磁盘事实矛盾；
- ``skipped`` 条目：
  - 原因 ``output_exists``：输出本就存在（非本次产物），属预期；
  - 其余原因（非视频/不可访问）：输出必须不存在；
- 全局：成功条目的输出路径不得重复；磁盘上实际存在的规划输出集合，必须
  恰好等于「本次成功产物 + 预存跳过产物」。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .contract import SkipReason
from .manifest import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    Manifest,
    ManifestEntry,
    default_manifest_paths,
    load_manifest,
)

__all__ = [
    "ReconciliationMismatch",
    "ReconciliationResult",
    "reconcile_manifest",
    "reconcile_manifest_files",
]


@dataclass(frozen=True)
class ReconciliationMismatch:
    """单条对账不符。"""

    index: int | None  # None 表示全局级不符
    input_path: str | None
    kind: str          # 稳定的不符类型码
    message: str       # 人读说明


@dataclass
class ReconciliationResult:
    """对账结果。"""

    ok: bool
    summary: dict[str, int]
    mismatches: list[ReconciliationMismatch] = field(default_factory=list)
    #: 磁盘上实际存在的规划输出数（成功产物 + 预存跳过产物）
    existing_outputs: int = 0
    #: 本次运行真正产出的文件数（应恰好等于 success 计数）
    produced_outputs: int = 0

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)


def reconcile_manifest(manifest: Manifest) -> ReconciliationResult:
    """把内存中的 :class:`Manifest` 与磁盘事实逐条对账。"""
    summary = manifest.summary()
    mismatches: list[ReconciliationMismatch] = []
    existing = 0
    produced = 0
    success_outputs: set[str] = set()

    for entry in sorted(manifest.entries, key=lambda e: e.index):
        output = entry.output_path
        output_exists = bool(output and os.path.isfile(output))

        if entry.status == STATUS_SUCCESS:
            produced += 1
            if not output:
                mismatches.append(ReconciliationMismatch(
                    entry.index, entry.input_path,
                    "success_without_output",
                    "成功条目缺少输出路径。",
                ))
                continue
            if output in success_outputs:
                mismatches.append(ReconciliationMismatch(
                    entry.index, entry.input_path,
                    "duplicate_output",
                    f"多个成功条目指向同一输出: {output}",
                ))
            success_outputs.add(output)
            _check_success_on_disk(entry, output, output_exists, mismatches)
            if output_exists:
                existing += 1

        elif entry.status == STATUS_FAILED:
            if output_exists:
                mismatches.append(ReconciliationMismatch(
                    entry.index, entry.input_path,
                    "failed_output_present",
                    f"失败条目却在磁盘上存在输出文件: {output}",
                ))
                existing += 1

        elif entry.status == STATUS_SKIPPED:
            if entry.skip_reason == SkipReason.OUTPUT_EXISTS.value:
                # 预存产物：存在才符合预期，但不计入本次 produced
                if not output_exists:
                    mismatches.append(ReconciliationMismatch(
                        entry.index, entry.input_path,
                        "skip_output_missing",
                        "因输出已存在而跳过，但该输出现在不存在。",
                    ))
                else:
                    existing += 1
            elif output_exists:
                mismatches.append(ReconciliationMismatch(
                    entry.index, entry.input_path,
                    "skipped_output_present",
                    f"跳过条目却在磁盘上存在输出文件: {output}",
                ))
                existing += 1
        else:  # pragma: no cover - 清单不变量已拦截非法状态
            mismatches.append(ReconciliationMismatch(
                entry.index, entry.input_path,
                "illegal_status",
                f"未知条目状态: {entry.status!r}",
            ))

    # 全局口径：success 计数必须与成功条目数、磁盘产物数完全相等
    if produced != summary["success"]:
        mismatches.append(ReconciliationMismatch(
            None, None, "count_mismatch",
            f"成功计数 {summary['success']} 与实际成功条目数 {produced} 不一致。",
        ))

    return ReconciliationResult(
        ok=not mismatches,
        summary=dict(summary),
        mismatches=mismatches,
        existing_outputs=existing,
        produced_outputs=produced,
    )


def _check_success_on_disk(
    entry: ManifestEntry,
    output: str,
    output_exists: bool,
    mismatches: list[ReconciliationMismatch],
) -> None:
    if not output_exists:
        mismatches.append(ReconciliationMismatch(
            entry.index, entry.input_path,
            "success_output_missing",
            f"清单标记成功但磁盘上找不到输出: {output}",
        ))
        return
    try:
        size = os.path.getsize(output)
    except OSError as exc:
        mismatches.append(ReconciliationMismatch(
            entry.index, entry.input_path,
            "output_inaccessible",
            f"输出文件无法读取大小: {exc}",
        ))
        return
    if size <= 0:
        mismatches.append(ReconciliationMismatch(
            entry.index, entry.input_path,
            "output_empty",
            f"成功产物为 0 字节: {output}",
        ))
    if entry.output_size is not None and size != entry.output_size:
        mismatches.append(ReconciliationMismatch(
            entry.index, entry.input_path,
            "size_mismatch",
            f"清单记录 {entry.output_size} 字节，磁盘实际 {size} 字节: {output}",
        ))


def reconcile_manifest_files(directory: str) -> ReconciliationResult:
    """交叉校验目录下三份清单文件的数字，再与磁盘对账。

    供独立复核程序/CI 使用：只依赖清单文件本身，不需要批量运行上下文。
    """
    import json

    paths = default_manifest_paths(directory)
    mismatches: list[ReconciliationMismatch] = []

    try:
        manifest = load_manifest(paths["full"])
    except (OSError, ValueError, KeyError) as exc:
        return ReconciliationResult(
            ok=False,
            summary={"total": 0, "success": 0, "failed": 0, "skipped": 0},
            mismatches=[ReconciliationMismatch(
                None, None, "manifest_unreadable",
                f"全量清单无法读取: {exc}",
            )],
        )

    for kind, key in (("failed", "failed"), ("skipped", "skipped")):
        try:
            with open(paths[key], "r", encoding="utf-8") as handle:
                sub = json.load(handle)
        except (OSError, ValueError) as exc:
            mismatches.append(ReconciliationMismatch(
                None, None, "submanifest_unreadable",
                f"{kind} 清单无法读取: {exc}",
            ))
            continue
        expected = manifest.summary()[key]
        if sub.get("count") != expected or len(sub.get("entries", [])) != expected:
            mismatches.append(ReconciliationMismatch(
                None, None, "submanifest_count_mismatch",
                f"{kind} 清单计数与全量清单不一致（期望 {expected}）。",
            ))
        # 子清单的汇总块必须与全量清单共享同一口径
        if sub.get("summary") != manifest.summary():
            mismatches.append(ReconciliationMismatch(
                None, None, "submanifest_summary_mismatch",
                f"{kind} 清单的 summary 与全量清单不一致。",
            ))

    disk_result = reconcile_manifest(manifest)
    mismatches.extend(disk_result.mismatches)
    return ReconciliationResult(
        ok=not mismatches,
        summary=disk_result.summary,
        mismatches=mismatches,
        existing_outputs=disk_result.existing_outputs,
        produced_outputs=disk_result.produced_outputs,
    )
