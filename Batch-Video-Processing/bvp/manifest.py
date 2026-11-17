"""结构化转码清单：成功/失败/跳过三类条目的落盘与回读。

清单是「数字与实际产出可复核」的载体，设计约束：

- **三份清单**：一份全量清单（每个被规划/扫描到的输入恰好一条），外加
  失败清单与跳过清单（全量清单的过滤视图），失败与跳过分别落盘；
- **数字来自条目计数**：``summary`` 中的 success/failed/skipped/total
  永远由条目实时统计，不允许手工维护另一份计数器，从根上杜绝口径漂移；
- **原子写**：先写同目录临时文件再 ``os.replace``，长任务中断也不会留下
  半截 JSON；
- **顺序确定**：条目按任务 ``index`` 排序后写出，多次运行 diff 友好。

批量层只通过本模块写清单，条目的 ``category`` 字段直接取自
:mod:`bvp.contract` 的稳定分类，不写任何解析自异常文本的内容。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from .contract import ErrorCategory, SkipReason

__all__ = [
    "STATUS_SUCCESS",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "TERMINAL_STATUSES",
    "ManifestEntry",
    "Manifest",
    "MANIFEST_VERSION",
    "default_manifest_paths",
    "load_manifest",
]

#: 清单条目终态（没有第四种状态；扫描问题也归 failed）
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED}
)

#: 清单格式版本，结构发生不兼容变化时递增
MANIFEST_VERSION = 1

#: 默认清单文件名
_FULL_NAME = "batch_manifest.json"
_FAILED_NAME = "batch_failures.json"
_SKIPPED_NAME = "batch_skipped.json"


@dataclass
class ManifestEntry:
    """清单中的单条记录，与一个输入一一对应。"""

    index: int
    input_path: str
    status: str
    output_path: str | None = None
    #: 稳定错误分类（ErrorCategory 值）；成功条目为 None
    category: str | None = None
    #: 跳过原因码（SkipReason 值）；仅跳过条目有值
    skip_reason: str | None = None
    #: 人读错误说明（str(exc)），只供排查，批量逻辑不读取
    detail: str | None = None
    #: 异常类型名，辅助排查；同样不承担契约职责
    error_type: str | None = None
    #: 失败时的 ffmpeg 完整日志路径（若存在）
    ffmpeg_log_path: str | None = None
    #: 输出文件字节数，成功后记录，供复核时与磁盘现状对照
    output_size: int | None = None
    #: 裁决该任务的规则名（规则引擎接入时写入；旧路径与旧清单为 None）
    rule_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "input_path": self.input_path,
            "status": self.status,
            "output_path": self.output_path,
            "category": self.category,
            "skip_reason": self.skip_reason,
            "detail": self.detail,
            "error_type": self.error_type,
            "ffmpeg_log_path": self.ffmpeg_log_path,
            "output_size": self.output_size,
            "rule_name": self.rule_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestEntry":
        return cls(
            index=int(data["index"]),
            input_path=data["input_path"],
            status=data["status"],
            output_path=data.get("output_path"),
            category=data.get("category"),
            skip_reason=data.get("skip_reason"),
            detail=data.get("detail"),
            error_type=data.get("error_type"),
            ffmpeg_log_path=data.get("ffmpeg_log_path"),
            output_size=data.get("output_size"),
            rule_name=data.get("rule_name"),
        )


@dataclass
class Manifest:
    """一次批量运行的全量清单。"""

    entries: list[ManifestEntry] = field(default_factory=list)
    run_info: dict[str, Any] = field(default_factory=dict)

    # ---- 计数：全部从条目实时统计，保证口径唯一 ------------------------

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def success_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_SUCCESS)

    @property
    def failed_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_FAILED)

    @property
    def skipped_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_SKIPPED)

    @property
    def success_entries(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.status == STATUS_SUCCESS]

    @property
    def failed_entries(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.status == STATUS_FAILED]

    @property
    def skipped_entries(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.status == STATUS_SKIPPED]

    def summary(self) -> dict[str, int]:
        """汇总数字；``total == 三项之和`` 是清单的不变量。"""
        return {
            "total": self.total,
            "success": self.success_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
        }

    def add(self, entry: ManifestEntry) -> None:
        if entry.status not in TERMINAL_STATUSES:
            raise ValueError(f"非法的条目终态: {entry.status!r}")
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.entries, key=lambda e: e.index)
        return {
            "manifest_version": MANIFEST_VERSION,
            "summary": self.summary(),
            "run_info": self.run_info,
            "entries": [e.to_dict() for e in ordered],
        }

    # ---- 落盘与回读 ----------------------------------------------------

    def write(self, directory: str, *, run_info: dict[str, Any] | None = None,
              ) -> dict[str, str]:
        """把全量/失败/跳过三份清单原子写入 ``directory``，返回路径映射。

        :raises ValueError: 条目 index 重复或汇总不变量被破坏
        """
        self._validate_invariants()
        if run_info is not None:
            self.run_info = dict(run_info)

        os.makedirs(directory, exist_ok=True)
        paths = default_manifest_paths(directory)

        full_dict = self.to_dict()
        _atomic_write_json(paths["full"], full_dict)
        _atomic_write_json(paths["failed"], _sub_manifest(
            self.failed_entries, full_dict["summary"], full_dict["run_info"],
            STATUS_FAILED,
        ))
        _atomic_write_json(paths["skipped"], _sub_manifest(
            self.skipped_entries, full_dict["summary"],
            full_dict["run_info"], STATUS_SKIPPED,
        ))
        return paths

    def _validate_invariants(self) -> None:
        indexes = [e.index for e in self.entries]
        if len(indexes) != len(set(indexes)):
            raise ValueError("清单条目 index 必须唯一。")
        summary = self.summary()
        if summary["total"] != (
            summary["success"] + summary["failed"] + summary["skipped"]
        ):
            raise ValueError("清单不变量被破坏：total 不等于三类计数之和。")
        for entry in self.entries:
            if entry.status == STATUS_SKIPPED and not entry.skip_reason:
                raise ValueError("跳过条目必须带 skip_reason。")
            if entry.status != STATUS_SUCCESS and not entry.category:
                raise ValueError("失败/跳过条目必须带稳定 category。")


def _sub_manifest(entries: list[ManifestEntry], parent_summary: dict,
                  run_info: dict, kind: str) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda e: e.index)
    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": kind,
        "summary": parent_summary,
        "run_info": run_info,
        "count": len(ordered),
        "entries": [e.to_dict() for e in ordered],
    }


def default_manifest_paths(directory: str) -> dict[str, str]:
    """返回某目录下三份清单的标准路径。"""
    base = os.path.abspath(directory)
    return {
        "full": os.path.join(base, _FULL_NAME),
        "failed": os.path.join(base, _FAILED_NAME),
        "skipped": os.path.join(base, _SKIPPED_NAME),
    }


def load_manifest(path: str) -> Manifest:
    """从全量清单 JSON 回读 :class:`Manifest`（供独立复核程序使用）。"""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError(f"清单文件结构无法识别: {path}")
    manifest = Manifest(
        entries=[ManifestEntry.from_dict(item) for item in data["entries"]],
        run_info=data.get("run_info", {}),
    )
    manifest._validate_invariants()
    return manifest


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """同目录临时文件 + os.replace 原子替换，避免半截 JSON。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # 写失败时清理临时文件，不污染输出目录
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def make_failure_entry(
    index: int,
    input_path: str,
    output_path: str | None,
    category: ErrorCategory,
    detail: str,
    error_type: str,
    *,
    ffmpeg_log_path: str | None = None,
    rule_name: str | None = None,
) -> ManifestEntry:
    """按稳定分类构造失败条目（批量层的统一入口）。"""
    return ManifestEntry(
        index=index,
        input_path=input_path,
        status=STATUS_FAILED,
        output_path=output_path,
        category=category.value,
        detail=detail,
        error_type=error_type,
        ffmpeg_log_path=ffmpeg_log_path,
        rule_name=rule_name,
    )


def make_skip_entry(
    index: int,
    input_path: str,
    output_path: str | None,
    category: ErrorCategory,
    reason: SkipReason,
    detail: str,
    error_type: str,
    *,
    rule_name: str | None = None,
) -> ManifestEntry:
    """按稳定分类构造跳过条目（批量层的统一入口）。"""
    return ManifestEntry(
        index=index,
        input_path=input_path,
        status=STATUS_SKIPPED,
        output_path=output_path,
        category=category.value,
        skip_reason=reason.value,
        detail=detail,
        error_type=error_type,
        rule_name=rule_name,
    )


def make_success_entry(
    index: int,
    input_path: str,
    output_path: str,
    output_size: int,
    *,
    rule_name: str | None = None,
) -> ManifestEntry:
    """构造成功条目，并记录产物字节数作为复核锚点。"""
    return ManifestEntry(
        index=index,
        input_path=input_path,
        status=STATUS_SUCCESS,
        output_path=output_path,
        output_size=output_size,
        rule_name=rule_name,
    )
