"""确定性的结果序列化：JSON 报告与 CSV 差异序列。

可复现性在输出层的三条保证：

1. **不含任何运行时刻信息**（无当前时间、无临时路径、无进程 id），
   报告内容完全由输入像素与固定参数决定；
2. **键顺序、列顺序、缩进与分隔符全部固定**，容器（dict/list）按
   既定顺序构造，不依赖集合迭代；
3. **浮点表示确定**：JSON 采用 Python 最短往返 repr（同一 IEEE 754 值
   必然产生同一字符串），CSV 统一固定 6 位小数、四舍六入五成双，
   不使用环境区域设置（始终点号小数点）。
"""

from __future__ import annotations

import csv
import io
import json

from .models import SegmentationResult

__all__ = [
    "FLOAT_DIGITS",
    "format_float",
    "result_to_dict",
    "write_json_report",
    "render_json_report",
    "write_boundaries_csv",
    "render_boundaries_csv",
    "render_summary",
]

#: CSV 中浮点字段的固定小数位数
FLOAT_DIGITS = 6

#: CSV 固定列顺序（同时锁定表头）
CSV_COLUMNS = [
    "frame_index",
    "time_seconds",
    "raw_distance",
    "score",
    "return_distance",
    "decision",
]


def format_float(value: float, digits: int = FLOAT_DIGITS) -> str:
    """固定小数位格式化；``f`` 格式遵循四舍六入五成双，结果与区域无关。"""
    return f"{value:.{digits}f}"


def result_to_dict(result: SegmentationResult) -> dict:
    """把结果转成键顺序固定的纯数据 dict（可直接 json 序列化）。"""
    return {
        "input_path": result.input_path,
        "frame_count": result.frame_count,
        "width": result.width,
        "height": result.height,
        "fps": result.fps,
        "fps_source": result.fps_source,
        "pixel_count": result.pixel_count,
        "threshold": result.threshold,
        "threshold_source": result.threshold_source,
        "flash_boundaries": list(result.flash_boundaries),
        "suppressed_boundaries": list(result.suppressed_boundaries),
        "cut_count": len(result.cuts),
        "cuts": [
            {
                "frame_index": cut.frame_index,
                "time_seconds": cut.time_seconds,
                "score": cut.score,
                "raw_distance": cut.raw_distance,
            }
            for cut in result.cuts
        ],
        "boundaries": [
            {
                "frame_index": b.frame_index,
                "time_seconds": b.time_seconds,
                "raw_distance": b.raw_distance,
                "score": b.score,
                "return_distance": b.return_distance,
                "decision": b.decision,
            }
            for b in result.boundaries
        ],
    }


def render_json_report(result: SegmentationResult) -> str:
    """渲染 JSON 报告字符串：固定缩进、不转义非 ASCII、键序由 dict 保证。"""
    return json.dumps(
        result_to_dict(result),
        ensure_ascii=False,
        indent=2,
        separators=(",", ": "),
    )


def write_json_report(result: SegmentationResult, path: str) -> None:
    """以 UTF-8（无 BOM）写入 JSON 报告，末尾固定一个换行。"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_json_report(result))
        fh.write("\n")


def render_boundaries_csv(result: SegmentationResult) -> str:
    """渲染完整差异序列 CSV（字符串形式，\\r\\n 行尾，符合 csv 模块约定）。"""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for b in result.boundaries:
        writer.writerow({
            "frame_index": b.frame_index,
            "time_seconds": format_float(b.time_seconds),
            "raw_distance": b.raw_distance,
            "score": format_float(b.score),
            "return_distance": b.return_distance,
            "decision": b.decision,
        })
    return buffer.getvalue()


def write_boundaries_csv(result: SegmentationResult, path: str) -> None:
    """以 UTF-8（无 BOM）写入差异序列 CSV。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(render_boundaries_csv(result))


def render_summary(result: SegmentationResult) -> str:
    """渲染面向终端的人类可读摘要（同样不含任何运行时刻信息）。"""
    lines = [
        f"输入: {result.input_path}",
        f"帧数: {result.frame_count}  尺寸: {result.width}x{result.height}  "
        f"帧率: {format_float(result.fps)} ({result.fps_source})",
        f"判定阈值: {format_float(result.threshold)} "
        f"[{result.threshold_source}]",
        f"硬切点数: {len(result.cuts)}",
    ]
    if result.flash_boundaries:
        lines.append(
            "闪光抑制边界: " + ", ".join(map(str, result.flash_boundaries))
        )
    if result.suppressed_boundaries:
        lines.append(
            "最小场景长度抑制边界: "
            + ", ".join(map(str, result.suppressed_boundaries))
        )
    if result.cuts:
        lines.append("")
        lines.append("切点帧号        时间(秒)    归一化差异    整数距离")
        for cut in result.cuts:
            lines.append(
                f"{cut.frame_index:>8}  {format_float(cut.time_seconds):>12}  "
                f"{format_float(cut.score):>10}  {cut.raw_distance:>10}"
            )
    return "\n".join(lines)
