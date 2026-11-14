#!/usr/bin/env python3
"""批量 MP4 → MKV 重封装编排（单文件能力之上的批量层）。

本模块**不重新实现任何容器逻辑**，只在 :mod:`remux_all_tracks` 的单文件
重试管线之上做三件事：

1. **发现（discover）**：输入可以是若干文件与目录混排；目录按 ``rglob``
   递归展开，只收集常见 MP4 容器后缀的普通文件，并按解析后的绝对路径
   去重（同一路径写多次、符号链接指向同一目标都只处理一次）。
2. **规划（plan）**：把每个输入映射成输出目录下的 ``同名.mkv``，确定性地
   解决两类碰撞——不同目录的同名文件（``movie.mp4`` 与 ``sub/movie.mp4``）
   以及输出文件已存在的情况。
3. **逐项隔离执行（execute）**：每个输入独立调用单文件管线，单个文件
   失败（确定性错误 / 重试用尽 / 甚至未预期异常）只记录在结果里，**绝不
   影响其他文件**；``stop_on_error`` 可切换为遇错即停。

碰撞策略（``on_existing``）：
- ``"rename"``（默认，完全非破坏）：输出名已被占用时，依次尝试
  ``stem_2.mkv``、``stem_3.mkv``…… 直到找到空闲名字；批内多个同名输入
  也按同一规则拿到互不相同的输出。
- ``"overwrite"``：直接覆盖已存在的输出（单文件管线自身仍走
  临时文件 + 原子替换，覆盖过程中不会出现半成品）。
- ``"skip"``：输出已存在则跳过该输入，结果中标记为 skipped。

命令行用法::

    python batch_remux.py OUT_DIR INPUT [INPUT ...] [--recursive/--no-recursive]
        [--on-existing {rename,overwrite,skip}] [--dry-run] [--stop-on-error]
        [--retries N] [--retry-delay SECONDS]

Python 用法::

    from batch_remux import remux_batch
    report = remux_batch(["a.mp4", "dir_in"], "out_dir")
    if report.failed:
        ...  # report.results 里逐项带中文失败原因
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 允许“脚本直接运行”与“作为模块导入”两种方式（与 remux_all_tracks 一致）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mp4_to_mkv.errors import RemuxError  # noqa: E402
from remux_all_tracks import (  # noqa: E402
    DEFAULT_RETRIES,
    DEFAULT_RETRY_DELAY,
    RemuxRetryError,
    remux_all_tracks,
)

# 仅用于目录扫描时的候选过滤；显式给出的文件不做后缀限制，
# 合法性最终仍由底层解析器判定（与 remuxer 的“后缀只是提示”一致）。
MP4_SUFFIXES = (".mp4", ".m4v", ".mov", ".m4a")
MKV_SUFFIX = ".mkv"

ON_RENAME = "rename"
ON_OVERWRITE = "overwrite"
ON_SKIP = "skip"
_VALID_ON_EXISTING = (ON_RENAME, ON_OVERWRITE, ON_SKIP)

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_BATCH_FAILED = 5  # 批量中至少一项失败（确定性或重试用尽）

__all__ = [
    "BatchItemResult",
    "BatchResult",
    "discover_inputs",
    "plan_outputs",
    "remux_batch",
    "build_parser",
    "main",
    "MP4_SUFFIXES",
    "ON_RENAME",
    "ON_OVERWRITE",
    "ON_SKIP",
    "STATUS_SUCCESS",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "EXIT_BATCH_FAILED",
]


@dataclass
class BatchItemResult:
    """单个输入的处理结果。

    :ivar source: 输入路径（保留发现时的原始写法）。
    :ivar output: 规划出的输出路径；被跳过且无需输出时可能为 None。
    :ivar status: ``success`` / ``failed`` / ``skipped``。
    :ivar reason: skipped 的原因，或 failed 的中文错误消息；成功时为 None。
    :ivar attempts: 实际重封装尝试次数（复用单文件重试统计），未执行时为 0。
    """

    source: Path
    output: Path | None
    status: str
    reason: str | None = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS


@dataclass
class BatchResult:
    """整批处理的汇总，逐项结果见 :attr:`results`（顺序与输入规划一致）。"""

    results: list[BatchItemResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> list[BatchItemResult]:
        return [r for r in self.results if r.status == STATUS_SUCCESS]

    @property
    def failed(self) -> list[BatchItemResult]:
        return [r for r in self.results if r.status == STATUS_FAILED]

    @property
    def skipped(self) -> list[BatchItemResult]:
        return [r for r in self.results if r.status == STATUS_SKIPPED]


def _is_mp4_candidate(path: Path) -> bool:
    return path.suffix.lower() in MP4_SUFFIXES


def discover_inputs(inputs, *, recursive: bool = True) -> list[Path]:
    """把文件/目录混排的输入展开为去重后的 MP4 文件清单。

    - 显式给出的文件**原样保留**（不做后缀过滤，交给底层判定）；
    - 目录按 ``rglob``（``recursive=True``，默认）或 ``iterdir``
      （``recursive=False``）展开，只收常见 MP4 后缀的普通文件，
      子目录在非递归模式下整体跳过；
    - 按 ``Path.resolve()`` 去重，保留首次出现的原始写法；结果按
      解析后路径排序，保证同一批输入的处理顺序确定、可复现。

    :raises RemuxError: 任一输入路径不存在，或目录在扫描时不可读。
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            raise RemuxError(f"输入路径不存在: {raw}")
        if p.is_dir():
            try:
                children = list(p.rglob("*")) if recursive else list(p.iterdir())
            except OSError as exc:
                raise RemuxError(f"目录无法读取: {raw}（{exc}）") from exc
            candidates = [c for c in children if c.is_file() and _is_mp4_candidate(c)]
        else:
            # 显式文件（含非常规后缀）不过滤；is_file 已排除目录/设备
            candidates = [p]

        for c in candidates:
            try:
                key = c.resolve()
            except OSError as exc:
                raise RemuxError(f"路径无法解析: {c}（{exc}）") from exc
            if key not in seen:
                seen.add(key)
                found.append(c)

    # 以解析后的绝对路径为键排序，避免不同平台/写法导致的顺序漂移
    found.sort(key=lambda c: str(c.resolve()).casefold())
    return found


def _allocate(
    source: Path,
    output_dir: Path,
    reserved: set[str],
    on_existing: str,
) -> tuple[Path | None, str | None]:
    """为单个输入分配输出路径。

    :param reserved: 本批已经占用的输出路径键集合（``str(path).casefold()``，
        大小写不敏感，兼容 Windows/macOS 并让 Linux 上的规划结果同样确定）。
    :return: (输出路径, 跳过原因)；跳过场景输出路径为 None。
    """
    base = output_dir / (source.stem + MKV_SUFFIX)

    def taken(candidate: Path) -> bool:
        return str(candidate).casefold() in reserved or candidate.exists()

    if on_existing == ON_SKIP and taken(base):
        return None, f"输出已存在，按 skip 策略跳过: {base}"

    candidate = base
    if on_existing == ON_RENAME:
        # 批内同名或磁盘已占用：stem_2、stem_3…… 确定性探测
        counter = 2
        while taken(candidate):
            candidate = output_dir / f"{source.stem}_{counter}{MKV_SUFFIX}"
            counter += 1

    reserved.add(str(candidate).casefold())
    return candidate, None


def plan_outputs(
    sources, output_dir, *, on_existing: str = ON_RENAME
) -> list[tuple[Path, Path | None, str | None]]:
    """把输入清单规划成 (源, 输出或 None, 跳过原因或 None) 的执行计划。

    纯规划、不触碰文件内容（只做 ``exists`` 探测），因此 ``dry-run``
    与真实执行共用同一份规划，保证“预演即所得”。
    """
    if on_existing not in _VALID_ON_EXISTING:
        raise RemuxError(
            f"on_existing 取值必须是 {_VALID_ON_EXISTING} 之一，得到: {on_existing!r}"
        )
    out = Path(output_dir)
    reserved: set[str] = set()
    plan: list[tuple[Path, Path | None, str | None]] = []
    for source in sources:
        dst, reason = _allocate(Path(source), out, reserved, on_existing)
        plan.append((Path(source), dst, reason))
    return plan


def remux_batch(
    inputs,
    output_dir,
    *,
    recursive: bool = True,
    on_existing: str = ON_RENAME,
    dry_run: bool = False,
    stop_on_error: bool = False,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    remux_fn=remux_all_tracks,
) -> BatchResult:
    """批量重封装一批文件/目录，返回逐项 :class:`BatchResult`。

    单文件的确定性错误（:class:`RemuxError`）、重试用尽
    （:class:`RemuxRetryError`）乃至未预期异常都被**限制在该项内**：
    记录中文原因后继续处理后续文件，除非 ``stop_on_error=True``。

    :param inputs: 文件/目录路径列表（目录按 recursive 展开）。
    :param output_dir: 统一输出目录，不存在会自动创建（dry-run 除外）。
    :param on_existing: 输出碰撞策略，见模块头说明。
    :param dry_run: 只做发现与规划，不创建目录、不执行转换，
        各项标记为 skipped（原因注明 dry-run）。
    :param stop_on_error: 首项失败后立即停止排队中的后续项，
        未执行的项标记为 skipped。
    :param remux_fn: 可注入的单文件处理函数（测试桩用），签名与
        :func:`remux_all_tracks` 一致。
    :raises RemuxError: 参数非法或输入发现阶段失败（整批前置错误）。
    """
    if on_existing not in _VALID_ON_EXISTING:
        raise RemuxError(
            f"on_existing 取值必须是 {_VALID_ON_EXISTING} 之一，得到: {on_existing!r}"
        )
    if retries < 0:
        raise RemuxError("retries 不能为负数")

    sources = discover_inputs(inputs, recursive=recursive)
    plan = plan_outputs(sources, output_dir, on_existing=on_existing)

    report = BatchResult()
    if not plan:
        # 空批不是错误：CLI 退出码仍为 0，仅汇总提示“没有可处理的文件”
        return report

    if not dry_run:
        # 输出目录在执行前统一创建；创建失败属于整批前置错误，直接上抛
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    aborted = False
    for source, dst, skip_reason in plan:
        if aborted:
            report.results.append(
                BatchItemResult(source, None, STATUS_SKIPPED, "前序项失败，已停止")
            )
            continue
        if skip_reason is not None:
            report.results.append(
                BatchItemResult(source, dst, STATUS_SKIPPED, skip_reason)
            )
            continue
        if dry_run:
            report.results.append(
                BatchItemResult(
                    source, dst, STATUS_SKIPPED, "dry-run：仅规划，未执行转换"
                )
            )
            continue

        try:
            item = remux_fn(
                str(source),
                str(dst),
                # rename 策略分配出的名字必然空闲；overwrite 才需要下放覆盖权
                overwrite=(on_existing == ON_OVERWRITE),
                retries=retries,
                retry_delay=retry_delay,
            )
        except RemuxError as exc:
            # 确定性输入错误（含“输出与输入同路径”等单文件前置校验）
            report.results.append(BatchItemResult(source, dst, STATUS_FAILED, str(exc)))
            aborted = stop_on_error
            continue
        except RemuxRetryError as exc:
            # 瞬时性 I/O 错误重试用尽
            report.results.append(
                BatchItemResult(source, dst, STATUS_FAILED, f"重试用尽：{exc}")
            )
            aborted = stop_on_error
            continue
        except Exception as exc:  # noqa: BLE001 - 批量隔离：未知错误也不得炸掉整批
            report.results.append(
                BatchItemResult(
                    source,
                    dst,
                    STATUS_FAILED,
                    f"未预期错误 {type(exc).__name__}: {exc}",
                )
            )
            aborted = stop_on_error
            continue

        report.results.append(
            BatchItemResult(source, dst, STATUS_SUCCESS, attempts=item.attempts)
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_remux",
        description="批量把 MP4（视频/音频/字幕全轨道）无损重封装为 MKV，"
        "单文件失败不影响其他文件",
    )
    parser.add_argument("output_dir", help="统一输出目录（不存在会自动创建）")
    parser.add_argument("inputs", nargs="+", help="一个或多个输入文件/目录（可混排）")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="目录是否递归扫描子目录（默认递归；--no-recursive 只看一层）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--on-existing",
        choices=_VALID_ON_EXISTING,
        default=ON_RENAME,
        help="输出名碰撞策略：rename 自动改名（默认）/ overwrite 覆盖 / skip 跳过",
    )
    # 两个布尔旗标作为 --on-existing 的快捷写法，互斥且互斥于它本身
    group.add_argument(
        "--overwrite",
        action="store_const",
        const=ON_OVERWRITE,
        dest="on_existing_flag",
        help="等价于 --on-existing overwrite",
    )
    group.add_argument(
        "--skip-existing",
        action="store_const",
        const=ON_SKIP,
        dest="on_existing_flag",
        help="等价于 --on-existing skip",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印发现与输出规划，不创建目录、不执行转换",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="首个文件失败后立即停止（默认继续处理其余文件）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"单文件瞬时错误的额外重试次数（默认 {DEFAULT_RETRIES}）",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=f"单文件线性退避基准秒数（默认 {DEFAULT_RETRY_DELAY}）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    on_existing = args.on_existing_flag or args.on_existing

    try:
        report = remux_batch(
            args.inputs,
            args.output_dir,
            recursive=args.recursive,
            on_existing=on_existing,
            dry_run=args.dry_run,
            stop_on_error=args.stop_on_error,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
    except RemuxError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except MemoryError:
        print("错误：内存不足，输入规模超出当前处理能力", file=sys.stderr)
        return EXIT_BATCH_FAILED

    if report.total == 0:
        print("未发现任何可处理的 MP4 文件，无需执行。")
        return EXIT_OK

    for r in report.results:
        if r.status == STATUS_SUCCESS:
            print(f"[成功] {r.source} -> {r.output}（尝试 {r.attempts} 次）")
        elif r.status == STATUS_SKIPPED:
            print(f"[跳过] {r.source}：{r.reason}")
        else:
            print(f"[失败] {r.source}：{r.reason}", file=sys.stderr)

    print(
        f"汇总：共 {report.total} 项，成功 {len(report.succeeded)}，"
        f"失败 {len(report.failed)}，跳过 {len(report.skipped)}"
        + ("（dry-run，未实际转换）" if args.dry_run else "")
    )
    return EXIT_BATCH_FAILED if report.failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
