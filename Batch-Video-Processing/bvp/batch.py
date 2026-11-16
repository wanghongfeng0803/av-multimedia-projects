"""批量转码编排层：扫描 → 规划 → 有界并发调度 → 清单落盘 → 自动复核。

本模块是纯编排，工程边界划得很硬：

- **只复用单文件核心** :func:`bvp.core.transcode`，本模块不拼任何 ffmpeg
  命令、不做媒体探测、不复核编码格式——这些都在核心里，批量层一行都不复制；
- **只按稳定契约分类汇总**：失败/跳过的划分只调用
  :func:`bvp.contract.classify_error` 与 :func:`category_is_skip`，
  不解析异常消息、不匹配类名字符串；
- **单文件失败永不中断整体**：任何异常（含核心契约外的陌生异常）都被
  转成一条清单条目，工作线程继续处理后续任务；
- **数字可复核**：清单计数全部来自条目，落盘后立即执行一次
  :mod:`bvp.verify` 对账，返回结果自带 ``reconciliation``。

并发使用标准库 :class:`concurrent.futures.ThreadPoolExecutor`，默认单线程
（视频编码是 CPU/内存密集型任务），任务按规划顺序提交、条目按 index 排序，
因此无论并发度多少，清单顺序都确定。
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from . import core as core_module
from .contract import (
    ErrorCategory,
    SkipReason,
    category_is_skip,
    classify_error,
)
from .errors import InputNotFoundError
from .manifest import (
    Manifest,
    ManifestEntry,
    make_failure_entry,
    make_skip_entry,
    make_success_entry,
)
from .planning import (
    EXIST_SKIP,
    PlannedJob,
    plan_outputs,
)
from .scanner import scan_inputs
from .verify import ReconciliationResult, reconcile_manifest

__all__ = ["BatchResult", "batch_transcode", "TranscodeFn"]

#: 核心转码函数签名（便于测试期注入假实现，纯逻辑测试不依赖 ffmpeg）
TranscodeFn = Callable[..., Any]

#: 容器扩展名（与 core 支持的注册表保持一致）
_CONTAINER_EXTENSIONS: dict[str, str] = {
    "mp4": ".mp4",
    "mkv": ".mkv",
}

#: 支持的视频编码（与 core 支持的注册表保持一致）
_SUPPORTED_CODECS: frozenset[str] = frozenset({"h264", "h265"})


@dataclass
class BatchResult:
    """一次批量运行的完整结果（清单 + 对账结论 + 清单文件路径）。"""

    manifest: Manifest
    reconciliation: ReconciliationResult
    manifest_paths: dict[str, str]
    output_dir: str | None
    interrupted: bool = False
    #: 已规划但未及执行的任务数（当前实现不停机，恒为 0，保留字段以便扩展）
    not_run: int = 0

    @property
    def success(self) -> bool:
        """整体成功：无失败条目且对账通过。"""
        return (
            not self.interrupted
            and self.manifest.failed_count == 0
            and self.reconciliation.ok
        )

    @property
    def summary(self) -> dict[str, int]:
        return self.manifest.summary()


def batch_transcode(
    inputs: list[str],
    *,
    output_dir: str | None = None,
    recursive: bool = True,
    jobs: int = 1,
    on_exist: str = EXIST_SKIP,
    codec: str = "h264",
    container: str = "mp4",
    crf: int | None = None,
    preset: str = "medium",
    audio_bitrate: str = "128k",
    resolution: str | None = None,
    extensions: frozenset[str] | None = None,
    manifest_dir: str | None = None,
    progress_callback: Callable[[Manifest], None] | None = None,
    transcode_fn: TranscodeFn | None = None,
) -> BatchResult:
    """对若干文件/目录执行批量转码，返回可复核的 :class:`BatchResult`。

    :param inputs: 文件或目录路径列表，目录按扩展名筛选并可递归嵌套
    :param output_dir: 统一输出目录，镜像输入子目录；None 时产物在输入旁
    :param recursive: 目录是否递归（默认 True）
    :param jobs: 并发工作线程数，默认 1；视频编码建议不超过物理核数
    :param on_exist: 输出已存在策略 ``skip`` / ``overwrite`` / ``suffix``
    :param codec/container/crf/preset/audio_bitrate/resolution:
        原样透传给单文件核心的转码参数
    :param extensions: 扫描扩展名白名单；None 用内置视频扩展名集合
    :param manifest_dir: 清单目录；None 时取输出目录，再退化为当前工作目录
    :param progress_callback: 每完成一个任务收到当前 :class:`Manifest` 快照
    :param transcode_fn: 测试注入点；生产代码留空即使用真实核心
    :returns: :class:`BatchResult`；任何单文件错误都不会向上抛，全部入清单
    :raises ValueError: 启动前参数不合法（在扫描与任何转码之前快速失败）
    """
    transcode = transcode_fn or core_module.transcode
    _validate_args(inputs, jobs, codec, container)
    extension = _CONTAINER_EXTENSIONS[container]

    # 输出目录嵌套在输入目录内时，递归扫描必须排除它，否则上一轮产物
    # （同为视频扩展名）会被当成新一轮输入反复转码
    exclude_dirs: frozenset[str] = frozenset()
    if output_dir is not None and recursive:
        exclude_dirs = frozenset({os.path.abspath(output_dir)})

    scan = scan_inputs(
        list(inputs),
        recursive=recursive,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
    )
    jobs_plan = plan_outputs(
        scan.files,
        list(inputs),
        output_dir=output_dir,
        extension=extension,
        codec=codec,
        on_exist=on_exist,
    )

    manifest = Manifest()
    entries_lock = threading.Lock()

    def _record(entry: ManifestEntry) -> None:
        with entries_lock:
            manifest.add(entry)
            if progress_callback is not None:
                try:
                    progress_callback(manifest)
                except Exception:
                    # 回调异常不允许影响批量调度
                    pass

    # --- 阶段 1：扫描期无法访问的路径直接记失败（不进入工作队列） ------
    # 索引空间显式划分：规划任务占用 0..len(jobs_plan)-1（见 plan_outputs），
    # 扫描问题紧随其后，避免两套编号隐式依赖长度巧合而撞号（清单要求 index 唯一）
    problem_base = len(jobs_plan)
    for offset, problem in enumerate(scan.problems):
        entry = make_failure_entry(
            index=problem_base + offset,
            input_path=problem.path,
            output_path=None,
            category=ErrorCategory.INPUT_NOT_FOUND,
            detail=f"[{problem.kind}] {problem.message}",
            error_type=InputNotFoundError.__name__,
        )
        _record(entry)

    # --- 阶段 2：规划期已裁决的「输出已存在」跳过 ----------------------
    runnable: list[PlannedJob] = []
    for job in jobs_plan:
        if job.pre_exists and on_exist == EXIST_SKIP:
            _record(make_skip_entry(
                index=job.index,
                input_path=job.input_path,
                output_path=job.output_path,
                category=ErrorCategory.OUTPUT_EXISTS,
                reason=SkipReason.OUTPUT_EXISTS,
                detail=f"输出文件已存在，按策略跳过: {job.output_path}",
                error_type="OutputExistsError",
            ))
            continue
        runnable.append(job)

    # --- 阶段 3：有界并发执行，单文件失败只入清单、绝不中断整体 --------
    worker_count = max(1, min(jobs, len(runnable))) if runnable else 0
    if worker_count:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="bvp-worker",
        ) as executor:
            futures = [
                executor.submit(
                    _run_one, job, transcode,
                    codec, container, crf, preset,
                    audio_bitrate, resolution, on_exist, _record,
                )
                for job in runnable
            ]
            # result() 把工作线程里任何未预期的异常也暴露出来；
            # _run_one 内部已兜底，这里再加一层防御，保证绝不向上逃逸
            for future in futures:
                future.result()

    # --- 阶段 4：落盘三份清单并立即对账 -------------------------------
    target_manifest_dir = os.path.abspath(
        manifest_dir or (output_dir or os.getcwd())
    )
    run_info = _build_run_info(
        inputs, output_dir, codec, container, on_exist, jobs, recursive
    )
    # 并发下条目按完成先后追加；统一按 index 排序，保证内存清单与落盘清单
    # 的顺序都只由确定性扫描顺序决定，与并发度无关
    manifest.entries.sort(key=lambda entry: entry.index)
    paths = manifest.write(target_manifest_dir, run_info=run_info)
    reconciliation = reconcile_manifest(manifest)

    return BatchResult(
        manifest=manifest,
        reconciliation=reconciliation,
        manifest_paths=paths,
        output_dir=os.path.abspath(output_dir) if output_dir else None,
    )


# ---------------------------------------------------------------------------
# 单任务执行：核心调用 + 契约分类，异常到清单条目的唯一转换点
# ---------------------------------------------------------------------------

def _run_one(
    job: PlannedJob,
    transcode: TranscodeFn,
    codec: str,
    container: str,
    crf: int | None,
    preset: str,
    audio_bitrate: str,
    resolution: str | None,
    on_exist: str,
    record: Callable[[ManifestEntry], None],
) -> None:
    """执行一个任务并把任何结果转成清单条目；本函数保证不抛出异常。"""
    try:
        result = transcode(
            job.input_path,
            job.output_path,
            codec=codec,
            container=container,
            crf=crf,
            preset=preset,
            audio_bitrate=audio_bitrate,
            resolution=resolution,
            overwrite=on_exist != EXIST_SKIP,
        )
    except Exception as exc:  # noqa: BLE001 - 批量层必须兜住所有普通错误
        # 注意只捕获 Exception：KeyboardInterrupt/SystemExit 属于控制流，
        # 必须继续向上传播，不能被当成一条失败任务静默吞掉。
        _record_exception(job, exc, record)
        return

    # 核心正常返回：以磁盘事实为准记录成功条目（大小作为复核锚点）
    output_path = getattr(result, "output_path", job.output_path)
    try:
        size = os.path.getsize(output_path)
    except OSError:
        # 核心声称成功但产物读不到：按转码失败入清单，触发对账失败
        record(make_failure_entry(
            index=job.index,
            input_path=job.input_path,
            output_path=output_path,
            category=ErrorCategory.TRANSCODE_FAILED,
            detail="核心返回成功但输出文件无法读取。",
            error_type="PostCheckError",
        ))
        return
    record(make_success_entry(
        index=job.index,
        input_path=job.input_path,
        output_path=output_path,
        output_size=size,
    ))


def _record_exception(
    job: PlannedJob,
    exc: Exception,
    record: Callable[[ManifestEntry], None],
) -> None:
    """按稳定契约把异常分类为失败条目或跳过条目（不解析消息文本）。"""
    category = classify_error(exc)
    error_type = type(exc).__name__
    detail = str(exc) or error_type
    ffmpeg_log = _resolve_ffmpeg_log(job.output_path)

    if category_is_skip(category):
        reason = _skip_reason_for(category)
        record(make_skip_entry(
            index=job.index,
            input_path=job.input_path,
            output_path=job.output_path,
            category=category,
            reason=reason,
            detail=detail,
            error_type=error_type,
        ))
        return

    record(make_failure_entry(
        index=job.index,
        input_path=job.input_path,
        output_path=job.output_path,
        category=category,
        detail=detail,
        error_type=error_type,
        ffmpeg_log_path=ffmpeg_log,
    ))


def _skip_reason_for(category: ErrorCategory) -> SkipReason:
    """分类 → 跳过原因码的稳定映射（只依据分类，不看异常细节）。"""
    if category == ErrorCategory.NOT_A_VIDEO:
        return SkipReason.NON_VIDEO
    if category == ErrorCategory.OUTPUT_EXISTS:
        return SkipReason.OUTPUT_EXISTS
    # 契约外的分类理论上不会进入跳过集合；兜底为不可访问并保持可枚举
    return SkipReason.INACCESSIBLE


def _resolve_ffmpeg_log(output_path: str | None) -> str | None:
    """尽力定位失败任务的 ffmpeg 完整日志（核心约定为 <输出>.ffmpeg.log）。"""
    if not output_path:
        return None
    candidate = output_path + ".ffmpeg.log"
    return candidate if os.path.exists(candidate) else None


# ---------------------------------------------------------------------------
# 参数校验与运行信息
# ---------------------------------------------------------------------------

def _validate_args(
    inputs: list[str],
    jobs: int,
    codec: str,
    container: str,
) -> None:
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs 必须是非空的文件/目录路径列表。")
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError(f"jobs 必须是 >= 1 的整数，收到 {jobs!r}。")
    if codec not in _SUPPORTED_CODECS:
        raise ValueError(f"不支持的编码 {codec!r}，可选 {sorted(_SUPPORTED_CODECS)}。")
    if container not in _CONTAINER_EXTENSIONS:
        raise ValueError(
            f"不支持的容器 {container!r}，可选 {', '.join(_CONTAINER_EXTENSIONS)}。"
        )


def _build_run_info(
    inputs: list[str],
    output_dir: str | None,
    codec: str,
    container: str,
    on_exist: str,
    jobs: int,
    recursive: bool,
) -> dict[str, Any]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [os.path.abspath(p) for p in inputs],
        "output_dir": os.path.abspath(output_dir) if output_dir else None,
        "codec": codec,
        "container": container,
        "on_exist": on_exist,
        "jobs": jobs,
        "recursive": recursive,
        "tool": "bvp",
    }
