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
    plan_outputs_per_file,
)
from .rules.engine import RuleSet
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

#: 无规则路径的内置全批默认值（与历史 CLI 默认严格一致）
_BUILTIN_BATCH_DEFAULTS: dict = {
    "codec": "h264",
    "container": "mp4",
    "crf": None,
    "preset": "medium",
    "audio_bitrate": "128k",
    "resolution": None,
    "on_exist": EXIST_SKIP,
    "output_dir": None,
}


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
    recursive: bool | None = None,
    jobs: int | None = None,
    on_exist: str | None = None,
    codec: str | None = None,
    container: str | None = None,
    crf: int | None = None,
    preset: str | None = None,
    audio_bitrate: str | None = None,
    resolution: str | None = None,
    extensions: frozenset[str] | None = None,
    manifest_dir: str | None = None,
    rule_set: RuleSet | None = None,
    progress_callback: Callable[[Manifest], None] | None = None,
    transcode_fn: TranscodeFn | None = None,
) -> BatchResult:
    """对若干文件/目录执行批量转码，返回可复核的 :class:`BatchResult`。

    支持两条参数来源，合并优先级（高 → 低）为：

    1. 本函数显式关键字参数（对应 CLI 显式选项，``None`` 表示未指定）；
    2. ``rule_set`` 中命中规则逐文件裁决出的参数（含继承/覆盖）；
    3. 规则配置的 ``defaults`` 段；
    4. 内置默认值（与历史 CLI 默认严格一致，旧调用方行为不变）。

    :param inputs: 文件或目录路径列表，目录按扩展名筛选并可递归嵌套
    :param output_dir: 统一输出目录；None 时由规则或核心默认决定
    :param recursive: 目录是否递归；None 时取规则 defaults，再退化为 True
    :param jobs: 并发工作线程数；None 时取规则 defaults，再退化为 1
    :param on_exist/codec/container/crf/preset/audio_bitrate/resolution:
        显式覆盖项，非 None 时压过规则对**所有**任务的同名字段
    :param extensions: 扫描扩展名白名单；None 时取规则并集，再退化为内置集合
    :param rule_set: 规则引擎不可变快照（见 :mod:`bvp.rules`）；None 走旧路径
    :param manifest_dir: 清单目录；None 时取输出目录，再退化为当前工作目录
    :param progress_callback: 每完成一个任务收到当前 :class:`Manifest` 快照
    :param transcode_fn: 测试注入点；生产代码留空即使用真实核心
    :returns: :class:`BatchResult`；任何单文件错误都不会向上抛，全部入清单
    :raises ValueError: 启动前参数不合法（在扫描与任何转码之前快速失败）
    """
    transcode = transcode_fn or core_module.transcode

    # --- 解析生效的批量级默认值（显式参数 > 规则 defaults > 内置） --------
    defaults = rule_set.defaults if rule_set is not None else {}
    effective_recursive = (
        recursive if recursive is not None
        else bool(defaults.get("recursive", True))
    )
    effective_jobs = (
        jobs if jobs is not None
        else int(defaults.get("jobs", 1))
    )

    # 显式覆盖层：只保留调用方真正给出的非 None 值。None 表示「未指定」，
    # 规则路径下绝不能用内置默认顶替，否则规则参数会被静默覆盖。
    explicit_overrides = {
        key: value
        for key, value in (
            ("codec", codec), ("container", container), ("crf", crf),
            ("preset", preset), ("audio_bitrate", audio_bitrate),
            ("resolution", resolution), ("on_exist", on_exist),
            ("output_dir", output_dir),
        )
        if value is not None
    }

    # 无规则路径（旧 CLI/库行为）：在内置默认上叠加显式覆盖，得到全批参数
    batch_params = dict(_BUILTIN_BATCH_DEFAULTS)
    batch_params.update({
        key: value for key, value in defaults.items()
        if key in _BUILTIN_BATCH_DEFAULTS
    })
    batch_params.update(explicit_overrides)
    _validate_args(inputs, effective_jobs,
                   batch_params["codec"], batch_params["container"])
    extension = _CONTAINER_EXTENSIONS[batch_params["container"]]

    # 规则驱动的扩展名白名单：规则并集；显式 extensions 仍优先
    scan_extensions = extensions
    if scan_extensions is None and rule_set is not None:
        rule_exts = rule_set.all_extensions()
        if rule_exts:
            scan_extensions = rule_exts

    # 输出目录嵌套在输入目录内时，递归扫描必须排除它，否则上一轮产物
    # （同为视频扩展名）会被当成新一轮输入反复转码
    exclude_dirs: frozenset[str] = frozenset()
    effective_output_dir = batch_params["output_dir"]
    if effective_output_dir is not None and effective_recursive:
        exclude_dirs = frozenset({os.path.abspath(effective_output_dir)})

    scan = scan_inputs(
        list(inputs),
        recursive=effective_recursive,
        extensions=scan_extensions,
        exclude_dirs=exclude_dirs,
    )

    # --- 规则求值：排除项不入选；入选文件逐任务裁决参数 ----------------
    selected_files, job_params = _resolve_files(
        scan.files, rule_set, explicit_overrides,
    )

    def _resolve_output(input_path: str):
        if rule_set is None:
            return (
                effective_output_dir, extension,
                batch_params["codec"], batch_params["on_exist"],
            )
        params, rule_name = job_params[input_path]
        job_ext = _CONTAINER_EXTENSIONS[params["container"]]
        return (
            params["output_dir"], job_ext, params["codec"],
            params["on_exist"], params, rule_name,
        )

    if rule_set is None:
        jobs_plan = plan_outputs(
            selected_files,
            list(inputs),
            output_dir=effective_output_dir,
            extension=extension,
            codec=batch_params["codec"],
            on_exist=batch_params["on_exist"],
        )
    else:
        jobs_plan = plan_outputs_per_file(
            selected_files, list(inputs), resolve=_resolve_output,
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
        if job.pre_exists:
            _record(make_skip_entry(
                index=job.index,
                input_path=job.input_path,
                output_path=job.output_path,
                category=ErrorCategory.OUTPUT_EXISTS,
                reason=SkipReason.OUTPUT_EXISTS,
                detail=f"输出文件已存在，按策略跳过: {job.output_path}",
                error_type="OutputExistsError",
                rule_name=job.rule_name,
            ))
            continue
        runnable.append(job)

    # --- 阶段 3：有界并发执行，单文件失败只入清单、绝不中断整体 --------
    worker_count = max(1, min(effective_jobs, len(runnable))) if runnable else 0
    if worker_count:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="bvp-worker",
        ) as executor:
            futures = [
                executor.submit(
                    _run_one, job, transcode, batch_params, _record,
                )
                for job in runnable
            ]
            # result() 把工作线程里任何未预期的异常也暴露出来；
            # _run_one 内部已兜底，这里再加一层防御，保证绝不向上逃逸
            for future in futures:
                future.result()

    # --- 阶段 4：落盘三份清单并立即对账 -------------------------------
    target_manifest_dir = os.path.abspath(
        manifest_dir or (effective_output_dir or os.getcwd())
    )
    run_info = _build_run_info(
        inputs, effective_output_dir, batch_params,
        effective_jobs, effective_recursive, rule_set,
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
        output_dir=(
            os.path.abspath(effective_output_dir)
            if effective_output_dir else None
        ),
    )


def _resolve_files(
    files: tuple[str, ...],
    rule_set: RuleSet | None,
    explicit_overrides: dict,
) -> tuple[tuple[str, ...], dict[str, tuple[dict, str | None]]]:
    """对扫描出的文件应用规则求值，返回入选文件与逐任务参数映射。

    无规则集时全部入选、复用全批参数（旧路径，``rule_name`` 为 None）。
    ``explicit_overrides`` 只包含调用方真正显式给出的非 None 值，在此叠加
    到每个任务上，保证 CLI 覆盖优先级高于规则，但不污染未覆盖字段。
    """
    if rule_set is None:
        return files, {}

    selected: list[str] = []
    job_params: dict[str, tuple[dict, str | None]] = {}

    for file_path in files:
        resolution = rule_set.resolve(file_path)
        if not resolution.selected:
            continue
        params = dict(resolution.params.core_kwargs())
        params["on_exist"] = resolution.params.on_exist
        params["output_dir"] = resolution.params.output_dir
        # 显式参数覆盖规则裁决结果（CLI > 规则）
        params.update(explicit_overrides)
        selected.append(file_path)
        job_params[file_path] = (params, resolution.winning_rule)
    return tuple(selected), job_params


# ---------------------------------------------------------------------------
# 单任务执行：核心调用 + 契约分类，异常到清单条目的唯一转换点
# ---------------------------------------------------------------------------

def _run_one(
    job: PlannedJob,
    transcode: TranscodeFn,
    batch_params: dict,
    record: Callable[[ManifestEntry], None],
) -> None:
    """执行一个任务并把任何结果转成清单条目；本函数保证不抛出异常。

    任务自带 :attr:`PlannedJob.params` 时（规则引擎路径）按任务参数转码；
    否则回退到全批统一参数（旧 CLI 路径），两条路径共用同一核心调用。
    """
    params = job.params or batch_params
    try:
        result = transcode(
            job.input_path,
            job.output_path,
            codec=params["codec"],
            container=params["container"],
            crf=params["crf"],
            preset=params["preset"],
            audio_bitrate=params["audio_bitrate"],
            resolution=params["resolution"],
            overwrite=params["on_exist"] != EXIST_SKIP,
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
            rule_name=job.rule_name,
        ))
        return
    record(make_success_entry(
        index=job.index,
        input_path=job.input_path,
        output_path=output_path,
        output_size=size,
        rule_name=job.rule_name,
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
            rule_name=job.rule_name,
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
        rule_name=job.rule_name,
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
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
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
    batch_params: dict,
    jobs: int,
    recursive: bool,
    rule_set: RuleSet | None,
) -> dict[str, Any]:
    info = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [os.path.abspath(p) for p in inputs],
        "output_dir": os.path.abspath(output_dir) if output_dir else None,
        "codec": batch_params["codec"],
        "container": batch_params["container"],
        "on_exist": batch_params["on_exist"],
        "jobs": jobs,
        "recursive": recursive,
        "tool": "bvp",
    }
    if rule_set is not None:
        info["rules_config"] = rule_set.config_path
        info["rules"] = [rule.name for rule in rule_set.rules]
    return info
