"""文件夹批量转码：目录扫描、任务规划、有界并发调度与聚合进度。

本模块在单文件 :func:`transcode_tool.transcoder.transcode` 之上做编排，
自身**不直接拼 ffmpeg 命令**，因此单文件转码的全部既有能力（编码/容器注册表、
码率与分辨率规划、分段转码、产物复核、半成品清理）在批量场景下逐字复用，
旧功能不会因批量能力的引入而退化。

设计上兼顾内存与时间开销：

- **内存**：扫描结果只保存路径字符串而非文件内容；并发度有界，默认仅 1 个
  ffmpeg 同时运行（视频编码是 CPU/内存密集型任务，盲目多开会把机器压垮）；
  每个文件的 ffmpeg 日志仍由 runner 边产生边落盘，批量层不缓存日志正文。
- **时间**：多文件可用 ``jobs`` 开启有界并发；任务队列基于 ``threading``
  条件变量手写，避免一次性起进程；同一文件只探测/编码一次，输出路径在
  启动前统一规划，冲突与覆盖策略在 ffmpeg 启动前就裁决完毕。
- **容错**：单文件失败不拖垮整批（可配置停止策略），失败现场写入结构化
  错误日志（含时间戳、文件、阶段、异常类型与消息、ffmpeg 日志路径）。

全部实现仅使用 Python 标准库，公开符号在包 ``__init__`` 重新导出。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .codecs import CodecSpec, resolve_codec
from .containers import (
    ContainerSpec,
    default_container_for,
    resolve_container,
)
from .errors import TranscodeError
from .transcoder import transcode

# ---------------------------------------------------------------------------
# 常量与策略名
# ---------------------------------------------------------------------------

#: 默认纳入批量扫描的视频扩展名（小写、含点）。
#: 只按扩展名做第一轮筛选；文件是否真为有效媒体仍由 ffprobe 按内容裁决，
#: 伪装文件会在转码阶段以 InvalidInputError 记入错误日志而非污染整批。
DEFAULT_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
    ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts",
    ".3gp", ".3g2", ".ogv", ".rmvb", ".rm", ".vob", ".dv",
})

#: 任务生命周期状态
STATUS_QUEUED = "queued"        # 排队中
STATUS_RUNNING = "running"      # 转码中
STATUS_DONE = "done"            # 成功
STATUS_FAILED = "failed"        # 失败（已写错误日志）
STATUS_SKIPPED = "skipped"      # 按策略跳过（输出已存在）
STATUS_CANCELLED = "cancelled"  # 中断或停止策略下未及执行

#: 失败策略：遇错继续其余文件 / 遇错立即停止（在跑的文件跑完后不再派发）
FAIL_CONTINUE = "continue"
FAIL_STOP = "stop"

#: 输出已存在策略：跳过 / 覆盖 / 自动改名加序号
EXIST_SKIP = "skip"
EXIST_OVERWRITE = "overwrite"
EXIST_SUFFIX = "suffix"

_FAIL_POLICIES = frozenset({FAIL_CONTINUE, FAIL_STOP})
_EXIST_POLICIES = frozenset({EXIST_SKIP, EXIST_OVERWRITE, EXIST_SUFFIX})

#: 聚合进度回调：参数为不可变快照 :class:`BatchSnapshot`
BatchProgressCallback = Callable[["BatchSnapshot"], None]


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchJob:
    """单个批量任务的规划结果（不可变，可在线程间安全共享）。"""

    index: int
    input_path: str          # 已转绝对路径
    output_path: str         # 启动前规划好的绝对输出路径
    source_root: str | None  # 用于镜像目录结构的输入根（文件或目录）


@dataclass(frozen=True)
class BatchSnapshot:
    """某次进度回调时刻的整批进度快照（纯数值，线程安全读取）。"""

    total: int
    done: int
    failed: int
    skipped: int
    cancelled: int
    running: int
    overall_ratio: float
    current_files: tuple[str, ...]

    @property
    def finished(self) -> int:
        """已进入终态的任务数（成功+失败+跳过+取消）。"""
        return self.done + self.failed + self.skipped + self.cancelled


@dataclass
class JobOutcome:
    """单个任务的终态记录。"""

    index: int
    input_path: str
    output_path: str | None
    status: str
    error: str | None = None
    error_type: str | None = None
    ffmpeg_log_path: str | None = None


@dataclass
class BatchReport:
    """整批转码的汇总结果。"""

    jobs: tuple[BatchJob, ...]
    outcomes: list[JobOutcome] = field(default_factory=list)
    log_path: str | None = None
    interrupted: bool = False

    @property
    def total(self) -> int:
        return len(self.jobs)

    @property
    def done_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == STATUS_DONE)

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == STATUS_FAILED)

    @property
    def skipped_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == STATUS_SKIPPED)

    @property
    def cancelled_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == STATUS_CANCELLED)

    @property
    def success(self) -> bool:
        """全部入队任务都成功或被策略跳过时视为 True（中断/失败为 False）。"""
        return (
            not self.interrupted
            and self.failed_count == 0
            and self.cancelled_count == 0
        )

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "done": self.done_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "cancelled": self.cancelled_count,
            "interrupted": self.interrupted,
            "success": self.success,
            "log_path": self.log_path,
            "outcomes": [
                {
                    "input_path": o.input_path,
                    "output_path": o.output_path,
                    "status": o.status,
                    "error": o.error,
                    "error_type": o.error_type,
                    "ffmpeg_log_path": o.ffmpeg_log_path,
                }
                for o in self.outcomes
            ],
        }


# ---------------------------------------------------------------------------
# 目录扫描（手写 os.scandir 迭代，不引入任何第三方封装）
# ---------------------------------------------------------------------------

def scan_video_files(
    inputs: list[str],
    *,
    recursive: bool = False,
    extensions: frozenset[str] | None = None,
    exclude_dirs: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """从若干文件/目录路径中收集待转码视频。

    :param inputs: 文件或目录路径列表。文件直接入选；目录按扩展名筛选其中
        的普通文件，``recursive=True`` 时递归进入子目录
    :param recursive: 目录是否递归
    :param extensions: 自定义扩展名集合（小写、含点，如 ``{".mp4"}``）；
        None 时使用 :data:`DEFAULT_VIDEO_EXTENSIONS`
    :param exclude_dirs: 递归时整体跳过的目录绝对路径集合（典型用途：
        输出目录位于输入目录之内时，避免把上一轮产物当成新输入）
    :returns: 去重、排序后的绝对路径元组；不存在的路径直接忽略
        （是否报错由上层规划阶段决定，这里保持扫描函数的纯粹性）

    扫描过程只持有 DirEntry 与最终的路径字符串列表，不缓存任何文件内容，
    万级文件目录的内存开销也只有路径名本身。
    """
    exts = _normalize_extensions(extensions)
    excluded = {os.path.abspath(p) for p in (exclude_dirs or frozenset())}
    seen: set[str] = set()
    collected: list[str] = []

    for raw in inputs:
        if not raw:
            continue
        path = os.path.abspath(raw)
        if os.path.isfile(path):
            # 显式点名的文件绕过扩展名筛选：用户既然直接给出，就应入选，
            # 是否为有效媒体交给后续 ffprobe 按内容裁决（与目录扫描区分开）
            _collect_path(path, seen, collected)
        elif os.path.isdir(path):
            if recursive:
                _scan_recursive(path, exts, seen, collected, excluded)
            else:
                _scan_flat(path, exts, seen, collected)
        # 不存在的路径静默跳过，批量场景下单个笔误不应阻止整批规划

    # 统一排序保证多次运行的任务顺序确定，便于测试与日志对照
    return tuple(sorted(collected))


def _normalize_extensions(extensions: frozenset[str] | None) -> frozenset[str]:
    if extensions is None:
        return DEFAULT_VIDEO_EXTENSIONS
    normalized = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else "." + ext)
    if not normalized:
        raise ValueError("扩展名过滤集合为空，至少需要一个扩展名。")
    return frozenset(normalized)


def _scan_flat(directory: str, exts: frozenset[str],
               seen: set[str], collected: list[str]) -> None:
    """只扫描目录一层（不递归），子目录不展开。"""
    try:
        entries = sorted(os.listdir(directory), key=lambda n: n.lower())
    except OSError:
        return
    for name in entries:
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            _collect_if_video(full, exts, seen, collected)


def _scan_recursive(directory: str, exts: frozenset[str],
                    seen: set[str], collected: list[str],
                    excluded: set[str]) -> None:
    """深度优先递归扫描；每一层先排序，保证跨平台顺序稳定。"""
    stack: list[str] = [directory]
    while stack:
        current = stack.pop()
        if os.path.abspath(current) in excluded:
            continue
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError:
            continue
        entries.sort(key=lambda entry: entry.name.lower())
        sub_dirs: list[str] = []
        for entry in entries:
            try:
                is_dir = entry.is_dir()
                is_file = entry.is_file()
            except OSError:
                continue
            if is_dir:
                if os.path.abspath(entry.path) not in excluded:
                    sub_dirs.append(entry.path)
            elif is_file:
                _collect_if_video(entry.path, exts, seen, collected)
        # 逆序压栈，使最先排序的子目录最先弹出
        stack.extend(reversed(sub_dirs))


def _collect_if_video(path: str, exts: frozenset[str],
                      seen: set[str], collected: list[str]) -> None:
    """目录扫描入口：按扩展名筛选后去重收集。"""
    abs_path = os.path.abspath(path)
    if os.path.splitext(abs_path)[1].lower() not in exts:
        return
    _collect_path(abs_path, seen, collected)


def _collect_path(abs_path: str, seen: set[str],
                  collected: list[str]) -> None:
    """去重收集一个已确认入选的绝对路径（不做扩展名判断）。"""
    if abs_path in seen:
        return
    seen.add(abs_path)
    collected.append(abs_path)


# ---------------------------------------------------------------------------
# 输出路径规划：镜像目录结构 + 冲突/覆盖裁决
# ---------------------------------------------------------------------------

def plan_jobs(
    input_files: tuple[str, ...],
    input_roots: list[str],
    *,
    output_dir: str | None,
    codec: CodecSpec,
    container: ContainerSpec,
    on_exist: str,
    output_suffix: str | None = None,
) -> list[BatchJob]:
    """把扫描出的输入文件规划成带输出路径的任务列表。

    :param input_files: :func:`scan_video_files` 的产物
    :param input_roots: 用户给定的原始输入根（文件或目录），用于计算镜像
        目录结构时的相对路径；取包含该文件的第一个根
    :param output_dir: 统一输出目录；None 时产物生成在各自输入旁，
        命名复刻单文件默认规则 ``<名>_<编码><容器扩展名>``
    :param on_exist: 输出已存在策略 ``skip`` / ``overwrite`` / ``suffix``
    :param output_suffix: 文件名中编码名前的自定义后缀段，None 时用编码
        规范名（与单文件默认命名 ``<名>_<编码><扩展名>`` 保持一致）

    规划阶段不启动任何子进程，纯字符串计算，可在无 ffmpeg 环境下单测。
    """
    if on_exist not in _EXIST_POLICIES:
        raise ValueError(
            f"on_exist 必须是 {', '.join(sorted(_EXIST_POLICIES))} 之一，"
            f"收到 {on_exist!r}。"
        )

    roots = [os.path.abspath(r) for r in input_roots if r]
    suffix = output_suffix if output_suffix is not None else codec.canonical
    used_outputs: set[str] = set()
    jobs: list[BatchJob] = []

    for index, input_path in enumerate(input_files):
        source_root = _find_containing_root(input_path, roots)

        if output_dir is None:
            # 与 transcoder._resolve_output_path 的默认命名保持一致，
            # 让批量层在启动前即可裁决 skip/suffix，而不是把冲突推迟到 ffmpeg
            output_path = _default_output_path(
                input_path, suffix, container.extension
            )
        else:
            output_path = _mirror_output_path(
                input_path, source_root, os.path.abspath(output_dir),
                suffix, container.extension,
            )

        # 同批两个输入映射到同一输出（如 a.mov 与 a.mp4 同名不同扩展名）：
        # 不论 on_exist 如何都给后者加序号，避免互相覆盖
        if output_path in used_outputs:
            output_path = _dedupe_path(output_path, used_outputs)
        used_outputs.add(output_path)

        if os.path.exists(output_path):
            if on_exist == EXIST_SKIP:
                jobs.append(BatchJob(index, input_path, output_path, source_root))
                continue
            if on_exist == EXIST_SUFFIX:
                output_path = _dedupe_path(output_path, used_outputs)
                used_outputs.add(output_path)
            # overwrite：保留原路径，转码时传 overwrite=True

        jobs.append(BatchJob(index, input_path, output_path, source_root))

    return jobs


def _find_containing_root(input_path: str, roots: list[str]) -> str | None:
    """找到包含该文件的输入根；文件独立给出时根就是其所在目录。"""
    for root in roots:
        if os.path.isfile(root):
            if os.path.abspath(root) == input_path:
                return os.path.dirname(root)
            continue
        # 公共前缀判断 + 路径分隔符边界，避免 /a/ab 误匹配 /a/a
        if input_path == root or input_path.startswith(root + os.sep):
            return root
    return os.path.dirname(input_path)


def _default_output_path(input_path: str, suffix: str,
                         extension: str) -> str:
    """复刻单文件转码的默认输出命名（含自覆盖保护）。"""
    stem = os.path.splitext(input_path)[0]
    default_path = os.path.abspath(f"{stem}_{suffix}{extension}")
    if default_path == os.path.abspath(input_path):
        default_path = os.path.abspath(f"{stem}_transcoded{extension}")
    return default_path


def _mirror_output_path(input_path: str, source_root: str | None,
                        output_dir: str, suffix: str,
                        extension: str) -> str:
    """按输入相对目录在输出目录中镜像同名文件，并替换扩展名。

    纯字符串计算，不依赖源目录在磁盘上是否存在（规划阶段可能先于落盘）；
    Windows 跨盘符等无法计算相对路径的情况退化为平铺到输出目录。
    """
    mirrored_dir = output_dir
    if source_root:
        try:
            relative = os.path.relpath(input_path, source_root)
            mirrored_dir = os.path.join(output_dir, os.path.dirname(relative))
        except ValueError:
            mirrored_dir = output_dir
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.abspath(
        os.path.join(mirrored_dir, f"{stem}_{suffix}{extension}")
    )


def _dedupe_path(path: str, used: set[str]) -> str:
    """在文件名 stem 后追加 _2、_3… 直到路径既未被占用也不存在。"""
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    counter = 2
    while True:
        candidate = os.path.join(directory, f"{stem}_{counter}{ext}")
        if candidate not in used and not os.path.exists(candidate):
            return os.path.abspath(candidate)
        counter += 1


# ---------------------------------------------------------------------------
# 手写有界任务队列（threading.Condition 实现，不依赖 queue.Queue）
# ---------------------------------------------------------------------------

class _JobQueue:
    """有界批量任务的线程安全队列。

    全部任务在工作线程启动前一次性入队，运行期没有新生产者，因此
    “队列被领空”即等价于“全部任务已被认领”，:meth:`get` 领空时直接
    返回 None 让工作线程退出——不能阻塞等待 :meth:`close`，否则工作
    线程等关闭、主线程等工作线程结束，形成死锁。锁仅用于保证多线程
    并发 ``pop`` 的互斥。
    """

    def __init__(self, jobs: list[BatchJob]):
        self._lock = threading.Lock()
        self._pending: list[BatchJob] = list(jobs)
        self._closed = False

    def get(self) -> BatchJob | None:
        """领取下一个任务；队列已领空时返回 None。"""
        with self._lock:
            if self._pending:
                return self._pending.pop(0)
            self._closed = True
            return None

    def close(self) -> None:
        """标记关闭（幂等）；领空语义下仅作显式收尾，不唤醒任何等待。"""
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


# ---------------------------------------------------------------------------
# 线程安全的聚合进度
# ---------------------------------------------------------------------------

class BatchProgress:
    """跟踪每个任务的状态与文件内进度，聚合成整批完成比例。

    终态任务（done/failed/skipped/cancelled）按 100% 计入总量——它们的
    工作量确实已经消耗完毕；running 任务按 ffmpeg 上报的实时比例计入；
    queued 任务计 0。等权聚合：``overall = Σ 单任务比例 / N``。
    """

    def __init__(self, jobs: list[BatchJob]):
        self._lock = threading.Lock()
        self._ratios: dict[int, float] = {job.index: 0.0 for job in jobs}
        self._status: dict[int, str] = {job.index: STATUS_QUEUED for job in jobs}
        self._jobs = jobs
        self._total = len(jobs)

    def update_ratio(self, index: int, ratio: float) -> None:
        ratio = min(max(ratio, 0.0), 1.0)
        with self._lock:
            self._ratios[index] = ratio

    def mark(self, index: int, status: str) -> None:
        with self._lock:
            self._status[index] = status
            if status in (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED,
                          STATUS_CANCELLED):
                self._ratios[index] = 1.0

    def snapshot(self) -> BatchSnapshot:
        with self._lock:
            counts = {
                STATUS_DONE: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0,
                STATUS_CANCELLED: 0, STATUS_RUNNING: 0,
            }
            weighted = 0.0
            current: list[str] = []
            for job in self._jobs:
                status = self._status[job.index]
                ratio = self._ratios[job.index]
                if status in counts:
                    counts[status] += 1
                if status == STATUS_RUNNING:
                    weighted += ratio
                    current.append(os.path.basename(job.input_path))
                elif status in (STATUS_DONE, STATUS_FAILED,
                                STATUS_SKIPPED, STATUS_CANCELLED):
                    weighted += 1.0
            overall = weighted / self._total if self._total else 1.0
            return BatchSnapshot(
                total=self._total,
                done=counts[STATUS_DONE],
                failed=counts[STATUS_FAILED],
                skipped=counts[STATUS_SKIPPED],
                cancelled=counts[STATUS_CANCELLED],
                running=counts[STATUS_RUNNING],
                overall_ratio=min(max(overall, 0.0), 1.0),
                current_files=tuple(current),
            )

    def make_callback(self, index: int) -> Callable[[float], None]:
        """生成传给单文件 transcode(progress_callback=...) 的闭包。"""
        def _on_progress(ratio: float) -> None:
            self.update_ratio(index, ratio)
        return _on_progress


# ---------------------------------------------------------------------------
# 错误日志（结构化、行缓冲、失败现场可追溯）
# ---------------------------------------------------------------------------

class BatchErrorLogger:
    """把批量过程中的跳过、失败、中断写入结构化文本日志。

    日志文件保持追加友好的纯文本格式（时间戳 + 级别 + 缩进字段），不依赖
    logging 配置体系，便于在库调用与 CLI 两种场景下直接使用。文件句柄
    行缓冲打开，长任务运行中也能实时 tail 查看；句柄随上下文管理器关闭。
    """

    def __init__(self, log_path: str):
        directory = os.path.dirname(os.path.abspath(log_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.log_path = log_path
        self._handle = open(
            log_path, "w", encoding="utf-8", errors="replace", buffering=1
        )
        self._lock = threading.Lock()
        self._write("INFO", "批量转码开始", {})

    def log_error(self, job: BatchJob, exc: BaseException,
                  *, phase: str = "transcode") -> None:
        fields = {
            "phase": phase,
            "input": job.input_path,
            "output": job.output_path,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        ffmpeg_log = _resolve_ffmpeg_log(job)
        if ffmpeg_log:
            fields["ffmpeg_log"] = ffmpeg_log
        self._write("ERROR", f"任务失败: {os.path.basename(job.input_path)}",
                    fields)

    def log_skip(self, job: BatchJob, reason: str) -> None:
        self._write("WARN", f"任务跳过: {os.path.basename(job.input_path)}", {
            "input": job.input_path,
            "output": job.output_path,
            "reason": reason,
        })

    def log_message(self, level: str, message: str,
                    fields: dict | None = None) -> None:
        self._write(level, message, fields or {})

    def _write(self, level: str, message: str, fields: dict) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{timestamp}] [{level}] {message}"]
        for key in sorted(fields):
            lines.append(f"    {key}: {fields[key]}")
        payload = "\n".join(lines) + "\n"
        with self._lock:
            self._handle.write(payload)
            self._handle.flush()

    def close(self) -> None:
        # 注意：threading.Lock 不可重入，这里持锁后不能再调用会自行加锁的
        # _write，否则同线程二次获取锁形成自死锁；结束行直接在锁内写。
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            if not self._handle.closed:
                self._handle.write(f"[{timestamp}] [INFO] 批量转码结束\n")
                self._handle.close()

    def __enter__(self) -> "BatchErrorLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 批量调度主入口
# ---------------------------------------------------------------------------

def batch_transcode(
    inputs: list[str],
    *,
    output_dir: str | None = None,
    recursive: bool = False,
    jobs: int = 1,
    on_fail: str = FAIL_CONTINUE,
    on_exist: str = EXIST_SKIP,
    extensions: frozenset[str] | None = None,
    log_path: str | None = None,
    progress_callback: BatchProgressCallback | None = None,
    # 以下参数原样透传给单文件 transcode()
    codec: str = "h264",
    container: str | None = None,
    crf: int | None = None,
    preset: str | None = None,
    audio_bitrate: str = "128k",
    target_bitrate: str | int | float | None = None,
    rate_mode: str = "vbr",
    resolution: str | None = None,
    allow_upscale: bool = False,
    segment_duration: float | None = None,
    segment_count: int | None = None,
    overwrite: bool | None = None,
) -> BatchReport:
    """对文件夹（或若干文件/文件夹）执行批量转码。

    :param inputs: 输入路径列表，元素可以是视频文件或目录
    :param output_dir: 统一输出目录，目录结构按输入根镜像；None 时每个
        产物生成在各自输入旁
    :param recursive: 输入目录是否递归扫描子目录
    :param jobs: 并发 ffmpeg 数，默认 1（顺序执行，内存峰值最低）；
        视频编码为 CPU 密集型，建议不超过物理核数
    :param on_fail: ``continue``（默认，单文件失败继续整批）或 ``stop``
        （失败后不再派发新任务，在跑的任务跑完）
    :param on_exist: 输出已存在时 ``skip``（默认）/ ``overwrite`` / ``suffix``
    :param extensions: 自定义扫描扩展名；None 用内置视频扩展名集合
    :param log_path: 错误日志路径；None 时写入 ``<输出目录>/_batch_errors.log``，
        没有输出目录时写到首个输入的同级目录
    :param progress_callback: 聚合进度回调，每次状态变化收到 :class:`BatchSnapshot`
    :param overwrite: 透传给 transcode 的覆盖标志；None 时由 ``on_exist``
        自动推导（overwrite 策略 → True，其余 → False）
    :returns: :class:`BatchReport` 汇总；每个失败任务的异常不会向上抛，
        全部记录在报告与错误日志中
    :raises ValueError: 参数不合法（jobs < 1、策略名错误等），在启动任何
        ffmpeg 前快速失败
    """
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs 必须是非空的文件/目录路径列表。")
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError(f"jobs 必须是 >= 1 的整数，收到 {jobs!r}。")
    if on_fail not in _FAIL_POLICIES:
        raise ValueError(
            f"on_fail 必须是 {', '.join(sorted(_FAIL_POLICIES))} 之一，"
            f"收到 {on_fail!r}。"
        )
    if on_exist not in _EXIST_POLICIES:
        raise ValueError(
            f"on_exist 必须是 {', '.join(sorted(_EXIST_POLICIES))} 之一，"
            f"收到 {on_exist!r}。"
        )

    codec_spec = resolve_codec(codec)
    container_spec = (
        resolve_container(container) if container is not None
        else default_container_for(codec_spec)
    )

    # 输出目录嵌套在输入目录内时，递归扫描必须排除它，否则上一轮产物
    # （同为视频扩展名）会被当成新一轮输入反复转码
    exclude_dirs: frozenset[str] = frozenset()
    if output_dir is not None and recursive:
        exclude_dirs = frozenset({os.path.abspath(output_dir)})

    input_files = scan_video_files(
        list(inputs), recursive=recursive,
        extensions=extensions, exclude_dirs=exclude_dirs,
    )
    planned = plan_jobs(
        input_files, list(inputs),
        output_dir=output_dir,
        codec=codec_spec,
        container=container_spec,
        on_exist=on_exist,
    )

    if log_path is None:
        log_path = _default_log_path(output_dir, input_files)

    effective_overwrite = (
        overwrite if overwrite is not None else on_exist == EXIST_OVERWRITE
    )

    progress = BatchProgress(planned)
    report = BatchReport(jobs=tuple(planned), log_path=log_path)

    # 先裁决跳过项：已存在且策略为 skip 的任务不进入工作队列
    queued_jobs: list[BatchJob] = []
    with BatchErrorLogger(log_path) as logger:
        if not input_files:
            logger.log_message(
                "WARN", "扫描后没有找到任何可转码的视频文件",
                {"inputs": ", ".join(inputs)},
            )

        for job in planned:
            if os.path.exists(job.output_path) and on_exist == EXIST_SKIP:
                progress.mark(job.index, STATUS_SKIPPED)
                report.outcomes.append(JobOutcome(
                    job.index, job.input_path, job.output_path, STATUS_SKIPPED,
                ))
                logger.log_skip(job, "输出文件已存在")
                continue
            queued_jobs.append(job)

        if not queued_jobs:
            _emit(progress, progress_callback)
            report.outcomes.sort(key=lambda o: o.index)
            return report

        queue = _JobQueue(queued_jobs)
        stop_event = threading.Event()      # on_fail=stop 时阻止继续派发
        cancel_event = threading.Event()    # 用户中断
        outcomes_lock = threading.Lock()

        def _run_one(job: BatchJob) -> None:
            progress.mark(job.index, STATUS_RUNNING)
            _emit(progress, progress_callback)
            try:
                transcode(
                    job.input_path,
                    job.output_path,
                    crf=crf,
                    preset=preset,
                    overwrite=effective_overwrite,
                    audio_bitrate=audio_bitrate,
                    codec=codec_spec,
                    container=container_spec,
                    segment_duration=segment_duration,
                    segment_count=segment_count,
                    progress_callback=progress.make_callback(job.index),
                    target_bitrate=target_bitrate,
                    rate_mode=rate_mode,
                    resolution=resolution,
                    allow_upscale=allow_upscale,
                )
            except TranscodeError as exc:
                progress.mark(job.index, STATUS_FAILED)
                with outcomes_lock:
                    report.outcomes.append(JobOutcome(
                        job.index, job.input_path, job.output_path,
                        STATUS_FAILED,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        ffmpeg_log_path=_resolve_ffmpeg_log(job),
                    ))
                    logger.log_error(job, exc)
                if on_fail == FAIL_STOP:
                    stop_event.set()
            except ValueError as exc:
                # 参数级错误（规划期未拦住的个别情况）同样按任务失败处理
                progress.mark(job.index, STATUS_FAILED)
                with outcomes_lock:
                    report.outcomes.append(JobOutcome(
                        job.index, job.input_path, job.output_path,
                        STATUS_FAILED,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    ))
                    logger.log_error(job, exc, phase="validation")
                if on_fail == FAIL_STOP:
                    stop_event.set()
            else:
                progress.mark(job.index, STATUS_DONE)
                with outcomes_lock:
                    report.outcomes.append(JobOutcome(
                        job.index, job.input_path,
                        job.output_path, STATUS_DONE,
                    ))
            finally:
                _emit(progress, progress_callback)

        def _worker() -> None:
            while True:
                if cancel_event.is_set() or stop_event.is_set():
                    # 不再领取新任务；队列中剩余项由主线程统一记为取消
                    return
                job = queue.get()
                if job is None:
                    return
                # 领取与信号之间存在竞态，领取后再检查一次
                if cancel_event.is_set() or stop_event.is_set():
                    with outcomes_lock:
                        _mark_cancelled(job, progress, report, logger,
                                        cancel_event.is_set())
                    continue
                _run_one(job)

        worker_count = min(jobs, len(queued_jobs))
        workers = [
            threading.Thread(target=_worker, name=f"batch-worker-{i + 1}",
                             daemon=True)
            for i in range(worker_count)
        ]
        for worker in workers:
            worker.start()

        interrupted = False
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            # 协作式中断：不再派发新任务，在跑的 ffmpeg 让其自然结束
            # （强杀子进程会留下不可播放的半成品，transcode 已有清理逻辑）
            interrupted = True
            cancel_event.set()
            logger.log_message("WARN", "收到中断信号，等待在跑任务结束后停止")
            for worker in workers:
                worker.join()

        queue.close()
        for worker in workers:
            worker.join()

        # 队列里没被领取的任务统一记为取消（中断）或未执行（fail=stop）
        claimed = {o.index for o in report.outcomes}
        for job in queued_jobs:
            if job.index not in claimed:
                _mark_cancelled(job, progress, report, logger, interrupted)

        report.interrupted = interrupted
        report.outcomes.sort(key=lambda o: o.index)
        _emit(progress, progress_callback)

    return report


def _mark_cancelled(job: BatchJob, progress: BatchProgress,
                    report: BatchReport, logger: BatchErrorLogger,
                    interrupted: bool) -> None:
    progress.mark(job.index, STATUS_CANCELLED)
    report.outcomes.append(JobOutcome(
        job.index, job.input_path, job.output_path, STATUS_CANCELLED,
        error="用户中断，任务未执行" if interrupted else "前序任务失败，任务未执行",
        error_type="Cancelled",
    ))
    logger.log_skip(job, "中断停止" if interrupted else "失败停止策略")


def _resolve_ffmpeg_log(job: BatchJob) -> str | None:
    """尽力定位失败任务的 ffmpeg 完整日志（transcode 约定为 <输出>.ffmpeg.log）。"""
    candidate = job.output_path + ".ffmpeg.log"
    if os.path.exists(candidate):
        return candidate
    return None


def _default_log_path(output_dir: str | None,
                      input_files: tuple[str, ...]) -> str:
    if output_dir:
        base = os.path.abspath(output_dir)
    elif input_files:
        base = os.path.dirname(input_files[0])
    else:
        base = os.getcwd()
    return os.path.join(base, "_batch_errors.log")


def _emit(progress: BatchProgress,
          callback: BatchProgressCallback | None) -> None:
    """进度回调异常不应影响调度，与 runner._safe_callback 同一策略。"""
    if callback is None:
        return
    try:
        callback(progress.snapshot())
    except Exception:
        pass
