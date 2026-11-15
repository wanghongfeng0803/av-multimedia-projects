"""批量水印调度：多输入共享一份水印，逐任务隔离失败。

设计要点（内存与时间开销）：

- **水印只解码一次**：调用方传入已解码的 :class:`Image`，所有任务共享
  同一份只读像素（引擎不会修改输入水印），不随任务数增长；
- **帧流式处理**：每个任务独立「读帧 → 叠加 → 写盘 → 释放」，
  任意时刻内存中只有有限帧（顺序模式 1 帧，并发模式至多 ``workers`` 帧），
  与任务总数无关；
- **失败隔离**：单个任务的任何异常（输入缺失、PNG 损坏、参数非法）
  只记录到该任务的 :class:`JobOutcome`，不影响其余任务继续执行；
- **可选并发**：``workers > 1`` 时用线程池并行，PNG 编解码的 zlib
  压缩/解压会释放 GIL，多线程对 I/O 与编解码有实际收益；
  默认 ``workers=1`` 走纯顺序路径，零线程开销，结果与并发模式一致。

任务清单（manifest）是一个 JSON 文件，描述共享水印、共享配置与任务列表，
相对路径一律按清单所在目录解析，便于迁移整个批处理目录。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .animation import MotionConfig, motion_from_dict
from .config import WatermarkConfig
from .engine import OverlayResult, ScatterResult, overlay_animated_frame, overlay_image, overlay_scattered
from .errors import InvalidParameterError
from .image import Image, read_png, write_png
from .scatter import ScatterConfig, scatter_from_dict

# WatermarkConfig 字段名 -> manifest 中的键（一一对应，便于校验未知键）
_CONFIG_KEYS = {
    "anchor",
    "offset_x",
    "offset_y",
    "target_width",
    "target_height",
    "opacity",
    "adaptive",
    "mask_strength",
    "detail_low",
    "detail_high",
    "mask_radius",
}


@dataclass
class BatchJob:
    """一个批量任务：一张输入帧 -> 一张输出帧。

    ``config`` / ``motion`` / ``scatter`` 为 ``None`` 时回落到
    :func:`run_batch` 的共享配置；``frame_index`` / ``fps`` 仅在
    运动模式下有意义。``scatter`` 与 ``motion`` 互斥。
    """

    input_path: str
    output_path: str
    config: WatermarkConfig | None = None
    motion: MotionConfig | None = None
    scatter: ScatterConfig | None = None
    frame_index: int = 0
    fps: float = 25.0


@dataclass
class JobOutcome:
    """单个任务的执行结果：成功时携带 :class:`OverlayResult`，失败时携带错误信息。"""

    job: BatchJob
    ok: bool
    result: OverlayResult | ScatterResult | None = None
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class BatchReport:
    """整批结果汇总，保持与任务列表相同的顺序。"""

    outcomes: list[JobOutcome] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded

    @property
    def ok(self) -> bool:
        """整批是否全部成功。"""
        return self.failed == 0

    def failures(self) -> list[JobOutcome]:
        """按原顺序返回失败的任务结果。"""
        return [o for o in self.outcomes if not o.ok]


def _run_one_job(
    job: BatchJob,
    watermark: Image,
    shared_config: WatermarkConfig | None,
    shared_motion: MotionConfig | None,
    shared_scatter: ScatterConfig | None = None,
) -> JobOutcome:
    """执行单个任务；任何异常都收敛为 ``ok=False`` 的结果，绝不向上抛。"""
    started = time.perf_counter()
    try:
        config = job.config if job.config is not None else shared_config
        motion = job.motion if job.motion is not None else shared_motion
        scatter = job.scatter if job.scatter is not None else shared_scatter
        if motion is not None and scatter is not None:
            raise InvalidParameterError("分散布局与运动轨迹不能同时使用")
        frame = read_png(job.input_path)
        if scatter is not None:
            result_frame, info = overlay_scattered(frame, watermark, config, scatter)
        elif motion is None:
            result_frame, info = overlay_image(frame, watermark, config)
        else:
            result_frame, info = overlay_animated_frame(
                frame, watermark, config, motion,
                frame_index=job.frame_index, fps=job.fps,
            )
        # 输出目录不存在时自动创建（批量场景下输出常落在独立目录）
        out_dir = os.path.dirname(os.path.abspath(job.output_path))
        os.makedirs(out_dir, exist_ok=True)
        write_png(job.output_path, result_frame)
        # frame / result_frame 在此函数返回后即可被回收
        return JobOutcome(job=job, ok=True, result=info,
                          elapsed=time.perf_counter() - started)
    except Exception as exc:  # 失败隔离：单任务异常不影响整批
        return JobOutcome(job=job, ok=False, error=str(exc),
                          elapsed=time.perf_counter() - started)


def run_batch(
    jobs: list[BatchJob],
    watermark: Image,
    config: WatermarkConfig | None = None,
    motion: MotionConfig | None = None,
    workers: int = 1,
    on_progress=None,
    scatter: ScatterConfig | None = None,
) -> BatchReport:
    """按顺序语义执行一批水印任务，返回 :class:`BatchReport`。

    - ``watermark``：已解码的水印图像，全批共享（只读）；
    - ``config`` / ``motion`` / ``scatter``：共享配置，任务自身的
      非 None 字段优先；``scatter`` 与 ``motion`` 互斥；
    - ``workers``：并发线程数，``<= 1`` 时纯顺序执行（零线程开销）；
    - ``on_progress``：可选回调 ``fn(outcome, done, total)``，
      每完成一个任务按完成顺序调用一次（并发模式下完成顺序不保证
      与任务顺序一致，但报告本身始终按任务顺序排列）。

    本函数自身不抛出任务级异常；所有失败都体现在报告中。
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise InvalidParameterError(f"workers 必须为正整数，收到 {workers!r}")

    total = len(jobs)
    started = time.perf_counter()
    report = BatchReport()

    if workers <= 1:
        # 顺序路径：无线程池开销，任意时刻只有一帧在内存中
        outcomes = [
            _run_one_job(job, watermark, config, motion, scatter) for job in jobs
        ]
    else:
        # 并发路径：线程池大小即内存中同时在处理的帧数上限；
        # executor.map 保持结果顺序与任务顺序一致
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(
                lambda j: _run_one_job(j, watermark, config, motion, scatter), jobs
            ))

    report.outcomes = outcomes
    report.elapsed = time.perf_counter() - started

    if on_progress is not None:
        for done, outcome in enumerate(outcomes, start=1):
            on_progress(outcome, done, total)
    return report


# ---------------------------------------------------------------- 清单加载


@dataclass
class BatchSpec:
    """清单文件解析结果：共享水印路径、共享配置与任务列表。"""

    watermark_path: str
    config: WatermarkConfig
    motion: MotionConfig | None
    jobs: list[BatchJob]
    scatter: ScatterConfig | None = None


def config_from_dict(data: object, context: str = "config") -> WatermarkConfig:
    """从 JSON 风格字典构建 :class:`WatermarkConfig` 并校验。

    未知键直接拒绝（防止拼写错误被静默忽略），非法值抛出
    :class:`InvalidParameterError`。
    """
    if not isinstance(data, dict):
        raise InvalidParameterError(f"{context} 必须是 JSON 对象")
    unknown = sorted(set(data) - _CONFIG_KEYS)
    if unknown:
        raise InvalidParameterError(
            f"{context} 含未知字段：{', '.join(unknown)}"
            f"（可选：{', '.join(sorted(_CONFIG_KEYS))}）"
        )
    config = WatermarkConfig(**data)
    try:
        config.validate()
    except InvalidParameterError as exc:
        raise InvalidParameterError(f"{context} 非法：{exc}") from exc
    return config


def _resolve(base_dir: str, path: object, context: str) -> str:
    """把清单中的相对路径解析为相对清单所在目录的绝对路径。"""
    if not isinstance(path, str) or not path:
        raise InvalidParameterError(f"{context} 必须是非空路径字符串")
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _load_json_object(base_dir: str, raw: object, kind: str) -> object:
    """把「内联对象或相对路径字符串」统一解析为 JSON 对象。"""
    if isinstance(raw, str):
        path = _resolve(base_dir, raw, kind)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as exc:
            raise InvalidParameterError(f"无法读取{kind}文件 {path!r}：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise InvalidParameterError(
                f"{kind}文件 {path!r} 不是合法 JSON：{exc}"
            ) from exc
    return raw


def load_batch_manifest(path: str) -> BatchSpec:
    """加载批量任务清单 JSON，返回 :class:`BatchSpec`。

    清单格式（``config`` / ``motion`` / ``scatter`` / ``fps`` 可省略）::

        {
          "watermark": "wm.png",
          "config": {"anchor": "bottom-right", "opacity": 0.7},
          "motion": "motion.json",            // 或内联对象
          "scatter": {"rows": 2, "cols": 3},  // 或 JSON 文件路径；与 motion 互斥
          "fps": 25.0,
          "jobs": [
            {"input": "a.png", "output": "out/a.png"},
            {"input": "b.png", "output": "out/b.png",
             "config": {"opacity": 0.5}, "frame_index": 12}
          ]
        }

    所有相对路径按清单文件所在目录解析；格式非法统一抛出
    :class:`InvalidParameterError`。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise InvalidParameterError(f"无法读取批量清单 {path!r}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidParameterError(f"批量清单 {path!r} 不是合法 JSON：{exc}") from exc

    if not isinstance(data, dict):
        raise InvalidParameterError("批量清单必须是 JSON 对象")
    base_dir = os.path.dirname(os.path.abspath(path))

    if "watermark" not in data:
        raise InvalidParameterError("批量清单缺少 watermark 字段")
    watermark_path = _resolve(base_dir, data["watermark"], "watermark")

    config = config_from_dict(data.get("config", {}), "config")

    raw_motion = data.get("motion")
    motion: MotionConfig | None = None
    if raw_motion is not None:
        motion = motion_from_dict(_load_json_object(base_dir, raw_motion, "运动配置"))

    raw_scatter = data.get("scatter")
    scatter: ScatterConfig | None = None
    if raw_scatter is not None:
        scatter = scatter_from_dict(
            _load_json_object(base_dir, raw_scatter, "分散布局配置")
        )
    if motion is not None and scatter is not None:
        raise InvalidParameterError("分散布局与运动轨迹不能同时使用")

    shared_fps = data.get("fps", 25.0)

    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise InvalidParameterError("批量清单缺少非空 jobs 数组")

    jobs: list[BatchJob] = []
    for i, item in enumerate(raw_jobs):
        ctx = f"jobs[{i}]"
        if not isinstance(item, dict):
            raise InvalidParameterError(f"{ctx} 必须是 JSON 对象")
        if "input" not in item or "output" not in item:
            raise InvalidParameterError(f"{ctx} 必须含 input 与 output 字段")
        job_config = (
            config_from_dict(item["config"], f"{ctx}.config")
            if "config" in item
            else None
        )
        job_motion = (
            motion_from_dict(item["motion"]) if "motion" in item else None
        )
        job_scatter = (
            scatter_from_dict(_load_json_object(base_dir, item["scatter"], "分散布局配置"))
            if "scatter" in item
            else None
        )
        jobs.append(
            BatchJob(
                input_path=_resolve(base_dir, item["input"], f"{ctx}.input"),
                output_path=_resolve(base_dir, item["output"], f"{ctx}.output"),
                config=job_config,
                motion=job_motion,
                scatter=job_scatter,
                frame_index=item.get("frame_index", 0),
                fps=item.get("fps", shared_fps),
            )
        )

    return BatchSpec(
        watermark_path=watermark_path,
        config=config,
        motion=motion,
        jobs=jobs,
        scatter=scatter,
    )
