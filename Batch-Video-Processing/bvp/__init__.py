"""Batch-Video-Processing：可复核的目录批量视频转码工具。

对外主要入口：

- :func:`bvp.core.transcode`：单文件转码核心（ffmpeg/ffprobe 子进程）；
- :func:`bvp.batch.batch_transcode`：批量编排（扫描→规划→并发→清单→复核）；
- :mod:`bvp.contract`：核心与批量层之间的稳定错误分类契约；
- :func:`bvp.verify.reconcile_manifest_files`：脱离运行上下文的清单对账。

工程边界：批量层只复用单文件核心、只按契约分类汇总，不复制转码逻辑、
不解析异常消息文本。
"""

from .contract import (
    ErrorCategory,
    SKIP_CATEGORIES,
    SkipReason,
    category_is_skip,
    classify_error,
)
from .core import (
    supported_codecs,
    supported_containers,
    transcode,
)
from .errors import (
    BinaryNotFoundError,
    EncoderUnavailableError,
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    TranscodeError,
)
from .manifest import (
    MANIFEST_VERSION,
    Manifest,
    ManifestEntry,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    default_manifest_paths,
    load_manifest,
)
from .planning import (
    EXIST_OVERWRITE,
    EXIST_SKIP,
    EXIST_SUFFIX,
    PlannedJob,
    plan_outputs,
)
from .scanner import (
    DEFAULT_VIDEO_EXTENSIONS,
    ScanProblem,
    ScanResult,
    scan_inputs,
)
from .verify import (
    ReconciliationMismatch,
    ReconciliationResult,
    reconcile_manifest,
    reconcile_manifest_files,
)
from .batch import BatchResult, batch_transcode
from .models import MediaInfo, StreamInfo, TranscodeResult
from .probe import probe_media

__all__ = [
    # 主入口
    "transcode",
    "batch_transcode",
    "probe_media",
    # 稳定契约
    "ErrorCategory",
    "SkipReason",
    "SKIP_CATEGORIES",
    "classify_error",
    "category_is_skip",
    # 批量结果与清单
    "BatchResult",
    "Manifest",
    "ManifestEntry",
    "MANIFEST_VERSION",
    "STATUS_SUCCESS",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "load_manifest",
    "default_manifest_paths",
    # 扫描与规划
    "scan_inputs",
    "ScanResult",
    "ScanProblem",
    "DEFAULT_VIDEO_EXTENSIONS",
    "plan_outputs",
    "PlannedJob",
    "EXIST_SKIP",
    "EXIST_OVERWRITE",
    "EXIST_SUFFIX",
    # 复核
    "reconcile_manifest",
    "reconcile_manifest_files",
    "ReconciliationResult",
    "ReconciliationMismatch",
    # 单文件核心数据与异常
    "MediaInfo",
    "StreamInfo",
    "TranscodeResult",
    "supported_codecs",
    "supported_containers",
    "TranscodeError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "OutputExistsError",
    "FFmpegRunError",
    "EncoderUnavailableError",
]
