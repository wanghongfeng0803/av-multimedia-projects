"""单文件核心与批量层之间的**稳定错误分类契约**。

本模块是整个工程里最不允许随意改动的文件：批量层（:mod:`bvp.batch`、
:mod:`bvp.manifest`）只允许依据这里定义的 :class:`ErrorCategory` 做汇总，
**不得解析异常消息文本、不得依赖异常类名的字符串形式**。这样单文件核心
内部无论怎么改写错误措辞、调整实现，只要异常携带的分类不变，批量清单的
口径就保持稳定、可复核。

契约要点：

- 每个核心异常都带一个 :class:`ErrorCategory` 分类（见 :mod:`bvp.errors`）；
- 分类集合是闭合枚举，新增分类必须显式加在这里，而不是在批量层临时发明；
- :data:`SKIP_CATEGORIES` 明确划定「跳过」与「失败」的边界：落在该集合内
  的分类记为跳过（非视频/无法识别/输出已存在等），其余分类一律记为失败；
- :func:`classify_error` 是批量层读取分类的**唯一入口**：核心异常走
  ``category`` 属性，任何核心契约之外的未知异常（如 OSError、第三方异常）
  一律归入 :data:`ErrorCategory.INTERNAL_ERROR`，按失败处理，绝不静默吞掉。
"""

from __future__ import annotations

import enum

__all__ = [
    "ErrorCategory",
    "SKIP_CATEGORIES",
    "SkipReason",
    "classify_error",
    "category_is_skip",
]


class ErrorCategory(enum.Enum):
    """单文件核心抛出错误的闭合分类集合（稳定契约，取值字符串不可复用改写）。"""

    #: 系统中找不到 ffmpeg / ffprobe 可执行文件
    BINARY_MISSING = "binary_missing"
    #: 输入路径不存在、不是普通文件或无法访问（权限/IO）
    INPUT_NOT_FOUND = "input_not_found"
    #: 输入存在但不是可转码视频：非媒体、无法识别、无视频流、0 字节
    NOT_A_VIDEO = "not_a_video"
    #: 输出文件已存在且未允许覆盖（批量层可据此记为跳过而非失败）
    OUTPUT_EXISTS = "output_exists"
    #: ffmpeg / ffprobe 执行失败或产物复核不过
    TRANSCODE_FAILED = "transcode_failed"
    #: 本机 ffmpeg 不支持所需编码器
    ENCODER_UNAVAILABLE = "encoder_unavailable"
    #: 核心契约之外的未预期错误（兜底分类，按失败处理）
    INTERNAL_ERROR = "internal_error"


#: 归类为「跳过」的分类集合。
#:
#: 跳过的语义是「该输入本就不产生/不必产生转码产物」，与「尝试转码但失败」
#: 严格区分。当前只有两类：输入不是可转码视频，以及输出已存在且策略要求跳过。
#: 该集合是批量层划分 failed/skipped 的**唯一依据**。
SKIP_CATEGORIES: frozenset[ErrorCategory] = frozenset({
    ErrorCategory.NOT_A_VIDEO,
    ErrorCategory.OUTPUT_EXISTS,
})


class SkipReason(enum.Enum):
    """跳过条目的原因码（写入清单，取值稳定、可被外部程序枚举）。"""

    #: 非视频或内容无法被 ffprobe 识别（对应 NOT_A_VIDEO）
    NON_VIDEO = "non_video"
    #: 输出文件已存在，按 on_exist=skip 策略跳过（对应 OUTPUT_EXISTS）
    OUTPUT_EXISTS = "output_exists"
    #: 扫描阶段无法进入/读取该路径（对应 INPUT_NOT_FOUND 的扫描期分支）
    INACCESSIBLE = "inaccessible"


def classify_error(exc: BaseException) -> ErrorCategory:
    """把任意异常映射为 :class:`ErrorCategory`（批量层分类的唯一入口）。

    - 核心异常（:class:`bvp.errors.TranscodeError` 子类）读取其稳定的
      ``category`` 属性；
    - 核心契约之外的异常一律归入 :data:`ErrorCategory.INTERNAL_ERROR`，
      保证批量层永远拿到一个合法分类，不会因为陌生异常而崩溃或漏记。

    批量层调用本函数即视为遵守契约：**不允许**在批量层出现
    ``isinstance`` 之外基于消息文本的判断。
    """
    category = getattr(exc, "category", None)
    if isinstance(category, ErrorCategory):
        return category
    return ErrorCategory.INTERNAL_ERROR


def category_is_skip(category: ErrorCategory) -> bool:
    """判断某分类是否属于跳过口径（批量层 failed/skipped 分界）。"""
    return category in SKIP_CATEGORIES
