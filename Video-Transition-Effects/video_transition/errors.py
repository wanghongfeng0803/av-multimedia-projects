"""转场工具的异常层级。

所有预期内的错误（探针失败、规划参数非法、管道中断）都继承
:class:`TransitionError`，调用方只需捕获这一个基类即可区分
「用户可修正的问题」与未知崩溃。
"""

from __future__ import annotations

__all__ = [
    "TransitionError",
    "ProbeError",
    "PlanError",
    "PipelineError",
]


class TransitionError(Exception):
    """转场工具全部自定义异常的基类。"""


class ProbeError(TransitionError):
    """ffprobe 无法读取素材参数，或素材缺少视频流。"""


class PlanError(TransitionError):
    """转场规划参数非法（重叠时长越界、尺寸/帧率不一致等）。"""


class PipelineError(TransitionError):
    """解码/编码管道运行失败，或实际帧数与探针结果不一致。"""
