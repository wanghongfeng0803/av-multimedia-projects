"""vmotion：纯标准库实现的视频运动检测（ffmpeg 经子进程调用）。

三个模块严格解耦，各自只认明确的数据结构：

- 检测（:mod:`vmotion.detector`）：分块帧差 + 局部 MAD 自适应阈值 +
  光照突变 holdoff，逐帧只产出二值掩码与
  :class:`~vmotion.morphology.Region` 区域列表，不做可视化与落盘；
- 标记（:mod:`vmotion.render`）：只消费检测 JSON 画区域框与状态行，
  可对同一份 JSON 反复重渲染；
- 事件（:mod:`vmotion.events` / :mod:`vmotion.model`）：只消费区域，
  沿状态边沿组装运动区间并落盘，区域几何同时是后续背景建模的复用
  单元。

光照处理的核心取舍：不做全局归一化（避免把车灯、阴影等局部光照
变化一棒子打死），而是用局部阈值让全局亮度变化自行抬高全帧门限被
滤除，再用显式的突变 holdoff 兜住大跳变。
"""

from .morphology import (
    Region,
    erode,
    dilate,
    open_mask,
    close_mask,
    connected_components,
)
from .detector import (
    MotionConfig,
    DEFAULT_CONFIG,
    FrameDetection,
    MotionDetector,
)
from .events import MotionEvent, EventRecorder, union_bbox
from .model import (
    SCHEMA_VERSION,
    FrameRecord,
    DetectionResult,
    build_result,
    save_detection_json,
    load_detection_json,
)
from .pipeline import detect_video, render_video, PipelineError

__all__ = [
    # morphology（形态学与连通域）
    "Region",
    "erode",
    "dilate",
    "open_mask",
    "close_mask",
    "connected_components",
    # detector（检测核心）
    "MotionConfig",
    "DEFAULT_CONFIG",
    "FrameDetection",
    "MotionDetector",
    # events（事件记录）
    "MotionEvent",
    "EventRecorder",
    "union_bbox",
    # model（数据契约）
    "SCHEMA_VERSION",
    "FrameRecord",
    "DetectionResult",
    "build_result",
    "save_detection_json",
    "load_detection_json",
    # pipeline（ffmpeg 流水线）
    "detect_video",
    "render_video",
    "PipelineError",
]
