"""vmotion：纯标准库实现的视频运动检测（ffmpeg 经子进程调用）。

两种检测模式共享同一套门控/形态学/事件机制，差异只在参考模型：

- **帧差模式**（:mod:`vmotion.detector`）：分块帧差 + 局部 MAD 自适应
  阈值 + 光照突变 holdoff，参考是逐帧滚动的上一帧，捕捉变化沿；
- **背景建模模式**（:mod:`vmotion.background`）：逐块运行背景 EMA，
  预热期累计平均防误报、背景块缓慢跟踪光照、前景块停留超阈值后吸收
  进背景，回答「这块区域现在算不算背景」。

三个模块严格解耦，各自只认明确的数据结构：

- 检测（:mod:`vmotion.detector` / :mod:`vmotion.background`）：逐帧只
  产出二值掩码与 :class:`~vmotion.morphology.Region` 区域列表，不做
  可视化与落盘；
- 标记（:mod:`vmotion.render`）：只消费检测 JSON 画区域框与状态行，
  可对同一份 JSON 反复重渲染；
- 事件（:mod:`vmotion.events` / :mod:`vmotion.model`）：只消费区域，
  沿状态边沿组装运动区间并落盘。

光照处理的核心取舍：不做全局归一化（避免把车灯、阴影等局部光照
变化一棒子打死），而是用局部阈值让全局亮度变化自行抬高全帧门限被
滤除，再用显式的突变 holdoff 兜住大跳变。两种模式的同输入量化对比
见 :mod:`vmotion.comparison`。
"""

from .morphology import (
    Region,
    erode,
    dilate,
    open_mask,
    close_mask,
    connected_components,
)
from .detection_common import (
    GateConfig,
    Hysteresis,
    block_means,
    gate_block_mask,
    refine_mask,
    upsample_block_mask,
)
from .detector import (
    MotionConfig,
    DEFAULT_CONFIG,
    FrameDetection,
    MotionDetector,
)
from .background import (
    BackgroundConfig,
    DEFAULT_BACKGROUND_CONFIG,
    BackgroundDetector,
)
from .comparison import (
    COMPARE_SCHEMA_VERSION,
    compare_detections,
    compare_video,
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
from .pipeline import (
    detect_video,
    detect_background_video,
    render_video,
    PipelineError,
)

__all__ = [
    # morphology（形态学与连通域）
    "Region",
    "erode",
    "dilate",
    "open_mask",
    "close_mask",
    "connected_components",
    # detection_common（两模式共享机制）
    "GateConfig",
    "Hysteresis",
    "block_means",
    "gate_block_mask",
    "refine_mask",
    "upsample_block_mask",
    # detector（帧差模式检测核心）
    "MotionConfig",
    "DEFAULT_CONFIG",
    "FrameDetection",
    "MotionDetector",
    # background（背景建模模式）
    "BackgroundConfig",
    "DEFAULT_BACKGROUND_CONFIG",
    "BackgroundDetector",
    # comparison（双模式量化对比）
    "COMPARE_SCHEMA_VERSION",
    "compare_detections",
    "compare_video",
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
    "detect_background_video",
    "render_video",
    "PipelineError",
]
