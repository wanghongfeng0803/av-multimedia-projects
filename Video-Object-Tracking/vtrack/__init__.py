"""vtrack：逐帧模板匹配目标追踪（零三方依赖，纯标准库 + ffmpeg 子进程）。

追踪与渲染严格分离：

- 追踪（:mod:`vtrack.tracker` / :mod:`vtrack.pipeline`）逐帧做模板匹配，
  产出轨迹 JSON（:mod:`vtrack.model`），匹配分数低于阈值即标记丢失、
  绝不乱跟，恢复后在扩大的搜索区内重新搜索并经连续确认再回到追踪；
- 渲染（:mod:`vtrack.render`）只消费轨迹 JSON 画框出视频，可对同一轨迹
  反复重渲染，也供后续光流追踪、多目标追踪复用同一数据契约。

模板匹配（:mod:`vtrack.template`）提供 SAD / SSD / ZNCC 三种分数，
全部纯 Python 字节平面实现，SAD/SSD 走积分图 O(1) 窗和，ZNCC 走
粗-精两级搜索控成本。
"""

from .template import (
    METHOD_SAD,
    METHOD_SSD,
    METHOD_ZNCC,
    TEMPLATE_METHODS,
    higher_is_better,
    match_template,
    integral_image,
    window_sums,
    zncc_at,
    best_match,
    best_match_coarse,
)
from .model import (
    TRACKED,
    LOST,
    FrameRecord,
    LossEvent,
    TrackResult,
    build_track_result,
    save_track_json,
    load_track_json,
)
from .tracker import TemplateTracker
from .pipeline import (
    probe,
    track_video,
    render_video,
    read_initial_template,
    PipelineError,
)

__all__ = [
    # template（模板匹配）
    "METHOD_SAD",
    "METHOD_SSD",
    "METHOD_ZNCC",
    "TEMPLATE_METHODS",
    "higher_is_better",
    "match_template",
    "integral_image",
    "window_sums",
    "zncc_at",
    "best_match",
    "best_match_coarse",
    # model（轨迹数据契约）
    "TRACKED",
    "LOST",
    "FrameRecord",
    "LossEvent",
    "TrackResult",
    "build_track_result",
    "save_track_json",
    "load_track_json",
    # tracker（追踪状态机）
    "TemplateTracker",
    # pipeline（ffmpeg 追踪 / 渲染流水线）
    "probe",
    "track_video",
    "render_video",
    "read_initial_template",
    "PipelineError",
]
