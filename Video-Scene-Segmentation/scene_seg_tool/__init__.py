"""基于相邻帧灰度直方图差异的场景切换检测工具。

对外主要入口：
- :func:`scene_seg_tool.detector.detect_scenes`：对视频文件做单遍流式检测
- :func:`scene_seg_tool.detector.detect_scenes_in_frames`：对内存灰度帧序列检测
- :class:`scene_seg_tool.detector.StreamingHistogramScorer`：流式评分器
- :func:`scene_seg_tool.histogram.build_histogram` /
  :func:`histogram_l1_distance`：自研直方图统计与整数 L1 差异度量
- :func:`scene_seg_tool.probe.probe_video`：ffprobe 元数据探测
"""

from .errors import (
    SceneSegError,
    BinaryNotFoundError,
    InputNotFoundError,
    InvalidInputError,
    FFmpegRunError,
)
from .histogram import (
    GRAY_LEVELS,
    DEFAULT_MAD_K,
    FALLBACK_THRESHOLD,
    FLASH_RATIO,
    THRESHOLD_ROBUST,
    THRESHOLD_GAP_MIDPOINT,
    THRESHOLD_FALLBACK,
    build_histogram,
    histogram_l1_distance,
    normalize_distance,
    median,
    median_absolute_deviation,
    robust_threshold,
    robust_threshold_with_source,
    suppress_flash_boundaries,
    enforce_min_scene_length,
)
from .models import FrameBoundary, SceneCut, SegmentationResult
from .detector import (
    StreamingHistogramScorer,
    detect_scenes_in_frames,
    detect_scenes,
    build_result,
)
from .probe import VideoProbe, probe_video, parse_frame_rate, DEFAULT_FPS
from .frames import stream_gray_frames, read_exact_frame
from .output import (
    render_json_report,
    render_boundaries_csv,
    write_json_report,
    write_boundaries_csv,
    render_summary,
)

__all__ = [
    # 主入口
    "detect_scenes",
    "detect_scenes_in_frames",
    "StreamingHistogramScorer",
    "build_result",
    # 直方图与差异度量
    "GRAY_LEVELS",
    "DEFAULT_MAD_K",
    "FALLBACK_THRESHOLD",
    "FLASH_RATIO",
    "THRESHOLD_ROBUST",
    "THRESHOLD_GAP_MIDPOINT",
    "THRESHOLD_FALLBACK",
    "build_histogram",
    "histogram_l1_distance",
    "normalize_distance",
    "median",
    "median_absolute_deviation",
    "robust_threshold",
    "robust_threshold_with_source",
    "suppress_flash_boundaries",
    "enforce_min_scene_length",
    # 数据模型
    "FrameBoundary",
    "SceneCut",
    "SegmentationResult",
    # 探测与解码
    "VideoProbe",
    "probe_video",
    "parse_frame_rate",
    "DEFAULT_FPS",
    "stream_gray_frames",
    "read_exact_frame",
    # 序列化
    "render_json_report",
    "render_boundaries_csv",
    "write_json_report",
    "write_boundaries_csv",
    "render_summary",
    # 异常
    "SceneSegError",
    "BinaryNotFoundError",
    "InputNotFoundError",
    "InvalidInputError",
    "FFmpegRunError",
]
