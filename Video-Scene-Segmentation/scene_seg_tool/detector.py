"""单遍流式直方图评分与场景判定编排。

本模块只处理「灰度帧字节序列 → 差异序列 + 切点」的纯逻辑，不启动子进程、
不读文件系统，因此可以在没有 ffmpeg 的环境下完整测试。ffmpeg 解码由
:mod:`scene_seg_tool.frames` 负责，二者通过生成器接口拼接。

流式不变量
----------
任意时刻只保留 3 条直方图（前一帧、当前帧、下一帧），与视频总帧数无关，
直方图内存占用恒为 O(灰度级数)。每帧只被消费一次：

* 帧到达时统计一次直方图（O(像素数) 的 C 级扫描）；
* 与前一帧做一次 256 项 L1（O(灰度级数)）；
* 第三帧起再做一次跨越中间帧的三帧回返 L1，供闪光抑制使用。

因此核心统计复杂度为 O(帧数 × 灰度级数)（直方图统计的像素扫描另计，
由 C 级循环完成），且不存在任何形式的重复解码或回放。
"""

from __future__ import annotations

from collections.abc import Iterable

from .errors import InvalidInputError
from .frames import stream_gray_frames
from .histogram import (
    build_histogram,
    enforce_min_scene_length,
    histogram_l1_distance,
    normalize_distance,
    robust_threshold_with_source,
    suppress_flash_boundaries,
)
from .models import FrameBoundary, SceneCut, SegmentationResult
from .probe import probe_video

__all__ = [
    "StreamingHistogramScorer",
    "detect_scenes_in_frames",
    "detect_scenes",
    "build_result",
]

#: 边界判定标签
DECISION_CUT = "cut"
DECISION_FLASH = "flash"
DECISION_SUPPRESSED = "suppressed"
DECISION_BELOW = "below_threshold"


class StreamingHistogramScorer:
    """消费灰度帧字节流，单遍产出整数距离序列与三帧回返距离。

    用法::

        scorer = StreamingHistogramScorer()
        for raw_frame in frame_generator:
            scorer.push(raw_frame)
        result = scorer.snapshot()

    输出序列的第 ``i`` 项对应边界 ``i+1``（帧 i -> 帧 i+1）：

    * ``distances[i]``：相邻帧直方图 L1 距离；
    * ``returns[i]``：以边界 ``i+1`` 为进入边界、跳过中间帧 ``i+1`` 的
      三帧回返距离 ``|h[i] - h[i+2]|``；最后一条边界之后没有帧，记 -1。
    """

    def __init__(self) -> None:
        self._prev: list[int] | None = None
        self._cur: list[int] | None = None
        self._histograms: int = 0
        self.distances: list[int] = []
        self.returns: list[int] = []
        self.pixel_count: int = 0

    def push(self, frame: bytes) -> None:
        """消费一帧 gray 原始字节。"""
        hist = build_histogram(frame)
        frame_index = self._histograms
        if frame_index == 0:
            # 首帧只记录像素规模，不产生边界
            self.pixel_count = len(frame)
        else:
            self.distances.append(histogram_l1_distance(self._cur, hist))
            self.returns.append(-1)
            if self._prev is not None:
                # 帧 t 到达时 L1(h[t-2], h[t]) 跳过的是帧 t-1，
                # 属于上一条边界 t-1（序列下标 t-2），而非当前边界
                self.returns[frame_index - 2] = histogram_l1_distance(
                    self._prev, hist
                )
        self._prev = self._cur
        self._cur = hist
        self._histograms += 1

    @property
    def frame_count(self) -> int:
        return self._histograms

    def snapshot(self) -> tuple[list[int], list[int], int]:
        """返回 ``(相邻距离序列, 回返距离序列, 单帧像素数)``。"""
        return list(self.distances), list(self.returns), self.pixel_count


def detect_scenes_in_frames(
    frames: Iterable[bytes],
    width: int,
    height: int,
    fps: float,
    *,
    input_path: str = "<memory>",
    fps_source: str = "caller",
    mad_k: float = 9.0,
    fallback_threshold: float = 0.35,
    min_scene_frames: int = 1,
    flash_suppression: bool = True,
) -> SegmentationResult:
    """对内存中的灰度帧序列执行完整检测（纯逻辑入口）。

    :param frames: gray 原始帧字节的可迭代对象，每帧长度须为
                   ``width * height``。
    :param min_scene_frames: 最小场景长度（帧）；默认 1 表示不做 NMS。
    :param flash_suppression: 是否启用曝光突变（闪光）抑制。
    :raises InvalidInputError: 帧序列为空，或帧长与 ``width*height`` 不符。
    """
    scorer = StreamingHistogramScorer()
    expected = width * height
    for frame in frames:
        if expected and len(frame) != expected:
            raise InvalidInputError(
                f"灰度帧长度 {len(frame)} 与尺寸 {width}x{height} "
                f"（期望 {expected} 字节）不一致，解码流可能已损坏"
            )
        scorer.push(frame)

    return build_result(
        scorer,
        input_path=input_path,
        width=width,
        height=height,
        fps=fps,
        fps_source=fps_source,
        mad_k=mad_k,
        fallback_threshold=fallback_threshold,
        min_scene_frames=min_scene_frames,
        flash_suppression=flash_suppression,
    )


def build_result(
    scorer: StreamingHistogramScorer,
    *,
    input_path: str,
    width: int,
    height: int,
    fps: float,
    fps_source: str,
    mad_k: float,
    fallback_threshold: float,
    min_scene_frames: int,
    flash_suppression: bool,
) -> SegmentationResult:
    """根据评分器快照完成阈值计算、闪光抑制、NMS 与结果组装。

    抽成独立函数是为了让 ffmpeg 流式编排（detector 层）与内存帧编排
    共用同一段确定性判定逻辑，两条路径的结果必须一致。
    """
    frame_count = scorer.frame_count
    distances, returns, pixel_count = scorer.snapshot()
    if frame_count == 0:
        raise InvalidInputError("未解码到任何视频帧，无法进行场景分割")

    if pixel_count != width * height:
        raise InvalidInputError(
            f"实际帧像素数 {pixel_count} 与探测尺寸 {width}x{height} 不一致"
        )

    # 单帧视频没有任何边界：阈值走兜底，序列为空，切点为空
    if not distances:
        return SegmentationResult(
            input_path=input_path,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            fps_source=fps_source,
            threshold=fallback_threshold,
            threshold_source="fallback",
            boundaries=[],
            cuts=[],
        )

    scores = [normalize_distance(d, pixel_count) for d in distances]
    threshold, threshold_source = robust_threshold_with_source(
        scores, mad_k=mad_k, fallback=fallback_threshold
    )

    candidate_cuts = [
        idx + 1 for idx, score in enumerate(scores) if score >= threshold
    ]

    if flash_suppression:
        flash_set = suppress_flash_boundaries(
            distances, returns, threshold, pixel_count
        )
    else:
        flash_set = set()

    after_flash = [b for b in candidate_cuts if b not in flash_set]
    kept = enforce_min_scene_length(after_flash, distances, min_scene_frames)
    kept_set = set(kept)
    suppressed_set = set(after_flash) - kept_set

    boundaries: list[FrameBoundary] = []
    for idx, score in enumerate(scores):
        boundary_no = idx + 1
        if boundary_no in kept_set:
            decision = DECISION_CUT
        elif boundary_no in flash_set:
            decision = DECISION_FLASH
        elif boundary_no in suppressed_set:
            decision = DECISION_SUPPRESSED
        else:
            decision = DECISION_BELOW
        boundaries.append(
            FrameBoundary(
                frame_index=boundary_no,
                time_seconds=(boundary_no / fps) if fps > 0 else 0.0,
                raw_distance=distances[idx],
                score=score,
                return_distance=returns[idx],
                decision=decision,
            )
        )

    cuts = [
        SceneCut(
            frame_index=b.frame_index,
            time_seconds=b.time_seconds,
            score=b.score,
            raw_distance=b.raw_distance,
        )
        for b in boundaries
        if b.decision == DECISION_CUT
    ]

    return SegmentationResult(
        input_path=input_path,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        fps_source=fps_source,
        threshold=threshold,
        threshold_source=threshold_source,
        boundaries=boundaries,
        cuts=cuts,
        flash_boundaries=tuple(sorted(flash_set)),
        suppressed_boundaries=tuple(sorted(suppressed_set)),
    )


def detect_scenes(
    path: str,
    *,
    mad_k: float = 9.0,
    fallback_threshold: float = 0.35,
    min_scene_frames: int = 1,
    flash_suppression: bool = True,
) -> SegmentationResult:
    """对视频文件执行单遍流式场景分割（库级主入口）。

    流程严格保证「一次探测 + 一次解码」：

    1. ffprobe 只读元数据，拿到宽高与帧率（不解码帧）；
    2. ffmpeg 输出 gray 原始流，:class:`StreamingHistogramScorer` 逐帧消费，
       帧数据在内存中只存活一次迭代即被释放，不缓存任何帧或完整视频；
    3. 流结束后用累积的整数距离序列计算阈值并组装结果。

    单帧视频、无变化长视频、曝光突变等边界均由
    :func:`build_result` / :mod:`scene_seg_tool.histogram` 统一处理。

    :raises InputNotFoundError: 文件不存在。
    :raises InvalidInputError: 无视频流、零帧或帧尺寸不一致。
    :raises FFmpegRunError: 解码进程失败或流被截断。
    """
    probe = probe_video(path)
    scorer = StreamingHistogramScorer()
    for frame in stream_gray_frames(path, probe.width, probe.height):
        scorer.push(frame)

    return build_result(
        scorer,
        input_path=path,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        fps_source=probe.fps_source,
        mad_k=mad_k,
        fallback_threshold=fallback_threshold,
        min_scene_frames=min_scene_frames,
        flash_suppression=flash_suppression,
    )
