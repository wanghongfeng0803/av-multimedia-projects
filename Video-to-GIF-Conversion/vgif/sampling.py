"""帧率采样：端点锚定 Bresenham 取帧，纯丢帧/重复、不插值造帧。

采样是抽帧之后的纯函数阶段：输入完整源帧序列与目标输出帧数，输出
重排后的帧序列，时间信息（每帧延迟）由 :mod:`vgif.timing` 单独规划，
因此「取哪些帧」与「每帧播多久」共用同一套端点锚定整数误差法，
画面节奏与时序严格一致，调色板 / 量化的任何改动都不会波及它。

提供两条同构路径：

- :func:`sample_frames`：rgb24 字节帧（缩放前或缩放后）；
- :func:`sample_indexed_frames`：已量化的 :class:`QuantizedFrame`，
  供量化产物改帧率复用时零成本重采样。
"""

from __future__ import annotations

from .quantize import QuantizedFrame
from .timing import FramePick, frame_pick_indices

__all__ = ["sample_frames", "sample_indexed_frames", "build_frame_picks"]


def build_frame_picks(pts: list, durations: list,
                      n_out: int) -> tuple:
    """按源帧 PTS 构造每个输出帧的时间区间与源帧下标。

    :param pts: 源帧 PTS 列表（秒）。
    :param durations: 源帧持续时长列表（秒），与 ``pts`` 等长。
    :param n_out: 输出帧数。
    :return: ``(indices, [FramePick, ...])``，下标列表与
        :func:`vgif.timing.frame_pick_indices` 结果一致；FramePick 的
        时间区间按输出帧在源时间轴上的均分切点计算。
    """
    n_src = len(pts)
    indices = frame_pick_indices(n_src, n_out)
    total_duration = pts[-1] + durations[-1]
    picks = []
    for j, src_idx in enumerate(indices):
        start = total_duration * j / n_out
        end = total_duration * (j + 1) / n_out
        picks.append(FramePick(index=src_idx, start_ts=start, end_ts=end))
    return indices, picks


def sample_frames(frames: list, n_out: int) -> tuple:
    """对 rgb24 字节帧序列做端点锚定采样。

    :return: ``(sampled_frames, indices)``，前者是复用原字节对象的
        新列表（不拷贝像素），后者是对应的源帧下标。
    """
    indices = frame_pick_indices(len(frames), n_out)
    return [frames[i] for i in indices], indices


def sample_indexed_frames(qframes: list, n_out: int) -> tuple:
    """对量化帧序列做端点锚定采样（索引域，零颜色损失）。

    :return: ``(sampled_qframes, indices)``。
    """
    for q in qframes:
        if not isinstance(q, QuantizedFrame):
            raise TypeError("sample_indexed_frames 要求 QuantizedFrame 序列")
    indices = frame_pick_indices(len(qframes), n_out)
    return [qframes[i] for i in indices], indices
