"""视频帧时间轴与字幕显示调度。

时间对齐是烧录正确性的核心，这里把语义钉死：

* 第 ``i`` 帧（从 0 开始）的**展示时刻**为 ``t_i = i / fps`` 秒。解码管道
  输出的是恒定帧率（CFR）流，帧序号即时间轴坐标，不依赖容器时间戳；
* 字幕在**半开区间** ``[start_ms, end_ms)`` 内有效：帧时刻落在起点上要显示，
  恰好落在终点上不显示。半开语义保证相邻字幕在边界帧不会同帧闪现；
* 所有比较统一在「毫秒」尺度上做整数/浮点混合比较，字幕时间本身是整数
  毫秒，帧时刻按 ``i*1000/fps`` 现算，不做预先取整，避免 23.976fps 这类
  非整数帧率上的累积偏差。

调度器用「入场游标 + 在场列表」两个指针增量推进，整段视频跑下来每条
字幕只进入、退出各一次，总复杂度 O(字幕数 + 帧数)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .srt import Cue

__all__ = ["FrameTimeline", "SubtitleScheduler", "frame_time_ms"]


def frame_time_ms(frame_index: int, fps: float) -> float:
    """第 frame_index 帧展示时刻（毫秒）。CFR 下与帧序号线性对应。"""
    return frame_index * 1000.0 / fps


@dataclass(frozen=True)
class FrameTimeline:
    """帧时间轴参数。``frame_count`` 未知（纯管道流）时可为 0。"""

    fps: float
    width: int
    height: int
    frame_count: int = 0

    def time_ms(self, frame_index: int) -> float:
        return frame_time_ms(frame_index, self.fps)


class SubtitleScheduler:
    """按帧查询当前应显示的字幕集合。

    用法::

        scheduler = SubtitleScheduler(cues, fps=25.0)
        for i in range(frame_count):
            for cue in scheduler.active_cues_at(i):
                ...  # 把 cue.text 画到第 i 帧上

    字幕按开始时间排序；同一时刻有多条字幕时，返回顺序即自下而上的
    堆叠顺序（先开始的在更下面），与主流播放器的重叠处理一致。
    """

    def __init__(self, cues: list[Cue], fps: float):
        if fps <= 0:
            raise ValueError(f"fps 必须为正数，收到 {fps!r}")
        self.fps = fps
        # 防御性排序：真实 SRT 一般有序，但不把正确性押在文件守规矩上
        self._cues = sorted(cues, key=lambda c: (c.start_ms, c.end_ms, c.index))
        self._enter_cursor = 0     # 尚未入场的第一条字幕下标
        self._active: list[Cue] = []  # 当前在场字幕（保持 cues 的相对顺序）
        self._last_frame = -1

    def active_cues_at(self, frame_index: int) -> list[Cue]:
        """返回第 frame_index 帧时刻处于半开区间内的字幕列表。

        要求 frame_index 单调递增（逐帧调用）；乱序访问会抛异常，因为
        游标式调度只支持前向扫描。
        """
        if frame_index < 0:
            raise ValueError(f"帧序号不能为负: {frame_index}")
        if frame_index <= self._last_frame:
            raise ValueError(
                f"帧序号必须单调递增，上次 {self._last_frame}，本次 {frame_index}"
            )
        self._last_frame = frame_index
        now_ms = frame_time_ms(frame_index, self.fps)

        # 1) 入场：start_ms <= 当前时刻。游标只前进，不回退。
        while (
            self._enter_cursor < len(self._cues)
            and self._cues[self._enter_cursor].start_ms <= now_ms
        ):
            self._active.append(self._cues[self._enter_cursor])
            self._enter_cursor += 1

        # 2) 出场：半开区间 —— 终点恰好等于帧时刻时不再显示。
        self._active = [c for c in self._active if c.end_ms > now_ms]
        return self._active
