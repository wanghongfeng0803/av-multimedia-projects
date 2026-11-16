"""运动事件记录：只消费检测结果的区域，按状态边沿组装事件区间。

本模块不做任何检测与绘图，输入是 :class:`~vmotion.detector.FrameDetection`
序列，输出是 :class:`MotionEvent` 列表——检测/标记/事件三模块解耦中
「事件只消费区域落盘」的一环。事件沿检测器滞后状态机的**状态边沿**
组装：

- 上升沿（``motion`` 由假变真）开启事件；
- 下降沿（由真变假）闭合事件，事件覆盖 ``start_frame..end_frame-1``；
- 视频结束仍未闭合则由 :meth:`EventRecorder.finish` 补一个
  ``end_of_stream=True`` 的开区间事件。

事件内逐帧累计：运动帧数、区域总数、峰值区域面积及其帧号、所有区域
外接框的并集框，供后续背景建模或告警分级直接使用，无需回读整份 JSON。
"""

__all__ = ["MotionEvent", "EventRecorder", "union_bbox"]


def union_bbox(a, b):
    """两个 ``(x, y, w, h)`` 外接框求并集框；``None`` 透传另一个。"""
    if a is None:
        return tuple(b)
    if b is None:
        return tuple(a)
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


class MotionEvent:
    """一段连续运动区间（上升沿 → 下降沿前一帧）。"""

    __slots__ = (
        "event_id", "start_frame", "end_frame", "start_pts", "end_pts",
        "motion_frames", "region_count", "peak_frame", "peak_area",
        "bbox", "end_of_stream",
    )

    def __init__(self, event_id, start_frame, start_pts):
        self.event_id = event_id
        self.start_frame = start_frame
        self.end_frame = None
        self.start_pts = start_pts
        self.end_pts = None
        self.motion_frames = 0
        self.region_count = 0
        self.peak_frame = start_frame
        self.peak_area = 0
        self.bbox = None
        self.end_of_stream = False

    def accumulate(self, detection):
        """用一帧检测结果累计事件统计（仅在运动态帧调用）。

        区域既可能是 :class:`~vmotion.morphology.Region` 对象（检测
        直连），也可能是 ``to_dict`` 后的扁平 dict（经数据契约回放），
        两种形态都接受，保证事件模块只依赖区域几何这一最小结构。
        """
        self.motion_frames += 1
        self.region_count += len(detection.regions)
        for region in detection.regions:
            if hasattr(region, "bbox"):
                bbox = region.bbox
                area = region.area
            else:
                bbox = (region["x"], region["y"], region["w"], region["h"])
                area = region.get("area", 0)
            self.bbox = union_bbox(self.bbox, bbox)
            if area > self.peak_area:
                self.peak_area = area
                self.peak_frame = detection.frame

    def close(self, end_frame, end_pts, end_of_stream=False):
        """闭合事件。``end_frame`` 为下降沿帧号（区间不含该帧）。"""
        self.end_frame = end_frame
        self.end_pts = end_pts
        self.end_of_stream = end_of_stream

    def to_dict(self):
        d = {
            "event_id": self.event_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_pts": round(self.start_pts, 6),
            "end_pts": round(self.end_pts, 6) if self.end_pts is not None
            else None,
            "duration": (round(self.end_pts - self.start_pts, 6)
                         if self.end_pts is not None else None),
            "motion_frames": self.motion_frames,
            "region_count": self.region_count,
            "peak_frame": self.peak_frame,
            "peak_area": self.peak_area,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "end_of_stream": self.end_of_stream,
        }
        return d


class EventRecorder:
    """逐帧消费 :class:`FrameDetection`，维护事件区间的状态机。

    用法::

        rec = EventRecorder()
        for det in detections:
            rec.update(det)
        events = rec.finish(last_frame_index, last_pts)
    """

    def __init__(self):
        self._events = []
        self._open = None
        self._prev_motion = False

    @property
    def events(self):
        """已闭合事件列表（不含尚未闭合的当前事件）。"""
        return list(self._events)

    def update(self, detection):
        """喂入一帧检测结果，返回本帧关联的事件 id（无则 ``None``）。

        返回值供渲染侧直接在画面上标注事件编号，保证标记视频与事件
        列表逐帧可对照。
        """
        if detection.motion and not self._prev_motion:
            self._open = MotionEvent(
                event_id=len(self._events) + 1,
                start_frame=detection.frame,
                start_pts=detection.pts_time,
            )
        if detection.motion and self._open is not None:
            self._open.accumulate(detection)
        if not detection.motion and self._prev_motion and self._open is not None:
            self._open.close(detection.frame, detection.pts_time)
            self._events.append(self._open)
            self._open = None
        self._prev_motion = detection.motion
        return self._open.event_id if self._open is not None else None

    def finish(self, last_frame, last_pts):
        """流结束收尾：闭合仍开放的事件，返回全部事件列表。"""
        if self._open is not None:
            self._open.close(None, None, end_of_stream=True)
            self._open.end_frame = None
            self._open.end_pts = None
            self._events.append(self._open)
            self._open = None
        self._prev_motion = False
        return list(self._events)
