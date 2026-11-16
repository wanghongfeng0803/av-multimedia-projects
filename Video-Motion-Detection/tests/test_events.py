"""事件记录器测试：状态边沿组装、峰值统计、未闭合事件。"""

from vmotion.detector import FrameDetection
from vmotion.events import EventRecorder, union_bbox


def _det(frame, motion, regions=None, pts=None):
    return FrameDetection(
        frame=frame,
        pts_time=frame * 0.1 if pts is None else pts,
        mask=b"", regions=regions or [], motion=motion,
        stats={"gated_pixels": 1 if motion else 0},
    )


def _region(x, y, w, h, area):
    from vmotion.morphology import Region
    return Region(1, (x, y, w, h), area, (x + w / 2, y + h / 2))


def test_no_motion_no_events():
    rec = EventRecorder()
    for i in range(5):
        assert rec.update(_det(i, False)) is None
    assert rec.finish(4, 0.4) == []


def test_single_closed_event_on_falling_edge():
    rec = EventRecorder()
    eid = []
    for i in range(6):
        motion = 2 <= i <= 4
        eid.append(rec.update(_det(
            i, motion, [_region(0, 0, 8, 8, 64)] if motion else [])))
    events = rec.finish(5, 0.5)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_id == 1
    assert ev.start_frame == 2 and ev.end_frame == 5
    assert ev.start_pts == 0.2 and ev.end_pts == 0.5
    assert ev.motion_frames == 3
    assert ev.end_of_stream is False
    # 上升沿前无事件 id，运动帧拿到 id 1，下降帧回到 None
    assert eid == [None, None, 1, 1, 1, None]


def test_event_accumulates_regions_and_peak():
    rec = EventRecorder()
    rec.update(_det(0, False))
    rec.update(_det(1, True, [_region(0, 0, 10, 10, 100)]))
    rec.update(_det(2, True, [_region(20, 20, 12, 12, 144),
                              _region(40, 40, 4, 4, 16)]))
    rec.update(_det(3, False))
    ev = rec.finish(3, 0.3)[0]
    assert ev.region_count == 3
    assert ev.peak_area == 144 and ev.peak_frame == 2
    # 并集框覆盖两个区域的外接范围（(40,40)+4 右下角为 44）
    assert ev.bbox == (0, 0, 44, 44)


def test_union_bbox_helpers():
    assert union_bbox(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert union_bbox((0, 0, 10, 10), None) == (0, 0, 10, 10)
    assert union_bbox((0, 0, 10, 10), (5, 8, 10, 10)) == (0, 0, 15, 18)


def test_unclosed_event_at_end_of_stream():
    rec = EventRecorder()
    rec.update(_det(0, False))
    rec.update(_det(1, True, [_region(0, 0, 8, 8, 64)]))
    rec.update(_det(2, True, [_region(0, 0, 8, 8, 64)]))
    events = rec.finish(2, 0.2)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_frame == 1
    assert ev.end_frame is None and ev.end_pts is None
    assert ev.end_of_stream is True
    assert ev.motion_frames == 2


def test_two_separate_events_get_sequential_ids():
    rec = EventRecorder()
    seq = [False, True, True, False, False, True, False]
    for i, m in enumerate(seq):
        rec.update(_det(i, m, [_region(0, 0, 8, 8, 64)] if m else []))
    events = rec.finish(6, 0.6)
    assert [e.event_id for e in events] == [1, 2]
    assert events[0].start_frame == 1 and events[1].start_frame == 5
