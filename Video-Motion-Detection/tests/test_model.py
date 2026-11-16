"""数据契约测试：FrameRecord 序列化往返、整段结果落盘与校验。"""

import json

import pytest

from vmotion.events import MotionEvent
from vmotion.model import (
    SCHEMA_VERSION,
    FrameRecord,
    build_result,
    load_detection_json,
    save_detection_json,
)


def _region_dict(label=1, x=2, y=3, w=8, h=6, area=48):
    return {"label": label, "x": x, "y": y, "w": w, "h": h,
            "area": area, "cx": x + w / 2, "cy": y + h / 2}


def _rec(idx, motion=False, event_id=None, regions=None):
    return FrameRecord(
        frame=idx, pts_time=idx * 0.1, motion=motion,
        regions=regions or [], stats={"gated_pixels": 0,
                                      "held_off": False},
        event_id=event_id,
    )


def test_frame_record_roundtrip():
    rec = _rec(3, motion=True, event_id=1, regions=[_region_dict()])
    d = rec.to_dict()
    back = FrameRecord.from_dict(d)
    assert back.frame == 3
    assert back.motion is True
    assert back.event_id == 1
    assert back.regions[0]["x"] == 2
    assert back.pts_time == pytest.approx(0.3)


def test_frame_record_tolerates_missing_optional_fields():
    d = {"frame": 0}
    rec = FrameRecord.from_dict(d)
    assert rec.motion is False and rec.regions == []
    assert rec.event_id is None and rec.stats == {}


def test_save_and_load_full_result(tmp_path):
    frames = [
        _rec(0),
        _rec(1, motion=True, event_id=1, regions=[_region_dict()]),
        _rec(2, motion=True, event_id=1, regions=[_region_dict()]),
        _rec(3),
    ]
    ev = MotionEvent(1, 1, 0.1)
    ev.accumulate(frames[1])
    ev.accumulate(frames[2])
    ev.close(3, 0.3)
    result = build_result(
        frames, [ev],
        meta={"source": "in.mp4", "width": 64, "height": 48},
        params={"block": 8, "gain": 3.0},
    )
    path = tmp_path / "detect.json"
    save_detection_json(result, path)
    loaded = load_detection_json(path)
    assert loaded.meta["source"] == "in.mp4"
    assert loaded.params["block"] == 8
    assert len(loaded.frames) == 4
    assert loaded.frames[2].event_id == 1
    assert loaded.events[0]["start_frame"] == 1
    assert loaded.stats["motion_frames"] == 2
    assert loaded.stats["events"] == 1
    # UTF-8 中文不转义
    raw = path.read_text(encoding="utf-8")
    assert "schema_version" in raw


def test_load_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "frames": []}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_detection_json(path)


def test_load_rejects_non_contiguous_frames(tmp_path):
    path = tmp_path / "gap.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "frames": [{"frame": 0}, {"frame": 2}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="不连续"):
        load_detection_json(path)


def test_build_result_stats_ratios():
    frames = [_rec(i, motion=(i % 2 == 1)) for i in range(4)]
    result = build_result(frames, [], {"width": 8}, {})
    assert result.stats["frames"] == 4
    assert result.stats["motion_frames"] == 2
    assert result.stats["motion_ratio"] == pytest.approx(0.5)
    assert result.stats["held_off_frames"] == 0
