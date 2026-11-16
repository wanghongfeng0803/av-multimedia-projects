"""轨迹数据契约测试：序列化往返、统计汇总、丢失事件与校验。"""

import json

import pytest

from vtrack.model import (
    LOST,
    TRACKED,
    SCHEMA_VERSION,
    FrameRecord,
    build_track_result,
    load_track_json,
    save_track_json,
)


def _rec(idx, status=TRACKED, score=5.0, confirmed=True, x=1, y=2):
    return FrameRecord(
        frame=idx, pts_time=idx * 0.1, status=status,
        bbox=(x, y, 8, 6), score=score, confirmed=confirmed,
        search=(0, 0, 10, 10),
    )


def test_frame_record_roundtrip():
    rec = _rec(3, status=LOST, score=42.0, confirmed=False)
    d = rec.to_dict()
    back = FrameRecord.from_dict(d)
    assert back.frame == 3
    assert back.status == LOST
    assert back.bbox == (1, 2, 8, 6)
    assert back.confirmed is False
    assert back.search == (0, 0, 10, 10)
    assert back.pts_time == pytest.approx(0.3)


def test_frame_record_from_dict_tolerates_missing_search():
    d = {"frame": 0, "status": TRACKED, "x": 1, "y": 2, "w": 8, "h": 6,
         "score": 0.0}
    rec = FrameRecord.from_dict(d)
    assert rec.search == (1, 2, 1, 2)
    assert rec.confirmed is True


def test_save_and_load_full_result(tmp_path):
    frames = [_rec(0), _rec(1, LOST, 55.0, False), _rec(2, LOST, 60.0, False),
              _rec(3, score=4.0)]
    from vtrack.model import LossEvent
    events = [LossEvent(1, 3, 0.1, 0.3, 2, False)]
    result = build_track_result(
        frames, events,
        meta={"source": "in.mp4", "width": 64, "height": 48},
        params={"method": "sad", "threshold": 30.0}, fps="10/1",
    )
    path = tmp_path / "track.json"
    save_track_json(result, path)
    loaded = load_track_json(path)
    assert loaded.meta["source"] == "in.mp4"
    assert loaded.params["method"] == "sad"
    assert len(loaded.frames) == 4
    assert [f.frame for f in loaded.frames] == [0, 1, 2, 3]
    assert loaded.frames[1].status == LOST
    assert len(loaded.loss_events) == 1
    ev = loaded.loss_events[0]
    assert (ev.lost_frame, ev.recovered_frame, ev.frames_lost) == (1, 3, 2)
    assert ev.end_of_stream is False


def test_stats_aggregation():
    frames = [_rec(0, score=2.0), _rec(1, score=8.0),
              _rec(2, LOST, 99.0, False), _rec(3, score=6.0)]
    result = build_track_result(frames, [], {}, {}, "10/1")
    stats = result.stats
    assert stats["frames"] == 4
    assert stats["tracked_frames"] == 3
    assert stats["lost_frames"] == 1
    assert stats["lost_ratio"] == pytest.approx(0.25)
    assert stats["loss_events"] == 0
    assert stats["still_lost_at_end"] is False
    assert stats["score_min"] == 2.0 and stats["score_max"] == 8.0
    assert stats["score_mean"] == pytest.approx(16.0 / 3)


def test_still_lost_at_end_flag():
    frames = [_rec(0), _rec(1, LOST, 99.0, False)]
    result = build_track_result(frames, [], {}, {}, "10/1")
    assert result.stats["still_lost_at_end"] is True
    assert result.stats["score_mean"] == 5.0


def test_empty_track_stats_safe():
    result = build_track_result([], [], {}, {}, "10/1")
    stats = result.stats
    assert stats["frames"] == 0 and stats["lost_ratio"] == 0.0
    assert stats["score_min"] is None


def test_end_of_stream_event_serializes_without_recovery():
    from vtrack.model import LossEvent
    ev = LossEvent(5, None, 0.5, None, 3, True)
    d = ev.to_dict()
    assert "recovered_frame" not in d
    assert d["end_of_stream"] is True
    assert d["frames_lost"] == 3


def test_load_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "frames": []}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_track_json(path)


def test_load_rejects_non_contiguous_frames(tmp_path):
    path = tmp_path / "gap.json"
    data = {
        "schema_version": SCHEMA_VERSION,
        "frames": [
            {"frame": 0, "status": TRACKED, "x": 0, "y": 0, "w": 4, "h": 4},
            {"frame": 2, "status": TRACKED, "x": 0, "y": 0, "w": 4, "h": 4},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="不连续"):
        load_track_json(path)


def test_load_rejects_illegal_status(tmp_path):
    path = tmp_path / "st.json"
    data = {
        "schema_version": SCHEMA_VERSION,
        "frames": [
            {"frame": 0, "status": "occluded", "x": 0, "y": 0,
             "w": 4, "h": 4},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="状态非法"):
        load_track_json(path)


def test_json_is_utf8_and_readable(tmp_path):
    result = build_track_result([_rec(0)], [],
                                {"target_id": "目标甲"}, {}, "10/1")
    path = tmp_path / "cn.json"
    save_track_json(result, path)
    raw = path.read_bytes()
    assert "目标甲".encode("utf-8") in raw
    loaded = load_track_json(path)
    assert loaded.meta["target_id"] == "目标甲"
