"""采样模块测试：端点锚定、RGB 与索引两条路径同构。"""

import pytest

from vgif.quantize import QuantizedFrame
from vgif.sampling import (
    build_frame_picks,
    sample_frames,
    sample_indexed_frames,
)


def test_sample_frames_endpoints():
    frames = [f"f{i}".encode() for i in range(10)]
    picked, idx = sample_frames(frames, 4)
    assert idx[0] == 0 and idx[-1] == 9
    assert picked[0] == frames[0] and picked[-1] == frames[9]
    assert len(picked) == 4


def test_sample_identity():
    frames = [bytes([i]) for i in range(5)]
    picked, idx = sample_frames(frames, 5)
    assert idx == [0, 1, 2, 3, 4]
    assert picked == frames


def test_sample_reuses_byte_objects_no_copy():
    frames = [bytes([i]) * 3 for i in range(6)]
    picked, idx = sample_frames(frames, 3)
    # 6→3 端点锚定：divmod(5,2)=q2,r1，首间隔多走一步 → 0、3、5。
    assert idx == [0, 3, 5]
    assert all(picked[i] is frames[j]
               for i, j in enumerate([0, 3, 5]))


def test_build_frame_picks_timeline():
    pts = [0.0, 0.1, 0.2, 0.3]
    durations = [0.1] * 4
    indices, picks = build_frame_picks(pts, durations, 2)
    assert indices == [0, 3]
    assert picks[0].start_ts == 0.0
    assert picks[1].end_ts == pytest.approx(0.4)
    # 两帧时间区间无缝拼接。
    assert picks[0].end_ts == picks[1].start_ts


def test_indexed_path_matches_rgb_mapping():
    qframes = [
        QuantizedFrame(indices=bytes([i]), width=1, height=1)
        for i in range(12)
    ]
    picked, idx = sample_indexed_frames(qframes, 5)
    raw = [bytes([i]) for i in range(12)]
    _, ref_idx = sample_frames(raw, 5)
    assert idx == ref_idx
    assert [q.indices for q in picked] == [raw[i] for i in ref_idx]


def test_indexed_path_type_guard():
    with pytest.raises(TypeError):
        sample_indexed_frames([b"not-a-qframe"], 1)
