"""M3U8 渲染与合规性测试（纯逻辑，不依赖 ffmpeg）。"""

from __future__ import annotations

import pytest

from hls_packager.errors import PlaylistError
from hls_packager.m3u8 import (
    HLS_VERSION,
    PlaylistConfig,
    compute_target_duration,
    render_playlist,
)
from hls_packager.models import SegmentInfo


def _seg(seq: int, duration: float, start: float = 0.0) -> SegmentInfo:
    return SegmentInfo(
        sequence_number=seq,
        start_time=start,
        duration=duration,
        filename=f"seg_{seq:04d}.ts",
        source_start=start,
    )


def _segments(count: int, duration: float = 4.0) -> list[SegmentInfo]:
    return [
        _seg(i, duration, start=i * duration) for i in range(count)
    ]


def test_target_duration_is_ceiling_of_max():
    segs = _segments(3, 4.2)
    assert compute_target_duration(segs) == 5
    segs_int = _segments(3, 4.0)
    assert compute_target_duration(segs_int) == 4


def test_target_duration_minimum_one():
    assert compute_target_duration([_seg(0, 0.2)]) == 1


def test_vod_playlist_contains_required_tags():
    segs = _segments(2, 4.0)
    text = render_playlist(segs, PlaylistConfig(target_duration=4))

    assert text.startswith("#EXTM3U\n")
    assert f"#EXT-X-VERSION:{HLS_VERSION}" in text
    assert "#EXT-X-TARGETDURATION:4" in text
    assert "#EXT-X-MEDIA-SEQUENCE:0" in text
    assert "#EXT-X-PLAYLIST-TYPE:VOD" in text
    assert "#EXT-X-INDEPENDENT-SEGMENTS" in text
    assert "#EXT-X-ENDLIST" in text
    # 每个分片前都有 EXTINF，且文件名跟随其后
    lines = text.strip().splitlines()
    assert "#EXTINF:4.000," in lines
    assert "seg_0000.ts" in lines and "seg_0001.ts" in lines


def test_extinf_precision_three_decimals():
    text = render_playlist(
        [_seg(0, 4.123456)], PlaylistConfig(target_duration=5)
    )
    assert "#EXTINF:4.123," in text


def test_sliding_window_keeps_last_n_and_omits_endlist():
    segs = _segments(6, 4.0)  # 序号 0..5
    text = render_playlist(
        segs, PlaylistConfig(target_duration=4, window_size=3)
    )
    lines = text.strip().splitlines()

    # 窗口只保留最后 3 片：序号 3,4,5
    assert "seg_0003.ts" in lines
    assert "seg_0000.ts" not in lines
    assert "seg_0005.ts" in lines
    # MEDIA-SEQUENCE 指向窗口首片
    assert "#EXT-X-MEDIA-SEQUENCE:3" in lines
    # 直播列表不能有 ENDLIST / PLAYLIST-TYPE
    assert "#EXT-X-ENDLIST" not in text
    assert "#EXT-X-PLAYLIST-TYPE" not in text


def test_window_larger_than_segments_lists_all():
    segs = _segments(2, 4.0)
    text = render_playlist(
        segs, PlaylistConfig(target_duration=4, window_size=10)
    )
    assert "seg_0000.ts" in text and "seg_0001.ts" in text
    assert "#EXT-X-MEDIA-SEQUENCE:0" in text


def test_custom_media_sequence():
    segs = [_seg(20, 4.0, start=80.0), _seg(21, 4.0, start=84.0)]
    text = render_playlist(segs, PlaylistConfig(target_duration=4))
    assert "#EXT-X-MEDIA-SEQUENCE:20" in text


def test_rejects_overlong_segment():
    # TARGETDURATION=4 但存在 5s 分片，违反 RFC
    segs = [_seg(0, 4.0), _seg(1, 5.2)]
    with pytest.raises(PlaylistError):
        render_playlist(segs, PlaylistConfig(target_duration=4))


def test_rejects_non_contiguous_sequence():
    segs = [_seg(0, 4.0), _seg(2, 4.0)]
    with pytest.raises(PlaylistError):
        render_playlist(segs, PlaylistConfig(target_duration=4))


def test_rejects_empty_and_bad_config():
    with pytest.raises(PlaylistError):
        render_playlist([], PlaylistConfig(target_duration=4))
    segs = _segments(1, 4.0)
    with pytest.raises(PlaylistError):
        render_playlist(segs, PlaylistConfig(target_duration=0))
    with pytest.raises(PlaylistError):
        render_playlist(segs, PlaylistConfig(target_duration=4, window_size=0))
