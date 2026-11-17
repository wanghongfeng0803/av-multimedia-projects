"""ABR 主列表与跨码率对齐的纯逻辑测试（不依赖 ffmpeg）。

用轻量假对象构造切割结果与码率规格，覆盖：默认阶梯、主列表渲染、
峰值码率计算、跨码率对齐复核的接受/拒绝路径。
"""

import pytest

from vhls.abr import (
    AbrError,
    Rendition,
    VariantResult,
    _check_alignment,
    _measure_bandwidth,
    default_ladder,
    render_master,
)
from vhls.cut import SegmentResult
from vhls.playlist import PlaylistResult
from vhls.timeline import SegmentSpec


def _spec(seq, duration_ms=2000, filename=None):
    return SegmentSpec(
        sequence=seq,
        filename=filename or f"seg_{seq:05d}.ts",
        start_frame=seq * 50, end_frame=(seq + 1) * 50,
        start_ms=seq * duration_ms, end_ms=(seq + 1) * duration_ms,
        planned_duration_ms=duration_ms,
        is_last=False, is_short=False,
    )


class _PseudoPath:
    """不落盘的路径桩：模拟 is_file/stat/name。"""

    def __init__(self, name, size=1000):
        self.name = name
        self._size = size

    def is_file(self):
        return True

    def stat(self):
        class _S:
            st_size = self._size
        return _S()

    def __fspath__(self):
        return self.name

    def __str__(self):
        return self.name


def _result(seq, duration_ms=2000, size=1000):
    spec = _spec(seq, duration_ms)
    return SegmentResult(
        spec=spec, path=_PseudoPath(spec.filename, size=size),
        actual_duration_ms=duration_ms,
    )


def _variant(name, bandwidth, width=640, height=360, results=None):
    if results is None:
        results = (_result(0),)
    playlist = PlaylistResult(
        path=_PseudoPath(f"{name}/index.m3u8"),
        target_duration=2, segment_count=len(results),
        media_sequence=0, last_segment_short=False,
        last_segment_ms=results[-1].actual_duration_ms,
        total_ms=sum(r.actual_duration_ms for r in results),
    )
    return VariantResult(
        rendition=Rendition(name=name, height=height, video_bitrate=bandwidth),
        playlist=playlist, results=results,
        bandwidth=bandwidth, width=width, height=height,
    )


class TestDefaultLadder:
    def test_1080p_source_gets_full_ladder(self):
        ladder = default_ladder(1080)
        assert [r.name for r in ladder] == ["360p", "480p", "720p", "1080p"]
        assert all(r.height <= 1080 for r in ladder)

    def test_720p_source_excludes_1080p(self):
        ladder = default_ladder(720)
        assert [r.name for r in ladder] == ["360p", "480p", "720p"]

    def test_480p_source_excludes_higher(self):
        ladder = default_ladder(480)
        assert [r.name for r in ladder] == ["360p", "480p"]

    def test_tiny_source_falls_back_to_native(self):
        ladder = default_ladder(240)
        assert len(ladder) == 1
        assert ladder[0].height == 240
        assert ladder[0].name == "240p"

    def test_bitrates_are_positive(self):
        for r in default_ladder(1080):
            assert r.video_bitrate > 0


class TestRenderMaster:
    def test_header_and_version(self):
        text = render_master((_variant("360p", 400_000),))
        lines = text.strip().split("\n")
        assert lines[0] == "#EXTM3U"
        assert any(l.startswith("#EXT-X-VERSION:") for l in lines)

    def test_stream_inf_format(self):
        text = render_master((
            _variant("360p", 400_000, width=640, height=360),
        ))
        assert "#EXT-X-STREAM-INF:BANDWIDTH=400000,RESOLUTION=640x360" in text
        assert "360p/index.m3u8" in text

    def test_sorted_by_bandwidth_ascending(self):
        variants = (
            _variant("720p", 2_500_000),
            _variant("360p", 400_000),
            _variant("480p", 800_000),
        )
        text = render_master(variants)
        lines = [l for l in text.splitlines()
                 if l.startswith("#EXT-X-STREAM-INF")]
        bandwidths = [
            int(l.split("BANDWIDTH=")[1].split(",")[0]) for l in lines
        ]
        assert bandwidths == sorted(bandwidths)

    def test_resolution_matches_variant(self):
        variants = (
            _variant("360p", 400_000, width=640, height=360),
            _variant("720p", 2_500_000, width=1280, height=720),
        )
        text = render_master(variants)
        assert "RESOLUTION=640x360" in text
        assert "RESOLUTION=1280x720" in text

    def test_uri_is_relative_path(self):
        text = render_master((_variant("480p", 800_000),))
        uris = [l for l in text.splitlines()
                if l and not l.startswith("#")]
        assert uris == ["480p/index.m3u8"]


class TestMeasureBandwidth:
    def test_peak_segment_bitrate(self):
        # 两片：片0 1000B/2s=4000bps，片1 2000B/2s=8000bps → 峰值 8000。
        results = (
            _result(0, duration_ms=2000, size=1000),
            _result(1, duration_ms=2000, size=2000),
        )
        assert _measure_bandwidth(results) == 8000

    def test_single_segment(self):
        results = (_result(0, duration_ms=2000, size=5000),)
        # 5000B * 8 / 2s = 20000bps
        assert _measure_bandwidth(results) == 20_000

    def test_shorter_segment_higher_bitrate(self):
        # 同样字节数，时长更短的片码率更高。
        results = (
            _result(0, duration_ms=2000, size=1000),
            _result(1, duration_ms=1000, size=1000),
        )
        assert _measure_bandwidth(results) == 8000

    def test_zero_duration_rejected(self):
        results = (_result(0, duration_ms=0, size=1000),)
        with pytest.raises(AbrError, match="时长非正"):
            _measure_bandwidth(results)


class TestCheckAlignment:
    def test_identical_durations_pass(self):
        a = (_result(0, 2000), _result(1, 2000), _result(2, 1080))
        b = (_result(0, 2000), _result(1, 2000), _result(2, 1080))
        _check_alignment((a, b), tolerance_ms=40)  # 不抛即通过

    def test_small_drift_within_tolerance_passes(self):
        a = (_result(0, 2000), _result(1, 2000))
        b = (_result(0, 2000), _result(1, 2020))
        _check_alignment((a, b), tolerance_ms=40)

    def test_drift_beyond_tolerance_rejected(self):
        a = (_result(0, 2000), _result(1, 2000))
        b = (_result(0, 2000), _result(1, 2100))
        with pytest.raises(AbrError, match="对齐被破坏"):
            _check_alignment((a, b), tolerance_ms=40)

    def test_segment_count_mismatch_rejected(self):
        a = (_result(0, 2000), _result(1, 2000))
        b = (_result(0, 2000),)
        with pytest.raises(AbrError, match="分片数不一致"):
            _check_alignment((a, b), tolerance_ms=40)

    def test_single_variant_no_check(self):
        a = (_result(0, 2000),)
        _check_alignment((a,), tolerance_ms=40)  # 单路不检查
