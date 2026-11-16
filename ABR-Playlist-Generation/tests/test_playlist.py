"""播放列表测试：EXTINF 舍入、TARGETDURATION 对齐、序号连续与原子写入。

用轻量假对象构造计划/切割结果，不触碰 ffmpeg；只在原子写盘用例里
落临时目录。
"""

import pytest

from vhls.cut import SegmentResult
from vhls.playlist import (
    PLAYLIST_VERSION,
    PlaylistError,
    render_playlist,
    validate_against_plan,
    write_playlist,
)
from vhls.probe import FrameInfo, SourceInfo
from vhls.timeline import SegmentSpec, build_timeline


def _source(n_frames=50, frame_ms=40):
    frames = tuple(
        FrameInfo(
            index=i,
            pts_time=i * frame_ms / 1000.0,
            duration_time=frame_ms / 1000.0,
        )
        for i in range(n_frames)
    )
    return SourceInfo(
        path="in.mp4", width=32, height=32, fps=25.0,
        frames=frames, has_audio=False,
    )


def _fake_result(spec, actual_ms=None, tmp_path=None, exists=True,
                 empty=False):
    """构造与真实 SegmentResult 同构的测试对象（含真实临时文件）。"""
    if actual_ms is None:
        actual_ms = spec.planned_duration_ms
    if tmp_path is not None:
        path = tmp_path / spec.filename
        if exists:
            path.write_bytes(b"" if empty else b"fake-ts-data")
    else:
        path = _PseudoPath(spec.filename, exists=exists, empty=empty)
    return SegmentResult(
        spec=spec, path=path, actual_duration_ms=actual_ms,
    )


class _PseudoPath:
    """不落盘的路径桩：模拟 is_file/stat/name。"""

    def __init__(self, name, exists=True, empty=False):
        self.name = name
        self._exists = exists
        self._empty = empty

    def is_file(self):
        return self._exists

    def stat(self):
        class _S:
            st_size = 0 if self._empty else 123
        return _S()

    def __fspath__(self):
        return self.name

    def __str__(self):
        return self.name


@pytest.fixture
def plan_and_results(tmp_path):
    source = _source(n_frames=127)  # 25fps，5.08s，目标 2s → 3 片带短尾
    plan = build_timeline(source, 2.0)
    results = tuple(
        _fake_result(spec, tmp_path=tmp_path) for spec in plan.segments
    )
    return plan, results


class TestRenderPlaylist:
    def test_header_and_tags(self, plan_and_results):
        plan, results = plan_and_results
        text = render_playlist(plan, results)
        lines = text.strip().split("\n")
        assert lines[0] == "#EXTM3U"
        assert f"#EXT-X-VERSION:{PLAYLIST_VERSION}" in lines
        assert "#EXT-X-MEDIA-SEQUENCE:0" in lines
        assert "#EXT-X-PLAYLIST-TYPE:VOD" in lines
        assert lines[-1] == "#EXT-X-ENDLIST"

    def test_targetduration_matches_worst_ceil(self, plan_and_results):
        plan, results = plan_and_results
        text = render_playlist(plan, results)
        worst_ms = max(r.actual_duration_ms for r in results)
        expected = -((-worst_ms) // 1000)
        assert f"#EXT-X-TARGETDURATION:{expected}" in text
        # 每个 EXTINF 都不得超过 TARGETDURATION。
        extinfs = [
            float(line.split(":")[1].rstrip(","))
            for line in text.splitlines() if line.startswith("#EXTINF")
        ]
        assert all(d <= expected for d in extinfs)

    def test_extinf_values_match_actual_ms(self, plan_and_results):
        plan, results = plan_and_results
        text = render_playlist(plan, results)
        extinfs = [
            line for line in text.splitlines() if line.startswith("#EXTINF")
        ]
        assert len(extinfs) == len(results)
        for line, result in zip(extinfs, results):
            expected = f"#EXTINF:{result.actual_duration_ms / 1000:.3f},"
            assert line == expected

    def test_segment_uris_in_sequence(self, plan_and_results):
        plan, results = plan_and_results
        text = render_playlist(plan, results)
        uris = [
            line for line in text.splitlines()
            if line and not line.startswith("#")
        ]
        assert uris == [r.spec.filename for r in results]
        assert uris == [f"seg_{i:05d}.ts" for i in range(len(uris))]

    def test_short_tail_extinf_is_accurate_not_padded(self, tmp_path):
        source = _source(n_frames=127)
        plan = build_timeline(source, 2.0)
        results = tuple(
            _fake_result(spec, tmp_path=tmp_path) for spec in plan.segments
        )
        text = render_playlist(plan, results)
        last_extinf = [
            line for line in text.splitlines() if line.startswith("#EXTINF")
        ][-1]
        # 尾片 27 帧 * 40ms = 1080ms，必须如实写 1.080 而非补成 2.000。
        assert last_extinf == "#EXTINF:1.080,"
        assert results[-1].spec.is_short

    def test_float_rounding_three_decimals(self, tmp_path):
        source = _source(n_frames=25)
        plan = build_timeline(source, 1.0)
        # 模拟实测 3333ms 的分片（验证浮点舍入口径）。
        results = (_fake_result(plan.segments[0], actual_ms=3333,
                                tmp_path=tmp_path),)
        text = render_playlist(plan, results)
        assert "#EXTINF:3.333," in text


class TestValidation:
    def test_valid_plan_passes(self, plan_and_results):
        plan, results = plan_and_results
        validate_against_plan(plan, results)  # 不抛即通过

    def test_rejects_count_mismatch(self, plan_and_results):
        plan, results = plan_and_results
        with pytest.raises(PlaylistError):
            validate_against_plan(plan, results[:-1])

    def test_rejects_disordered_sequence(self, plan_and_results):
        plan, results = plan_and_results
        swapped = (results[1], results[0], *results[2:])
        with pytest.raises(PlaylistError):
            validate_against_plan(plan, swapped)

    def test_rejects_missing_segment_file(self, tmp_path):
        source = _source(n_frames=50)
        plan = build_timeline(source, 1.0)
        results = (
            _fake_result(plan.segments[0], tmp_path=tmp_path),
            _fake_result(plan.segments[1], tmp_path=tmp_path, exists=False),
        )
        with pytest.raises(PlaylistError):
            validate_against_plan(plan, results)

    def test_rejects_empty_segment_file(self, tmp_path):
        source = _source(n_frames=50)
        plan = build_timeline(source, 1.0)
        results = (
            _fake_result(plan.segments[0], tmp_path=tmp_path),
            _fake_result(plan.segments[1], tmp_path=tmp_path, empty=True),
        )
        with pytest.raises(PlaylistError):
            validate_against_plan(plan, results)

    def test_rejects_filename_mismatch(self, tmp_path):
        source = _source(n_frames=50)
        plan = build_timeline(source, 1.0)
        good = _fake_result(plan.segments[0], tmp_path=tmp_path)
        bad_spec = SegmentSpec(
            sequence=1, filename="rogue.ts",
            start_frame=25, end_frame=50, start_ms=1000, end_ms=2000,
            planned_duration_ms=1000, is_last=True, is_short=False,
        )
        bad = _fake_result(bad_spec, tmp_path=tmp_path)
        with pytest.raises(PlaylistError):
            validate_against_plan(plan, (good, bad))


class TestWritePlaylist:
    def test_atomic_write_and_roundtrip_content(self, plan_and_results,
                                                tmp_path):
        plan, results = plan_and_results
        out = tmp_path / "nested" / "index.m3u8"
        summary = write_playlist(plan, results, out)
        assert out.is_file()
        # 磁盘内容与渲染文本逐字节一致（LF 行尾、末尾换行）。
        assert out.read_bytes() == render_playlist(
            plan, results
        ).encode("utf-8")
        assert summary.segment_count == len(results)
        assert summary.media_sequence == 0
        assert summary.last_segment_short is True
        assert summary.last_segment_ms == results[-1].actual_duration_ms
        assert summary.total_ms == sum(r.actual_duration_ms for r in results)
        # 临时文件已清理。
        assert not list(tmp_path.glob("**/*.tmp"))

    def test_summary_target_duration(self, plan_and_results, tmp_path):
        plan, results = plan_and_results
        summary = write_playlist(
            plan, results, tmp_path / "index.m3u8"
        )
        worst = max(r.actual_duration_ms for r in results)
        assert summary.target_duration == -((-worst) // 1000)

    def test_validation_failure_leaves_no_new_file(self, tmp_path):
        source = _source(n_frames=50)
        plan = build_timeline(source, 1.0)
        results = (
            _fake_result(plan.segments[0], tmp_path=tmp_path),
            _fake_result(plan.segments[1], tmp_path=tmp_path, exists=False),
        )
        out = tmp_path / "index.m3u8"
        with pytest.raises(PlaylistError):
            write_playlist(plan, results, out)
        assert not out.exists()
