"""内嵌多字幕轨：元数据解析、目标轨定位与端到端烧录的测试。

纯逻辑用例（parse_tracks/select_track）手工构造 ffprobe JSON，不依赖
ffmpeg；集成用例用 lavfi + 字幕文件现做多轨 MKV，无 ffmpeg 自动 skip。
"""

from __future__ import annotations

import json
import subprocess

import pytest

from subburn import (
    BitmapFont,
    TrackAmbiguousError,
    TrackNotFoundError,
    TrackProbeError,
    burn_track,
    parse_tracks,
    probe_subtitle_tracks,
    select_track,
)
from subburn.cli import main
from subburn.tracks import SubtitleTrack, extract_track

from .conftest import make_test_video
from .test_pipeline import bottom_region_diff, extract_frame_rgb


def _track(stream_index, sub_index, codec="subrip", language="", title="",
           default=False, forced=False, fmt="srt"):
    return SubtitleTrack(
        stream_index=stream_index, sub_index=sub_index, codec_name=codec,
        language=language, title=title, default=default, forced=forced,
        format=fmt,
    )


#: 三条轨的样例：英（默认）、中（forced）、图形轨（不可烧录）
SAMPLE_TRACKS = (
    _track(1, 0, language="eng", default=True),
    _track(2, 1, language="chi", title="简中", forced=True),
    _track(3, 2, codec="hdmv_pgs_subtitle", fmt=""),
)


class TestParseTracks:
    def test_full_metadata(self):
        data = {"streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
            {"index": 2, "codec_type": "subtitle", "codec_name": "subrip",
             "tags": {"language": "ENG", "title": "English"},
             "disposition": {"default": 1, "forced": 0}},
            {"index": 3, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "chi"},
             "disposition": {"default": 0, "forced": 1}},
        ]}
        tracks = parse_tracks(data)
        assert len(tracks) == 2
        eng, chi = tracks
        # 只有字幕流入表，相对序号按枚举顺序从 0 编
        assert (eng.stream_index, eng.sub_index) == (2, 0)
        assert (chi.stream_index, chi.sub_index) == (3, 1)
        assert eng.language == "eng"          # 小写归一化
        assert eng.title == "English"
        assert eng.default is True and eng.forced is False
        assert chi.forced is True and chi.default is False
        assert eng.format == "srt" and chi.format == "ass"

    def test_missing_tags_and_disposition_are_tolerated(self):
        data = {"streams": [
            {"index": 5, "codec_type": "subtitle", "codec_name": "mov_text"},
        ]}
        (t,) = parse_tracks(data)
        assert t.language == "" and t.title == ""
        assert t.default is False and t.forced is False
        assert t.format == "srt"              # mov_text 转 SRT 烧录

    def test_und_language_normalized_to_empty(self):
        data = {"streams": [
            {"index": 1, "codec_type": "subtitle", "codec_name": "subrip",
             "tags": {"language": "und"}},
        ]}
        assert parse_tracks(data)[0].language == ""

    def test_unknown_codec_marked_not_burnable(self):
        data = {"streams": [
            {"index": 1, "codec_type": "subtitle",
             "codec_name": "hdmv_pgs_subtitle"},
        ]}
        (t,) = parse_tracks(data)
        assert t.format == "" and t.burnable is False

    def test_stream_without_index_is_skipped(self):
        data = {"streams": [
            {"codec_type": "subtitle", "codec_name": "subrip"},
            {"index": "oops", "codec_type": "subtitle",
             "codec_name": "subrip"},
            {"index": 4, "codec_type": "subtitle", "codec_name": "subrip"},
        ]}
        tracks = parse_tracks(data)
        assert len(tracks) == 1 and tracks[0].stream_index == 4
        assert tracks[0].sub_index == 0       # 相对序号不受跳过影响

    def test_empty_or_malformed_container(self):
        assert parse_tracks({}) == ()
        assert parse_tracks({"streams": "not-a-list"}) == ()
        assert parse_tracks({"streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"}]}) == ()


class TestSelectTrack:
    def test_empty_track_list(self):
        with pytest.raises(TrackNotFoundError, match="不含任何内嵌字幕轨"):
            select_track([])

    def test_no_criteria_unique_default_wins(self):
        assert select_track(SAMPLE_TRACKS).language == "eng"

    def test_no_criteria_single_track(self):
        assert select_track(SAMPLE_TRACKS[:1]) is SAMPLE_TRACKS[0]

    def test_no_criteria_ambiguous_without_default(self):
        tracks = (SAMPLE_TRACKS[1], SAMPLE_TRACKS[2])
        with pytest.raises(TrackAmbiguousError) as ei:
            select_track(tracks)
        assert ei.value.tracks == tracks      # 携带可用轨供诊断

    def test_by_stream_index(self):
        assert select_track(SAMPLE_TRACKS, stream_index=2).language == "chi"

    def test_by_sub_index(self):
        assert select_track(SAMPLE_TRACKS, sub_index=1).language == "chi"

    def test_by_language_case_insensitive(self):
        assert select_track(SAMPLE_TRACKS, language="CHI").sub_index == 1

    def test_by_title(self):
        assert select_track(SAMPLE_TRACKS, title="简中").language == "chi"

    def test_combined_criteria_intersection(self):
        assert select_track(
            SAMPLE_TRACKS, language="chi", sub_index=1).stream_index == 2
        with pytest.raises(TrackNotFoundError):
            select_track(SAMPLE_TRACKS, language="chi", sub_index=0)

    def test_language_not_found_lists_available(self):
        with pytest.raises(TrackNotFoundError) as ei:
            select_track(SAMPLE_TRACKS, language="jpn")
        msg = str(ei.value)
        assert "语言=jpn" in msg and "lang=eng" in msg and "lang=chi" in msg

    def test_stream_index_not_found(self):
        with pytest.raises(TrackNotFoundError):
            select_track(SAMPLE_TRACKS, stream_index=99)

    def test_ambiguous_language(self):
        tracks = (
            _track(1, 0, language="eng"),
            _track(2, 1, language="eng", title="SDH"),
        )
        with pytest.raises(TrackAmbiguousError) as ei:
            select_track(tracks, language="eng")
        assert len(ei.value.tracks) == 2
        # 加标题收窄后唯一命中
        assert select_track(tracks, language="eng", title="sdh").title == "SDH"


class TestExtractTrack:
    def test_graphic_track_rejected_before_ffmpeg(self, tmp_path):
        with pytest.raises(Exception, match="图形字幕"):
            extract_track("whatever.mkv", SAMPLE_TRACKS[2], str(tmp_path))


# ---------------------------------------------------------------------------
# ffmpeg 集成：现做多轨 MKV，验证枚举 -> 选择 -> 抽取 -> 烧录全链路
# ---------------------------------------------------------------------------

_SRT_EN = "1\n00:00:00,200 --> 00:00:01,200\nENG TRACK\n"
_SRT_ZH = "1\n00:00:00,200 --> 00:00:01,200\nCHI TRACK\n"


def make_multi_track_mkv(path, video):
    """一条视频轨 + 两条 SRT 字幕轨（eng 默认 / chi 带标题）的 MKV。"""
    eng = path.parent / "eng.srt"
    chi = path.parent / "chi.srt"
    eng.write_text(_SRT_EN, encoding="utf-8")
    chi.write_text(_SRT_ZH, encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video), "-i", str(eng), "-i", str(chi),
        "-map", "0:v", "-map", "1:s", "-map", "2:s",
        "-c:v", "copy", "-c:s", "srt",
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:1", "language=chi",
        "-metadata:s:s:1", "title=简中",
        "-disposition:s:0", "default", "-disposition:s:1", "0",
        str(path),
    ], check=True)
    return path


@pytest.mark.usefixtures("ffmpeg")
class TestTrackPipelineIntegration:
    @pytest.fixture()
    def mkv(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_test_video(video)
        return make_multi_track_mkv(tmp_path / "multi.mkv", video)

    def test_probe_enumerates_all_tracks(self, mkv):
        tracks = probe_subtitle_tracks(str(mkv))
        assert len(tracks) == 2
        assert [t.language for t in tracks] == ["eng", "chi"]
        assert tracks[0].default is True and tracks[1].default is False
        assert tracks[1].title == "简中"
        assert all(t.format == "srt" and t.burnable for t in tracks)
        # 绝对流索引连续且不同于相对序号语义
        assert {t.stream_index for t in tracks} == {1, 2}
        assert [t.sub_index for t in tracks] == [0, 1]

    def test_extract_selected_track(self, mkv, tmp_path):
        track = select_track(probe_subtitle_tracks(str(mkv)), language="chi")
        out = extract_track(str(mkv), track, str(tmp_path))
        with open(out, encoding="utf-8-sig") as f:
            assert "CHI TRACK" in f.read()

    def test_probe_failure_raises_probe_error(self, tmp_path):
        bad = tmp_path / "not-a-video.mkv"
        bad.write_bytes(b"garbage")
        with pytest.raises(TrackProbeError):
            probe_subtitle_tracks(str(bad))

    def test_missing_input_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            probe_subtitle_tracks(str(tmp_path / "nope.mkv"))

    def test_container_without_subtitles(self, tmp_path):
        video = tmp_path / "plain.mp4"
        make_test_video(video)
        assert probe_subtitle_tracks(str(video)) == ()
        with pytest.raises(TrackNotFoundError):
            burn_track(str(video), str(tmp_path / "o.mp4"))

    def test_burn_default_track(self, mkv, tmp_path):
        out = tmp_path / "out.mp4"
        result = burn_track(str(mkv), str(out), font=BitmapFont(2))
        assert result.frames_written == 50
        assert "lang=eng" in result.track
        # 默认轨（eng）确实被烧进画面底部
        _, _, src = extract_frame_rgb(str(mkv), 0.7, tmp_path, "src")
        _, _, burned = extract_frame_rgb(str(out), 0.7, tmp_path, "burn")
        assert bottom_region_diff(src, burned, 320, 240) > 500

    def test_burn_track_selected_by_language(self, mkv, tmp_path):
        out = tmp_path / "out.mp4"
        result = burn_track(str(mkv), str(out), language="CHI",
                            font=BitmapFont(2), overwrite=True)
        assert "lang=chi" in result.track and result.cues == 1

    def test_burn_track_selected_by_stream_index(self, mkv, tmp_path):
        out = tmp_path / "out.mp4"
        result = burn_track(str(mkv), str(out), stream_index=2,
                            font=BitmapFont(2))
        assert "lang=chi" in result.track

    def test_burn_track_missing_language(self, mkv, tmp_path):
        with pytest.raises(TrackNotFoundError):
            burn_track(str(mkv), str(tmp_path / "o.mp4"), language="jpn")

    def test_burn_track_ambiguous_without_default(self, tmp_path):
        # 两条非默认轨 + 不给条件 -> 歧义
        video = tmp_path / "v.mp4"
        make_test_video(video)
        a = tmp_path / "a.srt"
        b = tmp_path / "b.srt"
        a.write_text(_SRT_EN, encoding="utf-8")
        b.write_text(_SRT_ZH, encoding="utf-8")
        mkv = tmp_path / "nodflt.mkv"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-i", str(a), "-i", str(b),
            "-map", "0:v", "-map", "1:s", "-map", "2:s",
            "-c:v", "copy", "-c:s", "srt",
            "-disposition:s:0", "0", "-disposition:s:1", "0", str(mkv),
        ], check=True)
        with pytest.raises(TrackAmbiguousError):
            burn_track(str(mkv), str(tmp_path / "o.mp4"))


@pytest.mark.usefixtures("ffmpeg")
class TestTrackCli:
    @pytest.fixture()
    def mkv(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_test_video(video)
        return make_multi_track_mkv(tmp_path / "multi.mkv", video)

    def test_tracks_listing(self, mkv, capsys):
        rc = main(["tracks", str(mkv)])
        out = capsys.readouterr().out
        assert rc == 0 and "lang=eng" in out and "lang=chi" in out

    def test_tracks_listing_json(self, mkv, capsys):
        rc = main(["tracks", str(mkv), "--json"])
        data = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert [t["language"] for t in data["tracks"]] == ["eng", "chi"]
        assert data["tracks"][1]["title"] == "简中"

    def test_tracks_on_bad_file_returns_2(self, tmp_path, capsys):
        bad = tmp_path / "x.mkv"
        bad.write_bytes(b"junk")
        assert main(["tracks", str(bad)]) == 2

    def test_burn_track_via_main(self, mkv, tmp_path, capsys):
        out = tmp_path / "o.mp4"
        rc = main(["burn-track", str(mkv), "-o", str(out),
                   "--lang", "chi", "-y", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert rc == 0 and out.exists()
        assert "lang=chi" in data["track"]

    def test_burn_track_missing_language_returns_2(self, mkv, tmp_path):
        rc = main(["burn-track", str(mkv), "-o", str(tmp_path / "o.mp4"),
                   "--lang", "jpn"])
        assert rc == 2

    def test_burn_track_refuse_overwrite_returns_1(self, mkv, tmp_path):
        out = tmp_path / "o.mp4"
        out.write_bytes(b"existing")
        rc = main(["burn-track", str(mkv), "-o", str(out), "--lang", "eng"])
        assert rc == 1
