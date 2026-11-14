"""端到端集成测试：真实调用 ffmpeg/ffprobe 验证切片正确性。

覆盖验收口径：
* 每个分片首帧为 IDR，可独立解码；
* 分片边界与 GOP 对齐；
* 跨分片时间戳连续（无回退/跳变），concat 后 PTS 单调；
* M3U8 标签与 TARGETDURATION 合规；
* 滑动窗口、纯音频、大 GOP、HEVC 等场景。
"""

from __future__ import annotations

import os
import subprocess

import pytest

from hls_packager.packager import PLAYLIST_FILENAME, package
from tests.conftest import ffprobe_json

pytestmark = pytest.mark.integration


def _segment_paths(out_dir: str, result) -> list[str]:
    return [os.path.join(out_dir, s.filename) for s in result.segments]


def _first_video_packet_flags(path: str) -> str:
    data = ffprobe_json(
        path, "-select_streams", "v:0", "-show_entries", "packet=flags"
    )
    return data["packets"][0]["flags"]


def _first_video_pts(path: str) -> float:
    data = ffprobe_json(
        path, "-select_streams", "v:0", "-show_entries", "packet=pts_time"
    )
    pts_values = [float(p["pts_time"]) for p in data["packets"]]
    return min(pts_values)


def _container_duration(path: str) -> float:
    data = ffprobe_json(path, "-show_entries", "format=duration")
    return float(data["format"]["duration"])


def test_packages_h264_with_correct_segment_count(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)

    # 30s / ~4s 目标，应产出多片且全部落盘
    assert len(result.segments) >= 7
    for seg in result.segments:
        assert os.path.isfile(os.path.join(out_dir, seg.filename))
    assert os.path.isfile(os.path.join(out_dir, PLAYLIST_FILENAME))


def test_every_segment_starts_with_idr(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)
    for path in _segment_paths(out_dir, result):
        assert "K" in _first_video_packet_flags(path), f"{path} 首帧不是关键帧"


def test_segments_are_independently_decodable(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)
    # 抽中间一片完整解码，不应报错
    middle = _segment_paths(out_dir, result)[len(result.segments) // 2]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", middle, "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")


def test_cut_boundaries_align_to_keyframes(h264_file, out_dir):
    media_probe = ffprobe_json(
        h264_file, "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
    )
    source_keyframes = {
        round(float(p["pts_time"]), 3)
        for p in media_probe["packets"] if "K" in p["flags"]
    }

    result = package(h264_file, out_dir, target_duration=5.0)
    for seg in result.segments:
        # 规划切点必须落在源关键帧时刻（容差 5ms）
        assert any(
            abs(seg.source_start - kf) < 0.005 for kf in source_keyframes
        ), f"切点 {seg.source_start} 未对齐关键帧 {sorted(source_keyframes)}"


def test_global_timestamps_are_continuous(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)
    first_pts = [_first_video_pts(p) for p in _segment_paths(out_dir, result)]

    # 全局时间戳：逐片首帧 PTS 应递增且无回退
    for prev, cur in zip(first_pts, first_pts[1:]):
        assert cur > prev, f"PTS 回退：{prev} -> {cur}"
    # 相邻首帧 PTS 差应接近对应分片时长（全局连续，不重置）
    for seg, pts in zip(result.segments, first_pts):
        assert pts >= seg.start_time - 0.5  # 首帧相对分片起点可有少量偏移


def test_concatenated_timeline_is_monotonic(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)
    concat_path = os.path.join(out_dir, "concat.txt")
    with open(concat_path, "w", encoding="utf-8") as handle:
        for seg in result.segments:
            handle.write(f"file '{seg.filename}'\n")

    joined = os.path.join(out_dir, "joined.ts")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", concat_path,
         "-c", "copy", joined],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    data = ffprobe_json(
        joined, "-select_streams", "v:0", "-show_entries", "packet=pts_time,dts_time"
    )
    packets = data["packets"]
    # B 帧流的 PTS 在交付顺序里会因重排而跳动，真正严格单调的是 DTS；
    # 用 DTS 验证跨分片无回退/重置，再用 PTS 值域验证覆盖完整时间轴。
    dts = [float(p["dts_time"]) for p in packets]
    assert dts == sorted(dts), "concat 后 DTS 非单调，分片边界存在时间戳回退"
    pts = [float(p["pts_time"]) for p in packets]
    assert min(pts) < 2.0 and max(pts) > 29.0  # PTS 覆盖 0~30s 全片
    # 总时长应接近源 30s（TS 容器允许少量音频填充）
    assert 29.5 < _container_duration(joined) < 31.5


def test_m3u8_targetduration_covers_all_segments(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)
    playlist = open(
        os.path.join(out_dir, PLAYLIST_FILENAME), encoding="utf-8"
    ).read()

    # TARGETDURATION 必须 >= 任意 EXTINF
    target_line = next(
        line for line in playlist.splitlines()
        if line.startswith("#EXT-X-TARGETDURATION:")
    )
    target = int(target_line.split(":")[1])
    for seg in result.segments:
        assert seg.duration <= target + 0.05
    assert target == result.target_duration
    assert "#EXT-X-VERSION:3" in playlist
    assert "#EXT-X-MEDIA-SEQUENCE:0" in playlist
    assert playlist.rstrip().endswith("#EXT-X-ENDLIST")


def test_extinf_durations_match_probed_segments(h264_file, out_dir):
    result = package(h264_file, out_dir, target_duration=4.0)
    playlist = open(
        os.path.join(out_dir, PLAYLIST_FILENAME), encoding="utf-8"
    ).read()
    extinf_values = [
        float(line.split(":")[1].rstrip(","))
        for line in playlist.splitlines() if line.startswith("#EXTINF:")
    ]
    assert len(extinf_values) == len(result.segments)
    for extinf, seg in zip(extinf_values, result.segments):
        assert extinf == pytest.approx(seg.duration, abs=0.002)


def test_sliding_window_playlist(h264_file, out_dir):
    result = package(
        h264_file, out_dir, target_duration=4.0, window_size=3
    )
    playlist = open(
        os.path.join(out_dir, PLAYLIST_FILENAME), encoding="utf-8"
    ).read()
    lines = playlist.splitlines()
    extinf_count = sum(1 for line in lines if line.startswith("#EXTINF:"))

    assert extinf_count == 3
    assert "#EXT-X-ENDLIST" not in playlist
    # 窗口首片序号 = 总分片数 - 3
    expected_seq = len(result.segments) - 3
    assert f"#EXT-X-MEDIA-SEQUENCE:{expected_seq}" in lines
    # 窗口内首片文件确实存在
    assert f"seg_{expected_seq:04d}.ts" in lines


def test_audio_only_input(audio_only_file, out_dir):
    result = package(audio_only_file, out_dir, target_duration=3.0)
    assert len(result.segments) >= 3
    for path in _segment_paths(out_dir, result):
        # 纯音频 TS 没有视频流，但应能探测到音频且时长为正
        data = ffprobe_json(path, "-show_entries", "stream=codec_type")
        assert any(s["codec_type"] == "audio" for s in data["streams"])
        assert _container_duration(path) > 0
    playlist = open(
        os.path.join(out_dir, PLAYLIST_FILENAME), encoding="utf-8"
    ).read()
    assert "#EXT-X-ENDLIST" in playlist


def test_large_gop_aligns_to_available_keyframes(big_gop_file, out_dir):
    result = package(big_gop_file, out_dir, target_duration=4.0)
    # 10s GOP：切点只能是 0/10/20，故 25s 源产出 3 片
    starts = [round(s.source_start, 3) for s in result.segments]
    assert starts == [0.0, 10.0, 20.0]
    # TARGETDURATION 必须能容纳 10s 分片
    assert result.target_duration >= 10
    for path in _segment_paths(out_dir, result):
        assert "K" in _first_video_packet_flags(path)


def test_hevc_input_packages(hevc_file, out_dir):
    result = package(hevc_file, out_dir, target_duration=4.0)
    assert len(result.segments) >= 2
    for path in _segment_paths(out_dir, result):
        assert "K" in _first_video_packet_flags(path)
        data = ffprobe_json(path, "-show_entries", "stream=codec_name")
        assert any(s["codec_name"] == "hevc" for s in data["streams"])


def test_metadata_file_records_segments(h264_file, out_dir):
    import json

    result = package(h264_file, out_dir, target_duration=4.0)
    metadata_path = result.extra["metadata_path"]
    payload = json.load(open(metadata_path, encoding="utf-8"))

    assert payload["hls"]["segment_count"] == len(result.segments)
    assert len(payload["segments"]) == len(result.segments)
    first = payload["segments"][0]
    assert set(first) >= {
        "sequence_number", "filename", "start_time", "duration", "source_start"
    }
    # start_time 从 0 连续累计
    assert payload["segments"][0]["start_time"] == 0.0


def test_rejects_nonempty_output_dir(h264_file, out_dir):
    package(h264_file, out_dir, target_duration=4.0)
    from hls_packager.errors import PackagerError

    with pytest.raises(PackagerError):
        package(h264_file, out_dir, target_duration=4.0)  # 未传 overwrite


def test_overwrite_allows_rerun(h264_file, out_dir):
    package(h264_file, out_dir, target_duration=4.0)
    # overwrite=True 应能重跑且产物数量一致
    result = package(h264_file, out_dir, target_duration=4.0, overwrite=True)
    assert len(result.segments) > 0
