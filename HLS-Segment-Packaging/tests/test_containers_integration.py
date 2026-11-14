"""多容器 / 命名模板 / 对齐模式的端到端集成测试（真实 ffmpeg/ffprobe）。

与 test_integration.py 同一验收口径：分片落盘、首帧 IDR、可独立解码、
播放列表版本与容器匹配。无 ffmpeg 时整个模块自动 skip。
"""

from __future__ import annotations

import os
import subprocess

import pytest

from hls_packager.containers import get_container
from hls_packager.packager import PLAYLIST_FILENAME, package
from hls_packager.planner import ALIGN_BEFORE
from tests.conftest import ffprobe_json

pytestmark = pytest.mark.integration


def _segment_paths(out_dir: str, result) -> list[str]:
    return [os.path.join(out_dir, s.filename) for s in result.segments]


def _first_video_packet_flags(path: str) -> str:
    data = ffprobe_json(
        path, "-select_streams", "v:0", "-show_entries", "packet=flags"
    )
    return data["packets"][0]["flags"]


def _container_duration(path: str) -> float:
    data = ffprobe_json(path, "-show_entries", "format=duration")
    return float(data["format"]["duration"])


def _read_playlist(out_dir: str) -> str:
    with open(os.path.join(out_dir, PLAYLIST_FILENAME), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------- 容器格式
def test_fmp4_packages_with_version_seven(h264_file, out_dir):
    result = package(
        h264_file, out_dir, target_duration=4.0, container="fmp4"
    )
    # 扩展名与落盘
    assert all(s.filename.endswith(".m4s") for s in result.segments)
    for path in _segment_paths(out_dir, result):
        assert os.path.isfile(path) and os.path.getsize(path) > 0
        # ftyp 盒子是 ISO-BMFF 的起始签名
        with open(path, "rb") as fh:
            assert fh.read(8)[4:8] == b"ftyp"
    playlist = _read_playlist(out_dir)
    assert "#EXT-X-VERSION:7" in playlist
    assert "#EXT-X-ENDLIST" in playlist


def test_fmp4_segments_are_independently_decodable(h264_file, out_dir):
    result = package(
        h264_file, out_dir, target_duration=4.0, container="fmp4"
    )
    middle = _segment_paths(out_dir, result)[len(result.segments) // 2]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", middle, "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")


def test_fmp4_cut_boundaries_still_align_keyframes(h264_file, out_dir):
    media_probe = ffprobe_json(
        h264_file, "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
    )
    source_keyframes = {
        round(float(p["pts_time"]), 3)
        for p in media_probe["packets"] if "K" in p["flags"]
    }
    result = package(
        h264_file, out_dir, target_duration=5.0, container="fmp4"
    )
    for seg in result.segments:
        assert any(
            abs(seg.source_start - kf) < 0.005 for kf in source_keyframes
        )


def test_mp4_container_packages(h264_file, out_dir):
    result = package(
        h264_file, out_dir, target_duration=4.0, container="mp4"
    )
    assert all(s.filename.endswith(".mp4") for s in result.segments)
    for path in _segment_paths(out_dir, result):
        with open(path, "rb") as fh:
            assert fh.read(8)[4:8] == b"ftyp"
        assert _container_duration(path) > 0
    assert "#EXT-X-VERSION:7" in _read_playlist(out_dir)


def test_mkv_container_packages(h264_file, out_dir):
    result = package(
        h264_file, out_dir, target_duration=4.0, container="mkv"
    )
    assert all(s.filename.endswith(".mkv") for s in result.segments)
    for path in _segment_paths(out_dir, result):
        # Matroska / WebM 的 EBML 起始签名 1A 45 DF A3
        with open(path, "rb") as fh:
            assert fh.read(4) == b"\x1a\x45\xdf\xa3"
        assert _container_duration(path) > 0
        assert "K" in _first_video_packet_flags(path)


def test_container_does_not_change_segment_count_semantics(h264_file, out_dir, tmp_path):
    ts_result = package(
        h264_file, str(tmp_path / "ts"), target_duration=4.0
    )
    fmp4_result = package(h264_file, out_dir, target_duration=4.0,
                          container="fmp4")
    # 同样的规划输入，切点数量应一致（容器只改封装不改规划）
    assert len(ts_result.segments) == len(fmp4_result.segments)
    for ts_seg, fmp4_seg in zip(ts_result.segments, fmp4_result.segments):
        assert ts_seg.source_start == fmp4_seg.source_start
        assert ts_seg.sequence_number == fmp4_seg.sequence_number


def test_unknown_container_rejected(h264_file, out_dir):
    from hls_packager.errors import PackagerError

    with pytest.raises(PackagerError):
        package(h264_file, out_dir, container="avi")


# --------------------------------------------------------------- 命名模板
def test_custom_name_template_on_disk(h264_file, out_dir):
    result = package(
        h264_file, out_dir,
        target_duration=6.0,
        name_template="{name}-{sequence:05d}.{ext}",
        filename_prefix="ep",
    )
    files = {s.filename for s in result.segments}
    assert "ep-00000.ts" in files
    for name in files:
        assert os.path.isfile(os.path.join(out_dir, name))
    # 播放列表引用的文件名与磁盘一致
    playlist = _read_playlist(out_dir)
    for name in files:
        assert name in playlist


def test_index_template_with_nonzero_sequence(h264_file, out_dir):
    result = package(
        h264_file, out_dir,
        target_duration=6.0,
        media_sequence=100,
        name_template="{name}_{index:03d}.{ext}",
    )
    names = [s.filename for s in result.segments]
    # 文件名按运行内序号从 0 起，而 MEDIA-SEQUENCE 从 100 起
    assert names[0] == "seg_000.ts"
    assert names[1] == "seg_001.ts"
    assert result.segments[0].sequence_number == 100
    playlist = _read_playlist(out_dir)
    assert "#EXT-X-MEDIA-SEQUENCE:100" in playlist
    assert "seg_000.ts" in playlist


def test_template_extension_matches_container(h264_file, out_dir):
    result = package(
        h264_file, out_dir,
        target_duration=6.0,
        container="mkv",
        name_template="{name}_{sequence:04d}.{ext}",
    )
    assert all(s.filename.endswith(".mkv") for s in result.segments)


def test_invalid_template_rejected_before_ffmpeg(h264_file, out_dir):
    from hls_packager.errors import NamingError

    # 缺少 {name} 占位符：在任何 IO / ffmpeg 调用之前即被拒绝
    with pytest.raises(NamingError):
        package(
            h264_file, out_dir,
            name_template="{sequence:04d}.{ext}",
        )
    # 输出目录中不应产生任何分片（参数在探测/切片前即被拒绝）
    assert not any(
        f.endswith((".ts", ".m4s", ".mp4", ".mkv"))
        for f in os.listdir(out_dir)
    )


# --------------------------------------------------------------- 对齐模式
def test_align_before_respects_target_ceiling(h264_file, out_dir):
    # 素材 GOP 固定 2s，目标 5s：before 模式非末片不得超过 5s
    result = package(
        h264_file, out_dir,
        target_duration=5.0,
        align_mode=ALIGN_BEFORE,
    )
    for seg in result.segments[:-1]:
        assert seg.duration <= 5.0 + 0.05
    # 切点仍是关键帧
    media_probe = ffprobe_json(
        h264_file, "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
    )
    keyframes = {
        round(float(p["pts_time"]), 3)
        for p in media_probe["packets"] if "K" in p["flags"]
    }
    for seg in result.segments:
        assert any(abs(seg.source_start - kf) < 0.005 for kf in keyframes)


def test_align_mode_recorded_in_metadata(h264_file, out_dir):
    import json

    result = package(
        h264_file, out_dir,
        target_duration=5.0,
        align_mode=ALIGN_BEFORE,
        container="fmp4",
        name_template="{name}_{sequence:04d}.{ext}",
    )
    payload = json.load(open(result.extra["metadata_path"], encoding="utf-8"))
    assert payload["hls"]["align_mode"] == "before"
    assert payload["hls"]["container"] == "fmp4"
    assert payload["hls"]["hls_version"] == 7
    assert payload["hls"]["name_template"] == "{name}_{sequence:04d}.{ext}"


def test_audio_only_align_mode_is_accepted(audio_only_file, out_dir):
    # 纯音频没有 GOP，对齐模式参数应被静默接受并走等长规划
    result = package(
        audio_only_file, out_dir,
        target_duration=3.0,
        align_mode=ALIGN_BEFORE,
    )
    assert len(result.segments) >= 3
    assert all(s.filename.endswith(".ts") for s in result.segments)
