"""ffprobe 探测逻辑测试：mock 子进程，喂构造的 JSON。"""

from __future__ import annotations

import json

import pytest

from hls_packager.errors import ProbeError, UnsupportedInputError
from hls_packager.process import CommandResult
from hls_packager import probe as probe_mod


def _ok(payload: dict) -> CommandResult:
    return CommandResult(
        argv=("ffprobe",),
        returncode=0,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=b"",
    )


def _fail() -> CommandResult:
    return CommandResult(
        argv=("ffprobe",), returncode=1, stdout=b"", stderr=b"bad file"
    )


def test_parse_streams_video_audio_fields(monkeypatch):
    monkeypatch.setattr(probe_mod.os.path, "isfile", lambda p: True)
    payload = {
        "format": {"duration": "12.5", "format_name": "mov,mp4", "bit_rate": "500000"},
        "streams": [
            {
                "index": 0, "codec_type": "video", "codec_name": "h264",
                "codec_tag_string": "avc1", "width": 1280, "height": 720,
                "avg_frame_rate": "30000/1001",
            },
            {
                "index": 1, "codec_type": "audio", "codec_name": "aac",
                "sample_rate": "48000", "channels": 2,
                "tags": {"language": "eng"},
            },
        ],
    }
    probe_mod.run_capture = lambda *a, **k: _ok(payload)
    media = probe_mod.probe_media("X:/in.mp4")

    assert media.duration == pytest.approx(12.5)
    assert media.bit_rate == 500000
    assert len(media.streams) == 2
    video, audio = media.streams
    assert video.is_video and video.width == 1280
    assert video.frame_rate == pytest.approx(30000 / 1001)
    assert audio.is_audio and audio.sample_rate == 48000
    assert audio.language == "eng"
    assert media.has_video and len(media.audio_streams) == 1


def test_keyframe_times_from_packets():
    packet_payload = {
        "packets": [
            {"pts_time": "0.000000", "flags": "K__"},
            {"pts_time": "0.041667", "flags": "___"},
            {"pts_time": "2.000000", "flags": "K__"},
            {"pts_time": "4.000000", "flags": "K__"},
        ]
    }
    probe_mod.run_capture = lambda *a, **k: _ok(packet_payload)
    times = probe_mod.extract_keyframe_times("X:/in.mp4", 0)
    assert times == (0.0, 2.0, 4.0)


def test_keyframe_fallback_to_frames_when_packets_empty():
    # 第一次调用（packets）返回空，第二次（frames）返回 I 帧
    payloads = [
        {"packets": []},
        {"frames": [
            {"pts_time": "0.0", "pict_type": "I"},
            {"pts_time": "0.04", "pict_type": "P"},
            {"pts_time": "1.0", "pict_type": "I"},
        ]},
    ]
    calls = {"n": 0}

    def fake_run(argv, timeout=None):
        result = _ok(payloads[calls["n"]])
        calls["n"] += 1
        return result

    probe_mod.run_capture = fake_run
    times = probe_mod.extract_keyframe_times("X:/in.ts", 0)
    assert times == (0.0, 1.0)
    assert calls["n"] == 2  # 确实触发了回退


def test_keyframe_dedup_and_sort():
    # 乱序 + 近似重复
    payload = {"packets": [
        {"pts_time": "4.0", "flags": "K__"},
        {"pts_time": "0.0", "flags": "K__"},
        {"pts_time": "4.0000001", "flags": "K__"},
        {"pts_time": "2.0", "flags": "K__"},
    ]}
    probe_mod.run_capture = lambda *a, **k: _ok(payload)
    times = probe_mod.extract_keyframe_times("X:/in.mp4", 0)
    assert times == (0.0, 2.0, 4.0)


def test_probe_missing_file_raises():
    with pytest.raises(ProbeError):
        probe_mod.probe_media("X:/does-not-exist.mp4")


def test_probe_ffprobe_failure_raises(monkeypatch):
    monkeypatch.setattr(probe_mod.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(probe_mod, "run_capture", lambda *a, **k: _fail())
    with pytest.raises(ProbeError):
        probe_mod.probe_media("X:/broken.mp4")


def test_probe_no_av_streams_raises(monkeypatch):
    monkeypatch.setattr(probe_mod.os.path, "isfile", lambda p: True)
    payload = {
        "format": {"duration": "1.0", "format_name": "data"},
        "streams": [{"index": 0, "codec_type": "data", "codec_name": "bin"}],
    }
    monkeypatch.setattr(probe_mod, "run_capture", lambda *a, **k: _ok(payload))
    with pytest.raises(UnsupportedInputError):
        probe_mod.probe_media("X:/data.bin")


def test_audio_only_has_empty_keyframes(monkeypatch):
    monkeypatch.setattr(probe_mod.os.path, "isfile", lambda p: True)
    payload = {
        "format": {"duration": "10.0", "format_name": "m4a"},
        "streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac"}],
    }
    monkeypatch.setattr(probe_mod, "run_capture", lambda *a, **k: _ok(payload))
    media = probe_mod.probe_media("X:/a.m4a")
    assert not media.has_video
    assert media.keyframe_times == ()
