"""probe_media 的纯逻辑测试（mock 掉 ffprobe 子进程）。"""

from __future__ import annotations

import json

import pytest

from compose_tool import probe as probe_module
from compose_tool.errors import InputNotFoundError, InvalidInputError
from compose_tool.probe import probe_media


def _fake_run_factory(payload: dict, returncode: int = 0, stderr: str = ""):
    class _Proc:
        def __init__(self):
            self.returncode = returncode
            self.stdout = json.dumps(payload)
            self.stderr = stderr

    def _run(cmd, **kwargs):
        return _Proc()

    return _run


def _patch(monkeypatch, payload, returncode=0, stderr=""):
    monkeypatch.setattr(probe_module, "find_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(
        probe_module.subprocess, "run", _fake_run_factory(payload, returncode, stderr)
    )


def _payload(streams, duration="2.000000"):
    return {
        "streams": streams,
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": duration},
    }


_VIDEO_STREAM = {
    "index": 0,
    "codec_type": "video",
    "codec_name": "h264",
    "width": 640,
    "height": 360,
    "pix_fmt": "yuv420p",
    "duration": "2.000000",
    "avg_frame_rate": "25/1",
    "r_frame_rate": "25/1",
    "sample_aspect_ratio": "1:1",
}

_AUDIO_STREAM = {
    "index": 1,
    "codec_type": "audio",
    "codec_name": "aac",
    "sample_rate": "44100",
    "channels": 2,
    "duration": "2.000000",
}


class TestProbeMedia:
    def test_parses_streams(self, tmp_path, monkeypatch):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        _patch(monkeypatch, _payload([_VIDEO_STREAM, _AUDIO_STREAM]))
        info = probe_media(str(f))
        assert info.duration_seconds == 2.0
        video = info.primary_video
        assert video is not None
        assert (video.width, video.height) == (640, 360)
        assert video.avg_frame_rate == "25/1"
        assert video.sample_aspect_ratio == "1:1"
        audio = info.primary_audio
        assert audio is not None
        assert audio.sample_rate == 44100
        assert audio.channels == 2

    def test_missing_file(self):
        with pytest.raises(InputNotFoundError):
            probe_media("definitely/not/here.mp4")

    def test_no_streams(self, tmp_path, monkeypatch):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        _patch(monkeypatch, _payload([]))
        with pytest.raises(InvalidInputError, match="没有任何媒体流"):
            probe_media(str(f))

    def test_no_video_stream(self, tmp_path, monkeypatch):
        f = tmp_path / "a.mp3"
        f.write_bytes(b"x")
        _patch(monkeypatch, _payload([_AUDIO_STREAM]))
        with pytest.raises(InvalidInputError, match="不含视频流"):
            probe_media(str(f))

    def test_unparseable_output(self, tmp_path, monkeypatch):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        monkeypatch.setattr(probe_module, "find_ffprobe", lambda: "ffprobe")

        class _Proc:
            returncode = 1
            stdout = "not json at all"
            stderr = "moov atom not found"

        monkeypatch.setattr(
            probe_module.subprocess, "run", lambda *a, **k: _Proc()
        )
        with pytest.raises(InvalidInputError, match="无法解析"):
            probe_media(str(f))

    def test_corrupt_but_partial_json_still_probes(self, tmp_path, monkeypatch):
        """returncode != 0 但 stdout 有部分 JSON 时仍给出结果并记录警告。"""
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        _patch(
            monkeypatch,
            _payload([_VIDEO_STREAM]),
            returncode=1,
            stderr="Invalid NAL unit",
        )
        info = probe_media(str(f))
        assert info.primary_video is not None
        assert info.warnings

    def test_na_values_become_none(self, tmp_path, monkeypatch):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        stream = dict(_VIDEO_STREAM)
        stream["duration"] = "N/A"
        stream["avg_frame_rate"] = "0/0"
        stream["r_frame_rate"] = "0/0"
        _patch(monkeypatch, _payload([stream]))
        info = probe_media(str(f))
        video = info.primary_video
        assert video is not None
        assert video.duration_seconds is None
        assert video.fps is None

    def test_fps_falls_back_to_r_frame_rate(self, tmp_path, monkeypatch):
        """avg_frame_rate 无效（0/0）时回退到 r_frame_rate。"""
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        stream = dict(_VIDEO_STREAM)
        stream["avg_frame_rate"] = "0/0"
        _patch(monkeypatch, _payload([stream]))
        info = probe_media(str(f))
        video = info.primary_video
        assert video is not None
        assert video.fps is not None
        assert str(video.fps) == "25"
