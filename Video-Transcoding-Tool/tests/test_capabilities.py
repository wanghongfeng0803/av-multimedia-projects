"""编码器能力探测测试：解析纯函数 + 进程探测的缓存/降级行为。"""

from __future__ import annotations

import subprocess

import pytest

from transcode_tool.capabilities import (
    available_encoders,
    is_encoder_available,
    parse_encoders_output,
    reset_cache,
    select_audio_encoder,
    select_video_encoder,
)
from transcode_tool.codecs import AV1, H265
from transcode_tool.errors import EncoderUnavailableError

_MODERN_OUTPUT = """ffmpeg version 9.0
Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V....D libx264              libx264 H.264
 V....D libx265              libx265 H.265
 V....D libvpx-vp9           libvpx VP9
 V..... libsvtav1            SVT-AV1 encoder
 A....D aac                  AAC
 A..... libopus             libopus
 S..... srt                 SubRip
"""

_LEGACY_OUTPUT = """Encoders:
 V..... = Video
 A..... = Audio
 ....E. = Supports encoding
 ------
 V....E libx264     H.264
 V.....  h264_dec   not really an encoder
 A....E aac         AAC
"""


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_cache()
    yield
    reset_cache()


def test_parse_modern_output():
    enc = parse_encoders_output(_MODERN_OUTPUT)
    for name in ("libx264", "libx265", "libvpx-vp9", "libsvtav1",
                 "aac", "libopus", "srt"):
        assert name in enc
    # 图例里的 "Video" 等单词不得被误收为编码器
    assert "Video" not in enc and "Audio" not in enc


def test_parse_legacy_e_bit_output():
    enc = parse_encoders_output(_LEGACY_OUTPUT)
    assert enc == frozenset({"libx264", "aac"})


@pytest.mark.parametrize("garbage", ["", "   ", "no header here\nfoo bar", None])
def test_parse_garbage_returns_empty(garbage):
    assert parse_encoders_output(garbage) == frozenset()


def test_available_encoders_caches_result(monkeypatch):
    calls = {"n": 0}

    class FakeProc:
        returncode = 0
        stdout = _MODERN_OUTPUT
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    first = available_encoders("fake-ffmpeg")
    second = available_encoders("fake-ffmpeg")
    assert first is second or first == second
    assert calls["n"] == 1  # 第二次命中缓存，不再拉起进程


def test_detection_failure_is_optimistic(monkeypatch):
    """探测进程失败：返回 None（未知），编码器选择乐观降级而非直接拒绝。"""
    def boom(cmd, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(subprocess, "run", boom)
    assert available_encoders("fake-ffmpeg") is None
    assert is_encoder_available("libx265", "fake-ffmpeg") is True
    assert select_video_encoder(H265, "fake-ffmpeg").name == "libx265"
    assert select_audio_encoder("libopus", ("aac",), "fake-ffmpeg") == "libopus"


def test_select_video_encoder_prefers_first_available(monkeypatch):
    class FakeProc:
        returncode = 0
        # SVT-AV1 缺失，libaom 存在 → 应跳过首选、选链上的第二个
        stdout = (
            _MODERN_OUTPUT
            .replace("V..... libsvtav1            SVT-AV1 encoder",
                     "V..... libaom-av1           libaom AV1")
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    chosen = select_video_encoder(AV1, "fake-ffmpeg")
    assert chosen.name == "libaom-av1"


def test_select_video_encoder_unavailable_raises(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = _MODERN_OUTPUT.replace("libx265", "libx265_gone")
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    with pytest.raises(EncoderUnavailableError) as exc:
        select_video_encoder(H265, "fake-ffmpeg")
    assert exc.value.codec == "h265"
    assert "libx265" in exc.value.candidates


def test_select_audio_encoder_uses_fallback(monkeypatch):
    class FakeProc:
        returncode = 0
        # libopus 缺失，aac 存在
        stdout = _MODERN_OUTPUT.replace("libopus", "libvorbis")
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    chosen = select_audio_encoder("libopus", ("libvorbis", "aac"), "fake-ffmpeg")
    assert chosen == "libvorbis"


def test_real_ffmpeg_detection_skips_when_binary_absent(monkeypatch):
    # 模拟 returncode 非零：按探测失败处理
    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    assert available_encoders("fake-ffmpeg") is None
