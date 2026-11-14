"""多编码、多容器、分段转码的端到端测试（真实启动 ffmpeg）。"""

from __future__ import annotations

import os

import pytest

from transcode_tool.containers import ContainerError
from transcode_tool.errors import (
    EncoderUnavailableError,
    InvalidInputError,
)
from transcode_tool.probe import probe_media
from transcode_tool.transcoder import transcode


def _probe(path: str):
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0
    return probe_media(path)


# ---------------------------------------------------------------------------
# 各编码端到端
# ---------------------------------------------------------------------------

def test_transcode_to_hevc_mp4(hevc_file, work_dir):
    # 输入本身是 HEVC，强制重编码一遍验证编码链路而不是流拷贝
    out = os.path.join(work_dir, "h265.mp4")
    result = transcode(hevc_file, out, codec="h265", preset="ultrafast")
    info = _probe(out)
    assert info.primary_video.codec_name == "hevc"
    assert info.primary_video.pix_fmt == "yuv420p"
    assert result.container == "mp4"
    assert result.encoder == "libx265"


def test_transcode_to_vp9_webm(hevc_file, work_dir):
    out = os.path.join(work_dir, "vp9.webm")
    result = transcode(hevc_file, out, codec="vp9", preset="ultrafast")
    info = _probe(out)
    assert info.primary_video.codec_name == "vp9"
    assert result.container == "webm"
    assert result.encoder == "libvpx-vp9"


def test_transcode_to_vp8_webm(hevc_file, work_dir):
    out = os.path.join(work_dir, "vp8.webm")
    result = transcode(hevc_file, out, codec="vp8", preset="ultrafast")
    info = _probe(out)
    assert info.primary_video.codec_name == "vp8"
    assert result.container == "webm"


def test_transcode_to_av1_mp4(hevc_file, work_dir):
    out = os.path.join(work_dir, "av1.mp4")
    result = transcode(hevc_file, out, codec="av1", preset="ultrafast")
    info = _probe(out)
    assert info.primary_video.codec_name == "av1"
    assert result.encoder  # svt/aom/rav1e 任一可用即可


def test_codec_aliases_resolve(hevc_file, work_dir):
    out = os.path.join(work_dir, "alias.mp4")
    transcode(hevc_file, out, codec="HEVC", preset="ultrafast")
    assert _probe(out).primary_video.codec_name == "hevc"


# ---------------------------------------------------------------------------
# 容器选择与输出路径
# ---------------------------------------------------------------------------

def test_default_output_name_per_codec(hevc_file, work_dir):
    r_h265 = transcode(hevc_file, codec="h265", preset="ultrafast")
    assert r_h265.output_path.endswith("_h265.mp4")
    r_vp9 = transcode(hevc_file, codec="vp9", preset="ultrafast")
    assert r_vp9.output_path.endswith("_vp9.webm")


def test_container_explicit_overrides_extension(hevc_file, work_dir):
    # 输出写 .mkv 但显式指定 webm：扩展名应被规范化为 .webm
    out = os.path.join(work_dir, "x.mkv")
    result = transcode(hevc_file, out, codec="vp9", container="webm",
                       preset="ultrafast")
    assert result.output_path.endswith(".webm")
    assert _probe(result.output_path).primary_video.codec_name == "vp9"


def test_container_inferred_from_output_extension(hevc_file, work_dir):
    out = os.path.join(work_dir, "x.mkv")
    result = transcode(hevc_file, out, codec="h265", preset="ultrafast")
    assert result.container == "mkv"
    assert result.output_path.endswith(".mkv")


def test_incompatible_container_raises(hevc_file, work_dir):
    out = os.path.join(work_dir, "bad.mp4")
    with pytest.raises(ContainerError, match="不能承载"):
        transcode(hevc_file, out, codec="vp9", container="mp4")
    assert not os.path.exists(out)


def test_unknown_codec_raises(hevc_file, work_dir):
    with pytest.raises(ValueError, match="未知视频编码"):
        transcode(hevc_file, os.path.join(work_dir, "a.mp4"), codec="mpeg1")


# ---------------------------------------------------------------------------
# CRF / 像素格式 / 音频
# ---------------------------------------------------------------------------

def test_crf_range_follows_codec(hevc_file, work_dir):
    # 52 对 H.264 非法，但对 VP9（0-63）合法
    with pytest.raises(ValueError):
        transcode(hevc_file, os.path.join(work_dir, "a.mp4"),
                  codec="h264", crf=52)
    out = os.path.join(work_dir, "b.webm")
    transcode(hevc_file, out, codec="vp9", crf=52, preset="ultrafast")
    assert _probe(out).primary_video.codec_name == "vp9"


def test_default_crf_used_when_omitted(hevc_file, work_dir):
    # 不传 crf 不应报错且能正常产出
    result = transcode(hevc_file, os.path.join(work_dir, "d.webm"),
                       codec="vp9", preset="ultrafast")
    assert result.success


def test_odd_size_rounded_for_hevc(odd_size_file, work_dir):
    out = os.path.join(work_dir, "odd_h265.mp4")
    result = transcode(odd_size_file, out, codec="h265", preset="ultrafast")
    info = _probe(out)
    assert info.primary_video.width % 2 == 0
    assert info.primary_video.height % 2 == 0
    assert any("奇数" in w for w in result.warnings)


def test_tenbit_input_downconverted(tenbit_file, work_dir):
    out = os.path.join(work_dir, "tenbit_h264.mp4")
    result = transcode(tenbit_file, out, preset="ultrafast")
    info = _probe(out)
    assert info.primary_video.pix_fmt == "yuv420p"
    assert any("yuv420p" in w for w in result.warnings)


def test_mkv_container_copies_audio(hevc_file, work_dir):
    # 源音频是 AAC，MKV 可直接承载，应流拷贝而非重编码
    out = os.path.join(work_dir, "audio.mkv")
    result = transcode(hevc_file, out, codec="h265", preset="ultrafast")
    info = _probe(out)
    assert info.primary_audio is not None
    assert info.primary_audio.codec_name == "aac"
    assert not any("重编码" in w for w in result.warnings)


def test_webm_reencodes_aac_to_opus(hevc_file, work_dir):
    out = os.path.join(work_dir, "audio.webm")
    result = transcode(hevc_file, out, codec="vp9", preset="ultrafast")
    info = _probe(out)
    assert info.primary_audio is not None
    assert info.primary_audio.codec_name == "opus"
    assert any("重编码" in w for w in result.warnings)


def test_no_audio_webm(no_audio_file, work_dir):
    out = os.path.join(work_dir, "silent.webm")
    result = transcode(no_audio_file, out, codec="vp9", preset="ultrafast")
    info = _probe(out)
    assert info.primary_audio is None
    assert any("无音频" in w for w in result.warnings)


def test_json_result_contains_container_and_encoder(hevc_file, work_dir):
    result = transcode(hevc_file, os.path.join(work_dir, "j.mp4"),
                       codec="h265", preset="ultrafast")
    payload = result.to_dict()
    assert payload["container"] == "mp4"
    assert payload["encoder"] == "libx265"
    assert payload["video_codec"] == "hevc"


# ---------------------------------------------------------------------------
# 异常输入
# ---------------------------------------------------------------------------

def test_zero_byte_input_raises(work_dir):
    empty = os.path.join(work_dir, "empty.mp4")
    open(empty, "wb").close()
    with pytest.raises(InvalidInputError):
        transcode(empty, os.path.join(work_dir, "o.mp4"))


def test_encoder_unavailable_error_carries_context(monkeypatch, hevc_file, work_dir):
    """模拟本机缺少目标编码全部编码器：抛 EncoderUnavailableError（退出码 7）。"""
    from transcode_tool import capabilities

    monkeypatch.setattr(
        capabilities, "available_encoders",
        lambda *a, **k: frozenset({"libx264", "aac"}),
    )
    with pytest.raises(EncoderUnavailableError) as exc:
        transcode(hevc_file, os.path.join(work_dir, "noenc.mp4"),
                  codec="h265", preset="ultrafast")
    assert exc.value.codec == "h265"
    assert exc.value.exit_code == 7


# ---------------------------------------------------------------------------
# 进度回调
# ---------------------------------------------------------------------------

def test_progress_callback_invoked(hevc_file, work_dir):
    seen: list[float] = []
    transcode(hevc_file, os.path.join(work_dir, "p.mp4"),
              preset="ultrafast", progress_callback=seen.append)
    assert seen, "进度回调至少应被触发一次"
    assert seen == sorted(seen)          # 单调不减
    assert all(0.0 <= v <= 1.0 for v in seen)
    assert seen[-1] == pytest.approx(1.0)


def test_progress_callback_exception_does_not_kill_transcode(hevc_file, work_dir):
    def boom(_ratio: float) -> None:
        raise RuntimeError("callback blew up")

    result = transcode(hevc_file, os.path.join(work_dir, "p2.mp4"),
                       preset="ultrafast", progress_callback=boom)
    assert result.success


# ---------------------------------------------------------------------------
# 大文件分段
# ---------------------------------------------------------------------------

def test_segmented_h264(long_file, work_dir):
    out = os.path.join(work_dir, "seg_h264.mp4")
    result = transcode(long_file, out, preset="ultrafast",
                       segment_duration=1.5)
    info = _probe(out)
    assert info.primary_video.codec_name == "h264"
    # 拼接后总时长应与源接近（concat 无损拼接）
    src = probe_media(long_file)
    assert info.duration_seconds == pytest.approx(
        src.duration_seconds, abs=0.5
    )
    assert result.success


def test_segmented_vp9_webm(long_file, work_dir):
    out = os.path.join(work_dir, "seg_vp9.webm")
    result = transcode(long_file, out, codec="vp9", preset="ultrafast",
                       segment_count=3)
    info = _probe(out)
    assert info.primary_video.codec_name == "vp9"
    assert result.container == "webm"
    src = probe_media(long_file)
    assert info.duration_seconds == pytest.approx(
        src.duration_seconds, abs=0.6
    )


def test_segmented_progress_monotonic(long_file, work_dir):
    seen: list[float] = []
    transcode(long_file, os.path.join(work_dir, "seg_p.mp4"),
              preset="ultrafast", segment_count=3,
              progress_callback=seen.append)
    assert len(seen) >= 3  # 三段各至少上报一次
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_segmented_temp_dir_cleaned(long_file, work_dir, monkeypatch):
    import tempfile

    created: list[str] = []
    original_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(prefix=None):  # type: ignore[no-untyped-def]
        path = original_mkdtemp(prefix=prefix or "vtt_seg_")
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)
    out = os.path.join(work_dir, "seg_clean.mp4")
    transcode(long_file, out, preset="ultrafast", segment_count=2)
    assert created and all(not os.path.exists(p) for p in created)


def test_segment_failure_cleans_and_raises(long_file, work_dir, monkeypatch):
    """分段中途失败：抛受控异常、清理半成品与临时目录、输出不留存。"""
    from transcode_tool import runner
    from transcode_tool.errors import FFmpegRunError

    original = runner.run_ffmpeg
    calls = {"n": 0}

    def fail_second_segment(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            return 1, "simulated segment crash"
        return original(cmd, **kwargs)

    monkeypatch.setattr(runner, "run_ffmpeg", fail_second_segment)
    # transcoder 模块内是 from .runner import run_ffmpeg，需同步替换
    monkeypatch.setattr(
        "transcode_tool.transcoder.run_ffmpeg", fail_second_segment
    )
    out = os.path.join(work_dir, "seg_fail.mp4")
    with pytest.raises(FFmpegRunError, match="段转码失败"):
        transcode(long_file, out, preset="ultrafast", segment_count=3)
    assert not os.path.exists(out)


def test_segmented_h265_mkv_sparse_gop_keeps_duration(
    sparse_gop_file, work_dir
):
    """稀疏 GOP + H.265/MKV 分段：时长必须精确，不得被关键帧回退拉长。

    回归测试：输入寻址回退到更早关键帧后，若时间戳未重置，MKV 中视频流
    与 copy 的音频流错位，会让每段（进而让拼接产物）多出一个关键帧间隔。
    """
    out = os.path.join(work_dir, "sparse_h265.mkv")
    result = transcode(
        sparse_gop_file, out, codec="h265", preset="ultrafast",
        segment_duration=1.5,
    )
    info = _probe(out)
    src = probe_media(sparse_gop_file)
    assert info.primary_video.codec_name == "hevc"
    # 容差收紧到 0.3s：旧的时间戳 bug 会把 6s 拉长到 ~14s，必被捕获
    assert info.duration_seconds == pytest.approx(
        src.duration_seconds, abs=0.3
    )
    assert result.container == "mkv"


@pytest.mark.parametrize("codec,container", [
    ("h264", "mp4"), ("h265", "mkv"), ("vp9", "webm"), ("av1", "mp4"),
])
def test_segmented_duration_accurate_across_codecs(
    long_file, work_dir, codec, container
):
    """各编码分段拼接后的总时长都应贴近源，容差 0.6s。"""
    ext = {"mp4": ".mp4", "mkv": ".mkv", "webm": ".webm"}[container]
    out = os.path.join(work_dir, f"acc_{codec}{ext}")
    transcode(long_file, out, codec=codec, preset="ultrafast",
              segment_count=3)
    info = _probe(out)
    src = probe_media(long_file)
    assert info.duration_seconds == pytest.approx(
        src.duration_seconds, abs=0.6
    )
