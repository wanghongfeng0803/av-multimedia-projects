"""端到端管道集成测试：ffmpeg 解码 -> 调度渲染 -> 编码回写。

这些测试会真实拉起 ffmpeg 子进程，无 ffmpeg 环境自动 skip。
"""

from __future__ import annotations

import json
import subprocess

import pytest

from subburn import (
    BitmapFont,
    Style,
    SubtitleScheduler,
    burn_subtitles,
    parse_srt,
    probe_video,
)

from .conftest import make_test_video


def ffprobe_streams(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,nb_frames,width,height",
        "-of", "json", path,
    ])
    return json.loads(out)["streams"]


def extract_frame_rgb(path, timestamp, tmp_path, name="f"):
    """抽取指定时刻的帧为 rgb24 裸数据，返回 (width, height, bytes)。"""
    out = tmp_path / f"{name}.rgb"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", str(timestamp),
        "-i", path, "-frames:v", "1", "-f", "rawvideo",
        "-pix_fmt", "rgb24", str(out),
    ], check=True)
    data = out.read_bytes()
    w, h = 320, 240
    assert len(data) == w * h * 3
    return w, h, data


def bottom_region_diff(rgb_a, rgb_b, w, h):
    """两帧在底部 1/3 区域的强差异像素数。

    经过 RGB->YUV420->RGB 的 H.264 有损往返后，即使没烧字幕每像素也有
    轻微色差（实测 <60）；字幕叠加造成的改动远大于此（实测 >100）。
    因此以单通道最大差值 >=100 作为"被字幕改写"的判据，滤掉编码噪声。
    """
    top = (h * 2 // 3) * w * 3
    return sum(
        1 for i in range(top, len(rgb_a), 3)
        if max(abs(rgb_a[i] - rgb_b[i]),
               abs(rgb_a[i + 1] - rgb_b[i + 1]),
               abs(rgb_a[i + 2] - rgb_b[i + 2])) >= 100
    )


@pytest.mark.usefixtures("ffmpeg")
class TestPipeline:
    def test_probe(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_test_video(video)
        tl = probe_video(str(video))
        assert (tl.width, tl.height) == (320, 240)
        assert tl.fps == pytest.approx(25.0)
        assert tl.frame_count == 50  # 2s * 25fps

    def test_burn_full_flow_with_audio(self, tmp_path):
        video = tmp_path / "v.mp4"
        out = tmp_path / "out.mp4"
        make_test_video(video)
        srt = "1\n00:00:00,200 --> 00:00:01,200\nHELLO SUBTITLE\n"
        cues, warns = parse_srt(srt)
        assert not warns

        result = burn_subtitles(
            str(video), str(out), cues,
            font=BitmapFont(2), style=Style(box=True),
            overwrite=True,
        )
        assert result.frames_written == 50
        assert result.had_audio is True

        streams = ffprobe_streams(str(out))
        codecs = {(s["codec_type"], s["codec_name"]) for s in streams}
        assert ("video", "h264") in codecs and ("audio", "aac") in codecs

    def test_frame_level_timing_alignment(self, tmp_path):
        """直接验证「正确时间区间内渲染到对应帧」：区间内亮、区间外暗。"""
        video = tmp_path / "v.mp4"
        out = tmp_path / "out.mp4"
        make_test_video(video, duration=2)
        # 0.4s ~ 1.2s 显示（25fps：帧 10 到帧 29）
        cues, _ = parse_srt("1\n00:00:00,400 --> 00:00:01,200\nTIMING\n")
        burn_subtitles(str(video), str(out), cues,
                       font=BitmapFont(2), overwrite=True)

        # 与原视频同刻帧对比：区间内应有大量差异像素，区间外应零差异
        _, _, src_in = extract_frame_rgb(str(video), 0.8, tmp_path, "s_in")
        _, _, inside = extract_frame_rgb(str(out), 0.8, tmp_path, "o_in")
        _, _, src_before = extract_frame_rgb(str(video), 0.1, tmp_path, "s_pre")
        _, _, before = extract_frame_rgb(str(out), 0.1, tmp_path, "o_pre")
        _, _, src_after = extract_frame_rgb(str(video), 1.6, tmp_path, "s_aft")
        _, _, after = extract_frame_rgb(str(out), 1.6, tmp_path, "o_aft")
        assert bottom_region_diff(src_in, inside, 320, 240) > 500
        assert bottom_region_diff(src_before, before, 320, 240) == 0
        assert bottom_region_diff(src_after, after, 320, 240) == 0
    def test_boundary_frame_half_open_semantics(self, tmp_path):
        """终点恰好落在帧时刻上：该帧不应再有字幕。"""
        video = tmp_path / "v.mp4"
        out = tmp_path / "out.mp4"
        make_test_video(video, duration=2)
        # 终点 1000ms = 帧 25 的展示时刻
        cues, _ = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nEDGE\n")
        burn_subtitles(str(video), str(out), cues,
                       font=BitmapFont(2), overwrite=True)
        # 1.0s 抽取到的是帧 25（半开区间外），与原帧应无差异
        _, _, src = extract_frame_rgb(str(video), 1.0, tmp_path, "s_edge")
        _, _, edge = extract_frame_rgb(str(out), 1.0, tmp_path, "o_edge")
        assert bottom_region_diff(src, edge, 320, 240) == 0

    def test_no_audio_input(self, tmp_path):
        video = tmp_path / "silent.mp4"
        make_test_video(video, with_audio=False)
        out = tmp_path / "out.mp4"
        cues, _ = parse_srt("1\n00:00:00,100 --> 00:00:01,000\nX\n")
        result = burn_subtitles(str(video), str(out), cues,
                                font=BitmapFont(1), overwrite=True)
        assert result.had_audio is False
        streams = ffprobe_streams(str(out))
        assert all(s["codec_type"] != "audio" for s in streams)

    def test_progress_callback_monotonic(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_test_video(video)
        out = tmp_path / "out.mp4"
        cues, _ = parse_srt("1\n00:00:00,100 --> 00:00:01,000\nX\n")
        seen = []
        burn_subtitles(str(video), str(out), cues, font=BitmapFont(1),
                       overwrite=True, progress_callback=seen.append)
        assert seen == sorted(seen) and seen[-1] == 1.0

    def test_refuse_overwrite(self, tmp_path):
        video = tmp_path / "v.mp4"
        out = tmp_path / "out.mp4"
        make_test_video(video)
        out.write_bytes(b"existing")
        cues, _ = parse_srt("1\n00:00:00,100 --> 00:00:01,000\nX\n")
        with pytest.raises(FileExistsError):
            burn_subtitles(str(video), str(out), cues, font=BitmapFont(1))

    def test_missing_input(self, tmp_path):
        cues, _ = parse_srt("1\n00:00:00,100 --> 00:00:01,000\nX\n")
        with pytest.raises(FileNotFoundError):
            burn_subtitles(str(tmp_path / "nope.mp4"),
                           str(tmp_path / "o.mp4"), cues, font=BitmapFont(1))

    def test_scheduler_feeds_renderer_consistently(self):
        """不经过 ffmpeg 的逻辑等价验证：逐帧调度结果与时间区间一一对应。"""
        cues, _ = parse_srt(
            "1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nB\n"
        )
        sch = SubtitleScheduler(cues, fps=25.0)
        texts_per_frame = [
            [c.text for c in sch.active_cues_at(i)] for i in range(50)
        ]
        assert texts_per_frame[0] == ["A"]
        assert texts_per_frame[24] == ["A"]
        assert texts_per_frame[25] == ["B"]   # 半开边界
        assert texts_per_frame[49] == ["B"]
