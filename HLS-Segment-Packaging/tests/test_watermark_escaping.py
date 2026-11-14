"""参数转义、引号包裹与 ffmpeg 水印命令构造测试（不实际执行 ffmpeg）。"""

from __future__ import annotations

import os

import pytest

from hls_packager.errors import UnsupportedInputError, WatermarkError
from hls_packager.models import MediaInfo, SegmentInfo, StreamInfo
from hls_packager.watermark.config import WatermarkConfig
from hls_packager.watermark.escaping import (
    escape_filter_path,
    posix_shell_quote,
    render_shell_command,
    windows_argv_quote,
)
from hls_packager.watermark.overlay import (
    build_filter_graph,
    build_watermarked_segment_command,
    overlay_position,
)

#: 典型注入载荷：分号、管道、反引号、$()、引号、冒号、反斜杠、空格
_INJECTION_PAYLOADS = [
    r"C:\bad dir;rm -rf / 'a'b:c.png",
    "movie=/etc/passwd; drawtext=text='pwned'",
    "file`whoami`.png",
    "file$(reboot).png",
    "a,b[c]d;e|f&g",
    "中文 路径#1;%PATH%.png",
    "it's a \"test\".png",
]


def test_filter_escape_wraps_in_single_quotes():
    for payload in _INJECTION_PAYLOADS:
        escaped = escape_filter_path(payload)
        assert escaped.startswith("'") and escaped.endswith("'")
        # 原始反斜杠在两级转义后必须被复制为偶数个反斜杠
        assert "\\\\" in escaped or "\\" not in payload


def test_filter_escape_backslash_count_doubled_twice():
    # 单个反斜杠：第一级变 2 个，第二级各复制一次 => 4 个
    escaped = escape_filter_path("a\\b")
    assert "\\\\\\\\" in escaped


def test_filter_escape_escapes_colon():
    escaped = escape_filter_path("C:windows")
    # 冒号在第二级前必须有反斜杠
    assert "\\:" in escaped


def test_posix_quote_neutralizes_metacharacters():
    for payload in _INJECTION_PAYLOADS:
        quoted = posix_shell_quote(payload)
        # 整体单引号包裹后，shell 无法在其中解释任何元字符
        assert quoted.startswith("'")
        # 单引号被规范地替换为 '\''
        if "'" in payload:
            assert "'\\''" in quoted


def test_posix_quote_plain_token_unchanged():
    assert posix_shell_quote("seg_0000.ts") == "seg_0000.ts"
    assert posix_shell_quote("") == "''"


def test_windows_quote_handles_spaces_and_quotes():
    # 无空白无引号的 token 不包裹
    assert windows_argv_quote("seg_0000.ts") == "seg_0000.ts"
    assert windows_argv_quote("") == '""'
    quoted = windows_argv_quote('C:\\my dir\\a"b.ts')
    assert quoted.startswith('"') and quoted.endswith('"')
    # 内嵌双引号必须被反斜杠转义
    assert '\\"' in quoted


def test_render_shell_command_keeps_argv_elements_separate():
    argv = ["ffmpeg", "-i", _INJECTION_PAYLOADS[0], "-f", "segment"]
    rendered = render_shell_command(argv)
    # 载荷被整体引号包裹，不会拆成多个词
    assert rendered.count('"') >= 2


# ------------------------------------------------------------ 锚点/过滤器

def test_overlay_position_all_anchors():
    assert overlay_position("top-left", 10, 20) == ("10", "20")
    assert overlay_position("top-center", 10, 20) == ("(W-w)/2", "20")
    assert overlay_position("bottom-right", 10, 20) == ("W-w-10", "H-h-20")
    assert overlay_position("center", 0, 0) == ("(W-w)/2", "(H-h)/2")


def test_overlay_position_rejects_unknown_anchor():
    with pytest.raises(WatermarkError):
        overlay_position("floating", 1, 1)


def test_filter_graph_contains_only_safe_tokens():
    config = WatermarkConfig(text="x", anchor="bottom-right", margin_x=5, margin_y=7)
    graph = build_filter_graph(config)
    assert graph.startswith("[1:v]format=rgba[wm];[0:v][wm]overlay=")
    assert graph.endswith(":shortest=1[v]")
    assert "W-w-5" in graph and "H-h-7" in graph
    # 过滤器图里绝不能出现文本内容（文本只通过 PNG 文件进入）
    assert "x" not in graph.split("overlay=")[1]


# ------------------------------------------------------------ 命令构造

def _media(codec: str = "h264") -> MediaInfo:
    return MediaInfo(
        path="X:/input.mp4",
        format_name="mov,mp4,m4a",
        duration=30.0,
        bit_rate=None,
        streams=(StreamInfo(index=0, codec_type="video", codec_name=codec),),
    )


def _planned() -> list[SegmentInfo]:
    return [
        SegmentInfo(0, 0.0, 8.0, "seg_0000.ts", source_start=0.0),
        SegmentInfo(1, 8.0, 8.0, "seg_0001.ts", source_start=8.0),
    ]


def test_watermark_command_uses_second_input_and_overlay(tmp_path):
    png = tmp_path / "wm.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    config = WatermarkConfig(text="hello")
    argv = build_watermarked_segment_command(
        _media(), _planned(), str(tmp_path), "seg",
        ["0:v:0?"], str(png), config,
    )

    # 两个独立 -i 输入，路径都是独立 argv 元素
    assert argv.count("-i") == 2
    assert str(png) in argv
    # 重编码而非流拷贝
    assert "libx264" in argv
    assert "-c:a" in argv and "copy" in argv
    # 切点同时驱动强制关键帧与 segment 切割
    kf_idx = argv.index("-force_key_frames")
    assert argv[kf_idx + 1] == "8.000000"
    st_idx = argv.index("-segment_times")
    assert argv[st_idx + 1] == "8.000000"
    # 输出仍是 mpegts segment
    assert argv[-1].replace("\\", "/").endswith("seg_%04d.ts")
    assert "-segment_format" in argv and "mpegts" in argv


def test_watermark_command_special_paths_are_passed_verbatim(tmp_path):
    # 含空格、分号、单引号的路径：作为独立 argv 元素不经 shell，原样传递
    nasty_dir = tmp_path / "out dir; rm -rf x"
    nasty_dir.mkdir()
    png = nasty_dir / "wm'inject.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    media = MediaInfo(
        path=str(tmp_path / "in;`whoami`.mp4"),
        format_name="mov,mp4", duration=10.0, bit_rate=None,
        streams=(StreamInfo(index=0, codec_type="video", codec_name="h264"),),
    )
    argv = build_watermarked_segment_command(
        media, _planned()[:1], str(nasty_dir), "seg",
        ["0:v:0?"], str(png), WatermarkConfig(text="t"),
    )
    # 输入与水印路径必须逐字出现在 argv 中，未被转义/截断
    assert media.path in argv
    assert str(png) in argv
    # 过滤器图本身不包含任何路径（杜绝 filtergraph 注入面）
    graph = argv[argv.index("-filter_complex") + 1]
    assert "whoami" not in graph and "inject" not in graph


def test_watermark_command_rejects_missing_png(tmp_path):
    with pytest.raises(WatermarkError):
        build_watermarked_segment_command(
            _media(), _planned(), str(tmp_path), "seg",
            ["0:v:0?"], str(tmp_path / "absent.png"),
            WatermarkConfig(text="t"),
        )


def test_watermark_command_rejects_empty_plan(tmp_path):
    png = tmp_path / "wm.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(WatermarkError):
        build_watermarked_segment_command(
            _media(), [], str(tmp_path), "seg",
            ["0:v:0?"], str(png), WatermarkConfig(text="t"),
        )


def test_watermark_command_rejects_audio_only_input(tmp_path):
    png = tmp_path / "wm.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    audio = MediaInfo(
        path="a.m4a", format_name="m4a", duration=10.0, bit_rate=None,
        streams=(StreamInfo(index=0, codec_type="audio", codec_name="aac"),),
    )
    with pytest.raises(UnsupportedInputError):
        build_watermarked_segment_command(
            audio, _planned(), str(tmp_path), "seg",
            ["0:0?"], str(png), WatermarkConfig(text="t"),
        )


def test_watermark_command_selects_encoder_per_codec(tmp_path):
    png = tmp_path / "wm.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    argv = build_watermarked_segment_command(
        _media("hevc"), _planned()[:1], str(tmp_path), "seg",
        ["0:v:0?"], str(png), WatermarkConfig(text="t"),
    )
    assert "libx265" in argv

    argv_mpeg2 = build_watermarked_segment_command(
        _media("mpeg2video"), _planned()[:1], str(tmp_path), "seg",
        ["0:v:0?"], str(png), WatermarkConfig(text="t"),
    )
    assert "mpeg2video" in argv_mpeg2
