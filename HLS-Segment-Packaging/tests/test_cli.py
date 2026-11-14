"""CLI 端到端测试：通过 main() 验证参数解析、退出码与产物。"""

from __future__ import annotations

import os
import shutil

import pytest

from hls_packager.cli import main

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)


def test_cli_vod_run(h264_file, out_dir, capsys):
    exit_code = main([h264_file, out_dir, "-d", "4", "-y"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert os.path.isfile(os.path.join(out_dir, "playlist.m3u8"))
    assert "VOD" in captured.out
    assert "seg_0000.ts" in captured.out


def test_cli_window_mode(h264_file, out_dir):
    exit_code = main([
        h264_file, out_dir, "-d", "4", "-w", "2",
        "-s", "10", "-p", "chunk", "-y",
    ])
    assert exit_code == 0
    files = os.listdir(out_dir)
    assert "playlist.m3u8" in files
    # 自定义前缀生效
    assert any(f.startswith("chunk_") for f in files)
    playlist = open(
        os.path.join(out_dir, "playlist.m3u8"), encoding="utf-8"
    ).read()
    lines = playlist.splitlines()

    # 直播列表不能有 ENDLIST，且窗口只保留最后 2 片
    assert "#EXT-X-ENDLIST" not in playlist
    chunk_lines = [ln for ln in lines if ln.startswith("chunk_")]
    assert len(chunk_lines) == 2
    # MEDIA-SEQUENCE 必须与窗口内首片文件名序号一致
    first_window_chunk = chunk_lines[0]
    first_seq = int(first_window_chunk.split("_")[1].split(".")[0])
    assert f"#EXT-X-MEDIA-SEQUENCE:{first_seq}" in lines
    # 已滚出窗口的早期分片（序号 10）不在列表中
    assert "chunk_0010.ts" not in lines
    assert first_seq > 10


def test_cli_missing_input_returns_error_code(tmp_path):
    exit_code = main([str(tmp_path / "nope.mp4"), str(tmp_path / "out")])
    assert exit_code == 2


def test_cli_bad_duration_returns_error_code(h264_file, out_dir):
    exit_code = main([h264_file, out_dir, "-d", "0", "-y"])
    assert exit_code == 2
