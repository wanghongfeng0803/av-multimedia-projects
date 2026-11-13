"""CLI 入口测试：退出码、JSON 输出、错误提示。"""

from __future__ import annotations

import json
import os
import sys

import pytest

from transcode_tool.cli import main


def test_cli_success(hevc_file, work_dir, capsys):
    out = os.path.join(work_dir, "cli.mp4")
    code = main([hevc_file, "-o", out, "--preset", "ultrafast"])
    assert code == 0
    captured = capsys.readouterr()
    assert "转码完成" in captured.out
    assert os.path.isfile(out)


def test_cli_json_output(hevc_file, work_dir, capsys):
    out = os.path.join(work_dir, "cli_json.mp4")
    code = main([hevc_file, "-o", out, "--preset", "ultrafast", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["video_codec"] == "h264"
    assert payload["output_path"] == os.path.abspath(out)


def test_cli_missing_input_exit_code(work_dir, capsys):
    code = main([os.path.join(work_dir, "ghost.mp4"),
                 "-o", os.path.join(work_dir, "o.mp4")])
    assert code == 3  # InputNotFoundError.exit_code
    captured = capsys.readouterr()
    assert "错误" in captured.err


def test_cli_invalid_media_json_error(text_disguised_file, work_dir, capsys):
    code = main([text_disguised_file,
                 "-o", os.path.join(work_dir, "o.mp4"), "--json"])
    assert code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert "error" in payload


def test_cli_output_exists_exit_code(hevc_file, work_dir):
    out = os.path.join(work_dir, "exists.mp4")
    assert main([hevc_file, "-o", out, "--preset", "ultrafast"]) == 0
    assert main([hevc_file, "-o", out, "--preset", "ultrafast"]) == 5
    assert main([hevc_file, "-o", out, "--preset", "ultrafast",
                 "--overwrite"]) == 0


def test_cli_bad_crf_exit_code(hevc_file, work_dir):
    code = main([hevc_file, "-o", os.path.join(work_dir, "o.mp4"),
                 "--crf", "99"])
    assert code == 1


def test_cli_default_output_name(hevc_file, work_dir):
    code = main([hevc_file, "--preset", "ultrafast"])
    assert code == 0
    assert os.path.isfile(os.path.join(work_dir, "sample_hevc_h264.mp4"))
