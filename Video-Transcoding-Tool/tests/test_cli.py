"""CLI 入口测试：退出码、JSON 输出、错误提示。"""

from __future__ import annotations

import json
import os

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


# ---------------------------------------------------------------------------
# 多编码 / 多容器 / 分段 / 进度
# ---------------------------------------------------------------------------

def test_cli_hevc_codec(hevc_file, work_dir):
    out = os.path.join(work_dir, "cli_hevc.mp4")
    assert main([hevc_file, "-o", out, "--codec", "h265",
                 "--preset", "ultrafast"]) == 0
    assert os.path.isfile(out)


def test_cli_vp9_webm_json(hevc_file, work_dir, capsys):
    out = os.path.join(work_dir, "cli_vp9.webm")
    code = main([hevc_file, "-o", out, "--codec", "vp9",
                 "--preset", "ultrafast", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["video_codec"] == "vp9"
    assert payload["container"] == "webm"
    assert payload["encoder"] == "libvpx-vp9"


def test_cli_incompatible_container_exit_1(hevc_file, work_dir):
    out = os.path.join(work_dir, "bad.mp4")
    code = main([hevc_file, "-o", out, "--codec", "vp9",
                 "--container", "mp4"])
    assert code == 1
    assert not os.path.exists(out)


def test_cli_unknown_codec_exit_1(hevc_file, work_dir):
    code = main([hevc_file, "-o", os.path.join(work_dir, "a.mp4"),
                 "--codec", "mpeg1"])
    assert code == 1


def test_cli_bad_crf_for_codec_exit_1(hevc_file, work_dir):
    code = main([hevc_file, "-o", os.path.join(work_dir, "a.webm"),
                 "--codec", "vp9", "--crf", "100"])
    assert code == 1


def test_cli_segmented(long_file, work_dir):
    out = os.path.join(work_dir, "cli_seg.mp4")
    code = main([long_file, "-o", out, "--preset", "ultrafast",
                 "--segment-count", "3"])
    assert code == 0
    assert os.path.isfile(out)


def test_cli_progress_prints_to_stderr(long_file, work_dir, capsys):
    out = os.path.join(work_dir, "cli_progress.mp4")
    code = main([long_file, "-o", out, "--preset", "ultrafast",
                 "--segment-count", "2", "--progress"])
    assert code == 0
    captured = capsys.readouterr()
    assert "转码进度" in captured.err
    # stdout 仍只输出人类可读结果，未被进度污染
    assert "转码完成" in captured.out


def test_cli_list_codecs_text(capsys):
    assert main(["--list-codecs"]) == 0
    out = capsys.readouterr().out
    for name in ("h264", "h265", "vp9", "av1", "vp8"):
        assert name in out


def test_cli_list_codecs_json(capsys):
    assert main(["--list-codecs", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {c["codec"] for c in payload["codecs"]} == \
        {"h264", "h265", "vp9", "vp8", "av1"}
    assert payload["ffmpeg_encoders"]  # 真实探测到本机编码器


def test_cli_missing_input_argument(work_dir):
    # argparse 错误会调用 SystemExit(2)
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
