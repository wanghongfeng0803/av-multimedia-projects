"""CLI 参数校验与退出码测试（不依赖 ffmpeg 的部分）。"""

from __future__ import annotations

import pytest

from scene_seg_tool.cli import build_parser, main, _validate_args


def _parse(argv):
    return build_parser().parse_args(argv)


def test_parser_defaults():
    args = _parse(["clip.mp4"])
    assert args.input == "clip.mp4"
    assert args.mad_k == 9.0
    assert args.fallback_threshold == 0.35
    assert args.min_scene_frames == 1
    assert args.no_flash_suppression is False


def test_validate_rejects_non_positive_mad_k():
    args = _parse(["clip.mp4", "--mad-k", "0"])
    with pytest.raises(SystemExit):
        _validate_args(args, build_parser())


def test_validate_rejects_out_of_range_fallback():
    args = _parse(["clip.mp4", "--fallback-threshold", "1.5"])
    with pytest.raises(SystemExit):
        _validate_args(args, build_parser())


def test_validate_rejects_min_scene_frames_below_one():
    args = _parse(["clip.mp4", "--min-scene-frames", "0"])
    with pytest.raises(SystemExit):
        _validate_args(args, build_parser())


def test_validate_rejects_unsupported_output_extension():
    args = _parse(["clip.mp4", "-o", "report.txt"])
    with pytest.raises(SystemExit):
        _validate_args(args, build_parser())


def test_main_missing_input_file_returns_exit_code_3():
    code = main(["definitely_not_here_12345.mp4", "--quiet"])
    assert code == 3
