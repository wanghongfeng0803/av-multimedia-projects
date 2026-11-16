"""CLI 测试：参数解析、配置映射、退出码约定（不依赖 ffmpeg）。"""

import pytest

from vmotion.cli import build_parser, main
from vmotion.detector import DEFAULT_CONFIG, MotionConfig


def parse(argv):
    return build_parser().parse_args(argv)


def test_detect_defaults_match_config():
    args = parse(["detect", "in.mp4", "out.json"])
    cfg = MotionConfig(
        block=args.block, local_radius=args.local_radius,
        pixel_threshold=args.pixel_threshold, gain=args.gain,
        min_area=args.min_area, open_radius=args.open_radius,
        close_radius=args.close_radius,
        trigger_frames=args.trigger_frames,
        release_frames=args.release_frames,
        illum_change_threshold=args.illum_change_threshold,
        illum_hold_frames=args.illum_hold_frames,
    )
    assert cfg == DEFAULT_CONFIG


def test_detect_all_thresholds_override():
    args = parse([
        "detect", "in.mp4", "out.json",
        "--block", "16", "--local-radius", "5",
        "--pixel-threshold", "20", "--gain", "4.5",
        "--min-area", "200", "--open-radius", "2", "--close-radius", "0",
        "--trigger-frames", "3", "--release-frames", "4",
        "--illum-change-threshold", "40", "--illum-hold-frames", "5",
    ])
    assert args.block == 16 and args.local_radius == 5
    assert args.pixel_threshold == 20 and args.gain == 4.5
    assert args.min_area == 200
    assert args.open_radius == 2 and args.close_radius == 0
    assert args.trigger_frames == 3 and args.release_frames == 4
    assert args.illum_change_threshold == 40
    assert args.illum_hold_frames == 5


def test_mark_parser():
    args = parse(["mark", "in.mp4", "d.json", "out.mp4",
                  "--font-scale", "2"])
    assert args.command == "mark"
    assert args.font_scale == 2
    assert args.detection_json == "d.json"


def test_subcommand_required():
    with pytest.raises(SystemExit):
        parse([])


def test_main_returns_error_on_missing_input(tmp_path):
    missing = tmp_path / "nope.mp4"
    rc = main(["detect", str(missing), str(tmp_path / "o.json")])
    assert rc == 1


def test_main_mark_missing_json_returns_error(tmp_path):
    rc = main(["mark", str(tmp_path / "in.mp4"),
               str(tmp_path / "missing.json"),
               str(tmp_path / "out.mp4")])
    assert rc == 1
