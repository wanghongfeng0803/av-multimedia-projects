"""CLI 层测试：参数解析与错误退出码；真实跑 ffmpeg 的路径由
test_pipeline.py 覆盖，这里只验证 argparse 行为，不依赖 ffmpeg。"""

import pytest

from vkeyer.cli import build_parser, main
from vkeyer.errors import OutputFormatError


def test_parse_explicit_key_color():
    args = build_parser().parse_args(["in.mp4", "-o", "out.mov",
                                      "--key", "0,200,0"])
    assert args.key_rgb == (0, 200, 0)
    assert args.output == "out.mov"


def test_parse_bg_color_and_bg_image_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["in.mp4", "-o", "o.mp4",
             "--bg-color", "0,0,0", "--bg-image", "b.png"]
        )


def test_parse_invalid_rgb_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["in.mp4", "-o", "o.mov",
                                   "--key", "0,300,0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["in.mp4", "-o", "o.mov",
                                   "--key", "0,200"])


def test_parse_sample_rect():
    args = build_parser().parse_args(
        ["in.mp4", "-o", "o.mov",
         "--sample-mode", "rect", "--sample-rect", "0.1,0.1,0.4,0.4"]
    )
    assert args.sample_rect == [0.1, 0.1, 0.4, 0.4]


def test_parse_sample_rect_bad_range():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["in.mp4", "-o", "o.mov",
             "--sample-mode", "rect", "--sample-rect", "0.5,0.5,0.4,0.6"]
        )


def test_default_matte_is_identity():
    args = build_parser().parse_args(["in.mp4", "-o", "o.mov"])
    assert args.choke == 0 and args.feather == 0
    assert args.threshold_low == 0 and args.threshold_high == 255


def test_despill_default_on():
    args = build_parser().parse_args(["in.mp4", "-o", "o.mov"])
    assert args.no_despill is False


def test_tolerance_and_despill_strength_defaults_neutral():
    """两新参数默认中性：与旧版本逐像素一致的基线。"""
    args = build_parser().parse_args(["in.mp4", "-o", "o.mov"])
    assert args.tolerance == 0.0
    assert args.despill_strength == 1.0


def test_parse_tolerance_and_despill_strength():
    args = build_parser().parse_args(
        ["in.mp4", "-o", "o.mov",
         "--tolerance", "0.3", "--despill-strength", "0.5"]
    )
    assert args.tolerance == 0.3
    assert args.despill_strength == 0.5


def test_mp4_alpha_returns_exit_code_9(tmp_path):
    """alpha 输出到不支持的容器时，CLI 返回 OutputFormatError 的退出码 9。

    用真实存在的文件绕过输入存在性检查；格式校验在 ffmpeg 子进程
    启动之前，故无需有效媒体内容。
    """
    src = tmp_path / "src.mp4"
    src.write_bytes(b"not-a-real-video")
    rc = main([str(src), "-o", str(tmp_path / "x.mp4"), "--overwrite"])
    assert rc == OutputFormatError.exit_code
