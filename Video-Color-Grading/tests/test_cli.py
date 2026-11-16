"""CLI 参数解析与错误路径的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

from color_grading.cli import build_parser, main
from color_grading.errors import InputNotFoundError, OutputExistsError


def test_parser_defaults_keep_legacy_behavior():
    """无参数调用的默认值：亮度/对比度均为 0（恒等映射）。"""
    args = build_parser().parse_args(["in.mp4", "-o", "out.mp4"])
    assert args.brightness == 0.0
    assert args.contrast == 0.0
    assert args.workers is None
    assert not args.overwrite


def test_parser_accepts_adjustment_args():
    args = build_parser().parse_args(
        ["in.mp4", "-o", "out.mp4", "-b", "25.5", "-c", "-40", "--workers", "4"]
    )
    assert args.brightness == 25.5
    assert args.contrast == -40.0
    assert args.workers == 4


def test_missing_input_returns_input_not_found(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.mp4"), "-o", str(tmp_path / "out.mp4")])
    assert rc == InputNotFoundError.exit_code
    assert "错误" in capsys.readouterr().err


def test_existing_output_returns_exists_code(tmp_path, capsys):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")  # 只用于越过存在性检查，不会走到探测
    out = tmp_path / "out.mp4"
    out.write_bytes(b"occupied")
    rc = main([str(src), "-o", str(out)])
    assert rc == OutputExistsError.exit_code
    assert out.read_bytes() == b"occupied"
