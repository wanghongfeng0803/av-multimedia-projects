"""3D LUT 节点的 ffmpeg 端到端集成测试。

覆盖：.cube 实文件接入、强度两端边界、节点位置、解析/域失败保留原图、
失败不留半成品、确定性、CLI 接线，以及节点化底层入口 adjust_nodes。
无 ffmpeg 时整模块自动跳过。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from color_grading import (
    BrightnessContrastNode,
    LUT3DNode,
    LUTDomainError,
    LUTParseError,
    adjust,
    adjust_nodes,
    load_cube,
)
from color_grading.binaries import find_ffmpeg
from color_grading.cli import main
from color_grading.lut3d import apply_chunk

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)


# --------------------------------------------------------------- .cube 夹具


def _write_cube(path, size: int, func) -> str:
    lines = [f"LUT_3D_SIZE {size}", 'TITLE "e2e"']
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rr, gg, bb = func(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append(f"{rr:.6f} {gg:.6f} {bb:.6f}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture()
def invert_cube(tmp_path):
    return _write_cube(tmp_path / "invert.cube", 8,
                       lambda r, g, b: (1 - r, 1 - g, 1 - b))


@pytest.fixture()
def warm_cube(tmp_path):
    # 暖调：压蓝、略提红，强度 0 时必须完全无影响。
    return _write_cube(tmp_path / "warm.cube", 6,
                       lambda r, g, b: (min(1, r * 1.1), g, b * 0.8))


def _decode_rgb24(path: str) -> bytes:
    proc = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-loglevel", "error",
         "-i", path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert proc.stdout
    return proc.stdout


# --------------------------------------------------------------- 基本接入


def test_lut3d_applied_and_result_fields(sample_av, invert_cube, tmp_path):
    out = str(tmp_path / "inv.mp4")
    result = adjust(sample_av, out, lut3d_path=invert_cube, workers=2, overwrite=True)
    assert result.frames == 10
    assert result.lut3d_path == invert_cube
    assert result.lut3d_strength == 1.0
    assert result.lut3d_position == "last"
    assert os.path.isfile(out)


def test_lut3d_strength_one_matches_direct_chunk(sample_av, warm_cube, tmp_path):
    """强度 1 的端到端输出，其像素等价于对解码帧直接 apply_chunk（编码容差内）。

    用温和暖调 LUT 而非反色 LUT：反色会把 yuv420p 的色度量化误差放大，
    不适合作为有损编码链路的精度基准。
    """
    out = str(tmp_path / "full.mp4")
    adjust(sample_av, out, lut3d_path=warm_cube, lut3d_strength=1.0,
           workers=2, overwrite=True)
    src = _decode_rgb24(sample_av)
    got = _decode_rgb24(out)
    cube = load_cube(warm_cube)
    expect = apply_chunk(src, cube, 1.0)
    mae = sum(abs(a - b) for a, b in zip(got, expect)) / len(got)
    assert mae < 3.0


def test_lut3d_strength_zero_keeps_original(sample_av, warm_cube, tmp_path):
    """强度 0：输出与不加 LUT 的基线一致（精确命中原图，编码容差内）。"""
    base = str(tmp_path / "base.mp4")
    out = str(tmp_path / "s0.mp4")
    adjust(sample_av, base, workers=2, overwrite=True)
    result = adjust(sample_av, out, lut3d_path=warm_cube, lut3d_strength=0.0,
                    workers=2, overwrite=True)
    assert result.lut3d_strength == 0.0
    s0 = _decode_rgb24(out)
    base_bytes = _decode_rgb24(base)
    assert len(s0) == len(base_bytes) and len(s0) > 0
    mae = sum(abs(a - b) for a, b in zip(s0, base_bytes)) / len(s0)
    assert mae < 1.5


def test_lut3d_warm_raises_red_blue_ratio(sample_av, warm_cube, tmp_path):
    out = str(tmp_path / "warm.mp4")
    adjust(sample_av, out, lut3d_path=warm_cube, workers=2, overwrite=True)
    src = _decode_rgb24(sample_av)
    got = _decode_rgb24(out)

    def ratio(data):
        rs = sum(data[0::3])
        bs = sum(data[2::3])
        return rs / max(1, bs)

    assert ratio(got) > ratio(src)


def test_lut3d_position_changes_output(sample_av, invert_cube, tmp_path):
    """first 与 last 位置产出不同文件，位置语义真实生效。"""
    out_first = str(tmp_path / "first.mp4")
    out_last = str(tmp_path / "last.mp4")
    adjust(sample_av, out_first, brightness=25, contrast=15,
           lut3d_path=invert_cube, lut3d_position="first",
           workers=2, overwrite=True)
    adjust(sample_av, out_last, brightness=25, contrast=15,
           lut3d_path=invert_cube, lut3d_position="last",
           workers=2, overwrite=True)
    # 像素均值不同即证明复合顺序改变了结果。
    assert sum(_decode_rgb24(out_first)) != sum(_decode_rgb24(out_last))


def test_lut3d_strength_is_monotonic(sample_av, warm_cube, tmp_path):
    """对加暖 LUT，强度 0/0.5/1 的红蓝比单调不减。"""
    ratios = []
    for s in (0.0, 0.5, 1.0):
        out = str(tmp_path / f"s{s}.mp4")
        adjust(sample_av, out, lut3d_path=warm_cube, lut3d_strength=s,
               workers=2, overwrite=True)
        data = _decode_rgb24(out)
        ratios.append(sum(data[0::3]) / max(1, sum(data[2::3])))
    assert ratios[0] <= ratios[1] <= ratios[2]


def test_lut3d_deterministic_two_runs(sample_av, invert_cube, tmp_path):
    """同一输入两次输出的解码像素一致（确定性，经两次独立进程池）。"""
    a = str(tmp_path / "a.mp4")
    b = str(tmp_path / "b.mp4")
    adjust(sample_av, a, lut3d_path=invert_cube, lut3d_strength=0.6,
           workers=2, overwrite=True)
    adjust(sample_av, b, lut3d_path=invert_cube, lut3d_strength=0.6,
           workers=2, overwrite=True)
    assert _decode_rgb24(a) == _decode_rgb24(b)


# --------------------------------------------------------------- 失败路径


def test_parse_failure_preserves_original_and_no_output(sample_av, tmp_path):
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 8\nonly three values\n", encoding="utf-8")
    out = str(tmp_path / "should_not_exist.mp4")
    with pytest.raises(LUTParseError) as ei:
        adjust(sample_av, out, lut3d_path=str(bad), workers=2)
    assert ei.value.line_number == 2
    assert not os.path.exists(out)  # 没有半成品
    assert os.path.isfile(sample_av)  # 原图保留


def test_domain_failure_preserves_original_and_no_output(sample_av, tmp_path):
    dom = tmp_path / "dom.cube"
    dom.write_text(
        "LUT_3D_SIZE 2\nDOMAIN_MIN -0.1 -0.1 -0.1\n"
        "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        encoding="utf-8",
    )
    out = str(tmp_path / "no_out.mp4")
    with pytest.raises(LUTDomainError):
        adjust(sample_av, out, lut3d_path=str(dom), workers=2)
    assert not os.path.exists(out)


def test_missing_cube_file_raises_before_output(sample_av, tmp_path):
    out = str(tmp_path / "x.mp4")
    with pytest.raises(LUTParseError):
        adjust(sample_av, out, lut3d_path=str(tmp_path / "missing.cube"), workers=2)
    assert not os.path.exists(out)


def test_pre_existing_output_untouched_on_lut_failure(sample_av, tmp_path):
    out = tmp_path / "occupied.mp4"
    out.write_bytes(b"keep-me")
    bad = tmp_path / "bad.cube"
    bad.write_text("not a cube at all\n", encoding="utf-8")
    with pytest.raises(LUTParseError):
        adjust(sample_av, str(out), lut3d_path=str(bad), overwrite=True, workers=2)
    assert out.read_bytes() == b"keep-me"  # 预先存在的文件绝不被破坏


def test_invalid_position_rejected(sample_av, invert_cube, tmp_path):
    with pytest.raises(ValueError):
        adjust(sample_av, str(tmp_path / "x.mp4"),
               lut3d_path=invert_cube, lut3d_position="middle", workers=2)


def test_strength_out_of_range_clamped_and_reported(sample_av, warm_cube, tmp_path):
    out = str(tmp_path / "clamp.mp4")
    with pytest.warns(UserWarning, match="strength"):
        result = adjust(sample_av, out, lut3d_path=warm_cube,
                        lut3d_strength=5.0, workers=2, overwrite=True)
    assert result.lut3d_strength == 1.0
    assert any("strength" in m for m in result.messages)


# --------------------------------------------------------------- 节点化入口


def test_adjust_nodes_runs_chain(sample_av, invert_cube, tmp_path):
    out = str(tmp_path / "nodes.mp4")
    cube = load_cube(invert_cube)
    chain = [
        BrightnessContrastNode(),
        LUT3DNode(cube, 0.7),
    ]
    result = adjust_nodes(sample_av, out, chain, workers=2, overwrite=True)
    assert result.frames == 10
    assert os.path.isfile(out)


# --------------------------------------------------------------- CLI


def test_cli_lut3d_end_to_end(sample_av, invert_cube, tmp_path, capsys):
    out = str(tmp_path / "cli_lut.mp4")
    rc = main([sample_av, "-o", out, "--lut3d", invert_cube,
               "--lut3d-strength", "0.5", "--lut3d-position", "first",
               "--workers", "2", "--overwrite"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "3DLUT[first]" in text and "0.5" in text


def test_cli_lut3d_parse_error_exit_code(sample_av, tmp_path, capsys):
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 4\n0 0 0\n", encoding="utf-8")
    rc = main([sample_av, "-o", str(tmp_path / "x.mp4"), "--lut3d", str(bad)])
    assert rc == LUTParseError.exit_code
    assert "错误" in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "x.mp4"))
