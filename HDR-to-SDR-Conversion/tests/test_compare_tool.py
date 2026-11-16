"""对比工具 compare_tonemappers 的冒烟测试。

合成模式不依赖 ffmpeg，验证产出文件齐全、数值可复算且结论满足
「Hable 高光细节不弱于 Reinhard」的硬指标；真实素材模式的覆盖在
test_pipeline.py 的端到端用例中。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_TOOLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import compare_tonemappers as cmp_tool  # noqa: E402


def test_synthetic_outputs_written(tmp_path):
    outdir = str(tmp_path)
    cmp_tool.write_anchor_csv(os.path.join(outdir, "anchor.csv"))
    report = cmp_tool.write_detail_json(os.path.join(outdir, "detail.json"))
    cmp_tool.write_curve_svg(os.path.join(outdir, "curves.svg"))

    assert os.path.getsize(os.path.join(outdir, "anchor.csv")) > 0
    assert os.path.getsize(os.path.join(outdir, "curves.svg")) > 0
    with open(os.path.join(outdir, "detail.json"), encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == report


def test_detail_report_says_hable_not_weaker():
    # 每个高光带：级数不减少、几何均值斜率不低于 Reinhard
    for band in cmp_tool.write_detail_json(os.devnull)["bands"]:
        assert band["hable_levels_8bit"] >= band["reinhard_levels_8bit"]
        assert band["slope_ratio_hable_over_reinhard"] >= 1.0
        assert band["hable_not_weaker"] is True


def test_anchor_table_is_deterministic(tmp_path):
    p1 = os.path.join(str(tmp_path), "a.csv")
    p2 = os.path.join(str(tmp_path), "b.csv")
    cmp_tool.write_anchor_csv(p1)
    cmp_tool.write_anchor_csv(p2)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() == f2.read()


def test_anchor_csv_contains_known_golden_row(tmp_path):
    path = os.path.join(str(tmp_path), "anchor.csv")
    cmp_tool.write_anchor_csv(path)
    with open(path, encoding="utf-8") as f:
        rows = {line.split(",")[0]: line for line in f.read().splitlines()}
    # x=1.0 行：Hable 0.8457401033、Reinhard 0.875（与单测 golden 同源）
    cells = rows["1.0000"].split(",")
    assert cells[1].startswith("0.8750000000")
    assert cells[3].startswith("0.8457401033")


def test_svg_is_well_formed(tmp_path):
    path = os.path.join(str(tmp_path), "c.svg")
    cmp_tool.write_curve_svg(path)
    with open(path, encoding="utf-8") as f:
        svg = f.read()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "Reinhard" in svg and "Hable" in svg


def test_frame_interval_parser():
    assert cmp_tool._parse_frame_interval("0-29") == (0, 29)
    assert cmp_tool._parse_frame_interval("10-10") == (10, 10)
