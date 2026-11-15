"""分散布局抗裁剪水印测试：布局确定性、网格覆盖、退化路径、参数校验与批量/CLI 集成。"""

import json

import pytest

from watermark_overlay.batch import BatchJob, load_batch_manifest, run_batch
from watermark_overlay.cli import main
from watermark_overlay.config import WatermarkConfig
from watermark_overlay.engine import overlay_files, overlay_image, overlay_scattered
from watermark_overlay.errors import InvalidParameterError
from watermark_overlay.image import read_png
from watermark_overlay.scatter import (
    ScatterConfig,
    compute_scatter_layout,
    scatter_from_dict,
)

from conftest import make_png, solid_image


def _frame(width=60, height=40):
    return solid_image(width, height, (40, 80, 120, 255))


def _wm(width=6, height=4):
    return solid_image(width, height, (255, 0, 0, 200))


# ---------------------------------------------------------------- 布局算法


def test_layout_count_and_determinism():
    config = ScatterConfig(rows=2, cols=3, jitter=0.5, angle_jitter=20, seed=42)
    a = compute_scatter_layout(60, 40, 6, 4, config)
    b = compute_scatter_layout(60, 40, 6, 4, config)
    assert len(a) == 6
    # 相同种子布局完全一致（位置与角度）
    assert [(p.x, p.y, p.angle) for p in a] == [(p.x, p.y, p.angle) for p in b]


def test_layout_seed_changes_layout():
    common = dict(rows=2, cols=3, jitter=1.0, angle_jitter=30)
    a = compute_scatter_layout(60, 40, 6, 4, ScatterConfig(seed=1, **common))
    b = compute_scatter_layout(60, 40, 6, 4, ScatterConfig(seed=2, **common))
    assert [(p.x, p.y, p.angle) for p in a] != [(p.x, p.y, p.angle) for p in b]


def test_layout_zero_jitter_centers_instances():
    """jitter=0 时实例严格位于单元格中心。"""
    config = ScatterConfig(rows=2, cols=3, jitter=0.0, angle_jitter=0.0)
    placements = compute_scatter_layout(60, 40, 6, 4, config)
    # 单元格 20x20，水印 6x4 → 每格居中偏移 (7, 8)
    expected = [(7 + 20 * c, 8 + 20 * r) for r in range(2) for c in range(3)]
    assert [(p.x, p.y) for p in placements] == expected
    assert all(p.angle == 0.0 for p in placements)


def test_layout_instances_stay_inside_cells_at_max_jitter():
    """jitter=1.0 时实例仍完整落在本单元格内（互不重叠）。"""
    config = ScatterConfig(rows=3, cols=4, jitter=1.0, seed=9)
    placements = compute_scatter_layout(80, 60, 6, 4, config)
    cell_w, cell_h = 80 / 4, 60 / 3
    for i, p in enumerate(placements):
        row, col = divmod(i, 4)
        assert col * cell_w <= p.x
        assert p.x + 6 <= (col + 1) * cell_w
        assert row * cell_h <= p.y
        assert p.y + 4 <= (row + 1) * cell_h


def test_layout_margin_shrinks_layout_area():
    config = ScatterConfig(rows=1, cols=1, margin=10, jitter=0.0)
    (p,) = compute_scatter_layout(60, 40, 6, 4, config)
    assert p.x >= 10 and p.y >= 10
    assert p.x + 6 <= 60 - 10 and p.y + 4 <= 40 - 10


def test_layout_angle_within_bounds():
    config = ScatterConfig(rows=4, cols=4, angle_jitter=45, seed=3)
    placements = compute_scatter_layout(100, 100, 4, 4, config)
    assert all(-45.0 <= p.angle <= 45.0 for p in placements)
    # 有抖动时角度不应全为 0
    assert any(p.angle != 0.0 for p in placements)


def test_layout_watermark_larger_than_cell_still_works():
    """水印大于单元格：按单元格中心居中，越界交给混合层裁剪，不报错。"""
    config = ScatterConfig(rows=2, cols=2, jitter=0.5)
    placements = compute_scatter_layout(20, 20, 16, 16, config)
    assert len(placements) == 4


# ---------------------------------------------------------------- 配置校验


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"cols": -1},
        {"rows": 2.5},
        {"rows": True},
        {"margin": -1},
        {"margin": 1.5},
        {"jitter": -0.1},
        {"jitter": 1.1},
        {"jitter": "0.5"},
        {"angle_jitter": -1},
        {"angle_jitter": 181},
        {"angle_jitter": float("nan")},
        {"seed": 1.5},
        {"seed": False},
    ],
)
def test_scatter_config_validation(kwargs):
    with pytest.raises(InvalidParameterError):
        ScatterConfig(**kwargs).validate()


def test_scatter_from_dict_roundtrip():
    config = scatter_from_dict(
        {"rows": 3, "cols": 4, "margin": 2, "jitter": 0.2,
         "angle_jitter": 15, "seed": 8}
    )
    assert (config.rows, config.cols, config.margin) == (3, 4, 2)
    assert (config.jitter, config.angle_jitter, config.seed) == (0.2, 15, 8)


def test_scatter_from_dict_rejects_unknown_key():
    with pytest.raises(InvalidParameterError):
        scatter_from_dict({"rows": 2, "col": 3})


def test_scatter_from_dict_rejects_non_object():
    with pytest.raises(InvalidParameterError):
        scatter_from_dict([1, 2, 3])


# ---------------------------------------------------------------- 引擎叠加


def test_overlay_scattered_deterministic_bytes():
    """相同参数两次叠加结果逐字节一致。"""
    config = WatermarkConfig(opacity=0.8)
    scatter = ScatterConfig(rows=2, cols=3, jitter=0.6, angle_jitter=25, seed=7)
    out1, _ = overlay_scattered(_frame(), _wm(), config, scatter)
    out2, _ = overlay_scattered(_frame(), _wm(), config, scatter)
    assert out1.pixels == out2.pixels


def test_overlay_scattered_anti_crop_coverage():
    """每个网格单元格内都有水印像素：裁掉任何单一区域，其余区域仍保留水印。"""
    frame, wm = _frame(90, 60), _wm(6, 4)
    scatter = ScatterConfig(rows=2, cols=3, jitter=0.0, seed=0)
    out, info = overlay_scattered(frame, wm, WatermarkConfig(), scatter)
    assert info.instance_count == 6
    assert info.blended_pixels == 6 * 6 * 4  # 6 个实例，每个 6x4 全部落在帧内

    cell_w, cell_h = 90 // 3, 60 // 2
    for row in range(2):
        for col in range(3):
            changed = 0
            for y in range(row * cell_h, (row + 1) * cell_h):
                for x in range(col * cell_w, (col + 1) * cell_w):
                    i = (y * 90 + x) * 4
                    if out.pixels[i:i + 4] != frame.pixels[i:i + 4]:
                        changed += 1
            assert changed == 6 * 4, f"单元格 ({row}, {col}) 缺少水印"


def test_overlay_scattered_matches_static_per_instance():
    """无抖动无旋转时，结果等同于在每个布局原点依次做静态叠加。"""
    frame, wm = _frame(60, 40), _wm(6, 4)
    scatter = ScatterConfig(rows=2, cols=3, jitter=0.0, angle_jitter=0.0)
    config = WatermarkConfig(opacity=0.7)
    scattered, _ = overlay_scattered(frame, wm, config, scatter)

    reference = frame.copy()
    for p in compute_scatter_layout(60, 40, 6, 4, scatter):
        reference, _ = overlay_image(
            reference,
            wm,
            WatermarkConfig(anchor="top-left", offset_x=p.x, offset_y=p.y,
                            opacity=0.7),
        )
    assert scattered.pixels == reference.pixels


def test_overlay_scattered_opacity_zero_passthrough():
    frame, wm = _frame(), _wm()
    out, info = overlay_scattered(
        frame, wm, WatermarkConfig(opacity=0.0), ScatterConfig()
    )
    assert out.pixels == frame.pixels
    assert info.blended_pixels == 0
    assert not info.overlapped


def test_overlay_scattered_with_angle_jitter():
    frame, wm = _frame(), _wm()
    out, info = overlay_scattered(
        frame, wm, WatermarkConfig(),
        ScatterConfig(rows=2, cols=2, angle_jitter=40, seed=5),
    )
    assert info.blended_pixels > 0
    assert any(angle != 0.0 for _, _, angle in info.instances)


def test_overlay_scattered_adaptive_mask():
    """自适应蒙版在分散布局下正常工作（细节图全帧只算一次）。"""
    frame, wm = _frame(), _wm()
    out, info = overlay_scattered(
        frame,
        wm,
        WatermarkConfig(adaptive=True, mask_strength=0.8),
        ScatterConfig(rows=2, cols=2, seed=1),
    )
    assert info.adaptive
    assert info.blended_pixels > 0


def test_overlay_scattered_oversized_watermark_clipped():
    """水印大于单元格甚至帧：逐像素裁剪，不报错。"""
    frame = _frame(20, 20)
    wm = _wm(18, 18)
    out, info = overlay_scattered(
        frame, wm, WatermarkConfig(), ScatterConfig(rows=2, cols=2)
    )
    assert info.blended_pixels > 0


def test_overlay_scattered_does_not_mutate_inputs():
    frame, wm = _frame(), _wm()
    frame_bytes, wm_bytes = bytes(frame.pixels), bytes(wm.pixels)
    overlay_scattered(
        frame, wm, WatermarkConfig(), ScatterConfig(angle_jitter=30, seed=2)
    )
    assert bytes(frame.pixels) == frame_bytes
    assert bytes(wm.pixels) == wm_bytes


# ---------------------------------------------------------------- 文件 / CLI


def test_overlay_files_scatter(tmp_path):
    frame_path = tmp_path / "frame.png"
    wm_path = tmp_path / "wm.png"
    out_path = tmp_path / "out.png"
    make_png(str(frame_path), 30, 20, channels=4, fill=(40, 80, 120, 255))
    make_png(str(wm_path), 4, 3, channels=4, fill=(255, 0, 0, 200))
    info = overlay_files(
        str(frame_path), str(wm_path), str(out_path),
        WatermarkConfig(), scatter=ScatterConfig(rows=2, cols=2, seed=1),
    )
    assert out_path.exists()
    assert info.instance_count == 4
    assert info.blended_pixels > 0


def test_overlay_files_scatter_motion_conflict(tmp_path):
    with pytest.raises(InvalidParameterError):
        overlay_files(
            "a", "b", "c",
            motion=object(), scatter=ScatterConfig(),
        )


def test_cli_scatter_success(tmp_path, capsys):
    frame = tmp_path / "frame.png"
    wm = tmp_path / "wm.png"
    out = tmp_path / "out.png"
    make_png(str(frame), 30, 20, channels=4, fill=(40, 80, 120, 255))
    make_png(str(wm), 4, 3, channels=4, fill=(255, 0, 0, 200))
    code = main([
        str(frame), str(wm), "-o", str(out),
        "--scatter", "--scatter-rows", "2", "--scatter-cols", "2",
        "--scatter-jitter", "0.5", "--scatter-seed", "3",
        "--scatter-angle-jitter", "10", "--opacity", "0.8",
    ])
    assert code == 0
    assert out.exists()
    assert "实例 4 个" in capsys.readouterr().out


def test_cli_scatter_motion_conflict_exit_2(tmp_path, capsys):
    frame = tmp_path / "frame.png"
    wm = tmp_path / "wm.png"
    motion = tmp_path / "motion.json"
    make_png(str(frame), 8, 6, channels=4)
    make_png(str(wm), 2, 2, channels=4)
    motion.write_text(
        json.dumps({"keyframes": [{"time": 0, "x": 0, "y": 0, "angle": 0}]}),
        encoding="utf-8",
    )
    code = main([
        str(frame), str(wm), "-o", str(tmp_path / "out.png"),
        "--scatter", "--motion", str(motion),
    ])
    assert code == 2
    assert "错误" in capsys.readouterr().err


def test_cli_scatter_invalid_params_exit_2(tmp_path, capsys):
    frame = tmp_path / "frame.png"
    wm = tmp_path / "wm.png"
    make_png(str(frame), 8, 6, channels=4)
    make_png(str(wm), 2, 2, channels=4)
    code = main([
        str(frame), str(wm), "-o", str(tmp_path / "out.png"),
        "--scatter", "--scatter-jitter", "1.5",
    ])
    assert code == 2
    assert "错误" in capsys.readouterr().err


# ---------------------------------------------------------------- 批量集成


def test_run_batch_with_scatter(tmp_path):
    frames = []
    for i in range(2):
        p = tmp_path / f"frame_{i}.png"
        make_png(str(p), 30, 20, channels=4, fill=(40 + i, 80, 120, 255))
        frames.append(str(p))
    wm_path = tmp_path / "wm.png"
    make_png(str(wm_path), 4, 3, channels=4, fill=(255, 0, 0, 200))

    jobs = [
        BatchJob(input_path=f, output_path=str(tmp_path / f"out_{i}.png"))
        for i, f in enumerate(frames)
    ]
    report = run_batch(
        jobs, read_png(str(wm_path)), WatermarkConfig(),
        scatter=ScatterConfig(rows=2, cols=2, seed=1),
    )
    assert report.ok
    for outcome in report.outcomes:
        assert outcome.result.instance_count == 4
        assert outcome.result.blended_pixels > 0


def test_run_batch_scatter_motion_conflict_isolated(tmp_path):
    """单个任务同时给出 scatter 与 motion：只该任务失败，其余继续。"""
    frame = tmp_path / "frame.png"
    make_png(str(frame), 30, 20, channels=4)
    wm_path = tmp_path / "wm.png"
    make_png(str(wm_path), 4, 3, channels=4)
    from watermark_overlay.animation import Keyframe, MotionConfig

    jobs = [
        BatchJob(input_path=str(frame), output_path=str(tmp_path / "ok.png")),
        BatchJob(
            input_path=str(frame), output_path=str(tmp_path / "bad.png"),
            motion=MotionConfig(keyframes=[Keyframe(0.0)]),
        ),
    ]
    report = run_batch(
        jobs, read_png(str(wm_path)), WatermarkConfig(),
        scatter=ScatterConfig(),
    )
    assert report.succeeded == 1
    assert report.failed == 1
    assert report.outcomes[0].ok
    assert not report.outcomes[1].ok


def test_batch_manifest_with_scatter(tmp_path):
    frame = tmp_path / "frame.png"
    make_png(str(frame), 30, 20, channels=4)
    wm_path = tmp_path / "wm.png"
    make_png(str(wm_path), 4, 3, channels=4)
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({
            "watermark": "wm.png",
            "scatter": {"rows": 2, "cols": 2, "seed": 4},
            "jobs": [{"input": "frame.png", "output": "out.png"}],
        }),
        encoding="utf-8",
    )
    spec = load_batch_manifest(str(manifest))
    assert spec.scatter is not None
    assert spec.scatter.rows == 2
    assert spec.scatter.seed == 4

    code = main(["--batch", str(manifest)])
    assert code == 0
    assert (tmp_path / "out.png").exists()


def test_batch_manifest_scatter_motion_conflict_exit_2(tmp_path, capsys):
    frame = tmp_path / "frame.png"
    make_png(str(frame), 8, 6, channels=4)
    wm_path = tmp_path / "wm.png"
    make_png(str(wm_path), 2, 2, channels=4)
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({
            "watermark": "wm.png",
            "motion": {"keyframes": [{"time": 0}]},
            "scatter": {"rows": 2},
            "jobs": [{"input": "frame.png", "output": "out.png"}],
        }),
        encoding="utf-8",
    )
    code = main(["--batch", str(manifest)])
    assert code == 2
    assert "错误" in capsys.readouterr().err
