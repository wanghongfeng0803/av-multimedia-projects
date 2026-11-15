"""批量调度测试：任务隔离、共享配置、并发一致性、清单解析与 CLI 批量模式。"""

import json

import pytest

from watermark_overlay.batch import (
    BatchJob,
    config_from_dict,
    load_batch_manifest,
    run_batch,
)
from watermark_overlay.cli import main
from watermark_overlay.config import WatermarkConfig
from watermark_overlay.engine import overlay_files
from watermark_overlay.errors import InvalidParameterError
from watermark_overlay.image import read_png

from conftest import make_png


def _make_frames(tmp_path, count=3, size=(8, 6)):
    """生成 count 张输入帧，返回路径列表。"""
    paths = []
    for i in range(count):
        p = tmp_path / f"frame_{i}.png"
        make_png(str(p), size[0], size[1], channels=4,
                 fill=(40 + i * 10, 80, 120, 255))
        paths.append(str(p))
    return paths


def _make_watermark(tmp_path):
    wm = tmp_path / "wm.png"
    make_png(str(wm), 2, 2, channels=4, fill=(255, 0, 0, 200))
    return str(wm)


# ---------------------------------------------------------------- run_batch


def test_run_batch_all_success(tmp_path):
    frames = _make_frames(tmp_path)
    wm_path = _make_watermark(tmp_path)
    watermark = read_png(wm_path)
    jobs = [
        BatchJob(input_path=f, output_path=str(tmp_path / f"out_{i}.png"))
        for i, f in enumerate(frames)
    ]
    report = run_batch(jobs, watermark, WatermarkConfig(opacity=0.5))
    assert report.ok
    assert report.succeeded == 3
    assert report.failed == 0
    assert report.elapsed >= 0.0
    for outcome in report.outcomes:
        assert outcome.ok
        assert outcome.result is not None
        assert outcome.result.blended_pixels > 0
        assert outcome.elapsed >= 0.0
    # 输出确实落盘
    for i in range(3):
        assert (tmp_path / f"out_{i}.png").exists()


def test_run_batch_matches_single_overlay(tmp_path):
    """批量结果与单文件 overlay_files 逐字节一致（共享路径无行为差异）。"""
    frames = _make_frames(tmp_path, count=1)
    wm_path = _make_watermark(tmp_path)
    config = WatermarkConfig(anchor="bottom-right", opacity=0.7)

    ref_out = tmp_path / "ref.png"
    overlay_files(frames[0], wm_path, str(ref_out), config)

    batch_out = tmp_path / "batch.png"
    report = run_batch(
        [BatchJob(input_path=frames[0], output_path=str(batch_out))],
        read_png(wm_path),
        config,
    )
    assert report.ok
    assert ref_out.read_bytes() == batch_out.read_bytes()


def test_run_batch_failure_isolated(tmp_path):
    """中间任务输入缺失：其余任务照常成功，失败被记录而不是中断。"""
    frames = _make_frames(tmp_path)
    wm_path = _make_watermark(tmp_path)
    missing = str(tmp_path / "missing.png")
    jobs = [
        BatchJob(input_path=frames[0], output_path=str(tmp_path / "o0.png")),
        BatchJob(input_path=missing, output_path=str(tmp_path / "o1.png")),
        BatchJob(input_path=frames[2], output_path=str(tmp_path / "o2.png")),
    ]
    report = run_batch(jobs, read_png(wm_path))
    assert not report.ok
    assert report.succeeded == 2
    assert report.failed == 1
    # 顺序保持与任务一致
    assert report.outcomes[0].ok
    assert not report.outcomes[1].ok
    assert report.outcomes[1].error  # 有可读错误信息
    assert report.outcomes[2].ok
    # 失败任务的输出未生成，其余正常落盘
    assert (tmp_path / "o0.png").exists()
    assert not (tmp_path / "o1.png").exists()
    assert (tmp_path / "o2.png").exists()
    assert len(report.failures()) == 1


def test_run_batch_corrupt_png_isolated(tmp_path):
    """损坏的 PNG 只让对应任务失败。"""
    frames = _make_frames(tmp_path, count=1)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png at all")
    wm_path = _make_watermark(tmp_path)
    jobs = [
        BatchJob(input_path=str(bad), output_path=str(tmp_path / "o_bad.png")),
        BatchJob(input_path=frames[0], output_path=str(tmp_path / "o_ok.png")),
    ]
    report = run_batch(jobs, read_png(wm_path))
    assert report.failed == 1
    assert report.succeeded == 1


def test_run_batch_per_job_config_override(tmp_path):
    """任务级配置覆盖共享配置。"""
    frames = _make_frames(tmp_path, count=2)
    wm_path = _make_watermark(tmp_path)
    jobs = [
        BatchJob(input_path=frames[0], output_path=str(tmp_path / "shared.png")),
        BatchJob(
            input_path=frames[1],
            output_path=str(tmp_path / "override.png"),
            config=WatermarkConfig(anchor="top-left", opacity=1.0),
        ),
    ]
    report = run_batch(
        jobs, read_png(wm_path), WatermarkConfig(anchor="bottom-right")
    )
    assert report.ok
    # 覆盖任务的原点应落在左上 (0,0)，共享任务在右下
    assert report.outcomes[1].result.origin == (0, 0)
    assert report.outcomes[0].result.origin != (0, 0)


def test_run_batch_motion_job(tmp_path):
    """运动任务按 frame_index/fps 渲染，角度写入结果。"""
    from watermark_overlay.animation import Keyframe, MotionConfig

    frames = _make_frames(tmp_path, count=1)
    wm_path = _make_watermark(tmp_path)
    motion = MotionConfig(
        keyframes=[Keyframe(0.0, angle=0), Keyframe(1.0, angle=90)]
    )
    jobs = [
        BatchJob(
            input_path=frames[0],
            output_path=str(tmp_path / "anim.png"),
            frame_index=12,
            fps=24.0,
        )
    ]
    report = run_batch(jobs, read_png(wm_path), motion=motion)
    assert report.ok
    assert report.outcomes[0].result.angle != 0.0


def test_run_batch_workers_parallel_same_results(tmp_path):
    """并发模式与顺序模式产出逐字节一致。"""
    frames = _make_frames(tmp_path, count=4)
    wm_path = _make_watermark(tmp_path)
    watermark = read_png(wm_path)
    config = WatermarkConfig(opacity=0.6, adaptive=True)

    seq_jobs = [
        BatchJob(input_path=f, output_path=str(tmp_path / f"seq_{i}.png"))
        for i, f in enumerate(frames)
    ]
    par_jobs = [
        BatchJob(input_path=f, output_path=str(tmp_path / f"par_{i}.png"))
        for i, f in enumerate(frames)
    ]
    seq_report = run_batch(seq_jobs, watermark, config, workers=1)
    par_report = run_batch(par_jobs, watermark, config, workers=3)
    assert seq_report.ok and par_report.ok
    for i in range(4):
        assert (tmp_path / f"seq_{i}.png").read_bytes() == (
            tmp_path / f"par_{i}.png"
        ).read_bytes()


def test_run_batch_progress_callback(tmp_path):
    frames = _make_frames(tmp_path, count=3)
    wm_path = _make_watermark(tmp_path)
    jobs = [
        BatchJob(input_path=f, output_path=str(tmp_path / f"p_{i}.png"))
        for i, f in enumerate(frames)
    ]
    calls = []
    run_batch(
        jobs,
        read_png(wm_path),
        on_progress=lambda outcome, done, total: calls.append((done, total)),
    )
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_run_batch_creates_output_dirs(tmp_path):
    """输出目录不存在时自动创建。"""
    frames = _make_frames(tmp_path, count=1)
    wm_path = _make_watermark(tmp_path)
    nested = tmp_path / "deep" / "nested" / "out.png"
    report = run_batch(
        [BatchJob(input_path=frames[0], output_path=str(nested))],
        read_png(wm_path),
    )
    assert report.ok
    assert nested.exists()


def test_run_batch_invalid_workers(tmp_path):
    with pytest.raises(InvalidParameterError):
        run_batch([], read_png(_make_watermark(tmp_path)), workers=0)


def test_run_batch_empty_jobs(tmp_path):
    report = run_batch([], read_png(_make_watermark(tmp_path)))
    assert report.ok
    assert report.total == 0


# ------------------------------------------------------- config_from_dict


def test_config_from_dict_valid():
    config = config_from_dict({"anchor": "top-left", "opacity": 0.5})
    assert config.anchor == "top-left"
    assert config.opacity == 0.5


def test_config_from_dict_unknown_key_rejected():
    with pytest.raises(InvalidParameterError, match="未知字段"):
        config_from_dict({"opactiy": 0.5})  # 拼写错误不应被静默忽略


def test_config_from_dict_invalid_value():
    with pytest.raises(InvalidParameterError):
        config_from_dict({"opacity": 1.5})


# ------------------------------------------------------- load_batch_manifest


def _write_manifest(tmp_path, payload):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_manifest_relative_paths_resolved(tmp_path):
    """清单中的相对路径按清单所在目录解析。"""
    frames = _make_frames(tmp_path, count=1)
    wm_path = _make_watermark(tmp_path)
    import os

    manifest = _write_manifest(tmp_path, {
        "watermark": "wm.png",
        "config": {"opacity": 0.5},
        "jobs": [{"input": "frame_0.png", "output": "out/result.png"}],
    })
    spec = load_batch_manifest(manifest)
    assert spec.watermark_path == os.path.normpath(wm_path)
    assert spec.config.opacity == 0.5
    assert len(spec.jobs) == 1
    assert spec.jobs[0].input_path == os.path.normpath(frames[0])
    assert spec.jobs[0].output_path == os.path.normpath(
        str(tmp_path / "out" / "result.png")
    )
    assert spec.motion is None


def test_manifest_motion_inline_and_shared_fps(tmp_path):
    manifest = _write_manifest(tmp_path, {
        "watermark": "wm.png",
        "motion": {"keyframes": [{"time": 0}, {"time": 1, "angle": 90}]},
        "fps": 30.0,
        "jobs": [{"input": "a.png", "output": "b.png", "frame_index": 5}],
    })
    spec = load_batch_manifest(manifest)
    assert spec.motion is not None
    assert spec.jobs[0].fps == 30.0
    assert spec.jobs[0].frame_index == 5


def test_manifest_motion_external_file(tmp_path):
    motion = tmp_path / "motion.json"
    motion.write_text(
        json.dumps({"keyframes": [{"time": 0}, {"time": 1, "x": 5}]}),
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, {
        "watermark": "wm.png",
        "motion": "motion.json",
        "jobs": [{"input": "a.png", "output": "b.png"}],
    })
    spec = load_batch_manifest(manifest)
    assert spec.motion is not None
    assert len(spec.motion.keyframes) == 2


def test_manifest_per_job_overrides(tmp_path):
    manifest = _write_manifest(tmp_path, {
        "watermark": "wm.png",
        "jobs": [{
            "input": "a.png", "output": "b.png",
            "config": {"opacity": 0.3},
            "motion": {"keyframes": [{"time": 0}]},
            "fps": 60.0,
        }],
    })
    spec = load_batch_manifest(manifest)
    job = spec.jobs[0]
    assert job.config.opacity == 0.3
    assert job.motion is not None
    assert job.fps == 60.0


def test_manifest_missing_watermark_rejected(tmp_path):
    manifest = _write_manifest(tmp_path, {
        "jobs": [{"input": "a.png", "output": "b.png"}],
    })
    with pytest.raises(InvalidParameterError, match="watermark"):
        load_batch_manifest(manifest)


def test_manifest_empty_jobs_rejected(tmp_path):
    manifest = _write_manifest(tmp_path, {"watermark": "wm.png", "jobs": []})
    with pytest.raises(InvalidParameterError, match="jobs"):
        load_batch_manifest(manifest)


def test_manifest_job_missing_output_rejected(tmp_path):
    manifest = _write_manifest(tmp_path, {
        "watermark": "wm.png",
        "jobs": [{"input": "a.png"}],
    })
    with pytest.raises(InvalidParameterError, match="output"):
        load_batch_manifest(manifest)


def test_manifest_bad_json_rejected(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(InvalidParameterError, match="JSON"):
        load_batch_manifest(str(path))


def test_manifest_missing_file_rejected(tmp_path):
    with pytest.raises(InvalidParameterError):
        load_batch_manifest(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------- CLI 批量模式


def _write_cli_manifest(tmp_path, jobs):
    _make_watermark(tmp_path)
    return _write_manifest(tmp_path, {
        "watermark": "wm.png",
        "config": {"opacity": 0.6},
        "jobs": jobs,
    })


def test_cli_batch_success(tmp_path, capsys):
    frames = _make_frames(tmp_path, count=2)
    manifest = _write_cli_manifest(tmp_path, [
        {"input": "frame_0.png", "output": "out/o0.png"},
        {"input": "frame_1.png", "output": "out/o1.png"},
    ])
    code = main(["--batch", manifest])
    assert code == 0
    printed = capsys.readouterr().out
    assert "成功 2/2" in printed
    assert (tmp_path / "out" / "o0.png").exists()
    assert (tmp_path / "out" / "o1.png").exists()


def test_cli_batch_partial_failure_exit_1(tmp_path, capsys):
    _make_frames(tmp_path, count=1)
    manifest = _write_cli_manifest(tmp_path, [
        {"input": "frame_0.png", "output": "out/o0.png"},
        {"input": "missing.png", "output": "out/o1.png"},
    ])
    code = main(["--batch", manifest])
    assert code == 1
    captured = capsys.readouterr()
    assert "成功 1/2" in captured.out
    assert "失败" in captured.err
    # 失败不影响其余任务落盘
    assert (tmp_path / "out" / "o0.png").exists()


def test_cli_batch_bad_manifest_exit_2(tmp_path, capsys):
    path = tmp_path / "batch.json"
    path.write_text("{broken", encoding="utf-8")
    code = main(["--batch", str(path)])
    assert code == 2
    assert "错误" in capsys.readouterr().err


def test_cli_batch_workers_flag(tmp_path):
    frames = _make_frames(tmp_path, count=3)
    manifest = _write_cli_manifest(tmp_path, [
        {"input": f"frame_{i}.png", "output": f"out/o{i}.png"} for i in range(3)
    ])
    code = main(["--batch", manifest, "--workers", "2"])
    assert code == 0
    for i in range(3):
        assert (tmp_path / "out" / f"o{i}.png").exists()


def test_cli_single_mode_still_requires_args(capsys):
    """不带 --batch 且缺少位置参数/-o 时给出明确错误（退出码 2）。"""
    code = main([])
    assert code == 2
    assert "错误" in capsys.readouterr().err
