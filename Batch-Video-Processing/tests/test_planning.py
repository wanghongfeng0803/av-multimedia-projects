"""输出路径规划的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import os

import pytest

from bvp.planning import (
    EXIST_OVERWRITE,
    EXIST_SKIP,
    EXIST_SUFFIX,
    default_output_path,
    mirror_output_path,
    plan_outputs,
)


def test_default_output_naming_and_self_overwrite_guard(tmp_path, monkeypatch):
    src = str(tmp_path / "clip.mp4")
    out = default_output_path(src, "h264", ".mp4")
    assert os.path.basename(out) == "clip_h264.mp4"

    # 自覆盖守卫：当拼出的默认路径与输入路径完全相同时，必须改名为
    # <stem>_transcoded<ext>，绝不能让输出覆盖输入。
    # 用 monkeypatch 把 abspath 对“默认候选路径”的判定固定为与输入相等，
    # 稳定且无关于具体 codec 命名地触发该分支。
    target = os.path.abspath(str(tmp_path / "clip_h264.mp4"))
    real_abspath = os.path.abspath

    def _fake_abspath(path):
        resolved = real_abspath(path)
        # 让函数内部对 default 的相等判断成立
        if resolved.endswith("clip_h264_h264.mp4"):
            return target
        return resolved

    monkeypatch.setattr(os.path, "abspath", _fake_abspath)
    guarded = default_output_path(
        str(tmp_path / "clip_h264.mp4"), "h264", ".mp4"
    )
    assert os.path.basename(guarded) == "clip_h264_transcoded.mp4"


def test_mirror_preserves_nested_relative_structure(tmp_path):
    root = str(tmp_path / "in")
    src = os.path.join(root, "sub", "deep", "clip.mov")
    out_dir = str(tmp_path / "out")
    out = mirror_output_path(
        os.path.abspath(src), os.path.abspath(root),
        os.path.abspath(out_dir), "h264", ".mp4",
    )
    rel = os.path.relpath(out, os.path.abspath(out_dir))
    assert rel.replace(os.sep, "/") == os.path.join(
        "sub", "deep", "clip_h264.mp4"
    ).replace(os.sep, "/")


def test_plan_marks_pre_existing_outputs_for_skip(tmp_path):
    src = str(tmp_path / "clip.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "clip_h264.mp4"
    existing.write_text("old", encoding="utf-8")

    jobs = plan_outputs(
        (os.path.abspath(src),), [str(tmp_path)],
        output_dir=str(out_dir), extension=".mp4",
        codec="h264", on_exist=EXIST_SKIP,
    )
    assert len(jobs) == 1
    assert jobs[0].pre_exists is True


def test_plan_overwrite_does_not_mark_skip(tmp_path):
    src = str(tmp_path / "clip.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip_h264.mp4").write_text("old", encoding="utf-8")

    jobs = plan_outputs(
        (os.path.abspath(src),), [str(tmp_path)],
        output_dir=str(out_dir), extension=".mp4",
        codec="h264", on_exist=EXIST_OVERWRITE,
    )
    assert jobs[0].pre_exists is False


def test_same_batch_name_collision_gets_unique_outputs(tmp_path):
    # a.mov 与 a.mp4 同名异源，镜像后 stem 相同 → 必须自动改名防互覆盖
    files = (
        os.path.abspath(str(tmp_path / "a.mov")),
        os.path.abspath(str(tmp_path / "a.mp4")),
    )
    out_dir = str(tmp_path / "out")
    jobs = plan_outputs(
        files, [str(tmp_path)], output_dir=out_dir,
        extension=".mp4", codec="h264", on_exist=EXIST_SKIP,
    )
    outputs = [j.output_path for j in jobs]
    assert len(set(outputs)) == 2
    assert any(p.endswith("a_h264_2.mp4") for p in outputs)


def test_suffix_policy_renames_pre_existing(tmp_path):
    src = str(tmp_path / "clip.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip_h264.mp4").write_text("old", encoding="utf-8")

    jobs = plan_outputs(
        (os.path.abspath(src),), [str(tmp_path)],
        output_dir=str(out_dir), extension=".mp4",
        codec="h264", on_exist=EXIST_SUFFIX,
    )
    assert jobs[0].output_path.endswith("clip_h264_2.mp4")
    assert jobs[0].pre_exists is False


def test_invalid_on_exist_raises_before_any_work(tmp_path):
    with pytest.raises(ValueError):
        plan_outputs(
            (str(tmp_path / "a.mp4"),), [str(tmp_path)],
            output_dir=None, extension=".mp4",
            codec="h264", on_exist="bogus",
        )


def test_job_order_matches_input_order(tmp_path):
    files = tuple(
        os.path.abspath(str(tmp_path / name))
        for name in ("c.mp4", "a.mp4", "b.mp4")
    )
    jobs = plan_outputs(
        files, [str(tmp_path)], output_dir=None,
        extension=".mp4", codec="h264",
    )
    assert [j.index for j in jobs] == [0, 1, 2]
    assert [os.path.basename(j.input_path) for j in jobs] == [
        "c.mp4", "a.mp4", "b.mp4"
    ]
