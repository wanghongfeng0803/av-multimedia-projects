"""清单与磁盘产出对账复核的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import json
import os

from bvp.contract import ErrorCategory, SkipReason
from bvp.manifest import (
    Manifest,
    default_manifest_paths,
    make_failure_entry,
    make_skip_entry,
    make_success_entry,
)
from bvp.verify import reconcile_manifest, reconcile_manifest_files


def _success(manifest: Manifest, tmp_path, index: int, *, size: int = 100,
             write: bool = True) -> str:
    out = str(tmp_path / f"out{index}.mp4")
    if write:
        with open(out, "wb") as handle:
            handle.write(b"x" * size)
    manifest.add(make_success_entry(index, f"in{index}.mp4", out, size))
    return out


def test_clean_manifest_reconciles(tmp_path):
    manifest = Manifest()
    _success(manifest, tmp_path, 0)
    _success(manifest, tmp_path, 1)
    result = reconcile_manifest(manifest)
    assert result.ok
    assert result.produced_outputs == 2
    assert result.mismatches == []


def test_success_entry_missing_on_disk_is_mismatch(tmp_path):
    manifest = Manifest()
    _success(manifest, tmp_path, 0, write=True)
    _success(manifest, tmp_path, 1, write=False)  # 清单说成功，磁盘没文件
    result = reconcile_manifest(manifest)
    assert not result.ok
    kinds = [m.kind for m in result.mismatches]
    assert "success_output_missing" in kinds


def test_size_recorded_must_match_disk(tmp_path):
    out = str(tmp_path / "out0.mp4")
    with open(out, "wb") as handle:
        handle.write(b"x" * 50)
    manifest = Manifest(entries=[
        make_success_entry(0, "in0.mp4", out, 999),  # 记录大小与实际不符
    ])
    result = reconcile_manifest(manifest)
    assert not result.ok
    assert result.mismatches[0].kind == "size_mismatch"


def test_failure_entry_must_have_no_output(tmp_path):
    out = str(tmp_path / "failed_out.mp4")
    with open(out, "wb") as handle:  # 半成品未清理
        handle.write(b"junk")
    manifest = Manifest(entries=[
        make_failure_entry(
            0, "in.mp4", out, ErrorCategory.TRANSCODE_FAILED,
            "boom", "FFmpegRunError",
        ),
    ])
    result = reconcile_manifest(manifest)
    assert not result.ok
    assert result.mismatches[0].kind == "failed_output_present"


def test_failure_without_output_reconciles(tmp_path):
    manifest = Manifest(entries=[
        make_failure_entry(
            0, "in.mp4", None, ErrorCategory.BINARY_MISSING,
            "no ffmpeg", "BinaryNotFoundError",
        ),
    ])
    result = reconcile_manifest(manifest)
    assert result.ok


def test_non_video_skip_must_have_no_output(tmp_path):
    out = str(tmp_path / "fake_out.mp4")
    with open(out, "wb") as handle:
        handle.write(b"junk")
    manifest = Manifest(entries=[
        make_skip_entry(
            0, "fake.mp4", out, ErrorCategory.NOT_A_VIDEO,
            SkipReason.NON_VIDEO, "not video", "InvalidInputError",
        ),
    ])
    result = reconcile_manifest(manifest)
    assert not result.ok
    assert result.mismatches[0].kind == "skipped_output_present"


def test_output_exists_skip_expects_preexisting_file(tmp_path):
    out = str(tmp_path / "pre.mp4")
    with open(out, "wb") as handle:
        handle.write(b"old")
    manifest = Manifest(entries=[
        make_skip_entry(
            0, "in.mp4", out, ErrorCategory.OUTPUT_EXISTS,
            SkipReason.OUTPUT_EXISTS, "exists", "OutputExistsError",
        ),
    ])
    result = reconcile_manifest(manifest)
    assert result.ok
    # 预存产物计入 existing 但不计入本次 produced
    assert result.existing_outputs == 1
    assert result.produced_outputs == 0


def test_duplicate_success_outputs_flagged(tmp_path):
    out = str(tmp_path / "shared.mp4")
    with open(out, "wb") as handle:
        handle.write(b"x")
    manifest = Manifest(entries=[
        make_success_entry(0, "a.mp4", out, 1),
        make_success_entry(1, "b.mp4", out, 1),
    ])
    result = reconcile_manifest(manifest)
    assert not result.ok
    assert "duplicate_output" in [m.kind for m in result.mismatches]


def test_file_based_reconciliation_cross_checks_submanifests(tmp_path):
    manifest = Manifest()
    _success(manifest, tmp_path, 0)
    manifest.add(make_failure_entry(
        1, "bad.mp4", None, ErrorCategory.TRANSCODE_FAILED,
        "boom", "FFmpegRunError",
    ))
    manifest.add(make_skip_entry(
        2, "fake.mp4", None, ErrorCategory.NOT_A_VIDEO,
        SkipReason.NON_VIDEO, "x", "InvalidInputError",
    ))
    manifest.write(str(tmp_path))

    result = reconcile_manifest_files(str(tmp_path))
    assert result.ok
    assert result.summary == {
        "total": 3, "success": 1, "failed": 1, "skipped": 1,
    }
    assert result.produced_outputs == 1


def test_file_based_reconciliation_detects_tampered_submanifest(tmp_path):
    manifest = Manifest()
    _success(manifest, tmp_path, 0)
    manifest.add(make_failure_entry(
        1, "bad.mp4", None, ErrorCategory.TRANSCODE_FAILED,
        "boom", "FFmpegRunError",
    ))
    paths = manifest.write(str(tmp_path))

    # 篡改失败清单的计数，交叉校验必须发现
    with open(paths["failed"], encoding="utf-8") as handle:
        tampered = json.load(handle)
    tampered["count"] = 99
    with open(paths["failed"], "w", encoding="utf-8") as handle:
        json.dump(tampered, handle)

    result = reconcile_manifest_files(str(tmp_path))
    assert not result.ok
    assert "submanifest_count_mismatch" in [
        m.kind for m in result.mismatches
    ]


def test_unreadable_full_manifest_returns_structured_failure(tmp_path):
    result = reconcile_manifest_files(str(tmp_path))
    assert not result.ok
    assert result.mismatches[0].kind == "manifest_unreadable"


def test_empty_success_output_is_flagged(tmp_path):
    out = str(tmp_path / "empty.mp4")
    open(out, "wb").close()  # 0 字节
    manifest = Manifest(entries=[
        make_success_entry(0, "in.mp4", out, 0),
    ])
    result = reconcile_manifest(manifest)
    assert not result.ok
    assert "output_empty" in [m.kind for m in result.mismatches]
