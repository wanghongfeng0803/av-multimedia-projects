"""结构化清单的计数不变量、原子落盘与回读测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import json
import os

import pytest

from bvp.contract import ErrorCategory, SkipReason
from bvp.manifest import (
    MANIFEST_VERSION,
    Manifest,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    default_manifest_paths,
    load_manifest,
    make_failure_entry,
    make_skip_entry,
    make_success_entry,
)


def _entry_success(index: int, out: str) -> dict:
    return make_success_entry(index, f"in{index}.mp4", out, 100 + index)


def test_summary_counts_derive_from_entries_only():
    manifest = Manifest()
    manifest.add(_entry_success(0, "out0.mp4"))
    manifest.add(make_failure_entry(
        1, "in1.mp4", "out1.mp4", ErrorCategory.TRANSCODE_FAILED,
        "boom", "FFmpegRunError",
    ))
    manifest.add(make_skip_entry(
        2, "in2.mp4", None, ErrorCategory.NOT_A_VIDEO,
        SkipReason.NON_VIDEO, "not a video", "InvalidInputError",
    ))
    assert manifest.summary() == {
        "total": 3, "success": 1, "failed": 1, "skipped": 1,
    }


def test_invariant_total_equals_three_buckets():
    manifest = Manifest()
    for i in range(5):
        manifest.add(_entry_success(i, f"out{i}.mp4"))
    summary = manifest.summary()
    assert summary["total"] == summary["success"] + summary["failed"] \
        + summary["skipped"]


def test_duplicate_index_rejected_at_write(tmp_path):
    manifest = Manifest()
    manifest.add(_entry_success(0, "a.mp4"))
    manifest.add(_entry_success(0, "b.mp4"))
    with pytest.raises(ValueError, match="index 必须唯一"):
        manifest.write(str(tmp_path))


def test_skip_entry_requires_reason():
    entry = make_skip_entry(
        0, "in.mp4", None, ErrorCategory.NOT_A_VIDEO,
        SkipReason.NON_VIDEO, "d", "InvalidInputError",
    )
    manifest = Manifest(entries=[entry])
    # 手工破坏 skip_reason 后必须被不变量拦截
    manifest.entries[0].skip_reason = None
    with pytest.raises(ValueError, match="skip_reason"):
        manifest.write(".")


def test_failure_entry_requires_category():
    entry = make_failure_entry(
        0, "in.mp4", None, ErrorCategory.INTERNAL_ERROR,
        "d", "RuntimeError",
    )
    entry.category = None
    manifest = Manifest(entries=[entry])
    with pytest.raises(ValueError, match="category"):
        manifest.write(".")


def test_three_manifests_written_and_consistent(tmp_path):
    manifest = Manifest()
    manifest.add(_entry_success(0, str(tmp_path / "ok.mp4")))
    manifest.add(make_failure_entry(
        1, "bad.mp4", str(tmp_path / "bad_out.mp4"),
        ErrorCategory.TRANSCODE_FAILED, "boom", "FFmpegRunError",
    ))
    manifest.add(make_skip_entry(
        2, "fake.mp4", None, ErrorCategory.NOT_A_VIDEO,
        SkipReason.NON_VIDEO, "not video", "InvalidInputError",
    ))
    paths = manifest.write(str(tmp_path), run_info={"tool": "bvp"})

    assert set(os.path.basename(p) for p in paths.values()) == {
        "batch_manifest.json", "batch_failures.json", "batch_skipped.json",
    }

    with open(paths["full"], encoding="utf-8") as handle:
        full = json.load(handle)
    with open(paths["failed"], encoding="utf-8") as handle:
        failed = json.load(handle)
    with open(paths["skipped"], encoding="utf-8") as handle:
        skipped = json.load(handle)

    assert full["manifest_version"] == MANIFEST_VERSION
    assert full["summary"] == {
        "total": 3, "success": 1, "failed": 1, "skipped": 1,
    }
    # 子清单计数与全量口径一致
    assert failed["count"] == 1 and failed["kind"] == "failed"
    assert skipped["count"] == 1 and skipped["kind"] == "skipped"
    assert failed["summary"] == full["summary"]
    assert skipped["summary"] == full["summary"]
    # 条目按 index 排序写出
    assert [e["index"] for e in full["entries"]] == [0, 1, 2]


def test_roundtrip_load_manifest_preserves_entries(tmp_path):
    manifest = Manifest()
    manifest.add(_entry_success(0, str(tmp_path / "ok.mp4")))
    manifest.add(make_skip_entry(
        1, "fake.mp4", None, ErrorCategory.NOT_A_VIDEO,
        SkipReason.NON_VIDEO, "x", "InvalidInputError",
    ))
    paths = manifest.write(str(tmp_path))

    loaded = load_manifest(paths["full"])
    assert loaded.summary() == manifest.summary()
    assert loaded.entries[0].status == STATUS_SUCCESS
    assert loaded.entries[1].status == STATUS_SKIPPED
    assert loaded.entries[1].skip_reason == "non_video"


def test_atomic_write_leaves_no_tmp_files(tmp_path):
    manifest = Manifest(entries=[_entry_success(0, str(tmp_path / "ok.mp4"))])
    manifest.write(str(tmp_path))
    leftovers = [
        name for name in os.listdir(str(tmp_path))
        if name.endswith(".tmp")
    ]
    assert leftovers == []


def test_default_manifest_paths_are_standard(tmp_path):
    paths = default_manifest_paths(str(tmp_path))
    assert paths["full"].endswith("batch_manifest.json")
    assert paths["failed"].endswith("batch_failures.json")
    assert paths["skipped"].endswith("batch_skipped.json")


def test_illegal_status_rejected():
    manifest = Manifest()
    with pytest.raises(ValueError):
        manifest.add(type(
            "Bad", (), {
                "index": 0, "input_path": "x", "status": "running",
            },
        )())
