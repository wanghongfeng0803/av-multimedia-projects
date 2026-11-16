"""批量编排层测试：用注入的假核心覆盖全部容错与契约路径（不依赖 ffmpeg）。

这些用例锁定本工程最关键的行为：

1. 单文件失败/陌生异常绝不中断整体，每个输入恰好一条清单条目；
2. 批量层只按 :func:`classify_error` 的稳定分类划分失败/跳过，不解析文本；
3. 清单计数与「假核心实际写出的文件」一一对得上，且自动复核通过；
4. 遍历与清单顺序确定，嵌套目录、不可访问路径、输出已存在都正确入账。
"""

from __future__ import annotations

import json
import os

import pytest

from bvp.batch import batch_transcode
from bvp.contract import ErrorCategory, SkipReason
from bvp.errors import (
    BinaryNotFoundError,
    FFmpegRunError,
    InvalidInputError,
    OutputExistsError,
)
from bvp.manifest import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    load_manifest,
)
from bvp.planning import EXIST_OVERWRITE, EXIST_SKIP, EXIST_SUFFIX


# ---------------------------------------------------------------------------
# 假核心：与真实 transcode 同签名，按输入名决定成功/抛错，并真实落盘
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, output_path: str):
        self.output_path = output_path


def make_fake_transcode(behavior=None, *, record_calls=None):
    """构造假转码函数。

    :param behavior: 字典，输入 basename → None(成功) / Exception 实例(抛出)；
        缺省全部成功
    """
    behavior = behavior or {}

    def _fake(input_path, output_path=None, **kwargs):
        if record_calls is not None:
            record_calls.append((input_path, output_path, kwargs))
        key = os.path.basename(input_path)
        outcome = behavior.get(key)
        if isinstance(outcome, BaseException):
            raise outcome
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(b"fake-bytes")
        return FakeResult(output_path)

    return _fake


def _touch_video(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\x00")
    return path


# ---------------------------------------------------------------------------
# 全成功路径：数字与产出文件一一对应
# ---------------------------------------------------------------------------

def test_all_success_counts_match_files_on_disk(tmp_path):
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        _touch_video(str(tmp_path / name))
    out_dir = str(tmp_path / "out")

    result = batch_transcode(
        [str(tmp_path)], output_dir=out_dir,
        transcode_fn=make_fake_transcode(),
    )

    summary = result.summary
    assert summary == {"total": 3, "success": 3, "failed": 0, "skipped": 0}
    # 磁盘上确实有 3 个产物，且清单记录的大小与之一致
    produced = [
        p for p in (os.path.join(out_dir, f) for f in os.listdir(out_dir))
        if p.endswith(".mp4")
    ]
    assert len(produced) == 3
    assert result.reconciliation.ok
    assert result.success is True
    for entry in result.manifest.success_entries:
        assert os.path.isfile(entry.output_path)
        assert entry.output_size == os.path.getsize(entry.output_path)


def test_manifest_files_written_and_reloadable(tmp_path):
    _touch_video(str(tmp_path / "a.mp4"))
    out_dir = str(tmp_path / "out")
    result = batch_transcode(
        [str(tmp_path)], output_dir=out_dir,
        transcode_fn=make_fake_transcode(),
    )
    loaded = load_manifest(result.manifest_paths["full"])
    assert loaded.summary() == result.summary


# ---------------------------------------------------------------------------
# 容错：单文件失败不中断整体
# ---------------------------------------------------------------------------

def test_single_failure_does_not_abort_batch(tmp_path):
    for name in ("good1.mp4", "bad.mp4", "good2.mp4"):
        _touch_video(str(tmp_path / name))
    fake = make_fake_transcode({
        "bad.mp4": FFmpegRunError("转码炸了"),
    })
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        transcode_fn=fake,
    )
    summary = result.summary
    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["failed"] == 1
    # 后续 good2 仍被处理 → 失败没有中断整体
    assert result.reconciliation.ok

    failed = result.manifest.failed_entries[0]
    assert failed.input_path.endswith("bad.mp4")
    assert failed.category == ErrorCategory.TRANSCODE_FAILED.value
    assert failed.error_type == "FFmpegRunError"
    # 批量层不解析文本，但人读 detail 仍保留现场
    assert "转码炸了" in failed.detail


def test_unknown_exception_is_classified_internal_and_continues(tmp_path):
    for name in ("weird.mp4", "good.mp4"):
        _touch_video(str(tmp_path / name))
    fake = make_fake_transcode({
        "weird.mp4": RuntimeError("核心契约外的意外"),
    })
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        transcode_fn=fake,
    )
    assert result.summary["failed"] == 1
    assert result.summary["success"] == 1
    entry = result.manifest.failed_entries[0]
    assert entry.category == ErrorCategory.INTERNAL_ERROR.value
    assert result.reconciliation.ok


def test_binary_missing_is_failure_not_skip(tmp_path):
    _touch_video(str(tmp_path / "a.mp4"))
    fake = make_fake_transcode({
        "a.mp4": BinaryNotFoundError("没有 ffmpeg"),
    })
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        transcode_fn=fake,
    )
    entry = result.manifest.failed_entries[0]
    assert entry.category == ErrorCategory.BINARY_MISSING.value
    assert entry.status == STATUS_FAILED


# ---------------------------------------------------------------------------
# 跳过：非视频/无法识别 与 输出已存在
# ---------------------------------------------------------------------------

def test_non_video_input_is_skipped_not_failed(tmp_path):
    _touch_video(str(tmp_path / "real.mp4"))
    # 伪装成 mp4 的文本文件
    fake_path = str(tmp_path / "pretend.mp4")
    with open(fake_path, "w", encoding="utf-8") as handle:
        handle.write("not a video" * 10)

    fake = make_fake_transcode({
        "pretend.mp4": InvalidInputError("ffprobe 无法解析"),
    })
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        transcode_fn=fake,
    )
    summary = result.summary
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert summary["success"] == 1

    skipped = result.manifest.skipped_entries[0]
    assert skipped.category == ErrorCategory.NOT_A_VIDEO.value
    assert skipped.skip_reason == SkipReason.NON_VIDEO.value
    assert skipped.status == STATUS_SKIPPED
    # 跳过条目不应有输出文件，复核据此通过
    assert result.reconciliation.ok


def test_output_exists_skip_is_recorded_without_calling_core(tmp_path):
    src = _touch_video(str(tmp_path / "clip.mp4"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    preexisting = out_dir / "clip_h264.mp4"
    preexisting.write_text("old output", encoding="utf-8")

    calls: list = []
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(out_dir),
        on_exist=EXIST_SKIP,
        transcode_fn=make_fake_transcode(record_calls=calls),
    )
    # 跳过的任务根本没调用核心
    assert calls == []
    entry = result.manifest.skipped_entries[0]
    assert entry.skip_reason == SkipReason.OUTPUT_EXISTS.value
    assert entry.category == ErrorCategory.OUTPUT_EXISTS.value
    assert result.summary == {
        "total": 1, "success": 0, "failed": 0, "skipped": 1,
    }
    assert result.reconciliation.ok


def test_overwrite_policy_invokes_core_for_preexisting(tmp_path):
    _touch_video(str(tmp_path / "clip.mp4"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip_h264.mp4").write_text("old", encoding="utf-8")

    calls: list = []
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(out_dir),
        on_exist=EXIST_OVERWRITE,
        transcode_fn=make_fake_transcode(record_calls=calls),
    )
    assert len(calls) == 1
    assert calls[0][2]["overwrite"] is True
    assert result.summary["success"] == 1


def test_suffix_policy_reroutes_output(tmp_path):
    _touch_video(str(tmp_path / "clip.mp4"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip_h264.mp4").write_text("old", encoding="utf-8")

    result = batch_transcode(
        [str(tmp_path)], output_dir=str(out_dir),
        on_exist=EXIST_SUFFIX,
        transcode_fn=make_fake_transcode(),
    )
    entry = result.manifest.success_entries[0]
    assert entry.output_path.endswith("clip_h264_2.mp4")


# ---------------------------------------------------------------------------
# 不可访问路径与嵌套目录
# ---------------------------------------------------------------------------

def test_missing_input_path_is_failure_and_batch_continues(tmp_path):
    _touch_video(str(tmp_path / "good.mp4"))
    ghost = str(tmp_path / "ghost.mp4")
    result = batch_transcode(
        [str(tmp_path), ghost], output_dir=str(tmp_path / "out"),
        transcode_fn=make_fake_transcode(),
    )
    # 扫描问题记一条失败，真实视频正常成功
    assert result.summary["failed"] == 1
    assert result.summary["success"] == 1
    problem_entry = next(
        e for e in result.manifest.entries if e.input_path == os.path.abspath(ghost)
    )
    assert problem_entry.category == ErrorCategory.INPUT_NOT_FOUND.value
    assert result.reconciliation.ok


def test_nested_directories_processed_in_deterministic_order(tmp_path):
    _touch_video(str(tmp_path / "z.mp4"))
    _touch_video(str(tmp_path / "sub" / "a.mp4"))
    _touch_video(str(tmp_path / "sub" / "deep" / "m.mp4"))

    calls: list = []
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        jobs=1, transcode_fn=make_fake_transcode(record_calls=calls),
    )
    # 清单条目按 index 排序，index 来自确定性扫描顺序
    inputs = [
        os.path.relpath(e.input_path, str(tmp_path)).replace(os.sep, "/")
        for e in result.manifest.entries
    ]
    assert inputs == [
        os.path.join("sub", "a.mp4").replace(os.sep, "/"),
        os.path.join("sub", "deep", "m.mp4").replace(os.sep, "/"),
        "z.mp4",
    ]
    # 输出目录镜像了嵌套结构
    assert os.path.isfile(
        os.path.join(str(tmp_path / "out"), "sub", "deep", "m_h264.mp4")
    )


def test_concurrent_jobs_produce_same_manifest_as_sequential(tmp_path):
    for i in range(12):
        _touch_video(str(tmp_path / f"v{i:02d}.mp4"))
    out_parallel = str(tmp_path / "out_p")

    result = batch_transcode(
        [str(tmp_path)], output_dir=out_parallel, jobs=4,
        transcode_fn=make_fake_transcode(),
    )
    assert result.summary["success"] == 12
    assert result.reconciliation.ok
    # 并发不影响条目顺序与计数
    assert [e.index for e in result.manifest.entries] == list(range(12))


# ---------------------------------------------------------------------------
# 输出目录在输入目录内：不得把上一轮产物当新输入
# ---------------------------------------------------------------------------

def test_output_dir_inside_input_is_excluded_from_rescan(tmp_path):
    _touch_video(str(tmp_path / "src.mp4"))
    out_dir = str(tmp_path / "outputs")
    first = batch_transcode(
        [str(tmp_path)], output_dir=out_dir,
        transcode_fn=make_fake_transcode(),
    )
    assert first.summary["success"] == 1

    # 第二轮：outputs 里的产物不能被当成新输入再次转码
    second = batch_transcode(
        [str(tmp_path)], output_dir=out_dir,
        on_exist=EXIST_OVERWRITE,
        transcode_fn=make_fake_transcode(),
    )
    assert second.summary["total"] == 1
    assert second.summary["success"] == 1


# ---------------------------------------------------------------------------
# 失败与跳过清单分别落盘
# ---------------------------------------------------------------------------

def test_failure_and_skip_manifests_are_separate_files(tmp_path):
    _touch_video(str(tmp_path / "good.mp4"))
    _touch_video(str(tmp_path / "bad.mp4"))
    _touch_video(str(tmp_path / "fake.mp4"))
    fake = make_fake_transcode({
        "bad.mp4": FFmpegRunError("fail"),
        "fake.mp4": InvalidInputError("not video"),
    })
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        transcode_fn=fake,
    )
    with open(result.manifest_paths["failed"], encoding="utf-8") as handle:
        failed_doc = json.load(handle)
    with open(result.manifest_paths["skipped"], encoding="utf-8") as handle:
        skipped_doc = json.load(handle)
    assert failed_doc["count"] == 1
    assert skipped_doc["count"] == 1
    assert failed_doc["entries"][0]["input_path"].endswith("bad.mp4")
    assert skipped_doc["entries"][0]["input_path"].endswith("fake.mp4")


def test_progress_callback_receives_each_terminal_entry(tmp_path):
    for name in ("a.mp4", "b.mp4"):
        _touch_video(str(tmp_path / name))
    snapshots: list = []
    batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        progress_callback=lambda manifest: snapshots.append(manifest.total),
        transcode_fn=make_fake_transcode(),
    )
    assert snapshots == [1, 2]


def test_callback_exception_does_not_break_batch(tmp_path):
    _touch_video(str(tmp_path / "a.mp4"))

    def _boom(_manifest):
        raise RuntimeError("回调炸了")

    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        progress_callback=_boom,
        transcode_fn=make_fake_transcode(),
    )
    assert result.summary["success"] == 1


def test_invalid_arguments_fail_fast_before_scan(tmp_path):
    with pytest.raises(ValueError):
        batch_transcode([], transcode_fn=make_fake_transcode())
    with pytest.raises(ValueError):
        batch_transcode(
            [str(tmp_path)], jobs=0,
            transcode_fn=make_fake_transcode(),
        )
    with pytest.raises(ValueError):
        batch_transcode(
            [str(tmp_path)], codec="av1",
            transcode_fn=make_fake_transcode(),
        )


def test_every_input_gets_exactly_one_manifest_entry(tmp_path):
    names = [f"f{i}.mp4" for i in range(7)]
    for name in names:
        _touch_video(str(tmp_path / name))
    behavior = {
        "f1.mp4": FFmpegRunError("x"),
        "f3.mp4": InvalidInputError("x"),
        "f5.mp4": OSError("陌生异常"),
    }
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        transcode_fn=make_fake_transcode(behavior),
    )
    assert result.manifest.total == 7
    assert len({e.index for e in result.manifest.entries}) == 7
    assert result.summary == {
        "total": 7, "success": 4, "failed": 2, "skipped": 1,
    }
    assert result.reconciliation.ok


def test_keyboard_interrupt_propagates_not_swallowed(tmp_path):
    # KeyboardInterrupt 属于控制流，必须穿透批量层，不能被记成一条失败
    for name in ("a.mp4", "b.mp4"):
        _touch_video(str(tmp_path / name))

    def _interrupting_core(input_path, output_path=None, **kwargs):
        if os.path.basename(input_path) == "a.mp4":
            raise KeyboardInterrupt()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(b"x")
        return FakeResult(output_path)

    with pytest.raises(KeyboardInterrupt):
        batch_transcode(
            [str(tmp_path)], output_dir=str(tmp_path / "out"),
            transcode_fn=_interrupting_core,
        )


def test_indexes_unique_when_scan_problems_mix_with_planned_jobs(tmp_path):
    # 两个真实规划任务 + 两个不可访问路径：两套编号不得撞号
    _touch_video(str(tmp_path / "a.mp4"))
    _touch_video(str(tmp_path / "b.mp4"))
    ghosts = [str(tmp_path / "g1.mp4"), str(tmp_path / "g2.mp4")]

    result = batch_transcode(
        [str(tmp_path)] + ghosts,
        output_dir=str(tmp_path / "out"),
        transcode_fn=make_fake_transcode(),
    )
    indexes = [e.index for e in result.manifest.entries]
    assert len(indexes) == len(set(indexes)) == 4
    assert result.summary == {
        "total": 4, "success": 2, "failed": 2, "skipped": 0,
    }
    # 排序后 index 连续无缺
    assert sorted(indexes) == [0, 1, 2, 3]
    assert result.reconciliation.ok
