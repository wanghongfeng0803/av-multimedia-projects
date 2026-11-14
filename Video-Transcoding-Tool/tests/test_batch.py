"""批量转码模块测试。

分为两部分：
- 纯逻辑测试（扫描、规划、队列、聚合进度、错误日志、参数校验）不启动
  任何子进程，无 ffmpeg 环境也能运行；
- 端到端测试复用 conftest 的 ffmpeg 样本夹具，验证真实批量转码、
  并发、跳过/覆盖/改名策略与 CLI 链路。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from transcode_tool.batch import (
    BatchErrorLogger,
    BatchProgress,
    EXIST_OVERWRITE,
    EXIST_SKIP,
    EXIST_SUFFIX,
    FAIL_STOP,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    _JobQueue,
    batch_transcode,
    plan_jobs,
    scan_video_files,
)
from transcode_tool.codecs import resolve_codec
from transcode_tool.containers import resolve_container

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="环境中未安装 ffmpeg"
)

H264 = resolve_codec("h264")
MP4 = resolve_container("mp4")


# ---------------------------------------------------------------------------
# 纯逻辑：目录扫描
# ---------------------------------------------------------------------------

def _touch(path: str, content: str = "x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def test_scan_flat_filters_by_extension(tmp_path):
    _touch(str(tmp_path / "a.mp4"))
    _touch(str(tmp_path / "b.mkv"))
    _touch(str(tmp_path / "note.txt"))
    _touch(str(tmp_path / "sub" / "c.mp4"))

    found = scan_video_files([str(tmp_path)])
    names = sorted(os.path.basename(p) for p in found)
    # 非递归：子目录中的 c.mp4 不入选；非视频扩展名被过滤
    assert names == ["a.mp4", "b.mkv"]


def test_scan_recursive_and_dedup(tmp_path):
    _touch(str(tmp_path / "sub" / "deep" / "d.webm"))
    _touch(str(tmp_path / "a.mp4"))

    found = scan_video_files([str(tmp_path)], recursive=True)
    assert {os.path.basename(p) for p in found} == {"a.mp4", "d.webm"}

    # 同一路径给两次只收一次
    again = scan_video_files([str(tmp_path), str(tmp_path)], recursive=True)
    assert len(again) == len(found)


def test_scan_custom_extensions_and_nonexistent_input(tmp_path):
    _touch(str(tmp_path / "a.mp4"))
    _touch(str(tmp_path / "b.xyz"))
    found = scan_video_files(
        [str(tmp_path), str(tmp_path / "ghost")],
        extensions=frozenset({".xyz"}),
    )
    assert [os.path.basename(p) for p in found] == ["b.xyz"]


def test_scan_exclude_dirs_skips_output_tree(tmp_path):
    out_dir = tmp_path / "out"
    _touch(str(out_dir / "product.mp4"))
    _touch(str(tmp_path / "src.mp4"))
    found = scan_video_files(
        [str(tmp_path)], recursive=True,
        exclude_dirs=frozenset({str(out_dir)}),
    )
    assert [os.path.basename(p) for p in found] == ["src.mp4"]


def test_scan_explicit_file_ignores_extension_filter(tmp_path):
    # 直接给定的文件即使扩展名不在集合内也入选（按内容交由 ffprobe 裁决）
    weird = tmp_path / "clip.data"
    _touch(str(weird))
    found = scan_video_files(
        [str(weird)], extensions=frozenset({".mp4"})
    )
    assert found == (os.path.abspath(str(weird)),)


def test_scan_empty_extension_set_rejected(tmp_path):
    with pytest.raises(ValueError):
        scan_video_files([str(tmp_path)], extensions=frozenset())


# ---------------------------------------------------------------------------
# 纯逻辑：输出路径规划
# ---------------------------------------------------------------------------

def test_plan_jobs_default_naming_and_mirror(tmp_path):
    src_root = tmp_path / "media"
    files = (
        os.path.abspath(str(src_root / "a.mp4")),
        os.path.abspath(str(src_root / "sub" / "b.mov")),
    )
    out_dir = tmp_path / "out"
    jobs = plan_jobs(
        files, [str(src_root)], output_dir=str(out_dir),
        codec=H264, container=MP4, on_exist=EXIST_SKIP,
    )
    assert jobs[0].output_path == os.path.abspath(str(out_dir / "a_h264.mp4"))
    # 子目录结构被镜像，扩展名统一为容器扩展名
    assert jobs[1].output_path == os.path.abspath(
        str(out_dir / "sub" / "b_h264.mp4")
    )


def test_plan_jobs_output_dir_none_uses_default_name(tmp_path):
    video = os.path.abspath(str(tmp_path / "a.mp4"))
    jobs = plan_jobs(
        (video,), [str(tmp_path)], output_dir=None,
        codec=H264, container=MP4, on_exist=EXIST_SKIP,
    )
    assert jobs[0].output_path == os.path.abspath(str(tmp_path / "a_h264.mp4"))


def test_plan_jobs_skip_overwrite_suffix(tmp_path):
    video = os.path.abspath(str(tmp_path / "a.mp4"))
    existing = os.path.abspath(str(tmp_path / "a_h264.mp4"))
    _touch(existing)

    skipped = plan_jobs(
        (video,), [str(tmp_path)], output_dir=None,
        codec=H264, container=MP4, on_exist=EXIST_SKIP,
    )
    assert skipped[0].output_path == existing  # 路径保留，跳过由调度层裁决

    overwritten = plan_jobs(
        (video,), [str(tmp_path)], output_dir=None,
        codec=H264, container=MP4, on_exist=EXIST_OVERWRITE,
    )
    assert overwritten[0].output_path == existing

    suffixed = plan_jobs(
        (video,), [str(tmp_path)], output_dir=None,
        codec=H264, container=MP4, on_exist=EXIST_SUFFIX,
    )
    assert suffixed[0].output_path == os.path.abspath(
        str(tmp_path / "a_h264_2.mp4")
    )


def test_plan_jobs_dedupes_same_stem_different_ext(tmp_path):
    # a.mov 与 a.mp4 同名不同扩展名，默认输出撞名时后者自动加序号
    files = (
        os.path.abspath(str(tmp_path / "a.mov")),
        os.path.abspath(str(tmp_path / "a.mp4")),
    )
    jobs = plan_jobs(
        files, [str(tmp_path)], output_dir=None,
        codec=H264, container=MP4, on_exist=EXIST_SKIP,
    )
    outputs = {j.output_path for j in jobs}
    assert len(outputs) == 2
    assert any(p.endswith("a_h264_2.mp4") for p in outputs)


def test_plan_jobs_rejects_bad_policy(tmp_path):
    with pytest.raises(ValueError):
        plan_jobs(
            (), [], output_dir=None, codec=H264,
            container=MP4, on_exist="bogus",
        )


# ---------------------------------------------------------------------------
# 纯逻辑：手写队列、聚合进度、错误日志
# ---------------------------------------------------------------------------

def test_job_queue_fifo_and_close(tmp_path):
    from transcode_tool.batch import BatchJob

    jobs = [BatchJob(i, str(tmp_path / f"f{i}.mp4"),
                     str(tmp_path / f"o{i}.mp4"), None) for i in range(3)]
    queue = _JobQueue(jobs)
    assert queue.get().index == 0
    assert queue.get().index == 1
    queue.close()
    assert queue.get().index == 2       # 已入队任务仍可领完
    assert queue.get() is None          # 排空后返回 None


def test_batch_progress_aggregation(tmp_path):
    from transcode_tool.batch import BatchJob

    jobs = [BatchJob(i, str(tmp_path / f"f{i}.mp4"),
                     str(tmp_path / f"o{i}.mp4"), None) for i in range(2)]
    progress = BatchProgress(jobs)

    snap0 = progress.snapshot()
    assert snap0.overall_ratio == 0.0
    assert snap0.total == 2

    progress.mark(0, STATUS_RUNNING)
    progress.update_ratio(0, 0.5)
    snap1 = progress.snapshot()
    assert snap1.running == 1
    assert abs(snap1.overall_ratio - 0.25) < 1e-9  # 0.5 / 2

    progress.mark(0, STATUS_DONE)
    progress.mark(1, STATUS_FAILED)
    snap2 = progress.snapshot()
    assert snap2.done == 1 and snap2.failed == 1
    assert snap2.overall_ratio == 1.0  # 终态任务按 100% 计入
    assert snap2.finished == 2


def test_progress_callback_clamps_and_marks(tmp_path):
    from transcode_tool.batch import BatchJob

    job = BatchJob(0, str(tmp_path / "f.mp4"),
                   str(tmp_path / "o.mp4"), None)
    progress = BatchProgress([job])
    cb = progress.make_callback(0)
    # QUEUED 状态在快照中按 0 计；先进入 RUNNING，比例才参与聚合
    progress.mark(0, STATUS_RUNNING)
    cb(9.9)   # 超界值应被夹到 1.0
    assert progress.snapshot().overall_ratio == 1.0
    cb(-1.0)  # 负值夹到 0.0
    assert progress.snapshot().overall_ratio == 0.0


def test_error_logger_writes_structured_lines(tmp_path):
    from transcode_tool.batch import BatchJob

    log_path = str(tmp_path / "errors.log")
    job = BatchJob(0, str(tmp_path / "in.mp4"),
                   str(tmp_path / "out.mp4"), None)
    with BatchErrorLogger(log_path) as logger:
        logger.log_skip(job, "输出文件已存在")
        logger.log_error(job, RuntimeError("模拟失败"), phase="validation")

    content = open(log_path, encoding="utf-8").read()
    assert "批量转码开始" in content
    assert "[WARN]" in content and "[ERROR]" in content
    assert "RuntimeError" in content and "模拟失败" in content
    assert "in.mp4" in content
    assert "批量转码结束" in content


def test_batch_transcode_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        batch_transcode([], output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        batch_transcode([str(tmp_path)], jobs=0)
    with pytest.raises(ValueError):
        batch_transcode([str(tmp_path)], on_fail="bogus")
    with pytest.raises(ValueError):
        batch_transcode([str(tmp_path)], on_exist="bogus")


# ---------------------------------------------------------------------------
# 端到端：真实 ffmpeg 批量转码
# ---------------------------------------------------------------------------

def _make_inputs(work_dir: str, ffmpeg: str, count: int) -> list[str]:
    paths = []
    for i in range(count):
        path = os.path.join(work_dir, f"clip_{i}.mp4")
        os.makedirs(work_dir, exist_ok=True)
        proc = subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc=size=160x120:rate=10:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-shortest", path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 0
        paths.append(path)
    return paths


def test_batch_transcode_end_to_end(work_dir, ffmpeg):
    inputs = _make_inputs(work_dir, ffmpeg, 3)
    out_dir = os.path.join(work_dir, "out")
    snapshots = []

    report = batch_transcode(
        inputs, output_dir=out_dir, preset="ultrafast",
        progress_callback=lambda s: snapshots.append(s),
    )

    assert report.success
    assert report.total == 3 and report.done_count == 3
    assert report.failed_count == 0
    for i in range(3):
        product = os.path.join(out_dir, f"clip_{i}_h264.mp4")
        assert os.path.isfile(product) and os.path.getsize(product) > 0
    # 进度回调被触发且最终收敛到 100%
    assert snapshots and snapshots[-1].overall_ratio == 1.0
    # 成功时批量日志仍存在（记录开始/结束与每条任务流水）
    assert os.path.isfile(report.log_path)


def test_batch_transcode_concurrent_jobs(work_dir, ffmpeg):
    inputs = _make_inputs(work_dir, ffmpeg, 4)
    out_dir = os.path.join(work_dir, "out")
    report = batch_transcode(
        inputs, output_dir=out_dir, jobs=2, preset="ultrafast",
        progress_callback=lambda s: None,
    )
    assert report.success and report.done_count == 4
    assert len([o for o in report.outcomes if o.status == STATUS_DONE]) == 4


def test_batch_skip_existing_output(work_dir, ffmpeg):
    inputs = _make_inputs(work_dir, ffmpeg, 2)
    out_dir = os.path.join(work_dir, "out")
    first = batch_transcode(inputs, output_dir=out_dir, preset="ultrafast")
    assert first.done_count == 2

    second = batch_transcode(
        inputs, output_dir=out_dir, preset="ultrafast", on_exist=EXIST_SKIP
    )
    assert second.done_count == 0
    assert second.skipped_count == 2
    assert second.success  # 全部跳过也算整批成功


def test_batch_overwrite_and_suffix(work_dir, ffmpeg):
    inputs = _make_inputs(work_dir, ffmpeg, 1)
    out_dir = os.path.join(work_dir, "out")
    batch_transcode(inputs, output_dir=out_dir, preset="ultrafast")
    product = os.path.join(out_dir, "clip_0_h264.mp4")
    size_before = os.path.getsize(product)

    batch_transcode(
        inputs, output_dir=out_dir, preset="ultrafast",
        on_exist=EXIST_OVERWRITE,
    )
    assert os.path.isfile(product)

    suffix_report = batch_transcode(
        inputs, output_dir=out_dir, preset="ultrafast",
        on_exist=EXIST_SUFFIX,
    )
    extra = os.path.join(out_dir, "clip_0_h264_2.mp4")
    assert os.path.isfile(extra)
    assert suffix_report.done_count == 1
    assert size_before > 0


def test_batch_continues_past_invalid_input(work_dir, ffmpeg, hevc_file):
    bogus = os.path.join(work_dir, "fake.mp4")
    with open(bogus, "w", encoding="utf-8") as fh:
        fh.write("not a video\n" * 10)
    good = os.path.join(work_dir, "good.mp4")
    shutil.copy(hevc_file, good)
    out_dir = os.path.join(work_dir, "out")

    report = batch_transcode(
        [bogus, good], output_dir=out_dir, preset="ultrafast",
    )
    assert not report.success
    assert report.failed_count == 1 and report.done_count == 1
    failed = next(o for o in report.outcomes if o.status == STATUS_FAILED)
    assert failed.input_path == os.path.abspath(bogus)
    assert failed.error_type  # 异常类型已记录
    # 失败现场写入错误日志
    log_content = open(report.log_path, encoding="utf-8").read()
    assert "fake.mp4" in log_content and "[ERROR]" in log_content
    # 好文件的产物不受影响
    assert os.path.isfile(os.path.join(out_dir, "good_h264.mp4"))


def test_batch_stop_on_first_failure(work_dir, ffmpeg):
    # 坏文件名排在最前（扫描结果按路径排序），单并发下它第一个被处理；
    # stop 策略触发后，后面两个好文件应保持未执行（cancelled）
    bogus = os.path.join(work_dir, "aaa_fake.mp4")
    with open(bogus, "w", encoding="utf-8") as fh:
        fh.write("bad\n")
    inputs = _make_inputs(work_dir, ffmpeg, 2)
    out_dir = os.path.join(work_dir, "out")

    report = batch_transcode(
        [bogus] + inputs, output_dir=out_dir, preset="ultrafast",
        on_fail=FAIL_STOP, jobs=1,
    )
    assert report.failed_count == 1
    assert report.cancelled_count == 2
    assert report.done_count == 0
    assert not report.success
    # 被取消的任务没有产生任何输出
    assert not os.path.exists(os.path.join(out_dir, "clip_0_h264.mp4"))


def test_batch_recursive_excludes_nested_output_dir(work_dir, ffmpeg):
    nested = os.path.join(work_dir, "nested")
    _make_inputs(nested, ffmpeg, 2)
    out_dir = os.path.join(work_dir, "out")
    report = batch_transcode(
        [work_dir], output_dir=out_dir, recursive=True, preset="ultrafast",
    )
    # 输出目录在输入目录内，递归扫描必须排除它，任务数只等于真实输入数
    assert report.total == 2
    assert report.done_count == 2


# ---------------------------------------------------------------------------
# CLI 链路
# ---------------------------------------------------------------------------

def test_cli_batch_success_json(work_dir, ffmpeg, capsys):
    from transcode_tool.cli import main

    _make_inputs(work_dir, ffmpeg, 2)
    out_dir = os.path.join(work_dir, "out")
    code = main([
        "batch", work_dir, "-o", out_dir, "--preset", "ultrafast",
        "--no-progress", "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["total"] == 2 and payload["done"] == 2


def test_cli_batch_failure_exit_code(work_dir, ffmpeg, capsys):
    from transcode_tool.cli import main

    bogus = os.path.join(work_dir, "fake.mp4")
    with open(bogus, "w", encoding="utf-8") as fh:
        fh.write("nope\n")
    out_dir = os.path.join(work_dir, "out")
    code = main([
        "batch", bogus, "-o", out_dir, "--preset", "ultrafast",
        "--no-progress",
    ])
    assert code == 6
    captured = capsys.readouterr()
    assert "失败" in captured.out


def test_cli_batch_bad_jobs_exit_1(work_dir, capsys):
    from transcode_tool.cli import main

    code = main(["batch", work_dir, "-j", "0", "--no-progress"])
    assert code == 1


def test_cli_batch_empty_directory(work_dir, ffmpeg, capsys):
    from transcode_tool.cli import main

    empty = os.path.join(work_dir, "empty")
    os.makedirs(empty, exist_ok=True)
    out_dir = os.path.join(work_dir, "out")
    code = main([
        "batch", empty, "-o", out_dir, "--no-progress", "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 0 and payload["success"] is True
