"""批量重封装（batch_remux）测试。

围绕三层职责分别验证边界：
- **发现**：文件/目录混排、递归开关、后缀过滤、路径去重、确定性排序；
- **规划**：批内同名改名、磁盘已占用的三种策略（rename/overwrite/skip）、
  大小写不敏感碰撞；
- **执行**：逐项故障隔离（确定性错误 / 重试用尽 / 未预期异常）、
  stop-on-error、dry-run、真实管线端到端、CLI 退出码。
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mp4_to_mkv.errors import RemuxError

import batch_remux as br
from batch_remux import (
    ON_OVERWRITE,
    ON_RENAME,
    ON_SKIP,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    BatchResult,
    discover_inputs,
    main,
    plan_outputs,
    remux_batch,
)
from remux_all_tracks import RemuxRetryError
from tests.mp4_builder import build_mp4
from tests.ebml_reader import parse_document


def _video_mp4() -> bytes:
    return build_mp4(
        [b"\x00\x00\x00\x01\x67" + bytes([i]) * 8 for i in range(3)],
        keyframes=[1],
    )


def _make_tree(root: Path) -> dict[str, Path]:
    """构造一棵含同名文件、非媒体文件与嵌套子目录的测试树。

    root/
      a.mp4, note.txt, sub/b.mp4, sub/movie.mp4, sub/deep/c.MP4, other/movie.mp4
    """
    paths = {}
    (root / "sub" / "deep").mkdir(parents=True)
    (root / "other").mkdir(parents=True)
    for name in ("a.mp4", "sub/b.mp4", "sub/movie.mp4",
                 "sub/deep/c.MP4", "other/movie.mp4"):
        p = root / name
        p.write_bytes(_video_mp4())
        paths[name] = p
    (root / "note.txt").write_bytes(b"not media")
    return paths


class DiscoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = _make_tree(self.root)

    def test_recursive_discovers_all_mp4_suffixes_case_insensitive(self):
        found = discover_inputs([self.root])
        names = sorted(p.name.lower() for p in found)
        self.assertEqual(names, ["a.mp4", "b.mp4", "c.mp4", "movie.mp4", "movie.mp4"])
        # 非媒体后缀不进入清单
        self.assertFalse(any(p.name == "note.txt" for p in found))

    def test_non_recursive_only_scans_top_level(self):
        found = discover_inputs([self.root], recursive=False)
        self.assertEqual([p.name for p in found], ["a.mp4"])

    def test_explicit_file_kept_even_with_unknown_suffix(self):
        weird = self.root / "strange.data"
        weird.write_bytes(_video_mp4())
        found = discover_inputs([weird])
        self.assertEqual(found, [weird])

    def test_duplicate_paths_dedup_by_resolved_target(self):
        # 同一文件用不同写法（相对 / 绝对 / 重复 / 目录+文件混排）只出现一次
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            found = discover_inputs(
                [Path("a.mp4"), (self.root / "a.mp4").resolve(),
                 self.root, Path("./a.mp4")]
            )
            self.assertEqual(len(found), 5)  # 整树 5 个媒体文件，a 只算一次
        finally:
            os.chdir(cwd)

    def test_order_is_deterministic(self):
        f1 = discover_inputs([self.root / "other", self.root / "sub"])
        f2 = discover_inputs([self.root / "sub", self.root / "other"])
        self.assertEqual([str(p) for p in f1], [str(p) for p in f2])

    def test_missing_path_raises_remux_error(self):
        with self.assertRaises(RemuxError) as ctx:
            discover_inputs([self.root / "nope.mp4"])
        self.assertIn("不存在", str(ctx.exception))

    def test_empty_input_list_returns_empty(self):
        self.assertEqual(discover_inputs([]), [])


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"

    def test_rename_for_same_stem_from_different_dirs(self):
        sources = [self.root / "sub" / "movie.mp4",
                   self.root / "other" / "movie.mp4"]
        plan = plan_outputs(sources, self.out, on_existing=ON_RENAME)
        dsts = [dst for _, dst, _ in plan]
        self.assertEqual([p.name for p in dsts], ["movie.mkv", "movie_2.mkv"])
        self.assertTrue(all(reason is None for _, _, reason in plan))

    def test_rename_when_output_already_on_disk(self):
        self.out.mkdir()
        (self.out / "movie.mkv").write_bytes(b"occupied")
        plan = plan_outputs([self.root / "movie.mp4"], self.out)
        self.assertEqual(plan[0][1].name, "movie_2.mkv")

    def test_rename_counter_skips_multiple_occupied_slots(self):
        self.out.mkdir()
        (self.out / "movie.mkv").write_bytes(b"x")
        (self.out / "movie_2.mkv").write_bytes(b"x")
        plan = plan_outputs([self.root / "movie.mp4"], self.out)
        self.assertEqual(plan[0][1].name, "movie_3.mkv")

    def test_skip_policy_marks_skipped_with_reason(self):
        self.out.mkdir()
        (self.out / "movie.mkv").write_bytes(b"x")
        plan = plan_outputs([self.root / "movie.mp4"], self.out,
                            on_existing=ON_SKIP)
        src, dst, reason = plan[0]
        self.assertIsNone(dst)
        self.assertIn("skip", reason)

    def test_skip_policy_runs_when_absent(self):
        plan = plan_outputs([self.root / "movie.mp4"], self.out,
                            on_existing=ON_SKIP)
        self.assertEqual(plan[0][1].name, "movie.mkv")
        self.assertIsNone(plan[0][2])

    def test_overwrite_policy_reuses_name_despite_existing_file(self):
        self.out.mkdir()
        (self.out / "movie.mkv").write_bytes(b"old")
        plan = plan_outputs([self.root / "movie.mp4"], self.out,
                            on_existing=ON_OVERWRITE)
        self.assertEqual(plan[0][1].name, "movie.mkv")

    def test_case_insensitive_collision_in_reserved_set(self):
        # 即便所在文件系统区分大小写，批内规划也按大小写不敏感分配
        sources = [self.root / "Movie.mp4", self.root / "movie.mp4"]
        plan = plan_outputs(sources, self.out)
        names = {dst.name.lower() for _, dst, _ in plan}
        self.assertEqual(names, {"movie.mkv", "movie_2.mkv"})

    def test_invalid_policy_rejected(self):
        with self.assertRaises(RemuxError):
            plan_outputs([], self.out, on_existing="bogus")


class _RecordingRemux:
    """单文件管线桩：按源路径决定结果，记录全部调用参数。"""

    def __init__(self, fail_map=None, retry_error_map=None, boom_map=None):
        self.calls: list[tuple[str, str, bool]] = []
        self.fail_map = set(fail_map or [])
        self.retry_error_map = set(retry_error_map or [])
        self.boom_map = set(boom_map or [])

    def __call__(self, src, dst, *, overwrite, retries, retry_delay):
        self.calls.append((src, dst, overwrite))
        if src in self.fail_map:
            raise RemuxError("模拟确定性错误")
        if src in self.retry_error_map:
            raise RemuxRetryError("模拟瞬时错误用尽")
        if src in self.boom_map:
            raise RuntimeError("模拟未预期异常")
        Path(dst).write_bytes(b"MKZ")

        class _R:
            attempts = 1
        return _R()


class ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = _make_tree(self.root)
        self.out = self.root / "out"

    def test_real_pipeline_end_to_end_all_outputs_valid_mkv(self):
        report = remux_batch([self.root], self.out)
        self.assertIsInstance(report, BatchResult)
        self.assertEqual(report.total, 5)
        self.assertEqual(len(report.succeeded), 5)
        self.assertEqual(len(report.failed), 0)
        mkv_files = sorted(p.name for p in self.out.glob("*.mkv"))
        # 两个 movie.mp4 经改名策略都有独立输出
        self.assertEqual(
            mkv_files, ["a.mkv", "b.mkv", "c.mkv", "movie.mkv", "movie_2.mkv"]
        )
        for p in self.out.glob("*.mkv"):
            parse_document(p.read_bytes())  # 输出必须是自洽 EBML/MKV

    def test_deterministic_failure_is_isolated(self):
        bad = str(self.paths["a.mp4"])
        stub = _RecordingRemux(fail_map={bad})
        report = remux_batch([self.root], self.out, remux_fn=stub)
        self.assertEqual(len(report.failed), 1)
        self.assertEqual(len(report.succeeded), 4)
        item = report.failed[0]
        self.assertEqual(item.status, STATUS_FAILED)
        self.assertIn("确定性错误", item.reason)
        # 失败项没有产出文件，成功项全部产出
        self.assertFalse((self.out / "a.mkv").exists())
        self.assertEqual(len(stub.calls), 5)

    def test_retry_exhaustion_is_isolated_and_reported(self):
        bad = str(self.paths["a.mp4"])
        stub = _RecordingRemux(retry_error_map={bad})
        report = remux_batch([self.root], self.out, remux_fn=stub)
        self.assertEqual(len(report.failed), 1)
        self.assertIn("重试用尽", report.failed[0].reason)
        self.assertEqual(len(report.succeeded), 4)

    def test_unexpected_exception_does_not_kill_batch(self):
        bad = str(self.paths["a.mp4"])
        stub = _RecordingRemux(boom_map={bad})
        report = remux_batch([self.root], self.out, remux_fn=stub)
        self.assertEqual(len(report.failed), 1)
        self.assertIn("RuntimeError", report.failed[0].reason)
        self.assertEqual(len(report.succeeded), 4)

    def test_stop_on_error_marks_remaining_as_skipped(self):
        # 排序后 a.mp4 处于首位；令其失败，后续 4 项都应被跳过且不再调用
        bad = str(self.paths["a.mp4"])
        stub = _RecordingRemux(fail_map={bad})
        report = remux_batch([self.root], self.out,
                             stop_on_error=True, remux_fn=stub)
        self.assertEqual(len(report.failed), 1)
        self.assertEqual(len(report.skipped), 4)
        self.assertTrue(all("已停止" in r.reason for r in report.skipped))
        self.assertEqual(len(stub.calls), 1)  # 失败后立即停止，未再调用

    def test_dry_run_creates_nothing_and_marks_all_skipped(self):
        report = remux_batch([self.root], self.out, dry_run=True)
        self.assertFalse(self.out.exists())       # 输出目录也不创建
        self.assertEqual(len(report.skipped), 5)
        self.assertTrue(all(r.status == STATUS_SKIPPED for r in report.results))
        self.assertTrue(all("dry-run" in r.reason for r in report.results))
        # 规划仍给出了输出名
        self.assertTrue(all(r.output is not None for r in report.results))

    def test_skip_existing_integration(self):
        # 先正常跑一遍，再以 skip 策略重跑：全部跳过且不改动现有输出
        first = remux_batch([self.root], self.out)
        self.assertEqual(len(first.succeeded), 5)
        before = {p.name: p.read_bytes() for p in self.out.glob("*.mkv")}
        stub = _RecordingRemux()
        second = remux_batch([self.root], self.out,
                             on_existing=ON_SKIP, remux_fn=stub)
        self.assertEqual(len(second.skipped), 5)
        self.assertEqual(stub.calls, [])  # 跳过的项不进入单文件管线
        after = {p.name: p.read_bytes() for p in self.out.glob("*.mkv")}
        self.assertEqual(before, after)

    def test_overwrite_passes_overwrite_flag_down(self):
        self.out.mkdir()
        (self.out / "a.mkv").write_bytes(b"old")
        stub = _RecordingRemux()
        report = remux_batch(
            [self.paths["a.mp4"]], self.out,
            on_existing=ON_OVERWRITE, remux_fn=stub,
        )
        self.assertEqual(len(report.succeeded), 1)
        self.assertTrue(stub.calls[0][2])  # overwrite=True 下放
        self.assertEqual(stub.calls[0][1], str(self.out / "a.mkv"))

    def test_rename_policy_never_passes_overwrite(self):
        stub = _RecordingRemux()
        remux_batch([self.paths["a.mp4"]], self.out, remux_fn=stub)
        self.assertFalse(stub.calls[0][2])

    def test_output_dir_created_including_nested_parents(self):
        deep = self.root / "nested" / "deeper" / "mkv_out"
        report = remux_batch([self.paths["a.mp4"]], deep)
        self.assertTrue(deep.is_dir())
        self.assertEqual(len(report.succeeded), 1)

    def test_empty_batch_returns_empty_report_without_creating_dir(self):
        empty = self.root / "empty_dir"
        empty.mkdir()
        report = remux_batch([empty], self.out)
        self.assertEqual(report.total, 0)
        self.assertFalse(self.out.exists())

    def test_negative_retries_rejected_before_any_work(self):
        stub = _RecordingRemux()
        with self.assertRaises(RemuxError):
            remux_batch([self.paths["a.mp4"]], self.out,
                        retries=-1, remux_fn=stub)
        self.assertEqual(stub.calls, [])

    def test_missing_input_is_batch_level_error(self):
        stub = _RecordingRemux()
        with self.assertRaises(RemuxError):
            remux_batch([self.root / "ghost.mp4"], self.out, remux_fn=stub)
        self.assertEqual(stub.calls, [])

    def test_source_equal_to_output_path_fails_item_not_batch(self):
        # 显式输入不做后缀限制：用 .mkv 命名的 MP4 内容，使规划出的输出
        # 与源完全同路径。单文件前置校验拦下该项，属于单项失败而非整批崩溃
        src = self.root / "same.mkv"
        src.write_bytes(_video_mp4())
        report = remux_batch([src], src.parent, on_existing=ON_OVERWRITE)
        self.assertEqual(len(report.failed), 1)
        self.assertIn("同一个路径", report.failed[0].reason)

    def test_mixed_files_and_dirs_in_one_batch(self):
        report = remux_batch(
            [self.paths["a.mp4"], self.root / "sub"], self.out
        )
        # a.mp4 显式一份 + sub 下 3 个（b/movie/c），去重后共 4
        self.assertEqual(report.total, 4)
        self.assertEqual(len(report.succeeded), 4)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = _make_tree(self.root)
        self.out = self.root / "out"

    def _run(self, argv):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            code = main(argv)
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_cli_success_exit_code_and_summary(self):
        code, out, err = self._run([str(self.out), str(self.root)])
        self.assertEqual(code, br.EXIT_OK)
        self.assertIn("汇总：共 5 项，成功 5，失败 0，跳过 0", out)
        self.assertEqual(err, "")

    def test_cli_partial_failure_exit_code(self):
        # 放入一个 0 字节的“mp4”，确定性失败；其余文件照常成功
        (self.root / "broken.mp4").write_bytes(b"")
        code, out, err = self._run([
            str(self.out), str(self.root),
            "--retries", "0", "--retry-delay", "0",
        ])
        self.assertEqual(code, br.EXIT_BATCH_FAILED)
        self.assertIn("[失败]", err)
        self.assertIn("成功 5", out)  # 其余 5 个不受影响

    def test_cli_dry_run_exit_ok_and_no_files(self):
        code, out, _ = self._run([
            str(self.out), str(self.root), "--dry-run",
        ])
        self.assertEqual(code, br.EXIT_OK)
        self.assertFalse(self.out.exists())
        self.assertIn("dry-run", out)

    def test_cli_empty_input_dir_reports_nothing_to_do(self):
        empty = self.root / "empty"
        empty.mkdir()
        code, out, _ = self._run([str(self.out), str(empty)])
        self.assertEqual(code, br.EXIT_OK)
        self.assertIn("未发现", out)

    def test_cli_missing_input_exit_invalid(self):
        code, _, err = self._run([str(self.out), str(self.root / "x.mp4")])
        self.assertEqual(code, br.EXIT_INVALID_INPUT)
        self.assertIn("不存在", err)

    def test_cli_mutually_existent_flags_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            main([str(self.out), str(self.root),
                  "--overwrite", "--skip-existing"])

    def test_cli_overwrite_shortcut_equivalent_to_policy(self):
        self.out.mkdir()
        (self.out / "a.mkv").write_bytes(b"old")
        code, _, _ = self._run([
            str(self.out), str(self.paths["a.mp4"]), "--overwrite",
        ])
        self.assertEqual(code, br.EXIT_OK)
        self.assertNotEqual((self.out / "a.mkv").read_bytes(), b"old")

    def test_cli_no_recursive_only_top_level(self):
        code, out, _ = self._run([
            str(self.out), str(self.root), "--no-recursive",
        ])
        self.assertEqual(code, br.EXIT_OK)
        self.assertIn("共 1 项", out)
        self.assertTrue((self.out / "a.mkv").exists())

    def test_cli_stop_on_error(self):
        # 名字以 '0' 开头保证字典序排在 a.mp4 之前，成为首个处理项即失败
        (self.root / "0broken.mp4").write_bytes(b"")
        code, out, _ = self._run([
            str(self.out), str(self.root),
            "--stop-on-error", "--retries", "0",
        ])
        self.assertEqual(code, br.EXIT_BATCH_FAILED)
        # 首项失败后其余全部跳过：成功数为 0
        self.assertIn("成功 0", out)


if __name__ == "__main__":
    unittest.main()
