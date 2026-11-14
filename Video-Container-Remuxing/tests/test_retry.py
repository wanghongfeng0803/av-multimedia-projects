"""失败重试编排（remux_all_tracks）测试。

重点验证重试的**边界判定**：
- 瞬时性 OSError：按线性退避重试，成功即返回，用尽抛 RemuxRetryError；
- 确定性 RemuxError：立即上抛，绝不重试；
- 半成品临时文件在失败后被清理；
- 真实文件端到端成功且输出可被独立解析。
"""

import tempfile
import unittest
from pathlib import Path

from mp4_to_mkv.errors import RemuxError

from tests.mp4_builder import build_mp4
import remux_all_tracks as rat
from remux_all_tracks import RemuxRetryError, remux_all_tracks


def _video_mp4() -> bytes:
    return build_mp4(
        [b"\x00\x00\x00\x01\x67" + bytes([i]) * 8 for i in range(4)],
        keyframes=[1],
    )


def _flaky_stub(fail_times: int, record: list, leave_part: bool = False):
    """前 fail_times 次抛 OSError，之后成功的桩；记录调用与 sleep。"""
    state = {"calls": 0}

    def stub(src, dst):
        state["calls"] += 1
        record.append(("attempt", state["calls"]))
        if state["calls"] <= fail_times:
            if leave_part:
                dst.with_name(dst.name + ".part").write_bytes(b"half")
            raise OSError(f"simulated transient error #{state['calls']}")
        return b"MKZ", 1, 4  # 成功时返回最小自洽三元组

    return stub


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "in.mp4"
        self.src.write_bytes(_video_mp4())
        self.dst = self.dir / "out.mkv"

    def test_success_first_attempt_uses_real_pipeline(self):
        result = remux_all_tracks(str(self.src), str(self.dst), retries=2)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(self.dst.exists())
        # 输出是合法 EBML/MKV（能被测试读取器解析）
        from tests.ebml_reader import parse_document

        parse_document(self.dst.read_bytes())

    def test_retry_after_transient_error_then_succeeds(self):
        record = []
        sleeps = []
        stub = _flaky_stub(fail_times=2, record=record)
        result = remux_all_tracks(
            str(self.src),
            str(self.dst),
            retries=3,
            retry_delay=0.01,
            sleep=sleeps.append,
            attempt_fn=stub,
        )
        self.assertEqual(result.attempts, 3)  # 2 次失败 + 1 次成功
        self.assertEqual([a for _, a in record], [1, 2, 3])
        # 线性退避：第 1、2 次失败后分别等待 1*delay、2*delay
        self.assertEqual(sleeps, [0.01, 0.02])

    def test_exhaust_retries_raises_retry_error(self):
        record = []
        sleeps = []
        stub = _flaky_stub(fail_times=99, record=record)
        with self.assertRaises(RemuxRetryError) as ctx:
            remux_all_tracks(
                str(self.src),
                str(self.dst),
                retries=2,
                retry_delay=0.01,
                sleep=sleeps.append,
                attempt_fn=stub,
            )
        # 总尝试数 = retries + 1
        self.assertEqual(len(record), 3)
        self.assertEqual(sleeps, [0.01, 0.02])
        # __cause__ 保留最后一次 OSError
        self.assertIsInstance(ctx.exception.__cause__, OSError)
        self.assertFalse(self.dst.exists())  # 失败不产出目标文件

    def test_deterministic_error_is_not_retried(self):
        record = []

        def stub(src, dst):
            record.append(1)
            raise RemuxError("不支持的结构")

        with self.assertRaises(RemuxError):
            remux_all_tracks(
                str(self.src),
                str(self.dst),
                retries=3,
                retry_delay=0.01,
                sleep=lambda s: self.fail("确定性错误不应触发等待"),
                attempt_fn=stub,
            )
        self.assertEqual(record, [1])  # 仅尝试一次

    def test_half_written_part_file_is_cleaned_up(self):
        stub = _flaky_stub(fail_times=99, record=[], leave_part=True)
        part = self.dst.with_name(self.dst.name + ".part")
        with self.assertRaises(RemuxRetryError):
            remux_all_tracks(
                str(self.src),
                str(self.dst),
                retries=1,
                retry_delay=0.01,
                sleep=lambda s: None,
                attempt_fn=stub,
            )
        self.assertFalse(part.exists())  # 半成品被清理
        self.assertFalse(self.dst.exists())

    def test_precondition_errors_skip_retry(self):
        # 输入不存在属于确定性前置校验，不进入尝试循环
        missing = self.dir / "nope.mp4"
        with self.assertRaises(RemuxError):
            remux_all_tracks(
                str(missing),
                str(self.dst),
                retries=3,
                attempt_fn=lambda *a: self.fail("不应执行尝试"),
            )

    def test_output_exists_requires_overwrite(self):
        self.dst.write_bytes(b"old")
        with self.assertRaises(RemuxError):
            remux_all_tracks(str(self.src), str(self.dst), retries=2)
        self.assertEqual(self.dst.read_bytes(), b"old")  # 原文件未被破坏
        result = remux_all_tracks(
            str(self.src), str(self.dst), overwrite=True, retries=2
        )
        self.assertEqual(result.attempts, 1)

    def test_negative_retries_rejected(self):
        with self.assertRaises(RemuxError):
            remux_all_tracks(str(self.src), str(self.dst), retries=-1)

    def test_empty_file_is_deterministic_failure(self):
        empty = self.dir / "empty.mp4"
        empty.write_bytes(b"")
        with self.assertRaises(RemuxError) as ctx:
            remux_all_tracks(str(empty), str(self.dst), retries=3)
        self.assertIn("为空", str(ctx.exception))
        self.assertFalse(self.dst.exists())

    def test_oversize_file_is_deterministic_failure(self):
        import remux_all_tracks as rat

        big = self.dir / "big.mp4"
        with open(big, "wb") as f:
            f.truncate(rat.MAX_INPUT_BYTES + 1)
        with self.assertRaises(RemuxError) as ctx:
            remux_all_tracks(str(big), str(self.dst), retries=2)
        self.assertIn("上限", str(ctx.exception))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "in.mp4"
        self.src.write_bytes(_video_mp4())
        self.dst = self.dir / "out.mkv"

    def test_cli_success_exit_code(self):
        code = rat.main([str(self.src), str(self.dst)])
        self.assertEqual(code, rat.EXIT_OK)
        self.assertTrue(self.dst.exists())

    def test_cli_deterministic_failure_exit_code(self):
        code = rat.main([str(self.src.parent / "missing.mp4"), str(self.dst)])
        self.assertEqual(code, rat.EXIT_INVALID_INPUT)

    def test_cli_retry_exhausted_exit_code(self):
        # 让真实管线必然遇到瞬时 OSError：输出目录不可写（用一个文件冒充目录）
        blocker = self.dir / "blocker"
        blocker.write_bytes(b"x")
        bad_dst = blocker / "out.mkv"  # blocker 是文件，其下无法创建
        code = rat.main(
            [str(self.src), str(bad_dst), "--retries", "1", "--retry-delay", "0.01"]
        )
        self.assertEqual(code, rat.EXIT_RETRY_EXHAUSTED)


if __name__ == "__main__":
    unittest.main()
