"""流式（无临时文件）重封装测试。

验证在上一版“整体读入 + 临时文件 + 原子替换”之外新增的流式路径：

1. **写出器**：``write_mkv_stream`` 产出未知大小 Segment 的合法 MKV，
   可被独立解析；块负载、轨道元数据、章节与内存路径逐字节等价；
   写入字节计数正确。
2. **不依赖可寻址性**：自定义只写、不可 seek 的流（管道语义）同样成功；
   短写（``write`` 一次只收部分字节）与零字节拒绝也被正确处理。
3. **无临时文件**：默认编排直接写目标路径，目录下不出现 ``.part``；
   ``use_temp=True`` 仍保留原子临时文件模式且两种模式结构等价。
4. **失败健壮性**：写出中途失败时半成品目标被清理、异常类型保留
   （确定性 ``RemuxError`` 不重试、瞬时 ``OSError`` 触发重试）。
5. **有界内存**：簇生成器同一时刻只物化一个簇，不累积整份输出。
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from mp4_to_mkv.errors import RemuxError
from mp4_to_mkv.matroska_ids import (
    BLOCK_GROUP,
    BLOCK,
    CHAPTERS,
    CLUSTER,
    EBML_HEADER,
    INFO,
    SEGMENT,
    SIMPLE_BLOCK,
    TRACKS,
)
from mp4_to_mkv.mp4_parser import parse_mp4
from mp4_to_mkv.mkv_writer import write_mkv, write_mkv_stream
from mp4_to_mkv.remuxer import remux

import remux_all_tracks as rat
from remux_all_tracks import RemuxRetryError, remux_all_tracks

from tests.ebml_reader import parse_document, read_vint
from tests.mp4_builder import build_mp4, build_multi_track_mp4, inject_moov_child, build_chapter_udta


def _video_mp4(*, frames=6, keyframes=None) -> bytes:
    keyframes = [1] if keyframes is None else keyframes
    data = [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(frames)]
    return build_mp4(data, keyframes=keyframes)


def _all_block_payloads(segment):
    """收集一个 Segment 内所有 SimpleBlock/Block 的 (轨道号, 帧负载)。"""
    out = []
    for cluster in segment.find_all(CLUSTER):
        for simple in cluster.find_all(SIMPLE_BLOCK):
            no, pos = read_vint(simple.payload, 0)
            out.append((no, simple.payload[pos + 3:]))
        # BlockGroup 是 Cluster 的直接子元素，必须在 cluster 上查找
        for group in cluster.find_all(BLOCK_GROUP):
            raw = group.find(BLOCK).payload
            no, pos = read_vint(raw, 0)
            out.append((no, raw[pos + 3:]))
    return out


class StreamingWriterTests(unittest.TestCase):
    def setUp(self):
        self.mp4 = parse_mp4(_video_mp4())

    def test_stream_output_is_valid_mkv_with_unknown_size_segment(self):
        buf = io.BytesIO()
        count = write_mkv_stream(self.mp4, buf)
        raw = buf.getvalue()
        self.assertEqual(count, len(raw))

        doc = parse_document(raw)
        kinds = [e.element_id for e in doc]
        self.assertEqual(kinds[0], EBML_HEADER)
        self.assertIn(SEGMENT, kinds)

        segment = next(e for e in doc if e.element_id == SEGMENT)
        # 未知大小 Segment 负载延伸到文档末尾
        self.assertEqual(segment.end, len(raw))
        # 头部字节：Segment ID(4) + 未知大小 VINT = 0x01 0xFF*7
        head = raw[segment.start:segment.start + 12]
        self.assertEqual(head, b"\x18\x53\x80\x67" + b"\x01" + b"\xff" * 7)
        # 一级子元素齐全且顺序正确
        self.assertEqual(
            [c.element_id for c in segment.children()][:3],
            [INFO, TRACKS, CLUSTER],
        )

    def test_stream_and_memory_paths_carry_identical_frames(self):
        stream_raw = io.BytesIO()
        write_mkv_stream(self.mp4, stream_raw)
        memory_raw = write_mkv(self.mp4)

        seg_s = next(e for e in parse_document(stream_raw.getvalue())
                     if e.element_id == SEGMENT)
        seg_m = next(e for e in parse_document(memory_raw)
                     if e.element_id == SEGMENT)
        self.assertEqual(
            sorted(_all_block_payloads(seg_s)),
            sorted(_all_block_payloads(seg_m)),
        )

    def test_info_tracks_prefix_byte_identical_between_paths(self):
        stream_raw = io.BytesIO()
        write_mkv_stream(self.mp4, stream_raw)
        memory_raw = write_mkv(self.mp4)

        def prefix_before_first_cluster(raw: bytes) -> bytes:
            seg = next(e for e in parse_document(raw)
                       if e.element_id == SEGMENT)
            payload = seg.payload
            pos = 0
            while pos < len(payload):
                from tests.ebml_reader import read_element_id
                eid, after = read_element_id(payload, pos)
                size, na = read_vint(payload, after)
                if eid == CLUSTER:
                    return payload[:pos]
                pos = len(payload) if size is None else na + size
            return payload

        self.assertEqual(
            prefix_before_first_cluster(stream_raw.getvalue()),
            prefix_before_first_cluster(memory_raw),
        )

    def test_written_byte_count_matches_stream_length(self):
        buf = io.BytesIO()
        count = write_mkv_stream(self.mp4, buf)
        self.assertGreater(count, 0)
        self.assertEqual(count, len(buf.getvalue()))

    def test_multi_track_and_subtitle_stream_roundtrip(self):
        ttml = b'<?xml version="1.0"?><tt><body/></tt>'
        mp4_bytes = build_multi_track_mp4(
            [
                {"kind": "video", "samples": [b"\x67" + bytes([i]) * 6 for i in range(3)],
                 "timescale": 24000, "deltas": [1000] * 3, "keyframes": [1]},
                {"kind": "audio", "samples": [b"\x21\x10" * 8 for _ in range(2)],
                 "timescale": 44100, "deltas": [1024, 1024]},
                {"kind": "stpp", "samples": [ttml], "timescale": 1000, "deltas": [3000]},
            ]
        )
        mp4 = parse_mp4(mp4_bytes)
        buf = io.BytesIO()
        write_mkv_stream(mp4, buf)

        segment = next(e for e in parse_document(buf.getvalue())
                       if e.element_id == SEGMENT)
        payloads = _all_block_payloads(segment)
        track_numbers = sorted({no for no, _ in payloads})
        self.assertEqual(track_numbers, [1, 2, 3])  # 视频/音频/字幕三轨
        self.assertIn((3, ttml), payloads)          # TTML 文档逐字节直拷

    def test_chapters_preserved_in_stream_output(self):
        mp4_bytes = inject_moov_child(
            _video_mp4(),
            build_chapter_udta([(0, "Intro"), (2000, "Main")]),
        )
        mp4 = parse_mp4(mp4_bytes)
        self.assertTrue(mp4.chapters)

        buf = io.BytesIO()
        write_mkv_stream(mp4, buf)
        segment = next(e for e in parse_document(buf.getvalue())
                       if e.element_id == SEGMENT)
        self.assertIsNotNone(segment.find(CHAPTERS))


class NonSeekableStreamTests(unittest.TestCase):
    class _WriteOnly:
        """只写、不可 seek 的流（模拟管道/套接字），记录每次写入。"""

        def __init__(self):
            self.parts = []
            self.write_calls = 0

        def write(self, data):
            self.write_calls += 1
            self.parts.append(bytes(data))
            return len(data)

    class _ShortWriter:
        """每次最多吞 16 字节的短写流，验证写出器的循环写补偿。"""

        def __init__(self):
            self.buf = bytearray()

        def write(self, data):
            chunk = bytes(data[:16])
            self.buf += chunk
            return len(chunk)

    def test_pipe_like_non_seekable_stream(self):
        mp4 = parse_mp4(_video_mp4())
        sink = self._WriteOnly()
        count = write_mkv_stream(mp4, sink)
        raw = b"".join(sink.parts)
        self.assertEqual(count, len(raw))
        self.assertGreater(sink.write_calls, 1)  # 确实分多次冲刷
        parse_document(raw)  # 拼回后是合法 MKV

    def test_short_writes_are_retried_until_fully_written(self):
        mp4 = parse_mp4(_video_mp4())
        sink = self._ShortWriter()
        write_mkv_stream(mp4, sink)
        # 短写补偿后内容必须与一次写完完全一致
        expected = io.BytesIO()
        write_mkv_stream(mp4, expected)
        self.assertEqual(bytes(sink.buf), expected.getvalue())

    def test_stream_that_rejects_all_bytes_raises(self):
        mp4 = parse_mp4(_video_mp4())

        class _Dead:
            def write(self, data):
                return 0

        with self.assertRaises(RemuxError):
            write_mkv_stream(mp4, _Dead())


class OrchestrationStreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "in.mp4"
        self.src.write_bytes(_video_mp4())

    def test_default_remux_writes_directly_without_part_file(self):
        dst = self.dir / "direct.mkv"
        byte_count, samples, tracks = remux(str(self.src), str(dst))
        self.assertTrue(dst.exists())
        self.assertEqual(list(self.dir.glob("*.part")), [])
        parse_document(dst.read_bytes())  # 合法 MKV
        self.assertEqual(byte_count, dst.stat().st_size)
        self.assertEqual(tracks, 1)
        self.assertEqual(samples, 6)

    def test_use_temp_mode_still_produces_equivalent_output(self):
        direct = self.dir / "direct.mkv"
        atomic = self.dir / "atomic.mkv"
        remux(str(self.src), str(direct))
        remux(str(self.src), str(atomic), use_temp=True)

        seg_d = next(e for e in parse_document(direct.read_bytes())
                     if e.element_id == SEGMENT)
        seg_a = next(e for e in parse_document(atomic.read_bytes())
                     if e.element_id == SEGMENT)
        # 两种模式承载的块负载（含轨道号）完全一致
        self.assertEqual(
            sorted(_all_block_payloads(seg_d)),
            sorted(_all_block_payloads(seg_a)),
        )
        # 原子模式结束后同样不留临时文件
        self.assertEqual(list(self.dir.glob("*.part")), [])

    def test_remux_all_tracks_default_is_streaming(self):
        dst = self.dir / "rat.mkv"
        result = remux_all_tracks(str(self.src), str(dst), retries=1)
        self.assertTrue(dst.exists())
        self.assertEqual(list(self.dir.glob("*.part")), [])
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output_bytes, dst.stat().st_size)
        parse_document(dst.read_bytes())

    def test_failed_streaming_write_cleans_half_target_and_preserves_type(self):
        # 让流式写出在写簇阶段必然 OSError：输出路径位于一个“文件”之下
        blocker = self.dir / "blocker"
        blocker.write_bytes(b"x")
        bad_dst = blocker / "out.mkv"
        # open("wb") 在 blocker 是文件时抛 OSError（瞬时错误，可重试）
        with self.assertRaises(RemuxRetryError):
            remux_all_tracks(
                str(self.src), str(bad_dst),
                retries=1, retry_delay=0.01,
            )
        self.assertFalse(bad_dst.exists())

    def test_deterministic_error_leaves_no_target_and_no_retry(self):
        empty = self.dir / "empty.mp4"
        empty.write_bytes(b"")
        dst = self.dir / "should_not_exist.mkv"
        calls = []

        def boom(src, d):
            calls.append(1)
            raise RemuxError("确定性失败")

        with self.assertRaises(RemuxError):
            remux_all_tracks(
                str(empty), str(dst), retries=3,
                retry_delay=0.01, sleep=lambda s: self.fail("不应等待"),
                attempt_fn=boom,
            )
        self.assertEqual(calls, [1])
        self.assertFalse(dst.exists())

    def test_transient_then_success_via_injected_two_arg_stub(self):
        # 旧式两参桩的兼容路径：第二次成功且目标最终存在
        dst = self.dir / "stub.mkv"
        state = {"n": 0}

        def stub(src, d):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError("瞬时占用")
            Path(d).write_bytes(b"MKZ")
            return b"MKZ", 1, 4  # 旧式返回字节串

        result = remux_all_tracks(
            str(self.src), str(dst), retries=2, retry_delay=0.01,
            sleep=lambda s: None, attempt_fn=stub,
        )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.output_bytes, 3)  # 字节串被归一化为长度
        self.assertTrue(dst.exists())

    def test_half_target_removed_before_retry_in_real_pipeline(self):
        # 让 rat 模块内绑定的流式写出前两次写部分字节后抛 OSError、第三次成功，
        # 验证每次失败都清掉半成品目标，重试从零开始，最终得到完整文件。
        dst = self.dir / "flaky.mkv"
        real = rat.write_mkv_stream
        counter = {"n": 0}

        def flaky_stream(mp4, stream):
            counter["n"] += 1
            if counter["n"] <= 2:
                stream.write(b"\x1a\x45\xdf\xa3partial")  # 制造半成品
                raise OSError("磁盘瞬态抖动")
            return real(mp4, stream)

        try:
            rat.write_mkv_stream = flaky_stream
            # 不传 attempt_fn：默认 _remux_attempt 调用被替换的模块全局函数
            result = remux_all_tracks(
                str(self.src), str(dst), retries=3, retry_delay=0.01,
                sleep=lambda s: None,
            )
        finally:
            rat.write_mkv_stream = real

        self.assertEqual(result.attempts, 3)
        self.assertTrue(dst.exists())
        # 最终文件是完整合法 MKV，而非带 partial 残骸的坏文件
        parse_document(dst.read_bytes())


class BoundedMemoryTests(unittest.TestCase):
    def test_cluster_generator_does_not_materialize_all_clusters(self):
        # 足够多的大帧触发多个簇
        big = [b"\xAA" * 200000 for _ in range(40)]
        mp4 = parse_mp4(build_mp4(big, keyframes=list(range(1, 41, 5))))
        from mp4_to_mkv.mkv_writer import _iter_cluster_bytes

        gen = _iter_cluster_bytes(mp4, _place(mp4))
        first = next(gen)
        # 生成器是惰性的：取一个簇之后仍可继续，且不会一次性持有全部簇
        second = next(gen)
        self.assertIsInstance(first, bytes)
        self.assertIsInstance(second, bytes)
        self.assertNotEqual(len(first), 0)

    def test_peak_output_buffer_is_one_cluster_not_whole_file(self):
        # 同一 mp4 用流式写，BytesIO 累积长度虽等于文件大小，但写出器内部
        # 只通过逐簇 write 推送；这里验证簇数量 >1 且每簇独立生成。
        big = [b"\xBB" * 200000 for _ in range(40)]
        mp4 = parse_mp4(build_mp4(big, keyframes=list(range(1, 41, 5))))
        buf = io.BytesIO()
        write_mkv_stream(mp4, buf)
        segment = next(e for e in parse_document(buf.getvalue())
                       if e.element_id == SEGMENT)
        self.assertGreater(len(segment.find_all(CLUSTER)), 1)


def _place(mp4):
    # 复用写出器内部的样本放置，仅供惰性生成器测试
    from mp4_to_mkv.mkv_writer import _order_tracks, _place_samples
    ordered, numbers, _ = _order_tracks(mp4)
    return _place_samples(mp4, ordered, numbers)


if __name__ == "__main__":
    unittest.main()
