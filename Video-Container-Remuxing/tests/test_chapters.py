"""章节（chpl）元数据解析与 Matroska Chapters 转写的测试。

覆盖四个维度：
1. **解析正确性**：标准/历史布局、100ns 事实时间基与 movie timescale
   兼容单位的纳秒换算、多层嵌套、多个 chpl 盒的归一化与去重；
2. **损坏与不规则输入**：负载截断、标题截断、计数不符、非法 UTF-8、
   兄弟盒损坏后的重同步、空清单、超大声明计数等——要求不抛异常且
   给出可解释的部分结果；
3. **写出正确性**：Matroska 元素结构、UID 确定性、ChapterTimeEnd、
   零时间戳的非空编码、UTF-8 标题逐字节保留、章节不影响媒体样本；
4. **可复现性与无退化**：同一输入多次解析/写出逐字节一致，无章节
   文件的轨道与样本输出保持原有行为。

时间单位约定（关键）：真实 ffmpeg/GPAC/AtomicParsley 的 chpl 用
**100ns tick**（每秒 1e7），这是默认路径；只有时间密度明显落在
movie timescale 区间时才切换兼容单位。测试据此分别构造。
"""

from __future__ import annotations

import struct
import unittest

from mp4_to_mkv.chapters import Chapter, extract_chapters
from mp4_to_mkv.matroska_ids import *  # noqa: F401,F403  直接使用元素 ID 常量
from mp4_to_mkv.mkv_writer import write_mkv
from mp4_to_mkv.mp4_parser import parse_mp4, parse_boxes

from tests.ebml_reader import parse_document
from tests.mp4_builder import (
    box,
    build_chapter_udta,
    build_chpl,
    build_mp4,
    fullbox,
    inject_moov_child,
)

# 测试用电影时间基（1000 → 1 单位 = 1 毫秒）
_TIMESCALE = 1000
# 100ns 事实时间基：每秒 1e7 tick
_TICKS_100NS = 10_000_000


def _video_mp4() -> bytes:
    """约 164ms 的短视频（4 帧），用于多数结构/容错测试。"""
    frames = [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(4)]
    return build_mp4(frames, keyframes=[1], timescale=_TIMESCALE)


def _long_video_mp4() -> bytes:
    """约 13 秒的视频，为时间基判定提供可靠的时长参照。"""
    frames = [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(4)]
    # 4 帧各 3250 tick（timescale=1000）→ 总时长 13 秒
    return build_mp4(
        frames, keyframes=[1], timescale=_TIMESCALE, deltas=[3250] * 4
    )


def _with_chapters(entries, *, base=None, **udta_kwargs) -> bytes:
    """把 chpl 章节注入 moov；entries 时间为 100ns tick（标准事实单位）。"""
    return inject_moov_child(
        base if base is not None else _video_mp4(),
        build_chapter_udta(entries, **udta_kwargs),
    )


def _chapter_atoms(mkv_bytes: bytes):
    doc = parse_document(mkv_bytes)
    segment = next(e for e in doc if e.element_id == SEGMENT)
    chapters_el = segment.find(CHAPTERS)
    assert chapters_el is not None, "输出 MKV 中缺少 Chapters 元素"
    edition = chapters_el.find(EDITION_ENTRY)
    return edition.find_all(CHAPTER_ATOM)


def _atom_fields(atom):
    """从 ChapterAtom 提取 (uid, start_ns, end_ns, title, hidden, enabled)。"""
    title = atom.find(CHAPTER_DISPLAY).find(CHAP_STRING).as_string()
    return (
        atom.find(CHAPTER_UID).as_uint(),
        atom.find(CHAPTER_TIME_START).as_uint(),
        atom.find(CHAPTER_TIME_END).as_uint(),
        title,
        atom.find(CHAPTER_FLAG_HIDDEN).as_uint(),
        atom.find(CHAPTER_FLAG_ENABLED).as_uint(),
    )


# ---------------------------------------------------------------------------
# 1. 解析正确性（100ns 事实时间基）
# ---------------------------------------------------------------------------

class ChplParseTests(unittest.TestCase):
    def test_standard_layout_100ns_ticks_to_nanoseconds(self):
        # start_time 为 100ns tick：0 / 50,000,000(=5s) / 123,450,000(=12.345s)
        entries = [(0, "开场"), (50_000_000, "正片"), (123_450_000, "尾声")]
        parsed = parse_mp4(_with_chapters(entries, base=_long_video_mp4()))
        self.assertEqual(
            [(c.start_ns, c.title) for c in parsed.chapters],
            [
                (0, "开场"),
                (5_000_000_000, "正片"),
                (12_345_000_000, "尾声"),
            ],
        )
        self.assertEqual(parsed.chapter_warnings, [])

    def test_legacy_non_fullbox_meta_and_chpl(self):
        # 无 FullBox 头的 meta 与 chpl（QuickTime 历史变体），100ns tick
        data = inject_moov_child(
            _video_mp4(),
            build_chapter_udta(
                [(10_000_000, "Legacy")],
                fullbox_meta=False,
                fullbox_chpl=False,
            ),
        )
        parsed = parse_mp4(data)
        self.assertEqual(len(parsed.chapters), 1)
        self.assertEqual(parsed.chapters[0].title, "Legacy")
        self.assertEqual(parsed.chapters[0].start_ns, 1_000_000_000)

    def test_nested_gnre_ilst_containers(self):
        # 多层嵌套：udta/meta/gnre/ilst/chpl
        entries = [(20_000_000, "嵌套 A"), (40_000_000, "嵌套 B")]
        data = _with_chapters(entries, extra_wrappers=("gnre", "ilst"))
        parsed = parse_mp4(data)
        self.assertEqual([c.title for c in parsed.chapters], ["嵌套 A", "嵌套 B"])
        self.assertEqual(
            [c.start_ns for c in parsed.chapters],
            [2_000_000_000, 4_000_000_000],
        )

    def test_unsorted_entries_are_stably_sorted(self):
        # 清单内时间乱序：归一化后按 (start_ns, title) 稳定排序
        entries = [
            (90_000_000, "C"),
            (10_000_000, "A"),
            (50_000_000, "B"),
            (10_000_000, "A2"),
        ]
        parsed = parse_mp4(_with_chapters(entries))
        self.assertEqual(
            [(c.start_ns, c.title) for c in parsed.chapters],
            [
                (1_000_000_000, "A"),
                (1_000_000_000, "A2"),
                (5_000_000_000, "B"),
                (9_000_000_000, "C"),
            ],
        )

    def test_duplicate_chpl_boxes_are_deduplicated(self):
        # 两个内容相同的 chpl 盒（不同包装器重复写入）合并为一份
        udta_a = build_chapter_udta([(0, "X"), (30_000_000, "Y")])
        udta_b = build_chapter_udta([(0, "X"), (30_000_000, "Y")])
        data = inject_moov_child(
            inject_moov_child(_video_mp4(), udta_a), udta_b
        )
        parsed = parse_mp4(data)
        self.assertEqual(
            [(c.start_ns, c.title) for c in parsed.chapters],
            [(0, "X"), (3_000_000_000, "Y")],
        )

    def test_uint8_title_length_is_used(self):
        # 真实 ffmpeg/GPAC 的标题长度是单字节 uint8（Pascal 风格）；
        # 构造长度恰好跨越 uint8/uint16 歧义边界的标题验证不会误读
        title = "T" * 10  # 长度 10：uint16 误读会把下一字节并入长度
        data = _with_chapters([(0, title)])
        parsed = parse_mp4(data)
        self.assertEqual(parsed.chapters[0].title, title)

    def test_chapter_model_is_frozen(self):
        chapter = Chapter(start_ns=0, title="t")
        with self.assertRaises(Exception):
            chapter.title = "other"  # type: ignore[misc]

    def test_no_chapters_yields_empty_lists(self):
        parsed = parse_mp4(_video_mp4())
        self.assertEqual(parsed.chapters, [])
        self.assertEqual(parsed.chapter_warnings, [])


# ---------------------------------------------------------------------------
# 2. 损坏与不规则输入
# ---------------------------------------------------------------------------

class ChplCorruptionTests(unittest.TestCase):
    def test_truncated_chpl_payload_recovers_partial_entries(self):
        # 外层 chpl 盒长自洽，但内部声明 3 个章节、第 3 个条目标题被截断：
        # 前两个完整条目必须保留，并产出截断诊断，主流程不抛异常
        body = b"\x00" + struct.pack(">I", 3)  # reserved + chapter_count=3
        body += struct.pack(">Q", 0) + bytes([1]) + b"A"
        body += struct.pack(">Q", 10_000_000) + bytes([1]) + b"B"
        # 第三个条目：声明标题 8 字节，实际只给 3 字节
        body += struct.pack(">Q", 20_000_000) + bytes([8]) + b"CDE"
        chpl = box("chpl", b"\x00\x00\x00\x00" + body)  # 外层长度与实际一致
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)
        self.assertEqual([c.title for c in parsed.chapters[:2]], ["A", "B"])
        # 第三个条目标题按可用字节恢复
        self.assertEqual(parsed.chapters[2].title, "CDE")
        self.assertTrue(any("截断" in w for w in parsed.chapter_warnings))

    def test_truncated_title_uses_available_bytes(self):
        # 手工构造：声明标题长 10 字节但只剩 3 字节
        body = (
            b"\x00" + struct.pack(">I", 1)
            + struct.pack(">Q", 5_000_000) + bytes([10]) + b"abc"
        )
        chpl = box("chpl", b"\x00\x00\x00\x00" + body)
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)
        self.assertEqual(len(parsed.chapters), 1)
        self.assertEqual(parsed.chapters[0].title, "abc")
        self.assertTrue(any("截断" in w for w in parsed.chapter_warnings))

    def test_declared_count_smaller_than_actual_entries(self):
        # 实际写 3 条但计数只声明 2：第 3 条成为尾随数据，前 2 条正常
        payload = b"\x00" + struct.pack(">I", 2)
        for t, name in [(0, "A"), (10_000_000, "B"), (20_000_000, "C")]:
            nb = name.encode()
            payload += struct.pack(">Q", t) + bytes([len(nb)]) + nb
        chpl = box("chpl", b"\x00\x00\x00\x00" + payload)
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)
        self.assertEqual([c.title for c in parsed.chapters], ["A", "B"])
        self.assertTrue(any("尾随" in w for w in parsed.chapter_warnings))

    def test_invalid_utf8_is_replaced_not_fatal(self):
        # 标题含非法 UTF-8 序列 0xFF：以 U+FFFD 替换，解析不中断
        raw_title = b"bad\xfftitle"
        payload = (
            b"\x00" + struct.pack(">I", 1)
            + struct.pack(">Q", 0) + bytes([len(raw_title)]) + raw_title
        )
        chpl = box("chpl", b"\x00\x00\x00\x00" + payload)
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)
        self.assertEqual(len(parsed.chapters), 1)
        self.assertIn("�", parsed.chapters[0].title)

    def test_empty_title_gets_deterministic_placeholder(self):
        parsed = parse_mp4(_with_chapters([(0, ""), (10_000_000, "")]))
        self.assertEqual(
            [c.title for c in parsed.chapters], ["Chapter 1", "Chapter 2"]
        )

    def test_nul_padded_title_is_trimmed(self):
        raw_title = b"real\x00\x00"
        payload = (
            b"\x00" + struct.pack(">I", 1)
            + struct.pack(">Q", 0) + bytes([len(raw_title)]) + raw_title
        )
        chpl = box("chpl", b"\x00\x00\x00\x00" + payload)
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)
        self.assertEqual(parsed.chapters[0].title, "real")

    def test_corrupt_sibling_box_resyncs_to_chpl(self):
        # udta 内先放一个长度越界的损坏盒，再放正常 chpl：
        # 严格解析会整体失败，重同步应恢复出其后的 chpl
        broken = b"\x00\x00\x00\x05brok"  # size=5 < 8 字节盒头，非法
        good = build_chpl([(7_000_000, "恢复")])
        meta = box("meta", broken + good)
        data = inject_moov_child(_video_mp4(), box("udta", meta))
        parsed = parse_mp4(data)
        self.assertEqual([c.title for c in parsed.chapters], ["恢复"])
        self.assertEqual(parsed.chapters[0].start_ns, 700_000_000)

    def test_garbage_chpl_payload_does_not_raise(self):
        # chpl 盒存在但负载是随机脏字节：返回空章节 + 诊断，不抛异常
        chpl = box("chpl", b"\x00" * 3)
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)
        self.assertEqual(parsed.chapters, [])
        self.assertTrue(parsed.chapter_warnings)

    def test_huge_declared_count_is_capped(self):
        # 声明计数远超实际上限：按保护值截断，不发生超大内存分配
        payload = b"\x00" + struct.pack(">I", 0xFFFFFFFF)
        chpl = box("chpl", b"\x00\x00\x00\x00" + payload)
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", chpl))
        )
        parsed = parse_mp4(data)  # 不抛 MemoryError / OverflowError
        self.assertEqual(parsed.chapters, [])
        self.assertTrue(parsed.chapter_warnings)

    def test_empty_chpl_list_is_valid(self):
        parsed = parse_mp4(_with_chapters([]))
        self.assertEqual(parsed.chapters, [])
        # 合法的空清单不应产生“无法解析”类告警
        self.assertFalse(
            any("无法解析" in w for w in parsed.chapter_warnings)
        )

    def test_chapter_parse_never_breaks_media_tracks(self):
        # 任何章节损坏都不得影响媒体轨道解析
        broken = b"\x00\x00\x00\x02bad!"
        data = inject_moov_child(
            _video_mp4(), box("udta", box("meta", broken))
        )
        parsed = parse_mp4(data)
        self.assertEqual(len(parsed.tracks), 1)
        self.assertEqual(len(parsed.tracks[0].samples), 4)


# ---------------------------------------------------------------------------
# 3. Matroska Chapters 写出
# ---------------------------------------------------------------------------

class ChaptersWriteTests(unittest.TestCase):
    def test_chapters_element_structure_and_values(self):
        entries = [(0, "开场"), (50_000_000, "第二章")]
        mkv = write_mkv(parse_mp4(_with_chapters(entries)))
        doc = parse_document(mkv)
        segment = next(e for e in doc if e.element_id == SEGMENT)

        # Segment 一级子元素顺序：Info → Tracks → Chapters → Cluster...
        top_ids = [e.element_id for e in segment.children()]
        self.assertLess(top_ids.index(CHAPTERS), top_ids.index(CLUSTER))
        self.assertLess(top_ids.index(TRACKS), top_ids.index(CHAPTERS))

        edition = segment.find(CHAPTERS).find(EDITION_ENTRY)
        self.assertEqual(edition.find(EDITION_FLAG_HIDDEN).as_uint(), 0)
        self.assertEqual(edition.find(EDITION_FLAG_DEFAULT).as_uint(), 1)
        atoms = edition.find_all(CHAPTER_ATOM)
        # 字段：(uid, start_ns, end_ns, title, hidden, enabled)
        self.assertEqual(
            [_atom_fields(a) for a in atoms],
            [
                (1, 0, 5_000_000_000, "开场", 0, 1),
                (2, 5_000_000_000, 5_000_000_000, "第二章", 0, 1),
            ],
        )
        # ChapLanguage 固定为 und（chpl 不携带语言标签）
        lang = atoms[0].find(CHAPTER_DISPLAY).find(CHAP_LANGUAGE)
        self.assertEqual(lang.as_string(), "und")

    def test_chapter_time_end_chain_and_last_uses_duration(self):
        # 每章 ChapterTimeEnd = 下一章起点；末章 = 影片总时长
        entries = [(0, "A"), (20_000_000, "B"), (80_000_000, "C")]
        mkv = write_mkv(parse_mp4(_with_chapters(entries, base=_long_video_mp4())))
        atoms = _chapter_atoms(mkv)
        ends = [a.find(CHAPTER_TIME_END).as_uint() for a in atoms]
        starts = [a.find(CHAPTER_TIME_START).as_uint() for a in atoms]
        self.assertEqual(starts, [0, 2_000_000_000, 8_000_000_000])
        # 前两章结束时间 = 下一章起点
        self.assertEqual(ends[0], starts[1])
        self.assertEqual(ends[1], starts[2])
        # 末章结束时间 = 影片时长（13 秒 = 13e9 ns）
        self.assertEqual(ends[2], 13_000_000_000)

    def test_zero_timestamp_is_encoded_with_nonempty_payload(self):
        # 回归：ChapterTimeStart=0 必须写为单字节 0x00，不能是空数据区，
        # 否则 libavformat 把必需元素当作“值缺失”而丢弃整个章节
        mkv = write_mkv(parse_mp4(_with_chapters([(0, "Zero")])))
        atom = _chapter_atoms(mkv)[0]
        start_el = atom.find(CHAPTER_TIME_START)
        self.assertEqual(start_el.as_uint(), 0)
        self.assertEqual(len(start_el.payload), 1)
        self.assertEqual(start_el.payload, b"\x00")
        # 编码字节中应出现 ID(0x91) + size(0x81) + 0x00
        self.assertIn(b"\x91\x81\x00", mkv)

    def test_utf8_title_bytes_are_preserved(self):
        title = "章节标题-日本語-😊"
        mkv = write_mkv(parse_mp4(_with_chapters([(0, title)])))
        atom = _chapter_atoms(mkv)[0]
        raw = atom.find(CHAPTER_DISPLAY).find(CHAP_STRING).payload
        self.assertEqual(raw, title.encode("utf-8"))

    def test_chapters_do_not_alter_media_frames(self):
        frames = [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(4)]
        plain = build_mp4(frames, keyframes=[1], timescale=_TIMESCALE)
        with_ch = inject_moov_child(plain, build_chapter_udta([(0, "标记")]))
        mkv_plain = write_mkv(parse_mp4(plain))
        mkv_ch = write_mkv(parse_mp4(with_ch))

        def block_frames(mkv_bytes):
            segment = next(
                e for e in parse_document(mkv_bytes) if e.element_id == SEGMENT
            )
            out = []
            for cluster in segment.find_all(CLUSTER):
                for block in cluster.find_all(SIMPLE_BLOCK):
                    # 跳过 track vint(1) + timecode(2) + flags(1)
                    out.append(block.payload[4:])
            return out

        self.assertEqual(block_frames(mkv_plain), block_frames(mkv_ch))
        self.assertEqual(block_frames(mkv_ch), frames)

    def test_no_chapters_means_no_chapters_element(self):
        mkv = write_mkv(parse_mp4(_video_mp4()))
        segment = next(e for e in parse_document(mkv) if e.element_id == SEGMENT)
        self.assertIsNone(segment.find(CHAPTERS))


# ---------------------------------------------------------------------------
# 4. 可复现性
# ---------------------------------------------------------------------------

class ReproducibilityTests(unittest.TestCase):
    def test_repeated_parse_is_identical(self):
        data = _with_chapters(
            [(90_000_000, "C"), (10_000_000, "A"), (50_000_000, "B")],
            extra_wrappers=("gnre",),
        )
        first = parse_mp4(data).chapters
        second = parse_mp4(data).chapters
        self.assertEqual(first, second)

    def test_repeated_write_is_byte_identical(self):
        data = _with_chapters([(0, "一"), (25_000_000, "二"), (75_000_000, "三")])
        out_1 = write_mkv(parse_mp4(data))
        out_2 = write_mkv(parse_mp4(data))
        self.assertEqual(out_1, out_2)

    def test_uid_is_position_based_and_deterministic(self):
        entries = [(30_000_000, "x"), (10_000_000, "y"), (20_000_000, "z")]
        mkv = write_mkv(parse_mp4(_with_chapters(entries)))
        atoms = _chapter_atoms(mkv)
        # UID 取归一化（排序后）位置，与源清单书写顺序无关
        uids = [a.find(CHAPTER_UID).as_uint() for a in atoms]
        self.assertEqual(uids, [1, 2, 3])
        starts = [a.find(CHAPTER_TIME_START).as_uint() for a in atoms]
        self.assertEqual(starts, [1_000_000_000, 2_000_000_000, 3_000_000_000])


# ---------------------------------------------------------------------------
# 5. 单元级：extract_chapters 直接调用与时间基判定
# ---------------------------------------------------------------------------

class ExtractChaptersUnitTests(unittest.TestCase):
    def _moov_box(self, data: bytes):
        parse_mp4(data)  # 先确认整体可解析
        return next(b for b in parse_boxes(data) if b.kind == "moov")

    def test_missing_timescale_falls_back_to_100ns(self):
        data = _with_chapters([(1_234_567, "t")])
        moov = self._moov_box(data)
        chapters, warnings = extract_chapters(moov, None)
        self.assertEqual(len(chapters), 1)
        # 无 mvhd timescale：回退 100ns 事实单位（1 tick = 100ns）
        self.assertEqual(chapters[0].start_ns, 123_456_700)
        self.assertTrue(warnings)

    def test_movie_timescale_unit_is_detected(self):
        # 兼容路径：时间戳密度明显落在 movie timescale（1000 tps）区间。
        # 13 秒影片、末章 13000 tick → 1000 tps，与 10MHz 相差 1e4 倍
        entries = [(0, "A"), (5000, "B"), (13000, "C")]
        data = inject_moov_child(_long_video_mp4(), build_chapter_udta(entries))
        moov = self._moov_box(data)
        chapters, warnings = extract_chapters(moov, _TIMESCALE, 13.0)
        self.assertEqual(
            [(c.start_ns, c.title) for c in chapters],
            [
                (0, "A"),
                (5_000_000_000, "B"),
                (13_000_000_000, "C"),
            ],
        )
        self.assertTrue(any("movie timescale" in w for w in warnings))

    def test_100ns_unit_is_default_without_duration(self):
        # 缺少时长参照时一律采用 100ns 事实单位
        moov = self._moov_box(_video_mp4())
        chapters, _ = extract_chapters(moov, _TIMESCALE, None)
        self.assertEqual(chapters, [])
        # 有章节但无时长：仍按 100ns
        data = _with_chapters([(20_000_000, "x")])
        moov = self._moov_box(data)
        chapters, _ = extract_chapters(moov, _TIMESCALE, None)
        self.assertEqual(chapters[0].start_ns, 2_000_000_000)

    def test_extract_never_raises_on_empty_tree(self):
        moov = self._moov_box(_video_mp4())
        chapters, warnings = extract_chapters(moov, _TIMESCALE)
        self.assertEqual(chapters, [])
        self.assertEqual(warnings, [])


# ---------------------------------------------------------------------------
# 6. QuickTime 章节文本轨（handler=text + gmhd）的优雅跳过
# ---------------------------------------------------------------------------

def _build_text_chapter_trak() -> bytes:
    """构造一条最小合法的 QuickTime 章节文本轨（handler=text + gmhd）。

    真实 ffmpeg 文件在 chpl 之外还会附带这样一条轨；解析器应识别并
    跳过它，而不是因不支持文本编码拒绝整个文件。
    """
    mdhd = fullbox("mdhd", 0, 0, struct.pack(">IIII", 0, 0, _TIMESCALE, 13000))
    # 标准 hdlr 布局：FullBox 头后 pre_defined(4)=0，handler_type 紧跟
    # 在 data[8:12]（与真实 ffmpeg 一致；勿用带两个零字段的错误布局）
    hdlr = fullbox(
        "hdlr", 0, 0,
        struct.pack(">I", 0) + b"text" + b"\x00" * 12 + b"name\x00",
    )
    # 通用媒体信息头 gmhd（章节轨用它替代 vmhd/smhd/sthd）
    gmhd = box("gmhd", box("gmin", b"\x00" * 16))
    url = fullbox("url ", 0, 1, b"")
    dref = fullbox("dref", 0, 0, struct.pack(">I", 1) + url)
    dinf = box("dinf", dref)
    # 空 stbl 即可：跳过判定发生在样本表解析之前，不会下钻到这里
    stbl = box("stbl", b"")
    minf = box("minf", gmhd + dinf + stbl)
    mdia = box("mdia", mdhd + hdlr + minf)
    tkhd = fullbox(
        "tkhd", 0, 0x07,
        struct.pack(">IIIIII", 0, 0, 2, 2, 0, 13000) + b"\x00" * 44,
    )
    return box("trak", tkhd + mdia)


class ChapterTrackSkipTests(unittest.TestCase):
    def test_text_gmhd_track_is_skipped_not_rejected(self):
        # moov 内含 1 条视频轨 + 1 条 text/gmhd 章节轨 + chpl：
        # 章节轨被识别跳过，视频轨与 chpl 章节都正常保留
        data = inject_moov_child(_video_mp4(), _build_text_chapter_trak())
        data = inject_moov_child(
            data, build_chapter_udta([(0, "唯一章节")])
        )
        parsed = parse_mp4(data)  # 不抛“不支持的 track 类型”
        self.assertEqual(len(parsed.tracks), 1)
        self.assertEqual(parsed.tracks[0].handler, "vide")
        self.assertEqual(parsed.skipped_tracks, ["text/gmhd 章节轨"])
        self.assertEqual([c.title for c in parsed.chapters], ["唯一章节"])

    def test_normal_file_has_no_skipped_tracks(self):
        parsed = parse_mp4(_video_mp4())
        self.assertEqual(parsed.skipped_tracks, [])
        self.assertEqual(len(parsed.tracks), 1)


if __name__ == "__main__":
    unittest.main()
