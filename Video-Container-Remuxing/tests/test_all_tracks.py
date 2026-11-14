"""全轨道（视频 / 多音频 / 字幕）重封装测试。

覆盖：
- 多轨道 MP4 的解析与轨道类型 / 编码族识别；
- 视频、AAC、AC-3、TTML、WebVTT 轨道在 MKV 中的 TrackEntry 与块结构；
- 所有样本逐字节无损（字幕文档直拷、音频裸帧直拷）；
- 轨道排列固定为 视频 → 音频 → 字幕，同类保持源顺序；
- WebVTT 的 cue id / settings 经 BlockAdditions 保留，空 cue 正确处理；
- mdhd 语言码透传到 MKV Language 元素。
"""

import unittest

from mp4_to_mkv.matroska_ids import (
    AUDIO,
    BLOCK,
    BLOCK_ADD_ID,
    BLOCK_ADDITIONAL,
    BLOCK_ADDITIONS,
    BLOCK_DURATION,
    BLOCK_GROUP,
    BLOCK_MORE,
    CHANNELS,
    CLUSTER,
    CODEC_ID,
    LANGUAGE,
    SAMPLING_FREQUENCY,
    SEGMENT,
    SIMPLE_BLOCK,
    TRACK_ENTRY,
    TRACK_NUMBER,
    TRACK_TYPE,
    TRACK_TYPE_AUDIO,
    TRACK_TYPE_SUBTITLE,
    TRACK_TYPE_VIDEO,
    TRACKS,
)
from mp4_to_mkv.mp4_parser import parse_mp4
from mp4_to_mkv.mkv_writer import write_mkv

from tests.ebml_reader import parse_document, read_vint
from tests.mp4_builder import box, build_multi_track_mp4


VIDEO_SAMPLES = [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(3)]
AAC_SAMPLES = [bytes([0x21, 0x10, i]) * 10 for i in range(2)]
AC3_SAMPLES = [b"\x0B\x77" + bytes([i]) * 18 for i in range(2)]
TTML_DOC = b'<?xml version="1.0" encoding="utf-8"?><tt><body/></tt>'


def _wvtt_cue(cue_id=b"", settings=b"", payload=b"") -> bytes:
    children = b""
    if cue_id:
        children += box("iden", cue_id)
    if settings:
        children += box("sttg", settings)
    children += box("payl", payload)
    return box("vttc", children)


def _build_five_track_mp4() -> bytes:
    return build_multi_track_mp4(
        [
            {
                "kind": "video",
                "samples": VIDEO_SAMPLES,
                "timescale": 24000,
                "deltas": [1000, 1000, 1000],
                "keyframes": [1],
            },
            {
                "kind": "audio",
                "samples": AAC_SAMPLES,
                "timescale": 44100,
                "deltas": [1024, 1024],
            },
            {
                "kind": "ac3",
                "samples": AC3_SAMPLES,
                "timescale": 48000,
                "deltas": [1536, 1536],
                "channels": 2,
                "sample_rate": 48000,
            },
            {
                "kind": "stpp",
                "samples": [TTML_DOC],
                "timescale": 1000,
                "deltas": [3000],
            },
            {
                "kind": "wvtt",
                "samples": [
                    _wvtt_cue(b"cue-1", b"align:left", b"Hello"),
                    box("vtte"),
                ],
                "timescale": 1000,
                "deltas": [1000, 1000],
            },
        ]
    )


class MultiTrackParseTests(unittest.TestCase):
    def setUp(self):
        self.mp4 = parse_mp4(_build_five_track_mp4())

    def test_track_recognition(self):
        families = [(t.handler, t.codec_family, t.codec_tag) for t in self.mp4.tracks]
        self.assertEqual(
            families,
            [
                ("vide", "avc", "avc1"),
                ("soun", "aac", "mp4a"),
                ("soun", "ac3", "ac-3"),
                ("subt", "ttml", "stpp"),
                ("subt", "webvtt", "wvtt"),
            ],
        )

    def test_sample_counts_and_offsets_in_range(self):
        counts = [len(t.samples) for t in self.mp4.tracks]
        self.assertEqual(counts, [3, 2, 2, 1, 2])
        total = len(self.mp4.data)
        for track in self.mp4.tracks:
            for sample in track.samples:
                self.assertGreaterEqual(sample.offset, 0)
                self.assertLessEqual(sample.offset + sample.size, total)
                self.assertGreater(sample.size, 0)

    def test_ac3_audio_fields(self):
        ac3 = next(t for t in self.mp4.tracks if t.codec_family == "ac3")
        self.assertEqual(ac3.channels, 2)
        self.assertEqual(ac3.sample_rate, 48000)
        self.assertEqual(ac3.codec_private, b"")  # AC-3 无 CodecPrivate


class MultiTrackMkvStructureTests(unittest.TestCase):
    def setUp(self):
        self.mp4 = parse_mp4(_build_five_track_mp4())
        self.mkv = write_mkv(self.mp4)
        self.doc = parse_document(self.mkv)
        self.segment = next(e for e in self.doc if e.element_id == SEGMENT)
        self.entries = self.segment.find(TRACKS).find_all(TRACK_ENTRY)

    def test_track_ordering_video_audio_subtitle(self):
        # MKV 轨道号顺序：视频 → 音频(AAC,AC3) → 字幕(TTML,WebVTT)
        types = [e.find(TRACK_TYPE).as_uint() for e in self.entries]
        self.assertEqual(
            types,
            [
                TRACK_TYPE_VIDEO,
                TRACK_TYPE_AUDIO,
                TRACK_TYPE_AUDIO,
                TRACK_TYPE_SUBTITLE,
                TRACK_TYPE_SUBTITLE,
            ],
        )
        numbers = [e.find(TRACK_NUMBER).as_uint() for e in self.entries]
        self.assertEqual(numbers, [1, 2, 3, 4, 5])

    def test_codec_ids(self):
        codec_ids = [e.find(CODEC_ID).as_string() for e in self.entries]
        self.assertEqual(
            codec_ids,
            [
                "V_MPEG4/ISO/AVC",
                "A_AAC",
                "A_AC3",
                "S_TEXT/TTML",
                "S_TEXT/WEBVTT",
            ],
        )

    def test_default_flag_only_first_per_kind(self):
        from mp4_to_mkv.matroska_ids import FLAG_DEFAULT

        defaults = [e.find(FLAG_DEFAULT).as_uint() for e in self.entries]
        # 视频 1 条(默)、音频 2 条(首条默)、字幕 2 条(首条默)
        self.assertEqual(defaults, [1, 1, 0, 1, 0])

    def test_ac3_track_has_audio_element_no_codec_private(self):
        from mp4_to_mkv.matroska_ids import CODEC_PRIVATE

        ac3_entry = self.entries[2]
        audio = ac3_entry.find(AUDIO)
        self.assertIsNotNone(audio)
        self.assertEqual(audio.find(CHANNELS).as_uint(), 2)
        self.assertEqual(audio.find(SAMPLING_FREQUENCY).as_float(), 48000.0)
        self.assertIsNone(ac3_entry.find(CODEC_PRIVATE))

    def test_all_samples_lossless(self):
        """所有非 WebVTT 样本必须在 MKV 中逐字节出现；WebVTT 正文也须保留。"""
        # 收集 MKV 中全部块负载（SimpleBlock 与 BlockGroup.Block）
        block_payloads = []
        for cluster in self.segment.find_all(CLUSTER):
            for simple in cluster.find_all(SIMPLE_BLOCK):
                track_no, pos = read_vint(simple.payload, 0)
                block_payloads.append((track_no, simple.payload[pos + 3:]))
            for group in cluster.find_all(BLOCK_GROUP):
                raw = group.find(BLOCK).payload
                track_no, pos = read_vint(raw, 0)
                block_payloads.append((track_no, raw[pos + 3:]))

        # 期望的 (轨道号, 负载) 集合，按 write_mkv 的轨道重排编号
        expected = []
        ordered = sorted(
            self.mp4.tracks,
            key=lambda t: (
                0 if t.is_video else (1 if t.is_audio else 2),
                t.track_id,
            ),
        )
        for number, track in enumerate(ordered, start=1):
            for sample in track.samples:
                raw = self.mp4.data[sample.offset:sample.offset + sample.size]
                if track.codec_family == "webvtt":
                    # WebVTT 块负载是 cue 正文而非原始 vttc 盒
                    from mp4_to_mkv.webvtt import convert_sample

                    raw = convert_sample(raw).payload
                expected.append((number, raw))

        self.assertEqual(sorted(block_payloads), sorted(expected))

    def test_ttml_document_verbatim(self):
        # TTML 轨道号为 4，其唯一块负载必须与源文档逐字节相同
        ttml_blocks = [p for n, p in self._all_block_payloads() if n == 4]
        self.assertEqual(ttml_blocks, [TTML_DOC])

    def _all_block_payloads(self):
        out = []
        for cluster in self.segment.find_all(CLUSTER):
            for simple in cluster.find_all(SIMPLE_BLOCK):
                no, pos = read_vint(simple.payload, 0)
                out.append((no, simple.payload[pos + 3:]))
            for group in cluster.find_all(BLOCK_GROUP):
                raw = group.find(BLOCK).payload
                no, pos = read_vint(raw, 0)
                out.append((no, raw[pos + 3:]))
        return out

    def test_webvtt_block_additions_and_empty_cue(self):
        # WebVTT 轨道号为 5
        groups = []
        for cluster in self.segment.find_all(CLUSTER):
            groups.extend(cluster.find_all(BLOCK_GROUP))
        wvtt_groups = []
        for group in groups:
            raw = group.find(BLOCK).payload
            no, _ = read_vint(raw, 0)
            if no == 5:
                wvtt_groups.append(group)
        self.assertEqual(len(wvtt_groups), 2)  # 两个 cue 各一个 BlockGroup

        first, second = wvtt_groups
        # 每条字幕都有 BlockDuration（呈现时长 1000ms）
        self.assertEqual(first.find(BLOCK_DURATION).as_uint(), 1000)
        self.assertEqual(second.find(BLOCK_DURATION).as_uint(), 1000)

        # 第一条：正文 + 两个 BlockMore（id=1 cue id，id=2 settings）
        first_raw = first.find(BLOCK).payload
        self.assertEqual(first_raw[4:], b"Hello")
        additions = first.find(BLOCK_ADDITIONS)
        self.assertIsNotNone(additions)
        mores = additions.find_all(BLOCK_MORE)
        self.assertEqual(len(mores), 2)
        self.assertEqual(mores[0].find(BLOCK_ADD_ID).as_uint(), 1)
        self.assertEqual(mores[0].find(BLOCK_ADDITIONAL).payload, b"cue-1")
        self.assertEqual(mores[1].find(BLOCK_ADD_ID).as_uint(), 2)
        self.assertEqual(mores[1].find(BLOCK_ADDITIONAL).payload, b"align:left")

        # 第二条：空 cue（vtte），无附加数据，块负载为空
        self.assertIsNone(second.find(BLOCK_ADDITIONS))
        self.assertEqual(second.find(BLOCK).payload[4:], b"")

    def test_webvtt_all_blocks_are_block_groups(self):
        # 字幕块统一用 BlockGroup（携带 BlockDuration），不使用 SimpleBlock
        wvtt_simple = []
        for cluster in self.segment.find_all(CLUSTER):
            for simple in cluster.find_all(SIMPLE_BLOCK):
                no, _ = read_vint(simple.payload, 0)
                if no == 5:
                    wvtt_simple.append(simple)
        self.assertEqual(wvtt_simple, [])


class LanguageTests(unittest.TestCase):
    def _build_ac3_mp4(self, language: str) -> bytes:
        return build_multi_track_mp4(
            [
                {
                    "kind": "ac3",
                    "samples": AC3_SAMPLES,
                    "timescale": 48000,
                    "deltas": [1536, 1536],
                    "channels": 2,
                    "sample_rate": 48000,
                    "language": language,
                }
            ]
        )

    def test_language_passthrough(self):
        mp4 = parse_mp4(self._build_ac3_mp4("eng"))
        self.assertEqual(mp4.tracks[0].language, "eng")
        doc = parse_document(write_mkv(mp4))
        segment = next(x for x in doc if x.element_id == SEGMENT)
        entry = segment.find(TRACKS).find_all(TRACK_ENTRY)[0]
        self.assertEqual(entry.find(LANGUAGE).as_string(), "eng")

    def test_und_language_omits_element(self):
        # 未指定语言（und）时不写 Language，交由 Matroska 缺省值处理
        mp4 = parse_mp4(self._build_ac3_mp4("und"))
        self.assertEqual(mp4.tracks[0].language, "und")
        doc = parse_document(write_mkv(mp4))
        segment = next(x for x in doc if x.element_id == SEGMENT)
        entry = segment.find(TRACKS).find_all(TRACK_ENTRY)[0]
        self.assertIsNone(entry.find(LANGUAGE))

    def test_subtitle_language_passthrough(self):
        mp4_bytes = build_multi_track_mp4(
            [
                {
                    "kind": "stpp",
                    "samples": [TTML_DOC],
                    "timescale": 1000,
                    "deltas": [1000],
                    "language": "fra",
                }
            ]
        )
        mp4 = parse_mp4(mp4_bytes)
        self.assertEqual(mp4.tracks[0].language, "fra")
        doc = parse_document(write_mkv(mp4))
        segment = next(x for x in doc if x.element_id == SEGMENT)
        entry = segment.find(TRACKS).find_all(TRACK_ENTRY)[0]
        self.assertEqual(entry.find(LANGUAGE).as_string(), "fra")


class WebVttConfigTests(unittest.TestCase):
    def _wvtt_mp4(self, vttc_config: bytes) -> bytes:
        return build_multi_track_mp4(
            [
                {
                    "kind": "wvtt",
                    "samples": [_wvtt_cue(b"", b"", b"Only text")],
                    "timescale": 1000,
                    "deltas": [1000],
                    "vttc_config": vttc_config,
                }
            ]
        )

    def test_vttc_config_box_becomes_codec_private(self):
        from mp4_to_mkv.matroska_ids import CODEC_PRIVATE

        config = b"WEBVTT\x00config"
        mp4 = parse_mp4(self._wvtt_mp4(config))
        # 解析器把 vttC 负载原样取出
        self.assertEqual(mp4.tracks[0].codec_private, config)

        doc = parse_document(write_mkv(mp4))
        segment = next(e for e in doc if e.element_id == SEGMENT)
        entry = segment.find(TRACKS).find_all(TRACK_ENTRY)[0]
        self.assertEqual(entry.find(CODEC_ID).as_string(), "S_TEXT/WEBVTT")
        # MKV CodecPrivate 与源 vttC 逐字节一致
        self.assertEqual(entry.find(CODEC_PRIVATE).payload, config)

    def test_no_vttc_omits_codec_private(self):
        from mp4_to_mkv.matroska_ids import CODEC_PRIVATE

        mp4 = parse_mp4(self._wvtt_mp4(b""))
        self.assertEqual(mp4.tracks[0].codec_private, b"")
        doc = parse_document(write_mkv(mp4))
        segment = next(e for e in doc if e.element_id == SEGMENT)
        entry = segment.find(TRACKS).find_all(TRACK_ENTRY)[0]
        self.assertIsNone(entry.find(CODEC_PRIVATE))


class WebVttConverterUnitTests(unittest.TestCase):
    def test_vtte_empty_cue(self):
        from mp4_to_mkv.webvtt import convert_sample

        cue = convert_sample(box("vtte"))
        self.assertEqual(cue.payload, b"")
        self.assertFalse(cue.has_additions)

    def test_payload_only_cue(self):
        from mp4_to_mkv.webvtt import convert_sample

        cue = convert_sample(box("vttc", box("payl", b"text")))
        self.assertEqual(cue.payload, b"text")
        self.assertFalse(cue.has_additions)

    def test_empty_sample(self):
        from mp4_to_mkv.webvtt import convert_sample

        cue = convert_sample(b"")
        self.assertEqual(cue.payload, b"")

    def test_corrupt_sample_raises(self):
        from mp4_to_mkv.errors import RemuxError
        from mp4_to_mkv.webvtt import convert_sample

        # 既非 vttc 也非 vtte 的合法盒 → 明确报错而非产出坏字幕
        with self.assertRaises(RemuxError):
            convert_sample(box("payl", b"orphan"))


if __name__ == "__main__":
    unittest.main()
