"""端到端重封装测试：MP4 → MKV，并解析输出校验结构与帧字节无损。"""

import struct
import tempfile
import unittest
from pathlib import Path

from mp4_to_mkv import __main__ as cli
from mp4_to_mkv.matroska_ids import (
    CLUSTER,
    CODEC_ID,
    CODEC_PRIVATE,
    DOC_TYPE,
    EBML_HEADER,
    FLAG_DEFAULT,
    INFO,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    SEGMENT,
    SIMPLE_BLOCK,
    TIMESTAMP,
    TRACK_ENTRY,
    TRACK_NUMBER,
    TRACK_TYPE,
    TRACK_TYPE_AUDIO,
    TRACK_TYPE_VIDEO,
    TRACKS,
    VIDEO,
)
from mp4_to_mkv.mp4_parser import parse_mp4
from mp4_to_mkv.mkv_writer import write_mkv

from tests.ebml_reader import parse_document, read_vint
from tests.mp4_builder import build_mp4


def video_frames():
    return [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(6)]


def extract_blocks(mkv_bytes: bytes):
    """解析 MKV，返回 (轨道元信息列表, 每个 SimpleBlock 的信息列表)。"""
    doc = parse_document(mkv_bytes)
    header = next(e for e in doc if e.element_id == EBML_HEADER)
    self_doc_type = header.find(DOC_TYPE).as_string()
    assert self_doc_type == "matroska"

    segment = next(e for e in doc if e.element_id == SEGMENT)
    tracks_el = segment.find(TRACKS)
    track_entries = tracks_el.find_all(TRACK_ENTRY)

    blocks = []
    for cluster in segment.find_all(CLUSTER):
        cluster_ts = cluster.find(TIMESTAMP).as_uint()
        for block_el in cluster.find_all(SIMPLE_BLOCK):
            payload = block_el.payload
            track_no, pos = read_vint(payload, 0)
            relative_tc = struct.unpack(">h", payload[pos:pos + 2])[0]
            flags = payload[pos + 2]
            frame = payload[pos + 3:]
            blocks.append(
                {
                    "track": track_no,
                    "pts_ms": cluster_ts + relative_tc,
                    "keyframe": bool(flags & 0x80),
                    "frame": frame,
                }
            )
    return track_entries, blocks


class RemuxVideoTests(unittest.TestCase):
    def test_structure_and_lossless_frames(self):
        frames = video_frames()
        mp4_bytes = build_mp4(frames, keyframes=[1, 4])
        mkv_bytes = write_mkv(parse_mp4(mp4_bytes))

        track_entries, blocks = extract_blocks(mkv_bytes)
        self.assertEqual(len(track_entries), 1)
        entry = track_entries[0]
        self.assertEqual(entry.find(TRACK_NUMBER).as_uint(), 1)
        self.assertEqual(entry.find(TRACK_TYPE).as_uint(), TRACK_TYPE_VIDEO)
        self.assertEqual(entry.find(CODEC_ID).as_string(), "V_MPEG4/ISO/AVC")
        video_el = entry.find(VIDEO)
        self.assertEqual(video_el.find(PIXEL_WIDTH).as_uint(), 320)
        self.assertEqual(video_el.find(PIXEL_HEIGHT).as_uint(), 240)
        self.assertEqual(entry.find(FLAG_DEFAULT).as_uint(), 1)
        # CodecPrivate 与 MP4 中的 avcC 负载一致
        avcc = parse_mp4(mp4_bytes).tracks[0].codec_private
        self.assertEqual(entry.find(CODEC_PRIVATE).payload, avcc)

        # 帧数一致、顺序一致、字节逐字节相同（无损的直接证据）
        self.assertEqual(len(blocks), len(frames))
        for block, frame in zip(blocks, frames):
            self.assertEqual(block["frame"], frame)
        # 关键帧标记
        self.assertEqual(
            [b["keyframe"] for b in blocks],
            [True, False, False, True, False, False],
        )

    def test_timestamps_in_milliseconds(self):
        # timescale=24000，每帧 delta=1000 → 每帧 1000/24000*1000 ≈ 41.67ms
        frames = video_frames()
        mp4_bytes = build_mp4(frames, keyframes=[1])
        mkv_bytes = write_mkv(parse_mp4(mp4_bytes))
        _, blocks = extract_blocks(mkv_bytes)
        # 四舍五入后的毫秒序列：0, 42, 83, 125, 167, 208
        expected = [0, 42, 83, 125, 167, 208]
        self.assertEqual([b["pts_ms"] for b in blocks], expected)

    def test_bframe_pts_dts_ordering(self):
        # ctts 让显示顺序与解码顺序不同；含负偏移必须用 version 1
        frames = video_frames()
        ctts = [2000, 1000, 0, 0, -1000, -2000]
        mp4_bytes = build_mp4(
            frames, ctts_offsets=ctts, ctts_version=1, keyframes=[1]
        )
        mkv_bytes = write_mkv(parse_mp4(mp4_bytes))
        _, blocks = extract_blocks(mkv_bytes)

        # 写出顺序 = 解码顺序（帧 0..5 原样顺序），B 帧重排由解码器完成
        self.assertEqual([b["frame"] for b in blocks], frames)
        # PTS = DTS + ctts：[2000,2000,2000,3000,3000,3000]（该夹具无 edit），
        # 毫秒四舍五入为 [83,83,83,125,125,125]
        self.assertEqual(
            [b["pts_ms"] for b in blocks], [83, 83, 83, 125, 125, 125]
        )

    def test_cluster_split_on_keyframe(self):
        # 构造足够多的大帧，触发目标大小后的关键帧切簇
        big_frames = [b"\xAA" * 200000 for _ in range(40)]
        mp4_bytes = build_mp4(
            big_frames,
            keyframes=list(range(1, 41, 5)),  # 每 5 帧一个关键帧
        )
        mkv_bytes = write_mkv(parse_mp4(mp4_bytes))
        doc = parse_document(mkv_bytes)
        segment = next(e for e in doc if e.element_id == SEGMENT)
        clusters = segment.find_all(CLUSTER)
        self.assertGreater(len(clusters), 1)
        # 每个簇的第一块必须是关键帧
        for cluster in clusters:
            first_block = cluster.find_all(SIMPLE_BLOCK)[0]
            flags = first_block.payload[
                # 跳过 track vint（测试轨道号 1，占 1 字节）+ timecode 2 字节
                1 + 2
            ]
            self.assertTrue(flags & 0x80)


class RemuxAudioTests(unittest.TestCase):
    def test_aac_lossless(self):
        packets = [bytes([0x21, 0x10, i]) * 10 for i in range(4)]
        mp4_bytes = build_mp4(
            packets,
            kind="audio",
            timescale=44100,
            deltas=[1024] * 4,
        )
        mkv_bytes = write_mkv(parse_mp4(mp4_bytes))
        track_entries, blocks = extract_blocks(mkv_bytes)
        entry = track_entries[0]
        self.assertEqual(entry.find(TRACK_TYPE).as_uint(), TRACK_TYPE_AUDIO)
        self.assertEqual(entry.find(CODEC_ID).as_string(), "A_AAC")
        self.assertEqual(entry.find(CODEC_PRIVATE).payload, b"\x12\x10")
        self.assertEqual([b["frame"] for b in blocks], packets)
        # 1024/44100 秒每包 ≈ 23.22ms，四舍五入：0, 23, 46, 70
        self.assertEqual([b["pts_ms"] for b in blocks], [0, 23, 46, 70])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.src = root / "in.mp4"
        self.dst = root / "out.mkv"
        self.src.write_bytes(build_mp4(video_frames(), keyframes=[1]))

    def test_success_exit_code(self):
        code = cli.main([str(self.src), str(self.dst)])
        self.assertEqual(code, 0)
        self.assertTrue(self.dst.exists())
        # 输出可被独立解析
        parse_document(self.dst.read_bytes())

    def test_missing_input(self):
        code = cli.main([str(self.src.parent / "nope.mp4"), str(self.dst)])
        self.assertEqual(code, cli.EXIT_INVALID_INPUT)

    def test_bad_extension(self):
        weird = self.src.parent / "in.txt"
        weird.write_bytes(b"hello")
        code = cli.main([str(weird), str(self.dst)])
        self.assertEqual(code, cli.EXIT_INVALID_INPUT)

    def test_output_exists_requires_overwrite(self):
        self.dst.write_bytes(b"old")
        code = cli.main([str(self.src), str(self.dst)])
        self.assertEqual(code, cli.EXIT_INVALID_INPUT)
        self.assertEqual(self.dst.read_bytes(), b"old")
        code = cli.main([str(self.src), str(self.dst), "--overwrite"])
        self.assertEqual(code, 0)

    def test_empty_file(self):
        empty = self.src.parent / "empty.mp4"
        empty.write_bytes(b"")
        code = cli.main([str(empty), str(self.dst)])
        self.assertEqual(code, cli.EXIT_INVALID_INPUT)

    def test_garbage_bytes(self):
        garbage = self.src.parent / "garbage.mp4"
        garbage.write_bytes(b"X" * 1024)
        code = cli.main([str(garbage), str(self.dst)])
        self.assertEqual(code, cli.EXIT_INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()
