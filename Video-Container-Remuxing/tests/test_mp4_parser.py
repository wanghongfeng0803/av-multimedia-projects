"""MP4 解析器测试：用手工构造的自洽 MP4 验证样本表还原与容错。"""

import unittest
from struct import pack

from mp4_to_mkv.errors import RemuxError
from mp4_to_mkv.mp4_parser import parse_boxes, parse_mp4

from tests.mp4_builder import box, build_mp4, fullbox


def video_frames():
    # 长度各不相同，便于验证 size/offset 还原
    return [b"\x00\x00\x00\x01\x67" + bytes([i + 1]) * 8 for i in range(6)]


class VideoTrackTests(unittest.TestCase):
    def test_basic_sample_table(self):
        frames = video_frames()
        data = build_mp4(frames, keyframes=[1, 4])
        mp4 = parse_mp4(data)
        self.assertEqual(len(mp4.tracks), 1)
        track = mp4.tracks[0]
        self.assertTrue(track.is_video)
        self.assertEqual(track.codec_family, "avc")
        self.assertEqual((track.width, track.height), (320, 240))
        self.assertEqual(len(track.samples), len(frames))

        # 逐样本核对 size / 偏移 / 帧字节原样可取回
        for i, sample in enumerate(track.samples):
            self.assertEqual(sample.size, len(frames[i]))
            self.assertEqual(
                data[sample.offset:sample.offset + sample.size], frames[i]
            )

        # DTS 为 delta 前缀和，默认等间隔 1000（24000/24）
        dts_list = [s.dts for s in track.samples]
        self.assertEqual(dts_list, [0, 1000, 2000, 3000, 4000, 5000])
        # 无 ctts 时 PTS == DTS
        self.assertTrue(all(s.pts == s.dts for s in track.samples))
        # 关键帧标记（1 起始序号 1、4）
        self.assertEqual(
            [s.is_keyframe for s in track.samples],
            [True, False, False, True, False, False],
        )

    def test_ctts_positive_offsets(self):
        frames = video_frames()
        offsets = [2000, 1000, 0, 0, -1000, -2000]
        # 含负偏移必须使用 version 1（version 0 的偏移为无符号整数）
        data = build_mp4(
            frames, ctts_offsets=offsets, ctts_version=1, keyframes=[1]
        )
        track = parse_mp4(data).tracks[0]
        for i, sample in enumerate(track.samples):
            self.assertEqual(sample.pts - sample.dts, offsets[i])

    def test_ctts_version1_signed(self):
        frames = video_frames()
        # version 1 下 0xFFFFFFFF 表示 -1
        data = build_mp4(
            frames,
            ctts_offsets=[0, 0, 0, 0, 0, -1],
            ctts_version=1,
            keyframes=[1],
        )
        track = parse_mp4(data).tracks[0]
        self.assertEqual(track.samples[-1].pts - track.samples[-1].dts, -1)

    def test_co64_offsets(self):
        frames = video_frames()
        data = build_mp4(frames, use_co64=True, keyframes=[1])
        track = parse_mp4(data).tracks[0]
        for i, sample in enumerate(track.samples):
            self.assertEqual(
                data[sample.offset:sample.offset + sample.size], frames[i]
            )

    def test_mdhd_version1(self):
        frames = video_frames()
        data = build_mp4(frames, mdhd_version=1, keyframes=[1])
        track = parse_mp4(data).tracks[0]
        self.assertEqual(track.timescale, 24000)
        self.assertEqual(track.duration, 6000)

    def test_variable_deltas(self):
        frames = video_frames()
        deltas = [500, 1500, 1000, 2000, 500, 2500]
        data = build_mp4(frames, deltas=deltas, keyframes=[1])
        track = parse_mp4(data).tracks[0]
        expected_dts = [0]
        for d in deltas[:-1]:
            expected_dts.append(expected_dts[-1] + d)
        self.assertEqual([s.dts for s in track.samples], expected_dts)


class AudioTrackTests(unittest.TestCase):
    def test_aac_track(self):
        packets = [bytes([0x21, 0x10, i]) * 10 for i in range(4)]
        data = build_mp4(
            packets,
            kind="audio",
            timescale=44100,
            deltas=[1024] * 4,
            channels=2,
            sample_rate=44100,
        )
        track = parse_mp4(data).tracks[0]
        self.assertTrue(track.is_audio)
        self.assertEqual(track.codec_family, "aac")
        self.assertEqual(track.channels, 2)
        self.assertEqual(track.sample_rate, 44100)
        # CodecPrivate 即 AudioSpecificConfig
        self.assertEqual(track.codec_private, b"\x12\x10")
        self.assertEqual(len(track.samples), 4)
        self.assertTrue(all(s.is_keyframe for s in track.samples))
        self.assertEqual([s.dts for s in track.samples], [0, 1024, 2048, 3072])


class InvalidInputTests(unittest.TestCase):
    def _assert_remux_error(self, data: bytes, fragment: str):
        with self.assertRaises(RemuxError) as ctx:
            parse_mp4(data)
        self.assertIn(fragment, str(ctx.exception))

    def test_too_small(self):
        self._assert_remux_error(b"\x00" * 4, "不足")

    def test_missing_ftyp(self):
        data = box("mdat", b"xxxx")
        self._assert_remux_error(data, "ftyp")

    def test_missing_moov(self):
        data = box("ftyp", b"isom" + b"\x00" * 4) + box("mdat", b"xxxx")
        self._assert_remux_error(data, "moov")

    def test_truncated_box(self):
        good = build_mp4(video_frames(), keyframes=[1])
        # 截掉最后 20 字节，mdat/stco 指向的数据将越界
        self._assert_remux_error(good[:-20], "损坏")

    def test_box_size_smaller_than_header(self):
        # 声明 size=4 但盒头本身就要 8 字节
        bad = b"\x00\x00\x00\x04ftyp"
        self._assert_remux_error(bad, "盒头")

    def test_fragmented_mp4_rejected(self):
        moof = box("moof", box("mfhd", fullbox("mfhd", 0, 0, b"\x00" * 4)[8:]))
        data = build_mp4(video_frames(), keyframes=[1], extra_top_boxes=[moof])
        self._assert_remux_error(data, "碎片化")

    def test_encrypted_entry_rejected(self):
        # 构造可解析的最小盒树：ftyp + moov(trak(mdia(sinf)))
        sinf = box("sinf", box("frma", b"avc1"))
        wrapped = (
            box("ftyp", b"isom" + b"\x00" * 8)
            + box("moov", box("trak", box("mdia", sinf)))
        )
        self._assert_remux_error(wrapped, "加密")

    def test_zero_sample_size_rejected(self):
        # stsz 中声明样本长度为 0：手工构造 stsz（uniform=0, count=1, size=0）
        bad_stsz = fullbox(
            "stsz", 0, 0,
            b"\x00\x00\x00\x00" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x00",
        )
        from mp4_to_mkv.mp4_parser import Box, _build_samples, _parse_stsz
        stsz_box = Box("stsz", 0, 8, bad_stsz[8:])
        self.assertEqual(_parse_stsz(stsz_box), [0])
        # 样本构建阶段应拒绝 0 长度
        with self.assertRaises(RemuxError) as ctx:
            _build_samples([0], [100], [1], [], None, True)
        self.assertIn("大小为 0", str(ctx.exception))


class LargeSizeBoxTests(unittest.TestCase):
    def test_largesize_box(self):
        payload = b"isom" + b"\x00" * 8
        # 手工构造 64 位 largesize 的 ftyp
        header = b"\x00\x00\x00\x01ftyp" + (16 + len(payload)).to_bytes(8, "big")
        data = header + payload
        boxes = parse_boxes(data)
        self.assertEqual(boxes[0].kind, "ftyp")
        self.assertEqual(boxes[0].header_size, 16)
        self.assertEqual(boxes[0].data, payload)


if __name__ == "__main__":
    unittest.main()
