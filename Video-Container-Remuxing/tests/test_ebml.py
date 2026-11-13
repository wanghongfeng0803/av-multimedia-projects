"""EBML 编码层的已知答案向量测试（不依赖外部工具）。"""

import unittest

from mp4_to_mkv import ebml
from mp4_to_mkv.errors import RemuxError  # noqa: F401  确保包可导入
from mp4_to_mkv.matroska_ids import (
    CLUSTER,
    DOC_TYPE,
    EBML_HEADER,
    SIMPLE_BLOCK,
    TRACK_ENTRY,
)


class VintTests(unittest.TestCase):
    def test_known_vectors(self):
        # (输入, 期望字节) 覆盖 1~4 字节长度与边界值
        cases = [
            (0, b"\x80"),
            (1, b"\x81"),
            (0x7F, b"\xFF"),
            (0x80, b"\x40\x80"),
            (0x3FFF, b"\x7F\xFF"),
            (0x4000, b"\x20\x40\x00"),
            (0x1FFFFF, b"\x3F\xFF\xFF"),
            (0x200000, b"\x10\x20\x00\x00"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(ebml.encode_vint(value), expected)

    def test_negative_rejected(self):
        with self.assertRaises(ebml.EbmlEncodeError):
            ebml.encode_vint(-1)

    def test_unknown_size(self):
        self.assertEqual(ebml.encode_vint(0, unknown=True), b"\xff" * 8)


class ElementIdTests(unittest.TestCase):
    def test_widths(self):
        self.assertEqual(ebml.encode_element_id(TRACK_ENTRY), b"\xae")          # 1 字节
        self.assertEqual(ebml.encode_element_id(SIMPLE_BLOCK), b"\xa3")         # 1 字节
        self.assertEqual(ebml.encode_element_id(CLUSTER), b"\x1f\x43\xb6\x75")  # 4 字节
        self.assertEqual(ebml.encode_element_id(EBML_HEADER), b"\x1a\x45\xdf\xa3")
        self.assertEqual(ebml.encode_element_id(DOC_TYPE), b"\x42\x82")         # 2 字节

    def test_invalid_id(self):
        with self.assertRaises(ebml.EbmlEncodeError):
            ebml.encode_element_id(0x05)  # 首字节无 VINT 长度标记


class ElementTests(unittest.TestCase):
    def test_uint_minimal_encoding(self):
        # value=0 → 空数据区
        self.assertEqual(ebml.uint_element(0x4286, 0), b"\x42\x86\x80")
        # value=1：ID(2) + size 0x81 + data 0x01
        self.assertEqual(ebml.uint_element(0x4286, 1), b"\x42\x86\x81\x01")
        # 256 需要 2 字节数据
        self.assertEqual(
            ebml.uint_element(0x4286, 256), b"\x42\x86\x82\x01\x00"
        )

    def test_sint_twos_complement(self):
        self.assertEqual(ebml.sint_element(0x73C5, 0), b"\x73\xc5\x80")
        self.assertEqual(ebml.sint_element(0x73C5, -1), b"\x73\xc5\x81\xff")
        self.assertEqual(ebml.sint_element(0x73C5, 127), b"\x73\xc5\x81\x7f")
        # 127 是单字节正数上限；128 需要 2 字节以保留符号位
        self.assertEqual(ebml.sint_element(0x73C5, 128), b"\x73\xc5\x82\x00\x80")
        # -128 的补码 0x80 已自带符号位，单字节即合法最短编码
        self.assertEqual(ebml.sint_element(0x73C5, -128), b"\x73\xc5\x81\x80")
        # -129 才需要 2 字节
        self.assertEqual(ebml.sint_element(0x73C5, -129), b"\x73\xc5\x82\xff\x7f")

    def test_string_and_master(self):
        el = ebml.string_element(DOC_TYPE, "matroska")
        # 42 82 = DocType，88 = size 8，后接 8 字节 ASCII
        self.assertEqual(el, b"\x42\x82\x88matroska")
        master = ebml.master_element(EBML_HEADER, [el])
        self.assertTrue(master.startswith(b"\x1a\x45\xdf\xa3"))
        # 主元素 size = 子元素全长
        self.assertEqual(master[4], 0x80 | len(el))

    def test_utf8(self):
        el = ebml.utf8_element(0x5741, "中文")
        encoded = "中文".encode("utf-8")
        self.assertEqual(el, b"\x57\x41" + ebml.encode_vint(len(encoded)) + encoded)


if __name__ == "__main__":
    unittest.main()
