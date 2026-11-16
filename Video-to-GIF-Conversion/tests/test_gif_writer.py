"""GIF 编码器测试：字节结构、LZW 往返、延迟原样保留。

测试自带一个最小 GIF 解析/LZW 解码器（独立于被测实现重写一遍），
用于验证写出的文件符合 GIF89a 布局且码流可还原为原索引。
"""

import struct

import pytest

from vgif.gif_writer import (
    GifEncodeError,
    lzw_encode,
    pad_palette,
    write_gif,
)
from vgif.quantize import QuantizedFrame


# ---------- 测试辅助：独立 LZW 解码 ----------

def lzw_decode(data: bytes, min_code_size: int) -> bytes:
    """最小 GIF LZW 解码器，仅用于测试核对（与实现独立）。"""
    clear = 1 << min_code_size
    end = clear + 1

    code_size = min_code_size + 1
    dict_entries = [bytes([i]) for i in range(clear)]
    dict_entries.append(b"")        # CLEAR 占位
    dict_entries.append(b"")        # END 占位
    next_code = end + 1

    bit_pos = 0

    def read_code():
        nonlocal bit_pos
        v = 0
        for b in range(code_size):
            byte_i = (bit_pos + b) // 8
            bit_i = (bit_pos + b) % 8
            if byte_i >= len(data):
                return None
            v |= ((data[byte_i] >> bit_i) & 1) << b
        bit_pos += code_size
        return v

    out = bytearray()
    prev = None
    while True:
        code = read_code()
        if code is None or code == end:
            break
        if code == clear:
            code_size = min_code_size + 1
            dict_entries = [bytes([i]) for i in range(clear)]
            dict_entries.append(b"")
            dict_entries.append(b"")
            next_code = end + 1
            prev = None
            continue
        if prev is None:
            entry = dict_entries[code]
            out.extend(entry)
            prev = entry
            continue
        if code < len(dict_entries):
            entry = dict_entries[code]
        else:
            # GIF 经典 KwKwK 情形。
            entry = prev + prev[:1]
        out.extend(entry)
        if next_code < 4096:
            dict_entries.append(prev + entry[:1])
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        prev = entry
    return bytes(out)


def parse_gif(path):
    """极简 GIF 解析：返回 (width, height, palette, [(delay, indices)])。"""
    raw = Path(path).read_bytes() if not isinstance(path, bytes) else path
    assert raw[:6] in (b"GIF87a", b"GIF89a")
    w, h = struct.unpack_from("<HH", raw, 6)
    packed = raw[10]
    has_gct = packed & 0x80
    k = packed & 0x07
    pos = 13
    palette = []
    if has_gct:
        n = 1 << (k + 1)
        for i in range(n):
            palette.append(tuple(raw[pos + 3 * i: pos + 3 * i + 3]))
        pos += 3 * n

    frames = []
    while raw[pos] != 0x3B:
        introducer = raw[pos]
        if introducer == 0x21:  # 扩展
            label = raw[pos + 1]
            pos += 2
            if label == 0xF9:  # GCE
                block_size = raw[pos]
                assert block_size == 4
                delay = struct.unpack_from("<H", raw, pos + 2)[0]
                pos += block_size + 1  # 跳到块终止
                pos += 1
                frames.append({"delay": delay, "indices": None})
            else:
                # 跳过其它扩展的所有子块。
                while raw[pos] != 0:
                    pos += 1 + raw[pos]
                pos += 1
        elif introducer == 0x2C:  # 图像
            left, top, iw, ih, ipacked = struct.unpack_from(
                "<HHHHB", raw, pos + 1
            )
            pos += 10
            assert not (ipacked & 0x80)  # 无局部颜色表
            min_code_size = raw[pos]
            pos += 1
            sub = bytearray()
            while raw[pos] != 0:
                size = raw[pos]
                sub.extend(raw[pos + 1: pos + 1 + size])
                pos += 1 + size
            pos += 1  # 块终止
            indices = lzw_decode(bytes(sub), min_code_size)
            frames[-1]["indices"] = indices
            frames[-1]["size"] = (iw, ih)
        else:
            raise AssertionError(f"未预期的字节 0x{introducer:02x} @ {pos}")
    return w, h, palette, frames


# 需要 Path 仅用于类型清晰。
from pathlib import Path  # noqa: E402


class TestPadPalette:
    def test_pads_to_power_of_two(self):
        pal = tuple((i, i, i) for i in range(3))
        out = pad_palette(pal)
        assert len(out) == 4
        assert out[-1] == pal[-1]

    def test_256_kept(self):
        pal = tuple((i, 0, 0) for i in range(256))
        assert len(pad_palette(pal)) == 256

    def test_empty_rejected(self):
        with pytest.raises(GifEncodeError):
            pad_palette(())


class TestLzw:
    def test_roundtrip_small(self):
        idx = bytes([0, 1, 2, 3, 0, 1, 2, 3])
        stream = lzw_encode(idx, 2)
        assert lzw_decode(stream, 2) == idx

    def test_roundtrip_8bit_long(self):
        idx = bytes((i * 37) % 256 for i in range(2000))
        stream = lzw_encode(idx, 8)
        assert lzw_decode(stream, 8) == idx

    def test_roundtrip_repetitive_clears_table(self):
        # 高度重复序列会撑码表触发 CLEAR，验证重置逻辑可往返。
        idx = bytes(range(256)) * 20
        stream = lzw_encode(idx, 8)
        assert lzw_decode(stream, 8) == idx

    def test_single_pixel(self):
        stream = lzw_encode(bytes([7]), 8)
        assert lzw_decode(stream, 8) == bytes([7])

    def test_bad_min_code_size(self):
        with pytest.raises(GifEncodeError):
            lzw_encode(bytes([0, 1]), 1)


class TestWriteGif:
    def _qframe(self, w, h, fill):
        return QuantizedFrame(indices=bytes([fill]) * (w * h),
                              width=w, height=h)

    def test_header_and_dimensions(self, tmp_path):
        out = tmp_path / "a.gif"
        q = self._qframe(4, 3, 0)
        write_gif(out, [q], ((10, 20, 30),), (5,), 4, 3)
        w, h, palette, frames = parse_gif(out)
        assert (w, h) == (4, 3)
        assert (10, 20, 30) in palette

    def test_delays_written_exactly(self, tmp_path):
        out = tmp_path / "b.gif"
        qs = [self._qframe(2, 2, 0) for _ in range(3)]
        delays = (8, 9, 8)
        write_gif(out, qs, ((0, 0, 0), (255, 255, 255)), delays, 2, 2)
        _, _, _, frames = parse_gif(out)
        assert [f["delay"] for f in frames] == [8, 9, 8]

    def test_frame_indices_roundtrip(self, tmp_path):
        out = tmp_path / "c.gif"
        idx = bytes([0, 1, 1, 0, 1, 0])
        q = QuantizedFrame(indices=idx, width=3, height=2)
        write_gif(out, [q], ((0, 0, 0), (255, 255, 255)), (10,), 3, 2)
        _, _, _, frames = parse_gif(out)
        assert frames[0]["indices"] == idx

    def test_frame_delay_count_mismatch_rejected(self, tmp_path):
        q = self._qframe(2, 2, 0)
        with pytest.raises(GifEncodeError):
            write_gif(tmp_path / "x.gif", [q], ((0, 0, 0),), (1, 2), 2, 2)

    def test_no_frames_rejected(self, tmp_path):
        with pytest.raises(GifEncodeError):
            write_gif(tmp_path / "x.gif", [], ((0, 0, 0),), (), 2, 2)

    def test_trailer_present(self, tmp_path):
        out = tmp_path / "d.gif"
        q = self._qframe(2, 2, 0)
        write_gif(out, [q], ((0, 0, 0),), (10,), 2, 2)
        assert out.read_bytes()[-1] == 0x3B

    def test_netscape_loop_extension(self, tmp_path):
        out = tmp_path / "e.gif"
        q = self._qframe(2, 2, 0)
        write_gif(out, [q], ((0, 0, 0),), (10,), 2, 2)
        assert b"NETSCAPE2.0" in out.read_bytes()
