"""测试辅助：手工构造最小但合法的 ISO-BMFF/MP4 字节流。

不依赖任何编码器：帧内容用可辨识的伪 NAL/AAC 字节代替，重点是让
盒结构、样本表与时间戳完全自洽，供解析器与重封装逻辑做正确性测试。

构建采用两遍法：第一遍用“等长占位偏移表”确定 moov/mdat 布局，
拿到 mdat 数据起点后第二遍填入真实样本偏移，避免回填导致的长度漂移。
"""

from __future__ import annotations

from struct import pack
from typing import Optional, Sequence


def box(kind: str, payload: bytes = b"") -> bytes:
    return pack(">I", 8 + len(payload)) + kind.encode("latin-1") + payload


def fullbox(kind: str, version: int, flags: int, payload: bytes) -> bytes:
    return box(kind, bytes([version]) + flags.to_bytes(3, "big") + payload)


def _ftyp() -> bytes:
    payload = b"isom" + pack(">I", 0x200) + b"isomavc1mp41"
    return box("ftyp", payload)


def _mdhd(timescale: int, duration: int, version: int) -> bytes:
    language = pack(">I", 0x55C40000)
    if version == 1:
        # v1：creation(8) modification(8) timescale(4) duration(8) + language(4)
        return fullbox(
            "mdhd", 1, 0,
            pack(">QQIQ", 0, 0, timescale, duration) + language,
        )
    return fullbox("mdhd", 0, 0, pack(">IIII", 0, 0, timescale, duration) + language)


def _hdlr(handler: str) -> bytes:
    payload = (
        pack(">I", 0)
        + pack(">I", 0)
        + handler.encode("latin-1")
        + b"\x00" * 12
        + b"name\x00"
    )
    return fullbox("hdlr", 0, 0, payload)


def _stts(deltas: Sequence[int]) -> bytes:
    payload = pack(">I", len(deltas)) + b"".join(pack(">II", 1, d) for d in deltas)
    return fullbox("stts", 0, 0, payload)


def _ctts(offsets: Sequence[int], version: int) -> bytes:
    payload = pack(">I", len(offsets))
    for off in offsets:
        payload += pack(">II", 1, off & 0xFFFFFFFF)
    return fullbox("ctts", version, 0, payload)


def _stsc_one_per_chunk() -> bytes:
    # entry_count=1；表项 (first_chunk=1, samples_per_chunk=1, desc_index=1)
    return fullbox("stsc", 0, 0, pack(">IIII", 1, 1, 1, 1))


def _stsz(sizes: Sequence[int]) -> bytes:
    return fullbox(
        "stsz", 0, 0, pack(">II", 0, len(sizes)) + b"".join(pack(">I", s) for s in sizes)
    )


def _stco(offsets: Sequence[int]) -> bytes:
    return fullbox("stco", 0, 0, pack(">I", len(offsets)) + b"".join(pack(">I", o) for o in offsets))


def _co64(offsets: Sequence[int]) -> bytes:
    return fullbox("co64", 0, 0, pack(">I", len(offsets)) + b"".join(pack(">Q", o) for o in offsets))


def _stss(keyframe_numbers: Sequence[int]) -> bytes:
    return fullbox(
        "stss", 0, 0,
        pack(">I", len(keyframe_numbers)) + b"".join(pack(">I", n) for n in keyframe_numbers),
    )


def _avc1_entry(width: int, height: int, avcc: bytes) -> bytes:
    # 视觉 sample entry 负载固定头 78 字节（不含盒头，整盒共 86 字节）
    fixed = (
        b"\x00" * 6
        + pack(">H", 1)                     # data_reference_index
        + b"\x00" * 16                      # pre_defined / reserved × 多组
        + pack(">HH", width, height)        # 负载偏移 24 / 26
        + pack(">II", 0x00480000, 0x00480000)
        + b"\x00" * 4
        + pack(">H", 1)                     # frame_count
        + b"\x00" * 32                      # compressorname
        + pack(">H", 0x0018)                # depth
        + pack(">h", -1)
    )
    assert len(fixed) == 78
    return box("avc1", fixed + box("avcC", avcc))


def _mp4a_entry(channels: int, sample_rate: int, asc: bytes) -> bytes:
    # 音频 sample entry v0 负载固定头 28 字节（整盒共 36 字节）
    fixed = (
        b"\x00" * 6
        + pack(">H", 1)                     # data_reference_index（偏移 6）
        + b"\x00" * 8                       # reserved（偏移 8..15）
        + pack(">H", channels)              # channelcount（偏移 16）
        + pack(">H", 16)                    # samplesize（偏移 18）
        + pack(">H", 0)                     # pre_defined（偏移 20）
        + pack(">H", 0)                     # reserved（偏移 22）
        + pack(">I", sample_rate << 16)     # samplerate 16.16（偏移 24）
    )
    assert len(fixed) == 28

    def descriptor(tag: int, content: bytes) -> bytes:
        assert len(content) < 128
        return bytes([tag, len(content)]) + content

    dec_specific = descriptor(0x05, asc)
    dec_config = descriptor(
        0x04,
        bytes([0x40, 0x15, 0x00, 0x00, 0x00]) + pack(">II", 0, 0) + dec_specific,
    )
    es_descr = descriptor(0x03, pack(">H", 1) + bytes([0x00]) + dec_config)
    return box("mp4a", fixed + fullbox("esds", 0, 0, es_descr))


def build_mp4(
    frames: Sequence[bytes],
    *,
    kind: str = "video",
    timescale: int = 24000,
    deltas: Optional[Sequence[int]] = None,
    ctts_offsets: Optional[Sequence[int]] = None,
    keyframes: Optional[Sequence[int]] = None,
    width: int = 320,
    height: int = 240,
    channels: int = 2,
    sample_rate: int = 44100,
    use_co64: bool = False,
    mdhd_version: int = 0,
    ctts_version: int = 0,
    extra_top_boxes: Sequence[bytes] = (),
) -> bytes:
    """构造一个单 track、每样本一个 chunk 的 MP4。"""
    count = len(frames)
    if count == 0:
        raise ValueError("至少需要一个样本")
    if deltas is None:
        deltas = [timescale // 24] * count
    duration = sum(deltas)

    if kind == "video":
        avcc = bytes([0x01, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x01, 0x67, 0x00])
        entry = _avc1_entry(width, height, avcc)
    elif kind == "audio":
        asc = bytes([0x12, 0x10])  # AAC-LC 44.1kHz stereo
        entry = _mp4a_entry(channels, sample_rate, asc)
    else:  # pragma: no cover - 测试自身参数错误
        raise ValueError(f"未知 kind: {kind}")

    def build_moov(offset_box: bytes) -> bytes:
        stsd = fullbox("stsd", 0, 0, pack(">I", 1) + entry)
        tables = [stsd, _stts(deltas), _stsc_one_per_chunk(), _stsz([len(f) for f in frames])]
        if ctts_offsets is not None:
            tables.append(_ctts(ctts_offsets, ctts_version))
        if kind == "video" and keyframes is not None:
            tables.append(_stss(keyframes))
        tables.append(offset_box)
        stbl = box("stbl", b"".join(tables))

        if kind == "video":
            media_header = box("vmhd", b"\x00" * 8)
        else:
            media_header = box("smhd", b"\x00" * 8)
        url = fullbox("url ", 0, 1, b"")
        dref = fullbox("dref", 0, 0, pack(">I", 1) + url)
        dinf = box("dinf", dref)
        minf = box("minf", media_header + dinf + stbl)
        mdia = box(
            "mdia",
            _mdhd(timescale, duration, mdhd_version)
            + _hdlr("vide" if kind == "video" else "soun")
            + minf,
        )

        if kind == "video":
            tkhd_body = (
                pack(">IIIIII", 0, 0, 1, 1, 0, duration)
                + b"\x00" * 8 + pack(">hh", 0, 0)
                + pack(">IIII", 0x00010000, 0, 0, 0x00010000)
                + pack(">II", width << 16, height << 16)
            )
        else:
            tkhd_body = (
                pack(">IIIIII", 0, 0, 1, 1, 0, duration)
                + b"\x00" * 8 + pack(">hh", 0, 0) + b"\x00" * 36
            )
        tkhd = fullbox("tkhd", 0, 0x07, tkhd_body)
        trak = box("trak", tkhd + mdia)
        mvhd = fullbox(
            "mvhd", 0, 0,
            pack(">IIII", 0, 0, timescale, duration) + b"\x00" * 80,
        )
        return box("moov", mvhd + trak)

    # 第一遍：用全零偏移表（与真实偏移表等长）确定整体布局
    placeholder = _co64([0] * count) if use_co64 else _stco([0] * count)
    moov_placeholder = build_moov(placeholder)
    mdat_data_start = len(_ftyp()) + len(moov_placeholder) + 8  # mdat 盒头 8 字节

    sample_offsets = []
    cursor = mdat_data_start
    for frame in frames:
        sample_offsets.append(cursor)
        cursor += len(frame)

    # 第二遍：填入真实偏移；偏移表长度与占位时完全一致，布局不再变化
    offset_box = _co64(sample_offsets) if use_co64 else _stco(sample_offsets)
    moov = build_moov(offset_box)
    assert len(moov) == len(moov_placeholder)

    mdat = box("mdat", b"".join(frames))
    return _ftyp() + moov + b"".join(extra_top_boxes) + mdat
