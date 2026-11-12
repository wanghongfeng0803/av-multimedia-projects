# -*- coding: utf-8 -*-
"""
fragmented MP4 增量读取器。

ffmpeg 把转码后的流以 fragmented mp4（ftyp/moov 后接若干 moof/mdat）写到管道，
本模块从任意边界的字节块里解析出：

- moov 事件：各 track 的 timescale、stsd（编码配置）、trex 默认值、媒体类型
- moof 事件：一个 fragment 内每个 track 的样本列表（dts/pts/duration/sync/原始字节）

只做「读」，如何切 HLS 分片由 pipeline 决定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import boxes as B
from .boxes import Box, Reader, iter_boxes, find_child, find_path, parse_fullbox


@dataclass
class TrackInfo:
    track_id: int
    media_type: str                 # "video" / "audio" / "other"
    timescale: int
    stsd_box: Box                  # stbl/stsd 盒子（含编码配置，重建 init/变化检测用）
    stsd_blob: bytes               # stsd payload 的规范化快照，配置变化对比用
    handler: str                   # hdlr name 中的类型 vide/soun
    width: int = 0                 # 视频：宽（像素）
    height: int = 0                # 视频：高（像素）
    sample_rate: int = 0           # 音频：采样率
    channels: int = 0
    default_sample_duration: int = 0
    default_sample_size: int = 0
    default_sample_flags: int = 0
    sample_description_index: int = 1
    # mdhd/tkhd 里构建 init 段需要的字段
    creation_time: int = 0
    modification_time: int = 0
    duration: int = 0
    language: int = B.LANGUAGE_UNDEFINED
    track_width: float = 0.0       # tkhd 16.16 定点宽高
    track_height: float = 0.0
    volume: float = 0.0


@dataclass
class Sample:
    track_id: int
    dts: int                # 轨道 timescale 单位
    pts: int
    duration: int
    sync: bool
    data: bytes
    cto: int = 0


@dataclass
class Fragment:
    sequence: int                       # mfhd sequence_number（ffmpeg 侧编号）
    samples: List[Sample] = field(default_factory=list)


@dataclass
class MoovEvent:
    tracks: Dict[int, TrackInfo]
    changed: bool                       # 相对上一个 moov，编码配置是否变化


# 回调签名
MoovCallback = Callable[[MoovEvent], None]
FragmentCallback = Callable[[Fragment], None]


class FragmentedMp4Reader:
    """喂入任意大小的字节块，按事件回调吐出 moov / fragment。"""

    def __init__(self, on_moov: Optional[MoovCallback] = None,
                 on_fragment: Optional[FragmentCallback] = None):
        self._buf = bytearray()
        self._on_moov = on_moov
        self._on_fragment = on_fragment
        self.tracks: Dict[int, TrackInfo] = {}
        self._prev_stsd: Dict[int, bytes] = {}

    def feed(self, chunk: bytes) -> None:
        self._buf.extend(chunk)
        self._parse()

    def flush(self) -> None:
        """流结束时调用；正常情况下缓冲应只剩 0 字节或一个不完整 box。"""
        self._parse(final=True)

    # ------------------------------------------------------------------
    def _parse(self, final: bool = False) -> None:
        buf = self._buf
        # 顶层序列为 ftyp/moov 后接成对的 moof/mdat。moof 必须等到紧随的
        # mdat 也完整后才能配对解析，否则绝不消费，避免半截配对被丢弃。
        boxes = list(iter_boxes(buf, 0))
        consumed = 0
        i = 0
        while i < len(boxes):
            top = boxes[i]
            if top.type == b"moov":
                self._parse_moov(top)
                consumed = top.payload_end
                i += 1
            elif top.type == b"moof":
                nxt = boxes[i + 1] if i + 1 < len(boxes) else None
                if nxt is not None and nxt.type == b"mdat":
                    self._parse_moof_mdat(top, nxt)
                    consumed = nxt.payload_end
                    i += 2
                else:
                    break  # mdat 尚未收全，保留 moof 等下次 feed
            else:
                # ftyp / free / skip / 孤立 mdat：直接跳过
                consumed = top.payload_end
                i += 1
        del buf[:consumed]

    # ------------------------------------------------------------------
    def _parse_moov(self, moov: Box) -> None:
        buf = self._buf
        tracks: Dict[int, TrackInfo] = {}

        for trak in B.iter_boxes(buf, moov.payload_start, moov.payload_end):
            if trak.type != b"trak":
                continue
            tkhd = find_child(buf, trak.payload_start, trak.payload_end, b"tkhd")
            if tkhd is None:
                continue
            ver, _, p = parse_fullbox(buf, tkhd.payload_start)
            r = Reader(buf, p, tkhd.payload_end)
            if ver == 1:
                creation = r.u64(); modification = r.u64()
                track_id = r.u32(); r.u32()  # reserved
                duration = r.u64()
            else:
                creation = r.u32(); modification = r.u32()
                track_id = r.u32(); r.u32()
                duration = r.u32()
            r.read(8)                     # reserved
            r.u16()                       # layer
            r.u16()                       # alternate_group
            volume = r.u16() / 256.0
            r.u16()                       # reserved
            r.read(36)                    # matrix
            tw = r.u32() / 65536.0
            th = r.u32() / 65536.0

            mdhd = find_path(buf, trak.payload_start, trak.payload_end, b"mdia", b"mdhd")
            md_ver, _, mp = parse_fullbox(buf, mdhd.payload_start)
            mr = Reader(buf, mp, mdhd.payload_end)
            if md_ver == 1:
                mr.u64(); mr.u64()
                timescale = mr.u32()
                mr.u64()
            else:
                mr.u32(); mr.u32()
                timescale = mr.u32()
                mr.u32()
            language = mr.u16()
            mr.u16()

            hdlr = find_path(buf, trak.payload_start, trak.payload_end,
                             b"mdia", b"hdlr")
            _, _, hp = parse_fullbox(buf, hdlr.payload_start)
            hr = Reader(buf, hp, hdlr.payload_end)
            hr.read(4)                    # pre_defined
            handler = hr.fourcc()

            stsd = find_path(buf, trak.payload_start, trak.payload_end,
                             b"mdia", b"minf", b"stbl", b"stsd")
            stsd_blob = bytes(buf[stsd.payload_start:stsd.payload_end])

            info = TrackInfo(
                track_id=track_id,
                media_type=_handler_to_type(handler),
                timescale=timescale,
                stsd_box=stsd,
                stsd_blob=stsd_blob,
                handler=handler.decode("ascii", "replace"),
                creation_time=creation,
                modification_time=modification,
                duration=duration,
                language=language,
                track_width=tw,
                track_height=th,
                volume=volume,
            )
            _read_sample_entry_params(buf, info)
            tracks[track_id] = info

        # mvex/trex：fragment 缺省值
        mvex = find_child(buf, moov.payload_start, moov.payload_end, b"mvex")
        if mvex is not None:
            for trex in B.iter_boxes(buf, mvex.payload_start, mvex.payload_end):
                if trex.type != b"trex":
                    continue
                _, _, tp = parse_fullbox(buf, trex.payload_start)
                tr = Reader(buf, tp, trex.payload_end)
                tid = tr.u32()
                if tid in tracks:
                    t = tracks[tid]
                    t.sample_description_index = tr.u32()
                    t.default_sample_duration = tr.u32()
                    t.default_sample_size = tr.u32()
                    t.default_sample_flags = tr.u32()

        changed = False
        for tid, info in tracks.items():
            old = self._prev_stsd.get(tid)
            if old is not None and old != info.stsd_blob:
                changed = True
            self._prev_stsd[tid] = info.stsd_blob

        self.tracks = tracks
        if self._on_moov:
            self._on_moov(MoovEvent(tracks=tracks, changed=changed))

    # ------------------------------------------------------------------
    def _parse_moof_mdat(self, moof: Box, mdat: Box) -> None:
        buf = self._buf
        mfhd = find_child(buf, moof.payload_start, moof.payload_end, b"mfhd")
        sequence = Reader(buf, mfhd.payload_start + 4, mfhd.payload_end).u32()
        frag = Fragment(sequence=sequence)

        mdat_payload = buf[mdat.payload_start:mdat.payload_end]
        data_cursor = 0

        for traf in B.iter_boxes(buf, moof.payload_start, moof.payload_end):
            if traf.type != b"traf":
                continue
            tfhd = find_child(buf, traf.payload_start, traf.payload_end, b"tfhd")
            _, flags, p = parse_fullbox(buf, tfhd.payload_start)
            r = Reader(buf, p, tfhd.payload_end)
            track_id = r.u32()
            info = self.tracks.get(track_id)

            base_data_offset = moof.start if (flags & B.TFHD_DEFAULT_BASE_IS_MOOF) else None
            if flags & B.TFHD_BASE_DATA_OFFSET_PRESENT:
                base_data_offset = r.u64()
            desc_idx = (r.u32() if flags & B.TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT
                        else (info.sample_description_index if info else 1))
            def_dur = (r.u32() if flags & B.TFHD_DEFAULT_SAMPLE_DURATION_PRESENT
                       else (info.default_sample_duration if info else 0))
            def_size = (r.u32() if flags & B.TFHD_DEFAULT_SAMPLE_SIZE_PRESENT
                        else (info.default_sample_size if info else 0))
            def_flags = (r.u32() if flags & B.TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT
                         else (info.default_sample_flags if info else 0))

            tfdt = find_child(buf, traf.payload_start, traf.payload_end, b"tfdt")
            base_dts = 0
            if tfdt is not None:
                tv, _, tpp = parse_fullbox(buf, tfdt.payload_start)
                tr = Reader(buf, tpp, tfdt.payload_end)
                base_dts = tr.u64() if tv == 1 else tr.u32()

            trun = find_child(buf, traf.payload_start, traf.payload_end, b"trun")
            sample_count = 0
            data_offset = 0
            first_sample_flags = def_flags
            per_sample: List[dict] = []
            if trun is not None:
                tv, tflags, tpp = parse_fullbox(buf, trun.payload_start)
                rr = Reader(buf, tpp, trun.payload_end)
                sample_count = rr.u32()
                if tflags & B.TRUN_DATA_OFFSET_PRESENT:
                    data_offset = rr.i32()
                if tflags & B.TRUN_FIRST_SAMPLE_FLAGS_PRESENT:
                    first_sample_flags = rr.u32()
                for i in range(sample_count):
                    s = {"duration": def_dur, "size": def_size,
                         "flags": def_flags, "cto": 0}
                    if tflags & B.TRUN_SAMPLE_DURATION_PRESENT:
                        s["duration"] = rr.u32()
                    if tflags & B.TRUN_SAMPLE_SIZE_PRESENT:
                        s["size"] = rr.u32()
                    if tflags & B.TRUN_SAMPLE_FLAGS_PRESENT:
                        s["flags"] = rr.u32()
                    if tflags & B.TRUN_SAMPLE_CTO_PRESENT:
                        s["cto"] = rr.i32() if tv == 1 else rr.u32()
                    per_sample.append(s)

            # ffmpeg 的 default-base-is-moof：mdat 起点 = moof.start + data_offset
            if base_data_offset is not None:
                expected = base_data_offset + data_offset - mdat.payload_start
                if expected != data_cursor:
                    # 多个 traf 时各自带 data_offset，以 trun 声明为准
                    data_cursor = expected

            dts = base_dts
            for i, s in enumerate(per_sample):
                size = s["size"]
                raw = bytes(mdat_payload[data_cursor:data_cursor + size])
                data_cursor += size
                sample_flags = first_sample_flags if i == 0 else s["flags"]
                sync = bool(sample_flags & 0x02000000)
                cto = s["cto"]
                frag.samples.append(Sample(
                    track_id=track_id,
                    dts=dts,
                    pts=dts + cto,
                    duration=s["duration"],
                    sync=sync,
                    data=raw,
                    cto=cto,
                ))
                dts += s["duration"]

        if self._on_fragment:
            self._on_fragment(frag)


def _handler_to_type(handler: bytes) -> str:
    if handler == b"vide":
        return "video"
    if handler == b"soun":
        return "audio"
    return "other"


def _read_sample_entry_params(buf: bytes, info: TrackInfo) -> None:
    """从 stsd 第一个 sample entry 里取宽高 / 采样率 / 声道。"""
    stsd = info.stsd_box
    entries = list(iter_boxes(buf, stsd.payload_start + 4, stsd.payload_end))
    if not entries:
        return
    entry = entries[0]
    r = Reader(buf, entry.payload_start, entry.payload_end)
    if info.media_type == "video":
        r.read(6)                 # reserved + data_reference_index
        r.u16(); r.u16()          # pre_defined, reserved
        r.read(12)                # pre_defined[3]
        info.width = r.u16()
        info.height = r.u16()
    elif info.media_type == "audio":
        r.read(6)                 # reserved + data_reference_index
        r.u32(); r.u32()          # reserved
        info.channels = r.u16()
        r.u16()                   # sampleSize
        r.u16(); r.u16()          # pre_defined / reserved
        info.sample_rate = r.u32() >> 16
