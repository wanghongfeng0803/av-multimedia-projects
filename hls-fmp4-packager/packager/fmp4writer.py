# -*- coding: utf-8 -*-
"""
fMP4 写入器：自己构建 init 段（ftyp+moov+mvex）与媒体段（styp+sidx+moof+mdat）。

关键点：
- init 段不沿用 ffmpeg 输出的 moov，而是根据解析出的 TrackInfo 自构，
  保证 trex/stsd/mvex 完全由本打包器控制；
- 每个媒体段的 moof/mdat 由本模块重新拼装，trun 的 data_offset、
  首样本 sync 标志、tfdt 基时间都在这里精确写入；
- 视频/音频各自一个 traf，共享同一个 mdat，切点由 pipeline 决定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import boxes as B
from .boxes import (
    Reader, box, fullbox, u16, u32, u64, i32, matrix_bytes, children,
)
from .mp4reader import Sample, TrackInfo

FTYP_PAYLOAD = (
    b"iso5" + u32(1) + b"iso5" + b"iso6" + b"mp41" + b"dash"
)
STYP_PAYLOAD = (
    b"msdh" + u32(0) + b"msdh" + b"msix"
)


@dataclass
class TrackSamples:
    info: TrackInfo
    samples: List[Sample]


class Fmp4Writer:
    def __init__(self, tracks: Sequence[TrackInfo]):
        self.tracks: Dict[int, TrackInfo] = {t.track_id: t for t in tracks}
        self.video_id = next(
            (t.track_id for t in tracks if t.media_type == "video"), None)
        self.audio_id = next(
            (t.track_id for t in tracks if t.media_type == "audio"), None)

    # ------------------------------------------------------------------
    # init 段
    # ------------------------------------------------------------------
    def build_init(self) -> bytes:
        infos = list(self.tracks.values())
        traks = [self._build_trak(info) for info in infos]
        mvex = self._build_mvex(infos)
        next_track_id = max(self.tracks) + 1
        moov = box(b"moov", children(
            self._build_mvhd(next_track_id), *traks, mvex))
        return box(b"ftyp", FTYP_PAYLOAD) + moov

    def _build_mvhd(self, next_track_id: int) -> bytes:
        payload = (
            u64(0) + u64(0)          # creation/modification
            + u32(1000)              # timescale
            + u64(0)                 # duration（fMP4 init 里固定 0）
            + u32(0x00010000)        # rate 1.0
            + u16(0x0100)            # volume 1.0
            + u16(0) + u32(0) + u32(0)   # reserved
            + matrix_bytes()
            + b"\x00" * 24           # pre_defined
            + u32(next_track_id)
        )
        return fullbox(b"mvhd", 1, 0, payload)

    def _build_trak(self, info: TrackInfo) -> bytes:
        tkhd = self._build_tkhd(info)
        mdia = box(b"mdia", children(
            self._build_mdhd(info),
            self._build_hdlr(info),
            self._build_minf(info),
        ))
        return box(b"trak", children(tkhd, mdia))

    def _build_tkhd(self, info: TrackInfo) -> bytes:
        payload = (
            u64(info.creation_time) + u64(info.modification_time)
            + u32(info.track_id) + u32(0)
            + u64(info.duration)
            + u32(0) + u32(0)             # reserved
            + u16(0) + u16(0)             # layer / alternate_group
            + u16(int(info.volume * 256)) + u16(0)
            + matrix_bytes()
            + u32(int(info.track_width * 65536))
            + u32(int(info.track_height * 65536))
        )
        flags = B.TKHD_TRACK_ENABLED | B.TKHD_TRACK_IN_MOVIE
        return fullbox(b"tkhd", 1, flags, payload)

    def _build_mdhd(self, info: TrackInfo) -> bytes:
        payload = (
            u64(info.creation_time) + u64(info.modification_time)
            + u32(info.timescale) + u64(info.duration)
            + u16(info.language) + u16(0)
        )
        return fullbox(b"mdhd", 1, 0, payload)

    def _build_hdlr(self, info: TrackInfo) -> bytes:
        handler = info.handler.encode("ascii") if isinstance(
            info.handler, str) else info.handler
        name = {"video": b"VideoHandler\x00",
                "audio": b"SoundHandler\x00"}.get(info.media_type, b"\x00")
        payload = u32(0) + handler + u32(0) + u32(0) + name
        return fullbox(b"hdlr", 0, 0, payload)

    def _build_minf(self, info: TrackInfo) -> bytes:
        if info.media_type == "video":
            media_header = fullbox(b"vmhd", 0, 1, u16(0) + b"\x00" * 6)
        elif info.media_type == "audio":
            media_header = fullbox(b"smhd", 0, 0, u16(0) + u16(0))
        else:
            media_header = fullbox(b"nmhd", 0, 0)
        dinf = box(b"dinf", fullbox(
            b"dref", 0, 0,
            u32(1) + box(b"url ", fullbox(b"url ", 0, 1))))
        stbl = box(b"stbl", children(
            # stsd 原样保留 ffmpeg 写出的编码配置（avcC/hevcC/esds...）
            box(b"stsd", info.stsd_blob),
            fullbox(b"stts", 0, 0, u32(0)),
            fullbox(b"stsc", 0, 0, u32(0)),
            fullbox(b"stsz", 0, 0, u32(0) + u32(0)),
            fullbox(b"stco", 0, 0, u32(0)),
        ))
        return box(b"minf", children(media_header, dinf, stbl))

    def _build_mvex(self, infos: List[TrackInfo]) -> bytes:
        trexs = []
        for info in infos:
            # 视频默认非同步；同步帧在每段第一个 trun 样本处单独打标志
            default_flags = (B.SAMPLE_FLAGS_NON_SYNC
                             if info.media_type == "video" else 0)
            payload = (
                u32(info.track_id) + u32(1)
                + u32(0) + u32(0) + u32(default_flags)
            )
            trexs.append(fullbox(b"trex", 0, 0, payload))
        return box(b"mvex", children(*trexs))

    # ------------------------------------------------------------------
    # 媒体段
    # ------------------------------------------------------------------
    def build_segment(self, sequence_number: int,
                      groups: List[TrackSamples]) -> bytes:
        """
        sequence_number: HLS 分片序号（同时写入 mfhd，从 0 开始单调递增）
        groups: 本段包含的各 track 样本（视频在前、音频在后）
        """
        styp = box(b"styp", STYP_PAYLOAD)
        mfhd = fullbox(b"mfhd", 0, 0, u32(sequence_number))  # 固定 16 字节

        # mdat 内按 groups 顺序拼接各 track 样本（视频在前、音频在后）。
        # 每个 traf 的 data_offset 指向「自己 track 数据」在段内的起点，
        # 因此要按前面各 track 的样本总字节数做前缀累加。
        prefix_bytes = {}
        running = 0
        for g in groups:
            prefix_bytes[id(g)] = running
            running += sum(len(s.data) for s in g.samples)

        # 第一遍：不带 data_offset 构建 traf。最终每个 trun 会多出 4 字节
        # 的 data_offset 字段，由此精确算出 moof 与 mdat payload 的相对位置。
        trafs_noff = [self._build_traf(g, data_offset=None) for g in groups]
        moof_header = 8                       # moof box header
        moof_size = (moof_header + len(mfhd)
                     + sum(len(t) + 4 for t in trafs_noff))

        # 第二遍：data_offset 相对 moof 起点（default-base-is-moof）
        trafs = []
        for g, t_noff in zip(groups, trafs_noff):
            offset = moof_size + 8 + prefix_bytes[id(g)]  # 8 = mdat header
            trafs.append(self._build_traf(g, data_offset=offset))
        moof = box(b"moof", children(mfhd, *trafs))
        assert len(moof) == moof_size, "moof 尺寸与 data_offset 计算不一致"

        mdat = box(b"mdat", b"".join(
            s.data for g in groups for s in g.samples))

        # sidx 引用 moof+mdat 的总字节数，必须在两者尺寸确定后构建
        sidx = self._build_sidx(groups, len(moof) + len(mdat))
        return styp + sidx + moof + mdat

    def _build_traf(self, group: TrackSamples,
                    data_offset: Optional[int]) -> bytes:
        info = group.info
        samples = group.samples

        tfhd_flags = B.TFHD_DEFAULT_BASE_IS_MOOF
        tfhd = fullbox(b"tfhd", 0, tfhd_flags, u32(info.track_id))

        first_dts = samples[0].dts if samples else 0
        tfdt = fullbox(b"tfdt", 1, 0, u64(first_dts))

        any_cto = any(s.cto != 0 for s in samples)
        trun_flags = (
            B.TRUN_DATA_OFFSET_PRESENT
            | B.TRUN_SAMPLE_DURATION_PRESENT
            | B.TRUN_SAMPLE_FLAGS_PRESENT
            | B.TRUN_SAMPLE_SIZE_PRESENT
        )
        if any_cto:
            trun_flags |= B.TRUN_SAMPLE_CTO_PRESENT

        payload = u32(len(samples))
        if data_offset is not None:
            payload += i32(data_offset)

        body = bytearray()
        for s in samples:
            flags = B.SAMPLE_FLAGS_SYNC if s.sync else B.SAMPLE_FLAGS_NON_SYNC
            body += u32(s.duration) + u32(len(s.data)) + u32(flags)
            if any_cto:
                body += i32(s.cto)
        trun = fullbox(b"trun", 1, trun_flags, payload + bytes(body))
        return box(b"traf", children(tfhd, tfdt, trun))

    def _build_sidx(self, groups: List[TrackSamples],
                    referenced_size: int) -> bytes:
        # 选视频 track 做 sidx 时间线；无视频时退化到音频
        ref_group = next(
            (g for g in groups if g.info.track_id == self.video_id),
            groups[0])
        info = ref_group.info
        samples = ref_group.samples
        earliest_pts = samples[0].pts if samples else 0
        duration = sum(s.duration for s in samples)

        payload = bytearray()
        payload += u32(info.track_id)
        payload += u32(info.timescale)
        payload += u64(earliest_pts)
        payload += u64(0)                 # first_offset = 0（sidx 之后即 moof）
        payload += u16(0)                 # reserved
        payload += u16(1)                 # reference_count
        # bit31 reference_type=0（media reference），低 31 位为引用字节数
        payload += u32(referenced_size & 0x7FFFFFFF)
        payload += u32(duration)
        # SAP: starts_with_SAP=1，SAP_type=1（IDR），SAP_delta_time=0
        payload += u32(0x90000000)
        return fullbox(b"sidx", 1, 0, bytes(payload))
