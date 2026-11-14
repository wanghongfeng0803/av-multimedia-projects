"""ISO-BMFF / MP4 容器解析器（只读、零第三方依赖）。

解析目标是 moov 盒中的样本表（sample table），把每个 track 的媒体样本
在文件中的字节偏移、长度、解码时间戳(DTS)、显示时间戳(PTS)以及关键帧
标记全部还原出来，并提取编码私有数据（avcC / hvcC / esds）。

支持常见盒类型与容错点：
- 32 位 largesize（64 位 size）盒；
- version 0/1 的 mdhd、version 0/1 的 ctts（有符号合成偏移）；
- stco（32 位块偏移）与 co64（64 位块偏移）；
- 标准 28 字节音频、78 字节视频 sample entry，并对子盒做扫描式容错；
- 单段 edit list（edts/elst）的时间轴偏移；
- 明确拒绝碎片化 MP4（moof）、加密（sinf/encv/enca）等不支持的结构。

覆盖的编码：
- 视频：H.264/AVC（avc1/avc3）、HEVC（hvc1/hev1）；
- 音频：AAC（mp4a + esds/ASC）、MP3（mp4a 无 ASC 或 objectType=0x6B）、
  AC-3（ac-3）、E-AC-3（ec-3）；
- 字幕：TTML（stpp，样本即完整 TTML 文档）、WebVTT（wvtt，vttc 盒）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from struct import unpack
from typing import List, Optional, Tuple

from .errors import RemuxError

# 支持的 sample entry 四字符码 → 编码族
_VIDEO_CODECS = {
    "avc1": "avc",
    "avc3": "avc",
    "hvc1": "hevc",
    "hev1": "hevc",
}
_AUDIO_CODECS = {
    "mp4a": "aac",   # 实际编码需结合 esds 判定：有 esds 为 AAC，否则为 MP3
    "ac-3": "ac3",
    "ec-3": "eac3",
}
_SUBTITLE_CODECS = {
    "stpp": "ttml",    # 样本即完整 TTML 文档，可直拷进 MKV
    "wvtt": "webvtt",  # 样本为 vttc/vtte 盒，写出时转换为 Matroska WebVTT
}


@dataclass
class Sample:
    """单个媒体样本（已编码的一帧 / 一个音频包）。"""

    offset: int          # 在源文件中的绝对字节偏移
    size: int            # 字节长度
    dts: int             # 解码时间戳（track timescale 单位）
    pts: int             # 显示时间戳 = dts + composition_offset
    is_keyframe: bool    # 是否关键帧（音频恒为 True）
    duration: int = 0    # 样本时长（来自 stts delta，timescale 单位）


@dataclass
class Track:
    """一条 track 的元数据与全部样本。"""

    track_id: int
    handler: str         # "vide" / "soun"
    codec_family: str    # "avc" / "hevc" / "aac"
    codec_tag: str       # 原始 sample entry 四字符码，如 avc1 / hvc1 / mp4a
    timescale: int
    duration: int        # mdhd duration（timescale 单位）
    # edit list（edts/elst）信息：单段全轨 edit 的 media_time
    # （track timescale 单位）。时间轴时间戳 = 媒体时间戳 - media_time。
    edit_media_time: int = 0
    has_edit: bool = False
    width: int = 0
    height: int = 0
    channels: int = 0
    sample_rate: int = 0
    language: str = "und"        # ISO 639-2/T 语言码（mdhd），未指定为 und
    codec_private: bytes = b""   # avcC / hvcC / AudioSpecificConfig / vttC 原始字节
    samples: List[Sample] = field(default_factory=list)

    @property
    def is_video(self) -> bool:
        return self.handler == "vide"

    @property
    def is_audio(self) -> bool:
        return self.handler == "soun"

    @property
    def is_subtitle(self) -> bool:
        return self.handler == "subt"


# ---------------------------------------------------------------------------
# 底层盒（box）读取工具
# ---------------------------------------------------------------------------

@dataclass
class Box:
    kind: str          # 四字符码
    start: int         # 盒头起始绝对偏移
    header_size: int   # 盒头长度（普通 8，largesize 16）
    data: bytes        # 盒负载（header 之后的全部字节，已整体读入内存）

    @property
    def data_start(self) -> int:
        """负载第一个字节在源文件中的绝对偏移。"""
        return self.start + self.header_size

    def children(self) -> List["Box"]:
        """把当前盒负载解析为一串子盒（要求当前盒是容器型盒）。"""
        return parse_boxes(self.data, base_offset=self.data_start)


def _fourcc(raw: bytes) -> str:
    return raw.decode("latin-1")


def parse_boxes(buf: bytes, base_offset: int = 0) -> List[Box]:
    """解析一段缓冲区中的同级盒序列。

    :param base_offset: buf[0] 对应在源文件中的绝对偏移，用于把盒内
        相对位置换算成文件绝对偏移。
    """
    boxes: List[Box] = []
    pos = 0
    n = len(buf)
    while pos < n:
        if n - pos < 8:
            raise RemuxError("盒结构损坏：剩余字节不足 8 字节的盒头")
        size32 = unpack(">I", buf[pos:pos + 4])[0]
        kind = _fourcc(buf[pos + 4:pos + 8])
        header_size = 8
        if size32 == 1:  # largesize：实际 64 位长度紧跟其后
            if n - pos < 16:
                raise RemuxError(f"[{kind}] 盒声明 64 位长度但字节不足")
            size = unpack(">Q", buf[pos + 8:pos + 16])[0]
            header_size = 16
        elif size32 == 0:  # 盒延伸到缓冲区末尾
            size = n - pos
        else:
            size = size32
        if size < header_size:
            raise RemuxError(f"[{kind}] 盒长度 {size} 小于盒头，文件损坏")
        end = pos + size
        if end > n:
            raise RemuxError(f"[{kind}] 盒长度超出所在容器边界，文件损坏")
        data = buf[pos + header_size:end]
        boxes.append(Box(kind, base_offset + pos, header_size, data))
        pos = end
    return boxes


def _find_child(box: Box, kind: str) -> Optional[Box]:
    for child in box.children():
        if child.kind == kind:
            return child
    return None


def _require_child(box: Box, kind: str) -> Box:
    child = _find_child(box, kind)
    if child is None:
        raise RemuxError(f"[{box.kind}] 中缺少必需的 [{kind}] 子盒")
    return child


# ---------------------------------------------------------------------------
# 各盒解析
# ---------------------------------------------------------------------------

def _parse_full_box_header(data: bytes) -> Tuple[int, int]:
    """返回 (version, flags)。"""
    if len(data) < 4:
        raise RemuxError("FullBox 负载不足 4 字节")
    return data[0], int.from_bytes(data[1:4], "big")


def _parse_mdhd(mdhd: Box) -> Tuple[int, int]:
    """返回 (timescale, duration)，自动适配 version 0/1。"""
    version, _ = _parse_full_box_header(mdhd.data)
    body = mdhd.data[4:]
    if version == 1:
        if len(body) < 28:
            raise RemuxError("[mdhd] version 1 长度不足")
        # creation(8) modification(8) timescale(4) duration(8)
        timescale = unpack(">I", body[16:20])[0]
        duration = unpack(">Q", body[20:28])[0]
    else:
        if len(body) < 16:
            raise RemuxError("[mdhd] version 0 长度不足")
        # creation(4) modification(4) timescale(4) duration(4)
        timescale = unpack(">I", body[8:12])[0]
        duration = unpack(">I", body[12:16])[0]
    if timescale <= 0:
        raise RemuxError("[mdhd] timescale 非法（必须为正整数）")
    return timescale, duration


def _parse_mdhd_language(mdhd: Box) -> str:
    """读取 mdhd 的 ISO 639-2/T 语言码；未指定（und/0x55C4）或非法时返回 'und'。

    语言字段为 3 个 5 位字符，各加上 0x60 得到小写 ASCII（如 0x55C4 → 'und'）。
    注意 mdhd.data 仍包含 4 字节 fullbox 头：
    - v0：creation(4)+modification(4)+timescale(4)+duration(4)=16，语言在 data[20]；
    - v1：creation(8)+modification(8)+timescale(4)+duration(8)=28，语言在 data[32]。
    """
    version, _ = _parse_full_box_header(mdhd.data)
    offset = 32 if version == 1 else 20
    if len(mdhd.data) < offset + 2:
        return "und"
    packed = unpack(">H", mdhd.data[offset:offset + 2])[0]
    chars = (
        0x60 + ((packed >> 10) & 0x1F),
        0x60 + ((packed >> 5) & 0x1F),
        0x60 + (packed & 0x1F),
    )
    if any(not (0x61 <= c <= 0x7A) for c in chars):
        return "und"
    code = bytes(chars).decode("ascii")
    return "und" if code == "und" else code


def _parse_hdlr(hdlr: Box) -> str:
    if len(hdlr.data) < 12:
        raise RemuxError("[hdlr] 长度不足")
    handler = _fourcc(hdlr.data[8:12])
    if handler not in ("vide", "soun", "hint", "meta", "text"):
        # 不常见的 handler 不当作致命错误，交由上层按 codec 判定
        return handler
    return handler


def _scan_subbox(entry: Box, wanted: Tuple[str, ...]) -> Optional[Box]:
    """在 sample entry 内查找目标子盒。

    entry.data 不含 sample entry 自身的 8 字节盒头。标准结构中，
    视频 entry 前 78 字节、音频 entry(version 0) 前 28 字节为固定头，
    子盒从负载偏移 78 / 28 开始。先按标准布局解析，失败再退化为对整个
    负载的四字符码特征扫描（要求盒长度自洽才采信），以容忍非标准封装。
    """
    # 视觉条目（avcC/hvcC）与字幕条目（vttC）同为 78 字节宽固定头；
    # 音频条目（esds/dac3/dec3）为 28 字节固定头
    wide_header = wanted[0] in ("avcC", "hvcC", "vttC")
    standard_offset = 78 if wide_header else 28
    for offset in (standard_offset, 0):
        if offset >= len(entry.data):
            continue
        try:
            for sub in parse_boxes(entry.data[offset:], base_offset=offset):
                if sub.kind in wanted:
                    return sub
        except RemuxError:
            # 该偏移不是合法盒序列（如把字幕条目的视觉式固定头当盒解析），
            # 继续尝试下一偏移，最终退化为四字符码特征扫描
            continue
    # 兜底：在整个负载中搜索四字符码，并要求其前 4 字节声明的长度自洽
    marker = wanted[0].encode("latin-1")
    idx = entry.data.find(marker)
    while idx != -1 and idx >= 4:
        size = unpack(">I", entry.data[idx - 4:idx])[0]
        payload_end = idx + 4 + (size - 8)
        if 8 <= size and payload_end <= len(entry.data):
            return Box(
                wanted[0],
                entry.data_start + idx - 4,
                8,
                entry.data[idx + 4:payload_end],
            )
        idx = entry.data.find(marker, idx + 1)
    return None


def _parse_visual_dimensions(entry: Box) -> Tuple[int, int]:
    # 视觉 sample entry 负载布局（ISO 14496-12）：偏移 24/26 为 width/height
    if len(entry.data) < 28:
        raise RemuxError(f"[{entry.kind}] 视频 sample entry 长度不足")
    width = unpack(">H", entry.data[24:26])[0]
    height = unpack(">H", entry.data[26:28])[0]
    return width, height


def _parse_audio_fixed_fields(entry: Box) -> Tuple[int, int]:
    """从 mp4a 固定头读取 (channels, sample_rate)。

    负载内偏移 16 为 channelcount(2)，偏移 24 为 samplerate(16.16 定点)。
    """
    if len(entry.data) < 28:
        raise RemuxError(f"[{entry.kind}] 音频 sample entry 长度不足")
    channels = unpack(">H", entry.data[16:18])[0]
    sample_rate_fixed = unpack(">I", entry.data[24:28])[0]
    return channels, sample_rate_fixed >> 16


# ---- ESDS（MPEG-4 描述符）解析，提取 AudioSpecificConfig -------------------

def _read_descriptor(data: bytes, pos: int) -> Tuple[int, bytes, int]:
    """读取一个 ES 描述符，返回 (tag, payload, next_pos)。

    描述符头为 1 字节 tag + 变长 size（每个 size 字节仅低 7 位有效）。
    某些早期文件在 tag 前会多写 1~4 字节（如 0x80/0xFE 等），做容错跳过。
    """
    if pos >= len(data):
        raise RemuxError("[esds] 描述符被截断")
    # 容错：跳过 MP4 特定的扩展前缀字节（>=0x80 的非 tag 字节，最多 4 个）
    guard = 0
    while data[pos] in (0x80, 0xFE, 0xFD) and guard < 4:
        pos += 1
        guard += 1
        if pos >= len(data):
            raise RemuxError("[esds] 描述符前缀后被截断")
    tag = data[pos]
    pos += 1
    size = 0
    for _ in range(4):
        if pos >= len(data):
            raise RemuxError("[esds] 描述符 size 被截断")
        byte = data[pos]
        pos += 1
        size = (size << 7) | (byte & 0x7F)
        if not byte & 0x80:
            break
    if pos + size > len(data):
        raise RemuxError("[esds] 描述符长度超出边界，文件损坏")
    return tag, data[pos:pos + size], pos + size


def _parse_esds(esds: Box) -> Tuple[int, bytes]:
    """解析 esds 盒，返回 (objectTypeIndication, DecoderSpecificInfo 字节)。

    objectTypeIndication 用于区分同为 mp4a 封装的不同编码：
    0x40 为 AAC（ISO/IEC 14496-3），0x6B 为 MPEG-1 Layer III（MP3）。
    MP3 通常没有 DecoderSpecificInfo，此时第二项返回空字节串。
    """
    _parse_full_box_header(esds.data)
    # ES_Descr tag=0x03
    tag, es_payload, _ = _read_descriptor(esds.data, 4)
    if tag != 0x03:
        raise RemuxError(f"[esds] 顶层描述符 tag 异常: 0x{tag:02X}")
    # ES_Descr 内先跳过 ES_ID(2)+flags(1)，再找 DecoderConfigDescr(0x04)
    if len(es_payload) < 3:
        raise RemuxError("[esds] ES 描述符内容过短")
    tag, dec_payload, _ = _read_descriptor(es_payload, 3)
    if tag != 0x04:
        raise RemuxError(f"[esds] 缺少 DecoderConfigDescriptor，实际 tag=0x{tag:02X}")
    if len(dec_payload) < 13:
        raise RemuxError("[esds] DecoderConfigDescriptor 内容过短")
    object_type = dec_payload[0]
    # 固定 13 字节头部之后是可选的 DecoderSpecificInfo(0x05)
    if len(dec_payload) == 13:
        return object_type, b""
    tag, asc, _ = _read_descriptor(dec_payload, 13)
    if tag != 0x05:
        raise RemuxError(f"[esds] 缺少 DecoderSpecificInfo，实际 tag=0x{tag:02X}")
    if not asc:
        # MP3 允许无 ASC；AAC 必须有至少 2 字节的配置
        return object_type, b""
    if len(asc) < 2:
        raise RemuxError("[esds] AudioSpecificConfig 至少需要 2 字节")
    return object_type, bytes(asc)


def _parse_audio_specific_config(asc: bytes) -> Tuple[int, int]:
    """解析 ASC 前两字节，返回 (sample_rate, channels)。

    无法完全确定（如显式 24 位采样率）时对应字段返回 0，交给固定头回退。
    """
    if len(asc) < 2:
        return 0, 0
    bits = (asc[0] << 8) | asc[1]
    freq_index = (bits >> 7) & 0x0F
    if freq_index == 0x0F:  # 显式 24 位采样率，声道位置在更后面
        if len(asc) < 5:
            return 0, 0
        sample_rate = (
            ((asc[1] & 0x7F) << 17) | (asc[2] << 9) | (asc[3] << 1) | (asc[4] >> 7)
        )
        return sample_rate, 0
    rates = (
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
        16000, 12000, 11025, 8000, 7350,
    )
    sample_rate = rates[freq_index] if freq_index < len(rates) else 0
    channels = (bits >> 3) & 0x0F
    return sample_rate, channels


# ---------------------------------------------------------------------------
# 样本表（stbl）展开
# ---------------------------------------------------------------------------

def _parse_stts(stts: Box) -> List[int]:
    """解码时间 → 样本 delta 数组（每样本一个 delta）。"""
    _parse_full_box_header(stts.data)
    entry_count = unpack(">I", stts.data[4:8])[0]
    pos = 8
    deltas: List[int] = []
    for _ in range(entry_count):
        if pos + 8 > len(stts.data):
            raise RemuxError("[stts] 表项被截断")
        count, delta = unpack(">II", stts.data[pos:pos + 8])
        pos += 8
        if count == 0:
            raise RemuxError("[stts] 存在 sample_count=0 的非法表项")
        deltas.extend([delta] * count)
    return deltas


def _parse_ctts(ctts: Box) -> List[int]:
    """合成偏移 → 每样本一个有符号偏移（version 1 为有符号）。"""
    version, _ = _parse_full_box_header(ctts.data)
    entry_count = unpack(">I", ctts.data[4:8])[0]
    pos = 8
    offsets: List[int] = []
    for _ in range(entry_count):
        if pos + 8 > len(ctts.data):
            raise RemuxError("[ctts] 表项被截断")
        count, raw = unpack(">II", ctts.data[pos:pos + 8])
        pos += 8
        # version 0：无符号；version 1：32 位有符号整数
        if version == 1 and raw >= 0x80000000:
            offset = raw - (1 << 32)
        else:
            offset = raw
        offsets.extend([offset] * count)
    return offsets


def _parse_stsc(stsc: Box) -> List[Tuple[int, int, int]]:
    """返回 (first_chunk, samples_per_chunk, sample_description_index) 表。"""
    _parse_full_box_header(stsc.data)
    entry_count = unpack(">I", stsc.data[4:8])[0]
    pos = 8
    entries = []
    for _ in range(entry_count):
        if pos + 12 > len(stsc.data):
            raise RemuxError("[stsc] 表项被截断")
        entries.append(unpack(">III", stsc.data[pos:pos + 12]))
        pos += 12
    return entries


def _parse_stsz(stsz: Box) -> List[int]:
    """每样本大小数组；sample_size 非 0 时所有样本等长。"""
    _parse_full_box_header(stsz.data)
    if len(stsz.data) < 12:
        raise RemuxError("[stsz] 长度不足")
    uniform_size, sample_count = unpack(">II", stsz.data[4:12])
    if uniform_size:
        return [uniform_size] * sample_count
    if len(stsz.data) < 12 + 4 * sample_count:
        raise RemuxError("[stsz] 样本大小表被截断")
    return [
        unpack(">I", stsz.data[12 + 4 * i:16 + 4 * i])[0]
        for i in range(sample_count)
    ]


def _parse_chunk_offsets(stbl: Box) -> List[int]:
    """优先 co64，其次 stco，返回每个 chunk 的绝对文件偏移。"""
    co64 = _find_child(stbl, "co64")
    if co64 is not None:
        _parse_full_box_header(co64.data)
        count = unpack(">I", co64.data[4:8])[0]
        if len(co64.data) < 8 + 8 * count:
            raise RemuxError("[co64] 偏移表被截断")
        return [unpack(">Q", co64.data[8 + 8 * i:16 + 8 * i])[0] for i in range(count)]
    stco = _find_child(stbl, "stco")
    if stco is None:
        raise RemuxError("[stbl] 中既无 stco 也无 co64，无法定位样本数据")
    _parse_full_box_header(stco.data)
    count = unpack(">I", stco.data[4:8])[0]
    if len(stco.data) < 8 + 4 * count:
        raise RemuxError("[stco] 偏移表被截断")
    return [unpack(">I", stco.data[8 + 4 * i:12 + 4 * i])[0] for i in range(count)]


def _parse_stss(stss: Box) -> set:
    """关键帧样本序号集合（1 起始）。"""
    _parse_full_box_header(stss.data)
    count = unpack(">I", stss.data[4:8])[0]
    if len(stss.data) < 8 + 4 * count:
        raise RemuxError("[stss] 表被截断")
    return {
        unpack(">I", stss.data[8 + 4 * i:12 + 4 * i])[0]
        for i in range(count)
    }


def _expand_sample_offsets(
    chunk_offsets: List[int],
    stsc_entries: List[Tuple[int, int, int]],
    sizes: List[int],
) -> List[int]:
    """依据 stsc 把 chunk + 槽位还原成每个样本的绝对字节偏移。"""
    chunk_count = len(chunk_offsets)
    # 对每个 chunk 求其 samples_per_chunk：stsc 表项对 first_chunk..下一表项前生效
    per_chunk: List[int] = []
    entry_idx = 0
    for chunk in range(1, chunk_count + 1):
        while (
            entry_idx + 1 < len(stsc_entries)
            and stsc_entries[entry_idx + 1][0] <= chunk
        ):
            entry_idx += 1
        if not stsc_entries or stsc_entries[entry_idx][0] > chunk:
            raise RemuxError("[stsc] 表项无法覆盖全部 chunk")
        per_chunk.append(stsc_entries[entry_idx][1])

    total = sum(per_chunk)
    if total != len(sizes):
        raise RemuxError(
            f"样本表计数不一致：stsc/stco 推导出 {total} 个样本，stsz 为 {len(sizes)}"
        )

    offsets: List[int] = []
    sample_idx = 0
    for chunk_idx, chunk_offset in enumerate(chunk_offsets):
        cursor = chunk_offset
        for _ in range(per_chunk[chunk_idx]):
            offsets.append(cursor)
            cursor += sizes[sample_idx]
            sample_idx += 1
    return offsets


def _build_samples(
    sizes: List[int],
    offsets: List[int],
    dts_deltas: List[int],
    ctts_offsets: List[int],
    keyframes: Optional[set],
    is_video: bool,
) -> List[Sample]:
    count = len(sizes)
    if not (len(offsets) == count == len(dts_deltas)):
        raise RemuxError(
            f"样本表计数不一致：size={count} offset={len(offsets)} stts={len(dts_deltas)}"
        )
    if ctts_offsets and len(ctts_offsets) != count:
        raise RemuxError(f"[ctts] 条目数 {len(ctts_offsets)} 与样本数 {count} 不一致")
    if keyframes is not None and not keyframes:
        raise RemuxError("[stss] 关键帧表为空，视频流无法定位随机访问点")

    samples: List[Sample] = []
    dts = 0
    for i in range(count):
        size = sizes[i]
        if size <= 0:
            raise RemuxError(f"第 {i + 1} 个样本大小为 0，文件损坏")
        composition = ctts_offsets[i] if ctts_offsets else 0
        pts = dts + composition
        if pts < 0:
            raise RemuxError(
                f"第 {i + 1} 个样本计算出负的显示时间戳(PTS={pts})，暂不支持"
            )
        if is_video:
            # 没有 stss 时按规范所有样本都是关键帧
            is_key = True if keyframes is None else (i + 1) in keyframes
        else:
            is_key = True
        samples.append(
            Sample(
                offsets[i],
                size,
                dts,
                pts,
                is_key,
                duration=dts_deltas[i],
            )
        )
        dts += dts_deltas[i]
    return samples


# ---------------------------------------------------------------------------
# trak / 顶层解析
# ---------------------------------------------------------------------------

def _parse_stsd(stsd: Box) -> Box:
    """返回唯一的 sample entry（基础版本每条 track 仅支持一个编码描述）。"""
    _parse_full_box_header(stsd.data)
    if len(stsd.data) < 8:
        raise RemuxError("[stsd] 长度不足")
    entry_count = unpack(">I", stsd.data[4:8])[0]
    if entry_count == 0:
        raise RemuxError("[stsd] 不包含任何 sample entry")
    entries = parse_boxes(stsd.data[8:], base_offset=stsd.data_start + 8)
    if not entries:
        raise RemuxError("[stsd] sample entry 解析为空")
    if entry_count != 1 or len(entries) != 1:
        raise RemuxError(
            f"基础版本仅支持单 sample entry 的 track，实际 {entry_count} 个"
        )
    return entries[0]


def _parse_elst(trak: Box) -> Tuple[bool, int, int]:
    """解析 trak/edts/elst，返回 (has_edit, media_time, segment_duration)。

    media_time 为 track timescale 单位、带符号；segment_duration 为
    movie timescale 单位。基础版本仅支持“单段、rate=1.0”的 edit
    （最常见的编码器延迟 / 启动偏移表达），空 edit 或多段/变速明确拒绝。
    """
    edts = _find_child(trak, "edts")
    if edts is None:
        return False, 0, 0
    elst = _find_child(edts, "elst")
    if elst is None:
        return False, 0, 0
    version, _ = _parse_full_box_header(elst.data)
    if len(elst.data) < 8:
        raise RemuxError("[elst] 长度不足")
    entry_count = unpack(">I", elst.data[4:8])[0]
    if entry_count != 1:
        raise RemuxError(
            f"基础版本仅支持单段 edit list，实际 {entry_count} 段"
        )
    pos = 8
    if version == 1:
        if len(elst.data) < pos + 20:
            raise RemuxError("[elst] v1 表项被截断")
        segment_duration = unpack(">Q", elst.data[pos:pos + 8])[0]
        media_time = unpack(">q", elst.data[pos + 8:pos + 16])[0]
        rate = unpack(">I", elst.data[pos + 16:pos + 20])[0]
    else:
        if len(elst.data) < pos + 12:
            raise RemuxError("[elst] v0 表项被截断")
        segment_duration = unpack(">I", elst.data[pos:pos + 4])[0]
        media_time = unpack(">i", elst.data[pos + 4:pos + 8])[0]
        rate = unpack(">I", elst.data[pos + 8:pos + 12])[0]
    if media_time < 0:
        raise RemuxError("[elst] 包含空 edit（media_time=-1），基础版本不支持")
    # rate 为 16.16 定点，1.0 = 0x00010000；非 1.0 即变速播放
    if rate != 0x00010000:
        raise RemuxError(
            f"[elst] 非 1.0 的播放速率 {rate / 65536:.3f}，基础版本不支持变速"
        )
    return True, media_time, segment_duration


def _parse_trak(trak: Box, file_size: int) -> Track:
    mdia = _require_child(trak, "mdia")
    mdhd = _require_child(mdia, "mdhd")
    hdlr = _require_child(mdia, "hdlr")
    minf = _require_child(mdia, "minf")
    stbl = _require_child(minf, "stbl")
    stsd = _require_child(stbl, "stsd")

    timescale, duration = _parse_mdhd(mdhd)
    language = _parse_mdhd_language(mdhd)
    handler = _parse_hdlr(hdlr)
    entry = _parse_stsd(stsd)
    tag = entry.kind

    has_edit, edit_media_time, _edit_segment_duration = _parse_elst(trak)

    if handler == "vide" or tag in _VIDEO_CODECS:
        if tag not in _VIDEO_CODECS:
            raise RemuxError(f"不支持的视频编码格式: {tag}（支持 avc1/avc3/hvc1/hev1）")
        track = _build_video_track(entry, stbl, timescale, duration, tag, file_size)
    elif handler == "soun" or tag in _AUDIO_CODECS:
        track = _build_audio_track(entry, stbl, timescale, duration, tag, file_size)
    elif handler == "subt" or tag in _SUBTITLE_CODECS:
        if tag not in _SUBTITLE_CODECS:
            raise RemuxError(f"不支持的字幕格式: {tag}（支持 stpp/TTML、wvtt/WebVTT）")
        track = _build_subtitle_track(entry, stbl, timescale, duration, tag, file_size)
    else:
        raise RemuxError(f"不支持的 track 类型: handler={handler} codec={tag}")

    track.has_edit = has_edit
    track.edit_media_time = edit_media_time
    track.language = language
    return track


def _build_video_track(entry, stbl, timescale, duration, tag, file_size) -> Track:
    family = _VIDEO_CODECS[tag]
    private_box = _scan_subbox(entry, ("avcC",) if family == "avc" else ("hvcC",))
    if private_box is None:
        raise RemuxError(f"[{tag}] 中找不到 {family} 编码私有数据盒")
    width, height = _parse_visual_dimensions(entry)
    if width <= 0 or height <= 0:
        raise RemuxError(f"[{tag}] 图像宽高非法: {width}x{height}")

    track = Track(
        track_id=0,
        handler="vide",
        codec_family=family,
        codec_tag=tag,
        timescale=timescale,
        duration=duration,
        width=width,
        height=height,
        codec_private=bytes(private_box.data),
    )
    _fill_samples(track, stbl, file_size, is_video=True)
    return track


def _build_audio_track(entry, stbl, timescale, duration, tag, file_size) -> Track:
    """构建音频轨。

    - mp4a + esds：AAC，私有数据为 AudioSpecificConfig；
    - mp4a 无 esds：MP3（裸 MPEG 音频帧，MKV 用 A_MPEG/L3，无私有数据）；
    - ac-3 / ec-3：Dolby AC-3 / E-AC-3，真实配置在 dac3/dec3 盒中，
      MKV 不需要 CodecPrivate，采样率/声道取自 sample entry 固定头。
    """
    if tag == "mp4a":
        esds = _scan_subbox(entry, ("esds",))
        if esds is not None:
            object_type, asc = _parse_esds(esds)
            if object_type == 0x6B or not asc:
                # MPEG-1 Audio（MP3）：esds 中无 AudioSpecificConfig，
                # objectTypeIndication=0x6B；裸帧直接拷贝，MKV 无私有数据
                channels, sample_rate = _parse_audio_fixed_fields(entry)
                family, private = "mp3", b""
            elif object_type == 0x40:
                asc_rate, asc_channels = _parse_audio_specific_config(asc)
                fixed_channels, fixed_rate = _parse_audio_fixed_fields(entry)
                channels = asc_channels or fixed_channels
                sample_rate = asc_rate or fixed_rate
                family, private = "aac", asc
            else:
                raise RemuxError(
                    f"[mp4a] 不支持的音频 objectTypeIndication=0x{object_type:02X}"
                    f"（仅支持 AAC 0x40 / MP3 0x6B）"
                )
        else:
            # 无 esds 的 mp4a：同样按 MP3 处理（MOV/MP4 中 .mp3 的另一种封装）
            channels, sample_rate = _parse_audio_fixed_fields(entry)
            family, private = "mp3", b""
    else:
        family = _AUDIO_CODECS[tag]  # ac3 / eac3
        channels, sample_rate = _parse_audio_fixed_fields(entry)
        private = b""

    if channels <= 0 or sample_rate <= 0:
        raise RemuxError(
            f"[{tag}] 无法确定音频参数: channels={channels} rate={sample_rate}"
        )

    track = Track(
        track_id=0,
        handler="soun",
        codec_family=family,
        codec_tag=tag,
        timescale=timescale,
        duration=duration,
        channels=channels,
        sample_rate=sample_rate,
        codec_private=private,
    )
    _fill_samples(track, stbl, file_size, is_video=False)
    return track


def _build_subtitle_track(entry, stbl, timescale, duration, tag, file_size) -> Track:
    """构建字幕轨。

    - stpp（TTML）：每个样本即一份完整 TTML 文档，可逐字节直拷进 MKV；
    - wvtt（WebVTT）：样本是 vttc/vtte 盒序列，转换在 MKV 写入器完成，
      条目内可选的 vttC 全局配置盒作为 CodecPrivate 原样保留。
    """
    family = _SUBTITLE_CODECS[tag]
    codec_private = b""
    if family == "webvtt":
        vttc = _scan_subbox(entry, ("vttC",))
        if vttc is not None:
            codec_private = bytes(vttc.data)

    track = Track(
        track_id=0,
        handler="subt",
        codec_family=family,
        codec_tag=tag,
        timescale=timescale,
        duration=duration,
        codec_private=codec_private,
    )
    _fill_samples(track, stbl, file_size, is_video=False)
    return track


def _fill_samples(track: Track, stbl: Box, file_size: int, *, is_video: bool) -> None:
    for kind in ("stts", "stsc", "stsz"):
        if _find_child(stbl, kind) is None:
            raise RemuxError(f"[stbl] 缺少必需的 [{kind}] 表")
    sizes = _parse_stsz(_find_child(stbl, "stsz"))
    if not sizes:
        raise RemuxError("该 track 不包含任何样本")
    chunk_offsets = _parse_chunk_offsets(stbl)
    stsc_entries = _parse_stsc(_find_child(stbl, "stsc"))
    offsets = _expand_sample_offsets(chunk_offsets, stsc_entries, sizes)

    dts_deltas = _parse_stts(_find_child(stbl, "stts"))
    ctts = _find_child(stbl, "ctts")
    ctts_offsets = _parse_ctts(ctts) if ctts is not None else []
    stss = _find_child(stbl, "stss")
    keyframes = _parse_stss(stss) if stss is not None else None

    # 边界校验：所有样本必须落在文件范围内
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if offset < 0 or offset + size > file_size:
            raise RemuxError(
                f"第 {idx + 1} 个样本的数据范围 [{offset}, {offset + size}) "
                f"超出文件边界（文件大小 {file_size}）"
            )

    track.samples = _build_samples(
        sizes, offsets, dts_deltas, ctts_offsets, keyframes, is_video
    )


# ---------------------------------------------------------------------------
# 顶层 Mp4File
# ---------------------------------------------------------------------------

class Mp4File:
    """已解析的 MP4 文件：保留文件字节缓冲与 track 列表。"""

    def __init__(self, data: bytes):
        self._data = data
        self.tracks: List[Track] = []
        self._parse()

    @property
    def data(self) -> bytes:
        return self._data

    def _parse(self) -> None:
        if len(self._data) < 8:
            raise RemuxError("文件过小，不可能是合法 MP4（不足 8 字节）")
        top_level = parse_boxes(self._data)
        kinds = {box.kind for box in top_level}
        if "ftyp" not in kinds:
            raise RemuxError("缺少 ftyp 盒，不是合法的 ISO-BMFF/MP4 文件")
        if "moov" not in kinds:
            raise RemuxError("缺少 moov 盒，无法获取媒体元数据（可能是未完成的录制文件）")

        # 明确拒绝基础版本不支持的结构，给出可操作的提示
        if kinds & {"moof"}:
            raise RemuxError("检测到碎片化 MP4（moof 盒），基础版本仅支持非分片 MP4")

        moov = next(b for b in top_level if b.kind == "moov")
        # 加密/保护检查先于后续结构解析，使不含完整元数据的异常文件也能被拦截
        if self._contains_any(moov, ("sinf", "saiz", "senc")):
            raise RemuxError("检测到加密/保护信息（sinf/saiz/senc），基础版本不支持加密媒体")
        if self._contains_any(moov, ("encv", "enca")):
            raise RemuxError("检测到加密 sample entry（encv/enca），基础版本不支持加密媒体")

        traks = [b for b in moov.children() if b.kind == "trak"]
        if not traks:
            raise RemuxError("moov 中没有任何 trak 盒")

        for index, trak in enumerate(traks, start=1):
            track = _parse_trak(trak, len(self._data))
            track.track_id = index
            self.tracks.append(track)

    @staticmethod
    def _contains_any(box: Box, targets: Tuple[str, ...]) -> bool:
        """深度优先扫描盒树，判断是否出现目标四字符码。"""
        stack = [box]
        while stack:
            current = stack.pop()
            try:
                children = current.children()
            except RemuxError:
                continue
            for child in children:
                if child.kind in targets:
                    return True
                stack.append(child)
        return False


def parse_mp4(data: bytes) -> Mp4File:
    """解析 MP4 字节缓冲，返回 :class:`Mp4File`。"""
    return Mp4File(data)
