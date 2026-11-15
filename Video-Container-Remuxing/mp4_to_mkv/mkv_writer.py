"""Matroska(MKV / EBML) 写入器。

把解析出的 track 元数据与样本重新组织为 Matroska 结构：

    EBML Header
    Segment
      ├─ Info（TimestampScale / Duration / 应用信息）
      ├─ Tracks（每条 track 一个 TrackEntry，含 CodecPrivate）
      └─ Cluster × N
           ├─ Timestamp（簇基准时间，毫秒）
           └─ SimpleBlock × N（帧字节原样写入，不重编码）

关键约定：
- 全局 TimestampScale 取默认值 1,000,000，即所有时间戳单位为 1 毫秒；
- SimpleBlock 的 16 位相对时间码承载 **显示时间戳(PTS)**（Matroska
  惯例，libavformat 亦如此），帧仍按 DTS 解码顺序写出，B 帧由块内
  编码顺序与参考关系自行表达；
- 仅使用 SimpleBlock（非 Block+BlockGroup），关键帧通过块头部 flags
  的 Keyframe 位标记，无需 ReferenceBlock；音频尾部裁剪与字幕块则使用
  BlockGroup（前者带 DiscardPadding，后者带 BlockDuration / BlockAdditions）；
- 应用 MP4 edit list 的 media_time 偏移，并丢弃时间轴起点之前的
  预卷样本（与主流工具行为一致）；
- 字幕轨（TTML 文档直拷、WebVTT 的 vttc 重组）与视频/音频统一按 DTS 交织，
  轨道排列固定为 视频 → 音频 → 字幕。

两条等价的写出路径：
- :func:`write_mkv`：在内存中拼出完整、带已知大小 Segment 的 MKV 字节串，
  输出与历史版本逐字节一致，适合小规模数据与需要完整字节的场景；
- :func:`write_mkv_stream`：面向二进制流的**流式**路径，Segment 采用未知
  大小 VINT，Cluster 逐簇写入、不产生临时文件、内存有界（同一时刻只持有
  一个簇），且不 seek，可写入管道等不可寻址流。两条路径共用同一份样本
  放置、时间戳与簇切分逻辑，块负载逐字节相同。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Sequence

from . import ebml
from .chapters import Chapter
from .errors import RemuxError
from .matroska_ids import *  # noqa: F401,F403  本模块大量使用元素 ID 常量
from .mp4_parser import Mp4File, Sample, Track
from .webvtt import BLOCK_ADD_ID_IDENTIFIER, BLOCK_ADD_ID_SETTINGS, convert_sample

# Cluster 切分参数
_TARGET_CLUSTER_SIZE = 5 * 1024 * 1024   # 达到该大小后在下一个视频关键帧切分
_HARD_CLUSTER_SIZE = 100 * 1024 * 1024   # 硬上限：即使没有关键帧也强制切分
_BLOCK_TC_MIN, _BLOCK_TC_MAX = -32768, 32767  # SimpleBlock 相对时间码范围

_WRITING_APP = "mp4-to-mkv-remuxer 0.3.0"


@dataclass
class _PlacedSample:
    """合并排序后的待写样本（时间戳已应用 edit list 偏移并归一化）。"""

    track_number: int   # MKV 中的 1 起始轨道号
    sample: Sample
    pts_ms: int         # 归一化后的显示时间戳（毫秒，非负）
    duration_ms: int    # 样本时长（毫秒）
    dts_time: Fraction  # 归一化后的 DTS（秒），用于跨 track 精确排序
    is_video: bool
    is_subtitle: bool = False
    # 末尾需裁剪的时长（纳秒），>0 时该块用 BlockGroup+DiscardPadding 表达
    discard_padding_ns: int = 0
    # WebVTT 字幕的块附加数据（cue id / settings），非空时用 BlockGroup 表达
    block_additions: list = field(default_factory=list)
    # WebVTT 转换后的块负载；为 None 时簇构建阶段按偏移从源缓冲切片
    frame_override: bytes | None = None


def _round_div(numerator: int, denominator: int) -> int:
    """对称四舍五入的整数除法（denominator > 0），正确处理负数。"""
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _adjust_ms(value: int, timescale: int, edit_media_time: int) -> int:
    """应用 edit 偏移并四舍五入到毫秒：(value - media_time) → ms。"""
    return _round_div((value - edit_media_time) * 1000, timescale)


def _build_simple_block(
    track_number: int, relative_timecode_ms: int, keyframe: bool, frame: bytes
) -> bytes:
    """构造一个完整的 SimpleBlock 元素。

    块数据布局：track(VINT) + timecode(int16 BE) + flags(1 byte) + 帧数据。
    """
    if not (_BLOCK_TC_MIN <= relative_timecode_ms <= _BLOCK_TC_MAX):
        raise RemuxError(
            f"SimpleBlock 相对时间码 {relative_timecode_ms}ms 超出 int16 范围"
        )
    block_data = (
        ebml.encode_vint(track_number)
        + struct.pack(">h", relative_timecode_ms)
        + bytes([0x80 if keyframe else 0x00])
        + frame
    )
    return ebml.binary_element(SIMPLE_BLOCK, block_data)


def _build_block_data(
    track_number: int, relative_timecode_ms: int, frame: bytes
) -> bytes:
    """构造 Block 元素的原始块数据（供 BlockGroup 使用，flags 固定为 0）。"""
    if not (_BLOCK_TC_MIN <= relative_timecode_ms <= _BLOCK_TC_MAX):
        raise RemuxError(
            f"Block 相对时间码 {relative_timecode_ms}ms 超出 int16 范围"
        )
    return (
        ebml.encode_vint(track_number)
        + struct.pack(">h", relative_timecode_ms)
        + b"\x00"
        + frame
    )


def _build_block_additions(additions: list) -> bytes:
    """构造 BlockAdditions 主元素。

    :param additions: (BlockAddID, 附加数据字节) 元组列表，
        WebVTT 中 1=cue identifier、2=cue settings。
    """
    block_mores = [
        ebml.master_element(
            BLOCK_MORE,
            [
                ebml.uint_element(BLOCK_ADD_ID, add_id),
                ebml.binary_element(BLOCK_ADDITIONAL, data),
            ],
        )
        for add_id, data in additions
    ]
    return ebml.master_element(BLOCK_ADDITIONS, block_mores)


def _build_block_group(
    track_number: int,
    relative_timecode_ms: int,
    frame: bytes,
    duration_ms: int,
    discard_padding_ns: int,
    additions: list | None = None,
) -> bytes:
    """构造 BlockGroup：AAC 尾部裁剪用 DiscardPadding，字幕用 BlockAdditions。

    两类附加信息都需要显式 BlockDuration 表达该块的呈现时长。
    """
    children = [
        ebml.binary_element(
            BLOCK,
            _build_block_data(track_number, relative_timecode_ms, frame),
        ),
        ebml.uint_element(BLOCK_DURATION, duration_ms),
    ]
    if discard_padding_ns > 0:
        children.append(ebml.sint_element(DISCARD_PADDING, discard_padding_ns))
    if additions:
        children.append(_build_block_additions(additions))
    return ebml.master_element(BLOCK_GROUP, children)


def _codec_id_for(track: Track) -> str:
    if track.codec_family == "avc":
        return CODEC_ID_AVC1
    if track.codec_family == "hevc":
        return CODEC_ID_HEVC
    if track.codec_family == "aac":
        return CODEC_ID_AAC
    if track.codec_family == "ac3":
        return CODEC_ID_AC3
    if track.codec_family == "eac3":
        return CODEC_ID_EAC3
    if track.codec_family == "mp3":
        return CODEC_ID_MP3
    if track.codec_family == "ttml":
        return CODEC_ID_TTML
    if track.codec_family == "webvtt":
        return CODEC_ID_WEBVTT
    raise RemuxError(f"内部错误：未知编码族 {track.codec_family}")


def _build_ebml_header() -> bytes:
    children = [
        ebml.uint_element(EBML_VERSION, 1),
        ebml.uint_element(EBML_READ_VERSION, 1),
        ebml.uint_element(EBML_MAX_ID_LENGTH, 4),
        ebml.uint_element(EBML_MAX_SIZE_LENGTH, 8),
        ebml.string_element(DOC_TYPE, "matroska"),
        ebml.uint_element(DOC_TYPE_VERSION, 4),
        ebml.uint_element(DOC_TYPE_READ_VERSION, 2),
    ]
    return ebml.master_element(EBML_HEADER, children)


def _track_type(track: Track) -> int:
    if track.is_video:
        return TRACK_TYPE_VIDEO
    if track.is_subtitle:
        return TRACK_TYPE_SUBTITLE
    return TRACK_TYPE_AUDIO


def _build_track_entry(
    track: Track, track_number: int, default: bool
) -> bytes:
    children = [
        ebml.uint_element(TRACK_NUMBER, track_number),
        # TrackUID 与轨道号取相同的确定性值即可，只需全文件唯一
        ebml.uint_element(TRACK_UID, track_number),
        ebml.uint_element(TRACK_TYPE, _track_type(track)),
        ebml.uint_element(FLAG_DEFAULT, 1 if default else 0),
        # 不使用 lacing：每块恰好一帧 / 一个音频包 / 一条字幕
        ebml.uint_element(FLAG_LACING, 0),
        ebml.string_element(CODEC_ID, _codec_id_for(track)),
    ]
    # 视频轨道的 MinCache 建议为 1（Matroska 规范对视频的推荐值）
    if track.is_video:
        children.append(ebml.uint_element(MIN_CACHE, 1))

    if track.codec_private:
        children.append(ebml.binary_element(CODEC_PRIVATE, track.codec_private))

    # 语言码：音频/字幕轨携带 mdhd 中的 ISO 639-2/T 语言码
    if not track.is_video and track.language and track.language != "und":
        children.append(ebml.string_element(LANGUAGE, track.language))

    if track.is_video:
        children.append(
            ebml.master_element(
                VIDEO,
                [
                    ebml.uint_element(PIXEL_WIDTH, track.width),
                    ebml.uint_element(PIXEL_HEIGHT, track.height),
                ],
            )
        )
    elif track.is_audio:
        # 编码器前导延迟（priming）：edit list 的 media_time 表示需在解码
        # 输出前裁掉的采样数，换算为 Matroska 的 CodecDelay（纳秒）。
        # 这是所有音频编码共有的机制（AAC/AC-3/E-AC-3/MP3 均可能携带），
        # 解码器据此跳过前导采样，使 PCM 输出与源严格对齐。
        if track.has_edit and track.edit_media_time > 0:
            codec_delay_ns = (
                track.edit_media_time * 1_000_000_000
                + track.timescale // 2
            ) // track.timescale
            children.append(ebml.uint_element(CODEC_DELAY, codec_delay_ns))
        audio_children = [
            ebml.float_element(SAMPLING_FREQUENCY, float(track.sample_rate)),
            ebml.uint_element(CHANNELS, track.channels),
        ]
        children.append(ebml.master_element(AUDIO, audio_children))
    # 字幕轨没有 VIDEO/AUDIO 主元素，其余元数据已在上方写齐

    return ebml.master_element(TRACK_ENTRY, children)


def _build_info(duration_ms: float) -> bytes:
    children = [
        ebml.uint_element(TIMESTAMP_SCALE, DEFAULT_TIMESTAMP_SCALE),
        # Duration 是浮点元素，单位由 TimestampScale 决定（此处为毫秒）
        ebml.float_element(DURATION, duration_ms),
        ebml.utf8_element(MUXING_APP, _WRITING_APP),
        ebml.utf8_element(WRITING_APP, _WRITING_APP),
    ]
    return ebml.master_element(INFO, children)


def _build_chapters(chapters: List[Chapter], duration_ms: float) -> bytes:
    """把 chpl 章节清单编码为 Matroska Chapters 主元素。

    结构（单个 EditionEntry 容纳全部章节，MP4 chpl 本身是单层清单）::

        Chapters
          EditionEntry
            EditionFlagHidden=0 / EditionFlagDefault=1
            ChapterAtom × N
              ChapterUID（确定性：取章节在归一化清单中的 1 起始序号）
              ChapterTimeStart（纳秒；来自 chpl 起始时间）
              ChapterTimeEnd（纳秒；下一章起点，末章取影片时长）
              ChapterFlagHidden=0 / ChapterFlagEnabled=1
              ChapterDisplay
                ChapString（标题，UTF-8）
                ChapLanguage="und"（chpl 不携带语言标签）

    chpl 只携带起始时间，而 Matroska 生态（libavformat/mkvmerge）要求
    每章带 ChapterTimeEnd，否则部分解复用器会丢弃起始为 0 的首章；
    因此用“下一章起点”补齐，末章结束时间取影片总时长。

    ChapterUID 取序号而非哈希，使同一输入在任意机器、任意次数下
    产出逐字节一致的输出（可复现性）。
    """
    duration_ns = round(duration_ms * 1_000_000)
    atoms: List[bytes] = []
    count = len(chapters)
    for index, chapter in enumerate(chapters, start=1):
        # 结束时间：优先取下一章起点；末章回退到影片时长，并保证不早于起点
        if index < count:
            end_ns = max(chapter.start_ns, chapters[index].start_ns)
        else:
            end_ns = max(chapter.start_ns, duration_ns)
        display = ebml.master_element(
            CHAPTER_DISPLAY,
            [
                ebml.utf8_element(CHAP_STRING, chapter.title),
                ebml.string_element(CHAP_LANGUAGE, "und"),
            ],
        )
        atom_children = [
            ebml.uint_element(CHAPTER_UID, index),
            # 时间戳是 ChapterAtom 必需元素：0 必须显式写为单字节 0x00，
            # 否则 libavformat 把零长度数据当作“值缺失”而丢弃整个章节
            ebml.uint_element(CHAPTER_TIME_START, chapter.start_ns, min_bytes=1),
            ebml.uint_element(CHAPTER_TIME_END, end_ns, min_bytes=1),
            ebml.uint_element(CHAPTER_FLAG_HIDDEN, 0),
            ebml.uint_element(CHAPTER_FLAG_ENABLED, 1),
            display,
        ]
        atoms.append(ebml.master_element(CHAPTER_ATOM, atom_children))

    edition = ebml.master_element(
        EDITION_ENTRY,
        [
            ebml.uint_element(EDITION_FLAG_HIDDEN, 0),
            ebml.uint_element(EDITION_FLAG_DEFAULT, 1),
            *atoms,
        ],
    )
    return ebml.master_element(CHAPTERS, [edition])


def _place_samples(
    mp4: Mp4File, ordered_tracks: Sequence[Track], track_numbers: Sequence[int]
) -> List[_PlacedSample]:
    """把所有 track 的样本按 DTS 合并为全局解码顺序。

    edit list / 编解码器 priming 处理（全程不丢样本）：
    1. 应用 media_time 偏移：时间轴时间 = 媒体时间 - media_time；
    2. 前导编码器延迟由 TrackEntry 的 CodecDelay 表达（见轨道构建），
       priming 样本保留，逐轨把负的最早时间戳钳到 0；
    3. 尾部填充：AAC 末包的 stts 时长常被 edit 截短（小于一个完整解码帧），
       但解码器仍会输出完整帧。差值以 BlockGroup + DiscardPadding（纳秒）
       表达，让解码器裁掉多出的尾部采样。
    """
    raw: List[_PlacedSample] = []
    for track, number in zip(ordered_tracks, track_numbers):
        edit = track.edit_media_time if track.has_edit else 0

        # 完整解码帧时长（track timescale 单位）：取样本时长的众数。
        # AAC 常规包均为 1024 采样，末包可能被截短为更小的值。
        full_frame_duration = _modal_duration(track)

        track_items: List[_PlacedSample] = []
        for sample in track.samples:
            discard_ns = 0
            # DiscardPadding 是音频 priming/填充机制，仅用于音频轨道：
            # 末包 stts 时长被截短，但解码器仍输出完整帧，差值需裁掉。
            if track.is_audio and full_frame_duration and sample.duration < full_frame_duration:
                padding_units = full_frame_duration - sample.duration
                discard_ns = _round_div(
                    padding_units * 1_000_000_000, track.timescale
                )

            # WebVTT：在放置阶段把 vttc 盒转换为 cue 正文与附加数据。
            # 其它编码的样本字节仍留到簇构建时按偏移零拷贝切片。
            additions = []
            frame_override = None
            if track.codec_family == "webvtt":
                raw_sample = mp4.data[
                    sample.offset:sample.offset + sample.size
                ]
                cue = convert_sample(raw_sample)
                frame_override = cue.payload
                if cue.has_additions:
                    if cue.identifier:
                        additions.append((BLOCK_ADD_ID_IDENTIFIER, cue.identifier))
                    if cue.settings:
                        additions.append((BLOCK_ADD_ID_SETTINGS, cue.settings))

            track_items.append(
                _PlacedSample(
                    track_number=number,
                    sample=sample,
                    # 暂时是 edit 调整后的毫秒，可能为负（priming）
                    pts_ms=_adjust_ms(sample.pts, track.timescale, edit),
                    duration_ms=_adjust_ms(
                        sample.duration, track.timescale, 0
                    ),
                    # DTS 可能为负（B 帧重排 / priming），仅用于排序
                    dts_time=Fraction(sample.dts - edit, track.timescale),
                    is_video=track.is_video,
                    is_subtitle=track.is_subtitle,
                    discard_padding_ns=discard_ns,
                    block_additions=additions,
                    frame_override=frame_override,
                )
            )

        # 逐轨把负的最早时间戳钳到 0：priming 包整体平移，而非丢弃。
        # 解码器依据 CodecDelay 裁掉编码器前导采样，平移不影响实际内容。
        if track_items:
            track_shift = -min(item.pts_ms for item in track_items)
            if track_shift > 0:
                for item in track_items:
                    item.pts_ms += track_shift
        raw.extend(track_items)

    if not raw:
        raise RemuxError("没有任何可写入的样本")

    # 归一化 DTS 升序；同一时刻视频 → 音频 → 字幕，再按轨道号，保证交织确定
    raw.sort(
        key=lambda p: (
            p.dts_time,
            0 if p.is_video else (1 if not p.is_subtitle else 2),
            p.track_number,
        )
    )
    return raw


def _modal_duration(track: Track) -> int:
    """返回轨道样本时长的众数（即一个完整解码帧的典型时长）。"""
    counts: dict = {}
    for sample in track.samples:
        counts[sample.duration] = counts.get(sample.duration, 0) + 1
    if not counts:
        return 0
    # 众数；并列时取较大值（完整帧时长通常大于被截短的末包时长）
    return max(counts, key=lambda d: (counts[d], d))


def _frame_for(mp4: Mp4File, item: _PlacedSample) -> bytes:
    """取出一个待写样本的块负载（帧字节）。

    WebVTT 在放置阶段已转换为 cue 正文（长度可能变化）；其它编码按
    样本偏移从源缓冲切片，零拷贝。到达文件末尾等截断情况明确报错。
    """
    if item.frame_override is not None:
        return item.frame_override
    frame = mp4.data[item.sample.offset:item.sample.offset + item.sample.size]
    if len(frame) != item.sample.size:  # 防御性校验，正常不会触发
        raise RemuxError("读取样本数据时到达文件末尾，源文件可能被截断")
    return frame


def _build_block(mp4: Mp4File, item: _PlacedSample, relative_tc: int) -> bytes:
    """按样本类型构造单个块（SimpleBlock 或 BlockGroup）。"""
    frame = _frame_for(mp4, item)
    if item.discard_padding_ns > 0:
        # 尾部需裁剪：用 BlockGroup + DiscardPadding 而非 SimpleBlock
        return _build_block_group(
            item.track_number,
            relative_tc,
            frame,
            item.duration_ms,
            item.discard_padding_ns,
        )
    if item.is_subtitle:
        # 字幕块统一用 BlockGroup：BlockDuration 表达 cue 呈现时长，
        # WebVTT 的 cue id/settings 通过 BlockAdditions 保留
        return _build_block_group(
            item.track_number,
            relative_tc,
            frame,
            item.duration_ms,
            0,
            item.block_additions,
        )
    return _build_simple_block(
        item.track_number,
        relative_tc,
        item.sample.is_keyframe,
        frame,
    )


def _needs_new_cluster(
    item: _PlacedSample, have_blocks: bool, cluster_ts: int, cluster_size: int
) -> bool:
    """判断给定样本是否应另起一个新簇（切分规则与历史版本完全一致）。"""
    if not have_blocks:
        return True
    relative_tc = item.pts_ms - cluster_ts
    if not (_BLOCK_TC_MIN <= relative_tc <= _BLOCK_TC_MAX):
        # 相对时间码装不下 int16，必须开新簇
        return True
    if cluster_size >= _HARD_CLUSTER_SIZE:
        return True
    if (
        cluster_size >= _TARGET_CLUSTER_SIZE
        and item.is_video
        and item.sample.is_keyframe
    ):
        # 达到目标大小后，在下一个视频关键帧处切分
        return True
    return False


def _iter_cluster_bytes(
    mp4: Mp4File, placed: Sequence[_PlacedSample]
):
    """按既定切分规则逐簇生成完整 Cluster 元素字节（生成器）。

    切分规则与历史一次性实现完全一致，区别仅在于这是惰性生成器：
    每次只在内存里保留**当前一个簇**的块，簇一闭合即可交给流式写出
    落盘，从而把峰值内存从“整段输出”降到“一个簇”。
    """
    blocks: List[bytes] = []
    # 首簇基准取 0 与最早 PTS 的较大者：priming 样本（PTS<0）以负相对
    # 时间码落入 Timestamp=0 的首簇，常规样本则从其自身 PTS 起簇。
    cluster_ts = max(0, min(item.pts_ms for item in placed))
    cluster_size = 0

    for item in placed:
        if _needs_new_cluster(item, bool(blocks), cluster_ts, cluster_size):
            if blocks:
                payload = (
                    ebml.uint_element(TIMESTAMP, cluster_ts) + b"".join(blocks)
                )
                yield ebml.master_element(CLUSTER, [payload])
                blocks = []
                cluster_size = 0
            # 新簇基准必须非负（Cluster Timestamp 是无符号整数）
            cluster_ts = max(0, item.pts_ms)

        relative_tc = item.pts_ms - cluster_ts
        block = _build_block(mp4, item, relative_tc)
        blocks.append(block)
        cluster_size += len(block)

    if blocks:
        payload = ebml.uint_element(TIMESTAMP, cluster_ts) + b"".join(blocks)
        yield ebml.master_element(CLUSTER, [payload])


def _order_tracks(
    mp4: Mp4File,
) -> tuple[List[Track], List[int], List[bool]]:
    """计算轨道排列：视频 → 音频 → 字幕，同类保持源顺序，轨道号从 1 开始。

    :return: (排序后的轨道, 对应的 1 起始 MKV 轨道号, 各轨是否为该类默认轨)。
    """
    def _track_order(track: Track) -> int:
        if track.is_video:
            return 0
        if track.is_audio:
            return 1
        return 2

    ordered_tracks = sorted(
        mp4.tracks, key=lambda t: (_track_order(t), t.track_id)
    )
    track_numbers = list(range(1, len(ordered_tracks) + 1))
    # 每类轨道的第一条为该类的默认轨道
    default_flags: List[bool] = []
    seen = {"video": False, "audio": False, "subtitle": False}
    for track in ordered_tracks:
        kind = "video" if track.is_video else (
            "audio" if track.is_audio else "subtitle"
        )
        default_flags.append(not seen[kind])
        seen[kind] = True
    return ordered_tracks, track_numbers, default_flags


def _prepare_segment(mp4: Mp4File):
    """计算内存路径与流式路径共用的全部前置结构。

    :return: ``(ebml_header, segment_prefix, placed, duration_ms)``，其中
        ``segment_prefix`` 是 Segment 内、Cluster 之前的全部一级元素
        （Info + Tracks + 可选 Chapters），两条写出路径复用同一字节，
        保证流式输出与内存输出在结构层面完全等价。
    """
    if not mp4.tracks:
        raise RemuxError("MP4 中没有可写入的 track")

    ordered_tracks, track_numbers, default_flags = _order_tracks(mp4)

    header = _build_ebml_header()
    placed = _place_samples(mp4, ordered_tracks, track_numbers)
    # 总时长 = 最后一个样本的显示结束时间（PTS + 自身时长）
    duration_ms = float(
        max(item.pts_ms + item.duration_ms for item in placed)
    )
    info = _build_info(duration_ms)
    tracks = ebml.master_element(
        TRACKS,
        [
            _build_track_entry(track, number, default)
            for track, number, default in zip(
                ordered_tracks, track_numbers, default_flags
            )
        ],
    )
    # 章节（chpl）作为 Segment 级主元素，位于 Tracks 之后、Cluster 之前；
    # 无章节时不产生任何字节，输出与旧版本逐字节一致（除 WritingApp 版本号）
    chapters = (
        _build_chapters(mp4.chapters, duration_ms)
        if getattr(mp4, "chapters", None)
        else b""
    )
    segment_prefix = info + tracks + chapters
    return header, segment_prefix, placed, duration_ms


def write_mkv(mp4: Mp4File) -> bytes:
    """把已解析的 :class:`Mp4File` 重封装为完整 MKV 字节串。

    一次性在内存中拼好整个文档（含已知大小的 Segment），适合小规模数据
    与需要拿到完整字节的场景（如测试、哈希比对）。其输出与历史版本
    逐字节兼容。
    """
    header, segment_prefix, placed, _duration = _prepare_segment(mp4)
    clusters = b"".join(_iter_cluster_bytes(mp4, placed))
    segment = ebml.master_element(SEGMENT, [segment_prefix + clusters])
    return header + segment


def write_mkv_stream(mp4: Mp4File, stream) -> int:
    """把 :class:`Mp4File` 以**流式**方式写入二进制流，不产生临时文件。

    与 :func:`write_mkv` 的区别：

    - Segment 使用 **未知大小（unknown-size）VINT** 头（8 字节，
      ``0x01`` 后随 7 个 ``0xFF``），因此无需预知总长度、无需先在内存或
      磁盘上拼出整份文档，这是 Matroska 规范允许的流式/直播布局，
      ffmpeg、mkvmerge 等均可正常读取；
    - Info/Tracks/Chapters 前缀只构造一次，随后 Cluster **逐簇**写入，
      内存里同一时刻只保留当前一个簇（上界即簇切分的硬上限），峰值内存
      与输出总大小无关；
    - 全程只调用 ``stream.write``，**不 seek、不 flush 之外的定位操作**，
      因此管道、套接字等不可寻址流同样适用。

    :param stream: 任意提供 ``write(bytes) -> int`` 的二进制对象。
    :return: 实际写入流的字节总数。
    :raises RemuxError: 输入不含可写轨道，或样本数据被截断等确定性错误。
    :raises OSError: 底层流写入失败（如磁盘满、管道关闭），原样上抛，
        由编排层判定是否值得重试，并负责清理半成品目标文件。
    """
    header, segment_prefix, placed, _duration = _prepare_segment(mp4)

    written = 0

    def _emit(chunk: bytes) -> None:
        nonlocal written
        if not chunk:
            return
        view = memoryview(chunk)
        # 循环写以兼容只接受有限长度、或 write 短写的类文件对象
        while view:
            n = stream.write(view)
            if n is None:  # 部分实现（如 io.TextIO）不返回写计数，按整段处理
                n = len(view)
            if n <= 0:
                raise RemuxError("输出流未接受任何字节，写出中止")
            view = view[n:]
            written += n

    # EBML 文档头
    _emit(header)
    # 未知大小的 Segment 头：ID + 8 字节未知大小 VINT（0x01 0xFF×7）。
    # 之后写入的 Info/Tracks/Chapters/Cluster 都成为该 Segment 的子元素，
    # 直到流结束（EOF）隐式闭合，无需回填长度，天然适合单向流式写出。
    _emit(ebml.unknown_size_header(SEGMENT))
    _emit(segment_prefix)
    # 逐簇落盘：生成器每次只物化一个 Cluster，写完即可被回收
    for cluster in _iter_cluster_bytes(mp4, placed):
        _emit(cluster)

    # 让缓冲型流（真实文件对象）尽快落盘；不支持 flush 的自定义流容错跳过
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()
    return written
