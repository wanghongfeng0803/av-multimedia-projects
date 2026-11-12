# -*- coding: utf-8 -*-
"""
HLS 媒体播放列表（m3u8）的手写生成与自校验。

单码率、fMP4（HLS 版本 7）：
- EXT-X-MAP 指向独立 init 段；
- 每个分片一条 EXTINF，断流/编码变化处在对应分片前插 EXT-X-DISCONTINUITY；
- 不使用 ffmpeg 的 hls muxer，标签与精度全部由本模块控制。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import List, Optional

HLS_VERSION = 7
INIT_FILENAME = "init.mp4"
MASTER_FILENAME = "master.m3u8"
SEGMENTS_JSON = "segments.json"


@dataclass
class SegmentRecord:
    sequence_number: int
    filename: str
    start_time: float          # 秒，浮点
    duration: float            # 秒，浮点
    discontinuity: bool        # 该分片前是否需要 EXT-X-DISCONTINUITY
    first_frame_idr: bool
    size_bytes: int
    # 该分片引用的 init 段；编码参数变化时会切到新的 initN.mp4
    init_file: str = INIT_FILENAME

    def to_json_dict(self) -> dict:
        return asdict(self)


def render_playlist(records: List[SegmentRecord],
                    target_duration: int,
                    media_sequence: int = 0,
                    ended: bool = True,
                    independent_segments: bool = True) -> str:
    lines: List[str] = ["#EXTM3U"]
    lines.append(f"#EXT-X-VERSION:{HLS_VERSION}")
    lines.append("#EXT-X-PLAYLIST-TYPE:VOD")
    lines.append(f"#EXT-X-TARGETDURATION:{target_duration}")
    lines.append(f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}")
    if independent_segments:
        lines.append("#EXT-X-INDEPENDENT-SEGMENTS")
    # 头部 EXT-X-MAP 指向第一份 init；后续编码变化处再用新的 EXT-X-MAP 覆盖
    if records:
        lines.append(f'#EXT-X-MAP:URI="{records[0].init_file}"')

    prev_init = records[0].init_file if records else INIT_FILENAME
    for rec in records:
        # 顺序：DISCONTINUITY →（init 变化时）EXT-X-MAP → EXTINF → URI
        if rec.discontinuity:
            lines.append("#EXT-X-DISCONTINUITY")
        if rec.init_file != prev_init:
            lines.append(f'#EXT-X-MAP:URI="{rec.init_file}"')
            prev_init = rec.init_file
        # EXTINF 浮点保留 3 位小数；RFC 8216 要求四舍五入到最近整数
        # 不超过 TARGETDURATION，浮点精度由校验器检查
        lines.append(f"#EXTINF:{rec.duration:.3f},")
        lines.append(rec.filename)

    if ended:
        lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def write_playlist(path: str, records: List[SegmentRecord],
                   target_duration: int, **kwargs) -> None:
    text = render_playlist(records, target_duration, **kwargs)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


class ManifestError(ValueError):
    pass


def validate_playlist_text(text: str) -> List[str]:
    """
    纯文本层面的 m3u8 规范校验，返回警告列表；遇到硬性错误抛 ManifestError。
    校验项：必备标签、版本、TARGETDURATION、EXTINF 精度与上限、
    MEDIA-SEQUENCE、DISCONTINUITY 位置、MAP、ENDLIST、URI 配对。
    """
    warnings: List[str] = []
    raw_lines = text.splitlines()
    lines = [ln.strip() for ln in raw_lines if ln.strip() != ""]
    if not lines or lines[0] != "#EXTM3U":
        raise ManifestError("缺少 #EXTM3U 头")

    def require(tag: str):
        if not any(ln.startswith(tag) for ln in lines):
            raise ManifestError(f"缺少必备标签 {tag}")

    require("#EXT-X-VERSION")
    require("#EXT-X-TARGETDURATION")
    map_lines = [ln for ln in lines if ln.startswith("#EXT-X-MAP:")]
    if not map_lines:
        raise ManifestError("缺少必备标签 #EXT-X-MAP")
    import re
    map_uris = []
    for ln in map_lines:
        m = re.search(r'URI="([^"]+)"', ln)
        if not m:
            raise ManifestError(f"EXT-X-MAP 缺少合法 URI 属性: {ln}")
        map_uris.append(m.group(1))
    if map_uris[0] != INIT_FILENAME:
        raise ManifestError(
            f"首个 EXT-X-MAP 必须指向 {INIT_FILENAME}，实际 {map_uris[0]}")

    version = _first_int(lines, "#EXT-X-VERSION:")
    if version < HLS_VERSION:
        raise ManifestError(f"fMP4 需要 VERSION>=7，当前 {version}")

    target = _first_int(lines, "#EXT-X-TARGETDURATION:")
    if target <= 0:
        raise ManifestError("EXT-X-TARGETDURATION 必须为正整数")

    seq_lines = [ln for ln in lines if ln.startswith("#EXT-X-MEDIA-SEQUENCE:")]
    if seq_lines:
        seq = int(seq_lines[0].split(":", 1)[1])
        if seq < 0:
            raise ManifestError("MEDIA-SEQUENCE 不能为负")

    # 逐行检查 EXTINF / URI / DISCONTINUITY 的配对与顺序
    extinf_count = 0
    uri_count = 0
    pending_discont = False
    last_was_uri = False
    max_dur = 0.0
    for i, ln in enumerate(lines):
        if ln == "#EXT-X-DISCONTINUITY":
            if last_was_uri is False and extinf_count > 0:
                raise ManifestError("DISCONTINUITY 位置非法（未紧贴分片）")
            pending_discont = True
        elif ln.startswith("#EXTINF:"):
            if pending_discont and extinf_count == 0:
                raise ManifestError("第一个分片前不应出现 DISCONTINUITY")
            value = ln[len("#EXTINF:"):].rstrip(",").split(",")[0]
            try:
                dur = float(value)
            except ValueError:
                raise ManifestError(f"EXTINF 时长无法解析: {ln}")
            if dur <= 0:
                raise ManifestError(f"EXTINF 时长必须为正: {ln}")
            # 精度：最多 3 位小数
            if "." in value and len(value.split(".", 1)[1]) > 3:
                raise ManifestError(f"EXTINF 精度超过 3 位小数: {ln}")
            # RFC 8216: 分片时长四舍五入后不得超过 TARGETDURATION
            if round(dur) > target:
                raise ManifestError(
                    f"分片 {dur}s 超过 TARGETDURATION={target}s")
            max_dur = max(max_dur, dur)
            extinf_count += 1
            last_was_uri = False
            pending_discont = False
        elif not ln.startswith("#"):
            uri_count += 1
            last_was_uri = True

    if extinf_count != uri_count:
        raise ManifestError(
            f"EXTINF 数量({extinf_count}) 与分片 URI 数量({uri_count}) 不一致")
    if extinf_count == 0:
        raise ManifestError("播放列表没有任何分片")

    if not any(ln == "#EXT-X-ENDLIST" for ln in lines):
        warnings.append("未出现 EXT-X-ENDLIST（直播列表可接受，VOD 应为结束态）")

    if max_dur > 0 and abs(max_dur - target) > 0.5:
        warnings.append(
            f"最大分片时长 {max_dur:.3f}s 与目标 {target}s 偏差较大")
    return warnings


def validate_playlist_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return validate_playlist_text(f.read())


def _first_int(lines: List[str], prefix: str) -> int:
    for ln in lines:
        if ln.startswith(prefix):
            return int(ln[len(prefix):].strip())
    raise ManifestError(f"找不到标签 {prefix}")


def validate_records(records: List[SegmentRecord],
                     segment_duration: float) -> List[str]:
    """对分片元数据做时间线层面的校验：序号连续、时间戳单调不回退。"""
    warnings: List[str] = []
    if not records:
        raise ManifestError("没有分片记录")
    prev_seq = None
    prev_end = None
    for rec in records:
        if prev_seq is not None and rec.sequence_number != prev_seq + 1:
            raise ManifestError(
                f"分片序号不连续: {prev_seq} -> {rec.sequence_number}")
        if prev_end is not None and rec.start_time + 1e-9 < prev_end:
            raise ManifestError(
                f"时间戳回退: 分片{rec.sequence_number} "
                f"start={rec.start_time} 早于上一片结束 {prev_end}")
        if rec.duration <= 0:
            raise ManifestError(f"分片{rec.sequence_number} 时长非法")
        if not rec.first_frame_idr:
            raise ManifestError(
                f"分片{rec.sequence_number} 首帧不是 IDR，违反切点约束")
        # 末片（或 DISCONTINUITY 收口片）偏短是正常的，只对超长片发警告
        if rec.duration > segment_duration + 0.6:
            warnings.append(
                f"分片{rec.sequence_number} 时长 {rec.duration:.3f}s "
                f"明显超过目标 {segment_duration}s")
        prev_seq = rec.sequence_number
        prev_end = rec.start_time + rec.duration
    return warnings
