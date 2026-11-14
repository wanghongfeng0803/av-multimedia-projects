"""M3U8 播放列表生成（RFC 8216，纯渲染、零 IO）。

支持两种播放列表形态：

* **VOD（点播）**：全量分片 + ``EXT-X-PLAYLIST-TYPE:VOD`` +
  ``EXT-X-ENDLIST``，文件一次成型；
* **滑动窗口（直播）**：只保留最近 N 个分片，通过
  ``EXT-X-MEDIA-SEQUENCE`` 表达窗口内首片的绝对序号，不写 ENDLIST。

标签合规要点
------------
* ``EXT-X-VERSION:3``：TS 分片 + 浮点 EXTINF 的最低可用版本；
* ``EXT-X-TARGETDURATION``：所有分片时长的**向上取整最大值**，
  保证每片 ``EXTINF`` 都不超过它（RFC 8216 §4.3.3.1 的硬性要求）；
* ``EXTINF``：保留 3 位小数的浮点秒，逗号后不写可选标题；
* ``EXT-X-MEDIA-SEQUENCE``：窗口首片的绝对分片序号；
* ``EXT-X-INDEPENDENT-SEGMENTS``：声明每片均可独立解码
  （本工具切点全部对齐 IDR，满足该承诺）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import PlaylistError
from .models import SegmentInfo

#: TS + 浮点 EXTINF 对应的最低 HLS 协议版本
HLS_VERSION = 3
#: EXTINF 浮点小数位数
_EXTINF_PRECISION = 3
#: TARGETDURATION 与分片时长的校验容差（秒）
_DURATION_EPS = 0.05


@dataclass(frozen=True)
class PlaylistConfig:
    """播放列表渲染配置。

    注：``EXT-X-MEDIA-SEQUENCE`` 不在这里配置，而是直接取窗口内首片
    的 :attr:`SegmentInfo.sequence_number`，保证标签与文件名序号永不脱节。
    """

    target_duration: int
    #: 滑动窗口大小；None 表示 VOD 全量列表
    window_size: int | None = None
    playlist_type: str = "VOD"
    independent_segments: bool = True


def compute_target_duration(segments: list[SegmentInfo]) -> int:
    """按 RFC 要求计算 TARGETDURATION：最大分片时长向上取整。"""
    if not segments:
        raise PlaylistError("没有分片，无法计算 TARGETDURATION")
    max_duration = max(seg.duration for seg in segments)
    return max(1, int(math.ceil(max_duration - 1e-9)))


def _validate(
    segments: list[SegmentInfo], config: PlaylistConfig
) -> None:
    """渲染前的合规性自检，违规直接抛错而不是产出非法播放列表。"""
    if not segments:
        raise PlaylistError("播放列表至少要包含一个分片")
    if config.target_duration <= 0:
        raise PlaylistError("TARGETDURATION 必须为正整数")

    overlong = [
        seg.filename
        for seg in segments
        if seg.duration > config.target_duration + _DURATION_EPS
    ]
    if overlong:
        raise PlaylistError(
            "以下分片时长超过 TARGETDURATION="
            f"{config.target_duration}s：{', '.join(overlong)}"
        )

    if config.window_size is not None and config.window_size <= 0:
        raise PlaylistError("滑动窗口大小必须为正整数（或使用 None 表示 VOD）")

    # 序号必须连续，否则 MEDIA-SEQUENCE 与窗口切片会错位
    for prev, cur in zip(segments, segments[1:]):
        if cur.sequence_number != prev.sequence_number + 1:
            raise PlaylistError(
                f"分片序号不连续：{prev.filename} -> {cur.filename}"
            )


def _select_window(
    segments: list[SegmentInfo], window_size: int | None
) -> list[SegmentInfo]:
    """直播模式取最后 window_size 片；VOD 模式返回全量。"""
    if window_size is None:
        return list(segments)
    return segments[-window_size:] if len(segments) > window_size else list(segments)


def render_playlist(
    segments: list[SegmentInfo], config: PlaylistConfig
) -> str:
    """把分片列表渲染为 M3U8 文本（末尾带换行，可直接落盘）。"""
    _validate(segments, config)
    window = _select_window(segments, config.window_size)
    first_sequence = window[0].sequence_number

    lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{HLS_VERSION}",
        f"#EXT-X-TARGETDURATION:{config.target_duration}",
        f"#EXT-X-MEDIA-SEQUENCE:{first_sequence}",
    ]
    if config.independent_segments:
        lines.append("#EXT-X-INDEPENDENT-SEGMENTS")

    is_vod = config.window_size is None
    if is_vod:
        lines.append(f"#EXT-X-PLAYLIST-TYPE:{config.playlist_type}")

    for seg in window:
        lines.append(f"#EXTINF:{seg.duration:.{_EXTINF_PRECISION}f},")
        lines.append(seg.filename)

    if is_vod:
        lines.append("#EXT-X-ENDLIST")

    return "\n".join(lines) + "\n"


def write_playlist(
    segments: list[SegmentInfo],
    output_path: str,
    config: PlaylistConfig,
) -> str:
    """渲染并写入 ``.m3u8`` 文件，返回文件绝对路径。"""
    text = render_playlist(segments, config)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return output_path
