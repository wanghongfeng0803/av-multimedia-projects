"""内嵌多字幕轨的枚举、元数据解析与目标轨定位。

容器（MKV/MP4 等）可以内嵌多条字幕轨，本模块负责「烧录哪一条」之前的
全部选择流程，分三层、各自可独立单测::

    ffprobe JSON ──► parse_tracks()        纯函数：元数据 → SubtitleTrack 索引表
    SubtitleTrack 列表 ──► select_track()  纯函数：按流索引/语言等条件定位目标轨
    目标轨 ──► extract_track()             ffmpeg 把该轨解出为临时 SRT/ASS 文件

关键设计：

* **双索引并存**：``stream_index`` 是容器内的绝对流序号（ffprobe 的
  ``index`` 字段，抽取时 ``-map 0:N`` 用它）；``sub_index`` 是字幕轨
  内部的相对序号（ffmpeg 的 ``s:N`` 写法语义，用户视角的「第几条字幕」）。
  两者都暴露给用户，避免「视频轨占位导致的索引错位」困惑；
* **元数据防御式归一化**：语言标签小写化、``und``/缺失统一为空串，
  disposition 缺省视为非 default/非 forced——ffprobe 输出字段经常缺席，
  解析层绝不因缺字段抛异常；
* **选择失败全部显式化**：无字幕轨、条件无匹配、多轨歧义、图形字幕轨
  不可烧录，分别对应不同的异常类型与中文诊断信息，并携带可用轨列表，
  调用方（CLI）可直接把异常文本呈现给用户。
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

__all__ = [
    "SubtitleTrack",
    "TrackError",
    "TrackProbeError",
    "TrackSelectionError",
    "TrackNotFoundError",
    "TrackAmbiguousError",
    "parse_tracks",
    "probe_subtitle_tracks",
    "select_track",
    "extract_track",
]

#: 可进入文本渲染管线的 codec → 烧录格式映射。
#: mov_text（MP4 内嵌文本轨）由 ffmpeg 转码为 SRT；ass/ssa 原样抽出。
#: 未收录的 codec（hdmv_pgs_subtitle、dvd_subtitle 等图形轨）format 为 ""，
#: 枚举时照常列出，选中烧录时才报「不支持」——枚举与可用性判断分离。
_CODEC_FORMATS = {
    "subrip": "srt",
    "srt": "srt",
    "mov_text": "srt",
    "ass": "ass",
    "ssa": "ass",
}


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------

class TrackError(RuntimeError):
    """多轨选择流程的统一异常基类。"""


class TrackProbeError(TrackError):
    """ffprobe 枚举失败（进程非零退出、输出不是合法 JSON 等）。"""


class TrackSelectionError(TrackError):
    """轨道定位失败的基类；``tracks`` 携带当时可用的轨道列表便于诊断。"""

    def __init__(self, message: str, tracks: tuple[SubtitleTrack, ...] = ()):
        super().__init__(message)
        self.tracks = tracks


class TrackNotFoundError(TrackSelectionError):
    """没有任何字幕轨，或给定条件（索引/语言/标题）匹配不到轨道。"""


class TrackAmbiguousError(TrackSelectionError):
    """条件过宽匹配到多条轨道，需要调用方进一步收窄。"""


# ---------------------------------------------------------------------------
# 轨道索引数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubtitleTrack:
    """一条内嵌字幕轨的索引记录（枚举阶段确定，之后只读）。"""

    stream_index: int   # 容器内绝对流索引（ffprobe index，-map 0:N 用）
    sub_index: int      # 字幕轨内相对序号（0 起，即 ffmpeg 的 s:N）
    codec_name: str     # ffprobe 原始 codec 名，小写
    language: str       # 归一化小写语言标签；缺失/und 统一为 ""
    title: str          # 轨标题元数据，缺失为 ""
    default: bool       # disposition.default
    forced: bool        # disposition.forced
    format: str         # "srt"/"ass"；不可烧录的类型为 ""

    @property
    def burnable(self) -> bool:
        """是否为可进入文本渲染管线的轨（图形字幕轨为 False）。"""
        return bool(self.format)

    def describe(self) -> str:
        """一行人读摘要，用于 CLI 列表与结果回显。"""
        parts = [f"s:{self.sub_index}", f"流{self.stream_index}",
                 self.codec_name]
        if self.language:
            parts.append(f"lang={self.language}")
        if self.title:
            parts.append(f"title={self.title!r}")
        flags = []
        if self.default:
            flags.append("default")
        if self.forced:
            flags.append("forced")
        if flags:
            parts.append("/".join(flags))
        if not self.burnable:
            parts.append("不可烧录(图形轨)")
        return " ".join(parts)


def _normalize_language(tag: object) -> str:
    """语言标签归一化：小写、去空白；``und``（未定）与缺失统一为空串。"""
    if not isinstance(tag, str):
        return ""
    tag = tag.strip().lower()
    return "" if tag in ("", "und") else tag


def parse_tracks(data: dict) -> tuple[SubtitleTrack, ...]:
    """把 ffprobe ``-show_streams`` 的 JSON 解析为字幕轨索引表（纯函数）。

    只收录 ``codec_type == "subtitle"`` 的流；tags/disposition 等字段
    全部防御式缺省，任何单条流的元数据残缺都不影响其余流的枚举。
    """
    tracks: list[SubtitleTrack] = []
    streams = data.get("streams")
    if not isinstance(streams, list):
        return ()
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") != "subtitle":
            continue
        try:
            stream_index = int(stream["index"])
        except (KeyError, TypeError, ValueError):
            continue  # 没有绝对索引的流无法被 -map 定位，跳过
        codec = str(stream.get("codec_name") or "").lower()
        tags = stream.get("tags")
        if not isinstance(tags, dict):
            tags = {}
        disposition = stream.get("disposition")
        if not isinstance(disposition, dict):
            disposition = {}
        title = tags.get("title")
        tracks.append(SubtitleTrack(
            stream_index=stream_index,
            sub_index=len(tracks),
            codec_name=codec,
            language=_normalize_language(tags.get("language")),
            title=str(title).strip() if isinstance(title, str) else "",
            default=bool(disposition.get("default")),
            forced=bool(disposition.get("forced")),
            format=_CODEC_FORMATS.get(codec, ""),
        ))
    return tuple(tracks)


def probe_subtitle_tracks(path: str) -> tuple[SubtitleTrack, ...]:
    """用 ffprobe 枚举容器内的全部字幕轨。

    :raises FileNotFoundError: 输入文件不存在
    :raises TrackProbeError: ffprobe 执行失败或输出无法解析
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ffprobe = os.environ.get("FFPROBE", "ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_tags=language,title"
        ":stream_disposition=default,forced",
        "-of", "json", path,
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "replace")[-400:]
        raise TrackProbeError(f"ffprobe 枚举字幕轨失败: {path}\n{tail}") from exc
    except OSError as exc:
        raise TrackProbeError(f"无法执行 ffprobe: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrackProbeError(f"ffprobe 输出不是合法 JSON: {path}") from exc
    return parse_tracks(data)


# ---------------------------------------------------------------------------
# 目标轨定位
# ---------------------------------------------------------------------------

def select_track(
    tracks: tuple[SubtitleTrack, ...] | list[SubtitleTrack],
    *,
    stream_index: int | None = None,
    sub_index: int | None = None,
    language: str | None = None,
    title: str | None = None,
) -> SubtitleTrack:
    """按给定条件从轨道索引表中定位唯一目标轨。

    所有给出的条件取**交集**（如同时给 ``language`` 与 ``title`` 须同时
    满足）；不给任何条件时按「唯一 default 轨 → 唯一轨」的次序取默认，
    否则视为歧义。

    :raises TrackNotFoundError: 无字幕轨或条件无匹配
    :raises TrackAmbiguousError: 条件匹配到多条轨道
    """
    tracks = tuple(tracks)
    if not tracks:
        raise TrackNotFoundError("输入不含任何内嵌字幕轨")

    criteria: list[str] = []
    matched = tracks
    if stream_index is not None:
        criteria.append(f"流索引={stream_index}")
        matched = [t for t in matched if t.stream_index == stream_index]
    if sub_index is not None:
        criteria.append(f"字幕轨序号={sub_index}")
        matched = [t for t in matched if t.sub_index == sub_index]
    if language is not None:
        want = _normalize_language(language)
        criteria.append(f"语言={language}")
        matched = [t for t in matched if t.language == want]
    if title is not None:
        want_title = title.strip().lower()
        criteria.append(f"标题={title!r}")
        matched = [t for t in matched if t.title.lower() == want_title]

    if not criteria:
        # 无条件的默认策略：唯一 default 轨优先，其次唯一的整条轨
        defaults = [t for t in tracks if t.default]
        if len(defaults) == 1:
            return defaults[0]
        if len(tracks) == 1:
            return tracks[0]
        raise TrackAmbiguousError(
            f"存在 {len(tracks)} 条字幕轨且无唯一默认轨，"
            "请用流索引/字幕轨序号/语言/标题指定目标轨",
            tracks,
        )

    if not matched:
        available = "、".join(t.describe() for t in tracks)
        raise TrackNotFoundError(
            f"没有匹配 {'，'.join(criteria)} 的字幕轨；可用轨: {available}",
            tracks,
        )
    if len(matched) > 1:
        hits = "、".join(t.describe() for t in matched)
        raise TrackAmbiguousError(
            f"{'，'.join(criteria)} 匹配到 {len(matched)} 条字幕轨"
            f"（{hits}），请增加条件收窄",
            tuple(matched),
        )
    return matched[0]


# ---------------------------------------------------------------------------
# 轨道抽取
# ---------------------------------------------------------------------------

def extract_track(input_path: str, track: SubtitleTrack, dest_dir: str) -> str:
    """用 ffmpeg 把目标轨从容器解出为临时字幕文件，返回文件路径。

    输出格式由 :attr:`SubtitleTrack.format` 决定（SRT/ASS），后续解析
    直接复用既有的 :func:`subburn.srt.parse_srt_file` /
    :func:`subburn.ass.parse_ass_file` 通道。

    :raises TrackError: 图形字幕轨（不可烧录）或 ffmpeg 抽取失败
    """
    if not track.burnable:
        raise TrackError(
            f"字幕轨 {track.describe()} 是图形字幕（codec="
            f"{track.codec_name or '未知'}），无法按文本轨烧录"
        )
    dest = os.path.join(dest_dir, f"track_{track.stream_index}.{track.format}")
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", input_path,
        "-map", f"0:{track.stream_index}",
        "-c:s", track.format,
        dest,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-400:]
        raise TrackError(
            f"抽取字幕轨 {track.describe()} 失败 "
            f"(exit {proc.returncode}):\n{tail}"
        )
    if not os.path.isfile(dest):
        raise TrackError(f"ffmpeg 未产出字幕文件: {dest}")
    return dest
