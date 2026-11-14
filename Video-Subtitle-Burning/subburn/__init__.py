"""subburn：从零实现 SRT/ASS 解析与帧级字幕烧录的工具包。"""

from .ass import (
    ASSCue,
    ASSColor,
    ASSDocument,
    ASSLine,
    ASSParseError,
    ASSRun,
    ASSStyle,
    parse_ass,
    parse_ass_file,
    parse_ass_time,
)
from .ass_render import ASSRenderer, numpad_align
from .fonts import (
    BitmapFont,
    Font,
    FontRegistry,
    PillowFont,
    load_font,
    resolve_font,
)
from .pipeline import BurnResult, burn_ass, burn_subtitles, burn_track, probe_video
from .render import OverlayRenderer, Style, blend_rgba_tile
from .srt import Cue, SRTParseError, parse_srt, parse_srt_file, parse_timecode
from .style import (
    Color,
    FontSpec,
    FontWeight,
    HAlign,
    Margins,
    Position,
    SubtitleStyle,
    VAlign,
)
from .timeline import FrameTimeline, SubtitleScheduler, frame_time_ms
from .tracks import (
    SubtitleTrack,
    TrackAmbiguousError,
    TrackError,
    TrackNotFoundError,
    TrackProbeError,
    TrackSelectionError,
    extract_track,
    parse_tracks,
    probe_subtitle_tracks,
    select_track,
)

__all__ = [
    # SRT
    "Cue", "SRTParseError", "parse_srt", "parse_srt_file", "parse_timecode",
    # ASS
    "ASSParseError", "ASSColor", "ASSStyle", "ASSRun", "ASSLine",
    "ASSCue", "ASSDocument", "parse_ass_time", "parse_ass", "parse_ass_file",
    "ASSRenderer", "numpad_align",
    # 时间轴
    "FrameTimeline", "SubtitleScheduler", "frame_time_ms",
    # 字体
    "Font", "BitmapFont", "PillowFont", "FontRegistry",
    "load_font", "resolve_font",
    # 渲染（SRT 路径）
    "Style", "OverlayRenderer", "blend_rgba_tile",
    # 样式
    "Color", "FontSpec", "FontWeight", "HAlign", "VAlign",
    "Margins", "Position", "SubtitleStyle",
    # 管道
    "BurnResult", "burn_subtitles", "burn_ass", "burn_track", "probe_video",
    # 内嵌多轨选择
    "SubtitleTrack", "TrackError", "TrackProbeError", "TrackSelectionError",
    "TrackNotFoundError", "TrackAmbiguousError",
    "parse_tracks", "probe_subtitle_tracks", "select_track", "extract_track",
]

__version__ = "1.3.0"
