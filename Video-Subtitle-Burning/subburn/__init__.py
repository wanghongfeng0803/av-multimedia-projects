"""subburn：从零实现 SRT 解析与帧级字幕烧录的工具包。"""

from .fonts import BitmapFont, Font, PillowFont, load_font
from .pipeline import BurnResult, burn_subtitles, probe_video
from .render import OverlayRenderer, Style, blend_rgba_tile
from .srt import Cue, SRTParseError, parse_srt, parse_srt_file, parse_timecode
from .timeline import FrameTimeline, SubtitleScheduler, frame_time_ms

__all__ = [
    "Cue", "SRTParseError", "parse_srt", "parse_srt_file", "parse_timecode",
    "FrameTimeline", "SubtitleScheduler", "frame_time_ms",
    "Font", "BitmapFont", "PillowFont", "load_font",
    "Style", "OverlayRenderer", "blend_rgba_tile",
    "BurnResult", "burn_subtitles", "probe_video",
]

__version__ = "1.0.0"
