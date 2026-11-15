"""探测结果与合成结果的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class StreamInfo:
    """单条媒体流的关键信息。"""

    index: int
    codec_type: str  # video / audio / subtitle / data / attachment
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    # 帧率保留 ffprobe 的原始 "分子/分母" 字符串，由 plan.parse_fps 精确解析
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    # 采样宽高比 "宽:高"，非 1:1 时显示宽度需要做 SAR 修正
    sample_aspect_ratio: str | None = None

    @property
    def is_video(self) -> bool:
        return self.codec_type == "video"

    @property
    def is_audio(self) -> bool:
        return self.codec_type == "audio"

    @property
    def fps(self) -> Fraction | None:
        """帧率（精确有理数），优先 avg_frame_rate，回退 r_frame_rate。"""
        from .plan import parse_fps  # 延迟导入避免循环依赖

        return parse_fps(self.avg_frame_rate) or parse_fps(self.r_frame_rate)

    @property
    def display_width(self) -> float | None:
        """含 SAR 修正的显示宽度（像素）；宽度未知时返回 None。"""
        if self.width is None:
            return None
        from .plan import parse_sar

        return self.width * parse_sar(self.sample_aspect_ratio)


@dataclass
class MediaInfo:
    """ffprobe 对输入文件的探测结果。"""

    file_path: str
    format_name: str | None
    duration_seconds: float | None
    streams: list[StreamInfo]
    # 探测过程中发现但不致命的问题（容器元数据损坏、时间戳异常等）
    warnings: list[str] = field(default_factory=list)

    @property
    def video_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.is_video]

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.is_audio]

    @property
    def primary_video(self) -> StreamInfo | None:
        """返回主视频流（取第一条视频流）。"""
        videos = self.video_streams
        return videos[0] if videos else None

    @property
    def primary_audio(self) -> StreamInfo | None:
        """返回默认/第一条音频流。"""
        audios = self.audio_streams
        return audios[0] if audios else None


@dataclass
class ComposeResult:
    """一次并排合成的结果摘要。"""

    left_path: str
    right_path: str
    output_path: str
    success: bool
    width: int
    height: int
    fps: str  # 输出帧率的有理数文本，如 "30000/1001"
    duration_seconds: float
    duration_mode: str  # pad / trim
    layout: str  # auto / fit
    audio_mode: str  # mix / left / right / none
    video_codec: str | None = None
    audio_codec: str | None = None
    warnings: list[str] = field(default_factory=list)
    ffmpeg_log_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "left_path": self.left_path,
            "right_path": self.right_path,
            "output_path": self.output_path,
            "success": self.success,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "duration_mode": self.duration_mode,
            "layout": self.layout,
            "audio_mode": self.audio_mode,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "warnings": self.warnings,
            "ffmpeg_log_path": self.ffmpeg_log_path,
        }
