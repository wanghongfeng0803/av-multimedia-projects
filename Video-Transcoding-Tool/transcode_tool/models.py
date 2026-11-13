"""探测结果与转码结果的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    @property
    def is_video(self) -> bool:
        return self.codec_type == "video"

    @property
    def is_audio(self) -> bool:
        return self.codec_type == "audio"


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
class TranscodeResult:
    """一次转码的结果摘要。"""

    input_path: str
    output_path: str
    success: bool
    video_codec: str | None
    audio_codec: str | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    warnings: list[str] = field(default_factory=list)
    ffmpeg_log_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "success": self.success,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "warnings": self.warnings,
            "ffmpeg_log_path": self.ffmpeg_log_path,
        }
