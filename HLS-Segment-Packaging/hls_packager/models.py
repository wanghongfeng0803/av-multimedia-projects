"""核心数据模型。

全部使用不可变 dataclass 描述，模块之间只通过这些结构传递数据，
避免字典 key 散落各处、方便测试构造。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamInfo:
    """单条流（轨道）的探测结果。"""

    index: int
    codec_type: str          # "video" / "audio" / "subtitle" / "data" ...
    codec_name: str          # ffprobe codec 名，如 h264 / aac / hevc
    codec_tag: str = ""      # 容器内 fourcc，如 avc1
    language: str = "und"
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None   # 平均帧率（fps），无法判定时为 None
    sample_rate: int | None = None
    channels: int | None = None

    @property
    def is_video(self) -> bool:
        return self.codec_type == "video"

    @property
    def is_audio(self) -> bool:
        return self.codec_type == "audio"


@dataclass(frozen=True)
class MediaInfo:
    """整个输入文件的探测结果。"""

    path: str
    format_name: str
    duration: float                    # 容器级时长（秒）
    bit_rate: int | None
    streams: tuple[StreamInfo, ...]
    #: 视频流全部关键帧的显示时间（秒），升序去重；纯音频输入为空元组
    keyframe_times: tuple[float, ...] = ()

    @property
    def video_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.is_video]

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.is_audio]

    @property
    def has_video(self) -> bool:
        return bool(self.video_streams)


@dataclass(frozen=True)
class SegmentInfo:
    """单个 HLS 分片的规划与产出信息。"""

    sequence_number: int       # EXT-X-MEDIA-SEQUENCE 视角下的绝对序号
    start_time: float          # 输出时间轴上的起点（秒），跨片连续、从 0 累计
    duration: float            # 分片实际时长（秒）
    filename: str              # 相对输出目录的文件名，如 seg_0000.ts
    source_start: float = 0.0  # 该分片在源文件中的起点（秒）

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass(frozen=True)
class PackagingResult:
    """一次完整打包的汇总结果。"""

    output_dir: str
    playlist_path: str
    segments: tuple[SegmentInfo, ...]
    media: MediaInfo
    target_duration: int
    media_sequence: int
    #: 滑动窗口大小；None 表示 VOD（全量列表 + ENDLIST）
    window_size: int | None = None
    extra: dict = field(default_factory=dict)
