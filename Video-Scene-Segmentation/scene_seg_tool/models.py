"""场景分割的结果数据模型。

全部为不可变 dataclass，字段只承载确定性数据（帧号、时间、距离、阈值），
不含任何运行时刻时间戳，保证同一输入多次运行产生逐字节相同的序列化结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FrameBoundary", "SceneCut", "SegmentationResult"]


@dataclass(frozen=True)
class FrameBoundary:
    """一条相邻帧边界 ``frame_index - 1 -> frame_index`` 的完整记录。"""

    #: 边界后的帧下标（从 1 开始；0 号帧之前不存在边界）
    frame_index: int
    #: 边界时间戳（秒），由帧率换算，VFR 视频为近似值
    time_seconds: float
    #: 整数 L1 直方图距离（未归一化）
    raw_distance: int
    #: 归一化差异，范围 [0.0, 1.0]
    score: float
    #: 三帧回返距离（|h[t-1]-h[t+1]|），无法构成三元组时为 -1
    return_distance: int
    #: 最终判定标签：cut / flash / below_threshold / suppressed
    decision: str


@dataclass(frozen=True)
class SceneCut:
    """一个被确认的场景硬切点。"""

    #: 切点边界编号（新场景第一帧的帧下标）
    frame_index: int
    time_seconds: float
    score: float
    raw_distance: int


@dataclass(frozen=True)
class SegmentationResult:
    """一次完整场景分割的全部产出。"""

    #: 输入视频路径（原样保留调用方传入的字符串）
    input_path: str
    #: 实际参与统计的解码帧数
    frame_count: int
    width: int
    height: int
    #: 时间戳换算所用帧率（帧/秒）
    fps: float
    #: 帧率来源：ffprobe 字段名或内置兜底标记
    fps_source: str
    #: 最终用于判定的归一化阈值
    threshold: float
    #: 阈值来源：robust / fallback
    threshold_source: str
    #: 完整差异序列（长度 frame_count - 1），单帧视频为空列表
    boundaries: list[FrameBoundary] = field(default_factory=list)
    #: 确认的硬切点（已做闪光抑制与最小场景长度 NMS）
    cuts: list[SceneCut] = field(default_factory=list)
    #: 被闪光抑制的边界编号
    flash_boundaries: tuple[int, ...] = ()
    #: 被最小场景长度 NMS 抑制的边界编号
    suppressed_boundaries: tuple[int, ...] = ()

    @property
    def pixel_count(self) -> int:
        return self.width * self.height
