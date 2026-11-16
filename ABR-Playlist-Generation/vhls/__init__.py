"""vhls：单码率 HLS 打包（零三方依赖），探测/规划/切割/列表四步解耦。

设计总览：

- **探测（probe）**：ffprobe 读取逐帧 PTS/时长（VFR 也成立）与音频
  存在性，时间戳以探测值为准，不由 ffmpeg 切割管线反推。
- **规划（timeline）**：纯函数模块，入口把浮点秒一次性换算为整数毫秒，
  之后所有切点/比较/求和全程整数运算；切点只落在真实帧边缘。产出的
  ``TimelinePlan`` 是切割与播放列表共用的**唯一时间轴**。
- **切割（cut）**：ffmpeg 重编码帧精确切片（``-ss/-t`` + 片首强制
  IDR），切完对每片独立 ffprobe 实测时长，超容差即失败，杜绝声明值
  与真实文件对不上。
- **列表（playlist）**：EXTINF 写实测毫秒的 3 位小数字符串，
  TARGETDURATION 取实测上限 ceil，序列号连续；校验通过后临时文件 +
  ``os.replace`` 原子落盘。

时序口径（关键）：内部时间单位是整数毫秒，浮点只在两个边界出现——
入口的 PTS 换算与出口的 EXTINF 格式化，因此不存在跨片浮点累加，
末片起点不会漂移；末片不足目标时长时由计划 ``is_short`` 标记，
时长仍按真实值准确写出。
"""

from .probe import (
    ProbeError,
    FrameInfo,
    SourceInfo,
    probe_source,
)
from .timeline import (
    MS_PER_SEC,
    TimelineError,
    SegmentSpec,
    TimelinePlan,
    build_timeline,
    ceil_div,
    format_extinf,
    ms_to_seconds,
)
from .cut import (
    DEFAULT_DURATION_TOLERANCE_MS,
    CutError,
    SegmentResult,
    cut_segments,
    probe_segment_duration_ms,
)
from .playlist import (
    PLAYLIST_VERSION,
    PlaylistError,
    PlaylistResult,
    render_playlist,
    validate_against_plan,
    write_playlist,
)
from .pipeline import (
    DEFAULT_TARGET_SECONDS,
    PackError,
    PackResult,
    pack_hls,
)

__all__ = [
    # probe
    "ProbeError",
    "FrameInfo",
    "SourceInfo",
    "probe_source",
    # timeline
    "MS_PER_SEC",
    "TimelineError",
    "SegmentSpec",
    "TimelinePlan",
    "build_timeline",
    "ceil_div",
    "format_extinf",
    "ms_to_seconds",
    # cut
    "DEFAULT_DURATION_TOLERANCE_MS",
    "CutError",
    "SegmentResult",
    "cut_segments",
    "probe_segment_duration_ms",
    # playlist
    "PLAYLIST_VERSION",
    "PlaylistError",
    "PlaylistResult",
    "render_playlist",
    "validate_against_plan",
    "write_playlist",
    # pipeline
    "DEFAULT_TARGET_SECONDS",
    "PackError",
    "PackResult",
    "pack_hls",
]
