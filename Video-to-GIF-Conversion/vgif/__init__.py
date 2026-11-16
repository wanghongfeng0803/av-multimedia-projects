"""vgif：视频转 GIF（零三方依赖），抽帧 / 量化 / 编码三步严格解耦。

设计总览：

- **抽帧（extract）**：ffmpeg 仅负责解码，逐帧 1:1 直通输出 rawvideo
  rgb24，帧的真实时间戳由 ffprobe 单独探测写入 ``manifest.json``。
  抽帧产物与后续帧率换算、调色板完全无关，可被任意目标帧率 / 尺寸 /
  色深重复复用。
- **量化（quantize）**：纯 Python 中位切分（median-cut）生成全局调色板，
  最近邻颜色映射为索引帧。缩放（双线性，RGB 域）与帧率采样（端点锚定
  Bresenham 取帧）都是纯函数，量化结果可在索引域做最近邻快速缩放复用。
- **编码（encode）**：纯 Python LZW 压缩 + GIF89a 字节封装，帧延迟以
  厘秒（1/100 秒）为单位按时间规划逐帧写出，不依赖 ffmpeg gif 编码器，
  因此不会被调色板滤镜或 fps 滤镜引入任何时序漂移。

时序口径（关键）：GIF 图形控制扩展里的 delay 只能是厘秒整数，本包以
「源时长 / 输出帧数」得到基值与余数，用 Bresenham 把余数均匀撒到各帧，
首尾帧端点锚定，总时长误差不超过 1 厘秒。
"""

from .timing import (
    MIN_GIF_DELAY_CS,
    TimingError,
    frame_pick_indices,
    plan_gif_delays,
)
from .extract import ExtractError, probe_frames, extract_rgb_frames
from .quantize import (
    MAX_GIF_COLORS,
    QuantizeError,
    median_cut_palette,
    map_frame_to_index,
    quantize_frames,
)
from .scale import bilinear_resize, nearest_resize_indexed
from .sampling import sample_frames, sample_indexed_frames
from .gif_writer import GifEncodeError, write_gif
from .pipeline import (
    DEFAULT_FPS,
    DEFAULT_MAX_WIDTH,
    DEFAULT_COLORS,
    ConversionResult,
    convert_to_gif,
)

__all__ = [
    # timing
    "MIN_GIF_DELAY_CS",
    "TimingError",
    "frame_pick_indices",
    "plan_gif_delays",
    # extract
    "ExtractError",
    "probe_frames",
    "extract_rgb_frames",
    # quantize
    "MAX_GIF_COLORS",
    "QuantizeError",
    "median_cut_palette",
    "map_frame_to_index",
    "quantize_frames",
    # scale
    "bilinear_resize",
    "nearest_resize_indexed",
    # sampling
    "sample_frames",
    "sample_indexed_frames",
    # encode
    "GifEncodeError",
    "write_gif",
    # pipeline
    "DEFAULT_FPS",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_COLORS",
    "ConversionResult",
    "convert_to_gif",
]
