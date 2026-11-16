"""视频人脸/隐私区域马赛克（打码）工具包。

三模块解耦：

- :mod:`vblur.mosaic` —— 打码：全局块网格上的一次性马赛克，
  :class:`BlockTable` 是三模块间唯一数据契约；
- :mod:`vblur.regions` —— 区域管理：钳制（负尺寸/越界）、线程安全
  的运行时增删、代际缓存的逐帧块表源；
- :mod:`vblur.pipeline` —— 编码：ffmpeg 解码→逐帧消费块表→编码，
  输出帧率/尺寸/像素格式/时间戳与输入一致，音轨直通。
"""

from .mosaic import BlockTable, apply_mosaic, build_block_table
from .pipeline import (
    DEFAULT_BLOCK,
    PipelineError,
    blur_video,
    pix_fmt_matches,
    probe,
)
from .regions import Region, RegionManager, clamp_region

__all__ = [
    "DEFAULT_BLOCK",
    "BlockTable",
    "PipelineError",
    "Region",
    "RegionManager",
    "apply_mosaic",
    "blur_video",
    "build_block_table",
    "clamp_region",
    "pix_fmt_matches",
    "probe",
]
