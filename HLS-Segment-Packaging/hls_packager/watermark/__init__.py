"""文字水印能力：自行完成文字栅格化与视频叠加。

模块划分（数据单向流动，全部仅依赖 Python 标准库）：

``utf8``         字节流 -> Unicode 码点序列（自研 UTF-8 解码器）
``bitmap_font``  零依赖内置 5x7 点阵字体（字体彻底缺失时的最终回退）
``ttf_font``     自研 TrueType 解析器（cmap/glyf/hmtx ...）+ 轮廓提取
``rasterizer``   扫描线 + 4x 超采样抗锯齿的字形栅格化器
``font_stack``   字体发现与按码点回退链（显式字体 -> 系统字体 -> 内置点阵）
``rasterize``    码点序列 -> RGBA 位图排版绘制 -> PNG 文件
``png_buf``      内存 RGBA 位图与自研 PNG 编码
``escaping``     过滤器路径转义与 shell/argv 引号包裹（注入防护）
``overlay``      叠加定位、ffmpeg overlay 命令构造
``config``       不可变水印配置与参数校验
"""

from __future__ import annotations

from ..errors import WatermarkError
from .config import WatermarkConfig
from .escaping import escape_filter_path, render_shell_command, shell_quote
from .overlay import (
    build_watermarked_segment_command,
    overlay_position,
)
from .rasterize import render_text_png
from .utf8 import decode_utf8

__all__ = [
    "WatermarkError",
    "WatermarkConfig",
    "render_text_png",
    "decode_utf8",
    "overlay_position",
    "escape_filter_path",
    "shell_quote",
    "render_shell_command",
    "build_watermarked_segment_command",
]
