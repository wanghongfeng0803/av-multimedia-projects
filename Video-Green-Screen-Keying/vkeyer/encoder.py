"""第三层：输出编码与背景替换（encoder / composite）。

本层不再关心键色与遮罩算法，只接收 RGBA 裸帧，负责两件事：

1. 决定 ffmpeg 编码参数：哪些容器/编码器能承载 alpha，图像序列
   如何落盘，音轨如何直通；
2. 背景替换：把 RGBA 前景合成到纯色或背景素材上，输出无 alpha
   的 rgb24 帧（复用前两层产出的 alpha，本层不重新键控）。

alpha 承载能力（本机 ffmpeg 9 实测可用的编码器）
=================================================
- ``.mov`` + ``qtrle``：QuickTime Animation，无损、支持 RGBA，
  合成/剪辑工作流的通用交换格式（默认）；
- ``.mkv`` + ``ffv1``：FFV1 无损，内部走 rgba，ffprobe 报告为
  ``bgra``（与 rgba 仅字节序不同，alpha 完整保留，已实测）；
- ``.webm`` + ``libvpx-vp9``：标称支持 yuva420p，但**本机
  ffmpeg 9 的 libvpx 构建会静默剥离 alpha**（编码日志声明
  yuva420p、产物却落 yuv420p，lossless / auto-alt-ref / gop
  各参数组合均无效）。本工具保留该出口并在结果 ``messages``
  给出明确警告；需要可靠透明网页视频请用 APNG 或 mov；
- 图像序列：``.png``（PNG 自带 alpha）、``.webp``（无损模式），
  输出路径含 ``%06d`` 等 ffmpeg 序列模式；
- ``.mp4`` **不能**承载 alpha：alpha 模式下直接抛
  :class:`OutputFormatError`，提示改用 mov/webm/mkv 或走背景
  替换（mp4 + 纯色/素材背景是合法的 rgb24 输出）。
"""

from __future__ import annotations

import os

from .errors import OutputFormatError

__all__ = [
    "IMAGE_SEQUENCE_EXTS",
    "ALPHA_VIDEO_SPECS",
    "ALPHA_UNRELIABLE_WARNING",
    "is_image_sequence",
    "resolve_video_spec",
    "composite_over_solid",
    "composite_over_frame",
    "composite_rgba_frame",
]

#: 支持 alpha 的图像序列扩展名
IMAGE_SEQUENCE_EXTS = (".png", ".webp")

#: 扩展名 → (编码器, 像素格式或 None, 额外参数)
ALPHA_VIDEO_SPECS = {
    ".mov": ("qtrle", "argb", []),
    ".mkv": ("ffv1", "rgba", []),
    # 见模块 docstring：本机构建 VP9 alpha 不可靠，保留出口但发警告
    ".webm": ("libvpx-vp9", "yuva420p",
              ["-lossless", "1", "-b:v", "0", "-auto-alt-ref", "0"]),
}

#: 选中不可靠 alpha 容器时附加到结果 messages 的警告
ALPHA_UNRELIABLE_WARNING = (
    "当前 ffmpeg 构建的 libvpx-vp9 会静默剥离 alpha，"
    "webm 产物可能不含透明通道；可靠透明输出请用 .mov(qtrle)、"
    ".mkv(ffv1) 或 .png/.apng 序列"
)

#: 背景替换（无 alpha 视频）的默认编码规格
RGB_VIDEO_SPECS = {
    ".mp4": ("libx264", "yuv420p", ["-crf", "18", "-preset", "medium"]),
    ".mov": ("libx264", "yuv420p", ["-crf", "18", "-preset", "medium"]),
    ".mkv": ("libx264", "yuv420p", ["-crf", "18", "-preset", "medium"]),
    ".webm": ("libvpx-vp9", "yuv420p", ["-crf", "32", "-b:v", "0"]),
}

#: ffmpeg 图像序列通配（%06d 之类）
_SEQ_MARK = "%"


def is_image_sequence(path) -> bool:
    """路径是否为图像序列模式（含 ``%`` 帧号通配，扩展名为 png/webp）。"""
    name = os.fspath(path)
    base = os.path.basename(name)
    if _SEQ_MARK not in base:
        return False
    ext = os.path.splitext(base)[1].lower()
    if ext not in IMAGE_SEQUENCE_EXTS:
        raise OutputFormatError(
            f"图像序列仅支持 {', '.join(IMAGE_SEQUENCE_EXTS)}，得到 {ext}"
        )
    return True


def resolve_video_spec(path, alpha: bool):
    """按输出扩展名解析 ``(编码器, 像素格式, 额外参数)``。

    ``alpha=True`` 只接受 :data:`ALPHA_VIDEO_SPECS` 中的容器；
    ``alpha=False``（背景替换）接受 :data:`RGB_VIDEO_SPECS`。
    """
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if alpha:
        if ext not in ALPHA_VIDEO_SPECS:
            raise OutputFormatError(
                f"{ext} 容器无法承载 alpha 通道，请改用 "
                f"{', '.join(ALPHA_VIDEO_SPECS)} 输出透明视频，"
                "或使用 --bg-color/--bg-image 做背景替换后输出普通视频"
            )
        return ALPHA_VIDEO_SPECS[ext]
    if ext not in RGB_VIDEO_SPECS:
        raise OutputFormatError(
            f"不支持的输出容器 {ext}，可选：{', '.join(RGB_VIDEO_SPECS)}"
        )
    return RGB_VIDEO_SPECS[ext]


def composite_rgba_frame(fg_rgba, bg_rgb, width, height):
    """单帧 alpha 合成（source-over，定点运算）。

    ``fg_rgba`` 为 width*height*4 的 RGBA 前景，``bg_rgb`` 为
    width*height*3 的 RGB 背景（尺寸已由调用方对齐）。输出
    width*height*3 的 rgb24 字节。公式::

        out = fg*a + bg*(255-a)，各通道 (x*a + bg*(255-a) + 127)//255

    全透明像素结果精确等于背景，全不透明精确等于前景 RGB——
    两端无中间舍入误差，是本函数的锁定基线。
    """
    n = width * height
    if len(fg_rgba) != n * 4:
        raise ValueError("前景 RGBA 帧长度与尺寸不符")
    if len(bg_rgb) != n * 3:
        raise ValueError("背景 RGB 帧长度与尺寸不符")
    out = bytearray(n * 3)
    for i in range(n):
        a = fg_rgba[i * 4 + 3]
        if a == 0:
            out[i * 3] = bg_rgb[i * 3]
            out[i * 3 + 1] = bg_rgb[i * 3 + 1]
            out[i * 3 + 2] = bg_rgb[i * 3 + 2]
            continue
        if a == 255:
            out[i * 3] = fg_rgba[i * 4]
            out[i * 3 + 1] = fg_rgba[i * 4 + 1]
            out[i * 3 + 2] = fg_rgba[i * 4 + 2]
            continue
        inv = 255 - a
        for c in range(3):
            fv = fg_rgba[i * 4 + c]
            bv = bg_rgb[i * 3 + c]
            out[i * 3 + c] = (fv * a + bv * inv + 127) // 255
    return bytes(out)


def composite_over_solid(fg_rgba, width, height, rgb):
    """把 RGBA 前景合成到纯色背景上，返回 rgb24 帧字节。"""
    r, g, b = rgb
    bg = bytearray((r, g, b)) * (width * height)
    return composite_rgba_frame(fg_rgba, bytes(bg), width, height)


def composite_over_frame(fg_rgba, bg_rgb, width, height):
    """把 RGBA 前景合成到背景素材帧上（:func:`composite_rgba_frame`
    的语义化别名，供流水线按背景类型分派时阅读）。"""
    return composite_rgba_frame(fg_rgba, bg_rgb, width, height)
