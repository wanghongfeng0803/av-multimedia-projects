"""键色采样：从输入画面里估计幕布键色，而非写死绿色。

采样模式
========
- 矩形区域（默认）：``rect``，按归一化坐标 ``[x0, y0, x1, y1]``
  （0–1，左上原点）在若干采样帧的该区域内统计像素；
- 画面比例区域：``topleft`` / ``top`` / ``topright`` / ``left`` /
  ``right`` / ``bottomleft`` / ``bottom`` / ``bottomright``，
  取画面边缘固定比例的条带，幕布通常铺满边缘，主体很少顶到边；
- 全帧：``all``，整帧参与统计（适合纯色测试素材）。

统计量
======
多帧、多像素汇总后取**每通道直方图中位数**而非均值：中位数对
前景误入采样区、压缩噪点、幕布上的阴影都更稳健。采样像素总数
低于下限时抛 :class:`SampleError`，不会静默拿噪声当键色。

帧抽取
======
通过 ``ffmpeg -vf select='eq(n,..)'`` 只解码指定帧号，输出
rgb24 裸流，避免整段视频载入内存。帧号越界（短视频）自动跳过，
全部越界才报错。
"""

from __future__ import annotations

import subprocess

from .errors import FFmpegRunError, SampleError
from .probe import find_tool, probe_video

__all__ = [
    "REGION_MODES",
    "DEFAULT_REGION",
    "DEFAULT_SAMPLE_FRAMES",
    "region_pixels",
    "median_rgb",
    "sample_key_color",
]

#: 边缘条带占画面宽/高的比例
_EDGE_RATIO = 0.1

#: 区域模式名 → 归一化矩形 (x0, y0, x1, y1)
_REGION_RECTS = {
    "all": (0.0, 0.0, 1.0, 1.0),
    "topleft": (0.0, 0.0, _EDGE_RATIO, _EDGE_RATIO),
    "top": (0.0, 0.0, 1.0, _EDGE_RATIO),
    "topright": (1.0 - _EDGE_RATIO, 0.0, 1.0, _EDGE_RATIO),
    "left": (0.0, 0.0, _EDGE_RATIO, 1.0),
    "right": (1.0 - _EDGE_RATIO, 0.0, 1.0, 1.0),
    "bottomleft": (0.0, 1.0 - _EDGE_RATIO, _EDGE_RATIO, 1.0),
    "bottom": (0.0, 1.0 - _EDGE_RATIO, 1.0, 1.0),
    "bottomright": (1.0 - _EDGE_RATIO, 1.0 - _EDGE_RATIO, 1.0, 1.0),
}

#: 全部内置区域模式（``rect`` 为显式矩形，不在此表内单独处理）
REGION_MODES = tuple(_REGION_RECTS) + ("rect",)

#: 默认采样区域：画面顶部条带（绿/蓝幕通常铺满上部）
DEFAULT_REGION = "top"

#: 默认采样帧号（首帧 + 中段 + 尾段的代表，0 基）
DEFAULT_SAMPLE_FRAMES = (0, 30, 60)

#: 采样像素数下限，低于它认为采样不可信
_MIN_SAMPLE_PIXELS = 16


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def region_pixels(width, height, mode, rect=None):
    """把区域模式解析成像素坐标包围盒 ``(x0, y0, x1, y1)``（含端点）。

    ``rect`` 模式下 ``rect`` 须给出 4 个归一化坐标；其余模式使用
    :data:`_REGION_RECTS` 内置条带。返回的坐标裁在帧内并保证
    宽高至少 1 像素。
    """
    if mode == "rect":
        if not rect or len(rect) != 4:
            raise SampleError("rect 模式需要 4 个归一化坐标 [x0,y0,x1,y1]")
        nx0, ny0, nx1, ny1 = (_clamp01(v) for v in rect)
        if nx1 <= nx0 or ny1 <= ny0:
            raise SampleError("rect 区域宽高必须为正（x1>x0 且 y1>y0）")
    else:
        if mode not in _REGION_RECTS:
            raise SampleError(
                f"未知采样区域 {mode!r}，可选：{', '.join(REGION_MODES)}"
            )
        nx0, ny0, nx1, ny1 = _REGION_RECTS[mode]
    x0 = int(nx0 * width)
    y0 = int(ny0 * height)
    x1 = max(int(nx1 * width), x0 + 1)
    y1 = max(int(ny1 * height), y0 + 1)
    # 右端点含在采样范围内，钳到帧内最后一个像素
    return x0, y0, min(x1, width), min(y1, height)


def median_rgb(pixels) -> tuple[int, int, int]:
    """对 ``(R, G, B)`` 像素序列求每通道直方图中位数。

    用 256 桶计数而非排序，像素量大时复杂度 O(n+256)；偶数个
    样本取中间两值的低者（与标准库 ``statistics.median_low`` 同
    语义，下标 (n-1)//2），结果可复现、无浮点歧义。
    """
    hist = [[0] * 256 for _ in range(3)]
    n = 0
    for r, g, b in pixels:
        hist[0][r] += 1
        hist[1][g] += 1
        hist[2][b] += 1
        n += 1
    if n == 0:
        raise SampleError("采样区域内没有任何像素")
    # median_low：与标准库 statistics.median_low 同语义——
    # 奇数 n 取正中（下标 n//2），偶数 n 取中间两值的低者
    # （下标 (n-1)//2），统一下标 (n-1)//2。
    target_index = (n - 1) // 2
    result = []
    for c in range(3):
        seen = 0
        for value, count in enumerate(hist[c]):
            seen += count
            if seen >= target_index + 1:
                result.append(value)
                break
    return result[0], result[1], result[2]


def _decode_sample_frames(src, frame_numbers, width, height):
    """用 ffmpeg 只解码指定帧号为 rgb24 裸帧，返回帧字节列表。

    越界帧号自动缺失（短视频帧数不足时 ffmpeg 少输出帧），
    由调用方按实际拿到的帧数判断是否足够。
    """
    ffmpeg = find_tool("ffmpeg")
    frame_set = sorted({int(n) for n in frame_numbers if int(n) >= 0})
    if not frame_set:
        raise SampleError("没有有效的采样帧号")
    select_expr = "+".join(f"eq(n\\,{n})" for n in frame_set)
    cmd = [
        ffmpeg, "-v", "error", "-i", str(src),
        "-vf", f"select='{select_expr}'",
        # ffmpeg 9 已移除 -vsync，帧选择后必须用 -fps_mode passthrough
        # 透传选中帧，否则默认 CFR 模式会补帧/丢帧打乱采样
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegRunError(
            "ffmpeg 采样解码失败",
            stderr=proc.stderr.decode("utf-8", "replace"),
            cmd=cmd,
        )
    frame_size = width * height * 3
    usable = len(proc.stdout) // frame_size
    if usable == 0:
        return []
    return [
        proc.stdout[i * frame_size:(i + 1) * frame_size]
        for i in range(usable)
    ]


def sample_key_color(src, mode=DEFAULT_REGION, rect=None,
                     frame_numbers=DEFAULT_SAMPLE_FRAMES):
    """从 ``src`` 采样键色，返回 ``(R, G, B)`` 三元组。

    ``mode`` 见 :data:`REGION_MODES`；``rect`` 仅 rect 模式使用；
    ``frame_numbers`` 为 0 基帧号序列，越界帧自动跳过。采样像素
    少于 :data:`_MIN_SAMPLE_PIXELS` 时抛 :class:`SampleError`。
    """
    info = probe_video(src)
    w, h = info["width"], info["height"]
    x0, y0, x1, y1 = region_pixels(w, h, mode, rect)
    frames = _decode_sample_frames(src, frame_numbers, w, h)
    if not frames:
        raise SampleError(
            f"采样帧 {list(frame_numbers)} 均超出视频实际帧数，无法采样键色"
        )

    pixels = []
    for frame in frames:
        for y in range(y0, y1):
            row_start = (y * w + x0) * 3
            row_end = (y * w + x1) * 3
            row = frame[row_start:row_end]
            for p in range(0, len(row), 3):
                pixels.append((row[p], row[p + 1], row[p + 2]))
    if len(pixels) < _MIN_SAMPLE_PIXELS:
        raise SampleError(
            f"采样像素仅 {len(pixels)} 个，低于下限 {_MIN_SAMPLE_PIXELS}"
        )
    return median_rgb(pixels)
