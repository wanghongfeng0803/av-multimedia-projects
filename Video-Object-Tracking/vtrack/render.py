"""轨迹渲染：读取轨迹 JSON，在视频帧上画框与状态标记后重新编码。

渲染与追踪完全解耦——本模块不做任何匹配，只消费
:class:`vtrack.model.TrackResult`，因此同一条轨迹可以反复换样式重渲染，
后续光流追踪、多目标追踪只要产出同一 JSON 契约即可共用本模块。

绘图全部在 rgb24 打包字节平面上手工完成（零三方依赖）：

- 确认帧画绿色矩形框 + 分数文字；
- 丢失帧不画跟踪框（避免把「最后已知位置」误呈现为当前位置），
  左上角画红色 ``LOST`` 与连续丢失计数；``show_last_known=True`` 时
  额外画黄色虚线框标出最后已知位置，语义与确认框刻意区分；
- 文字用 :mod:`vtrack.font` 的内置 5×7 点阵，按 ``font_scale`` 整数放大，
  带 1 像素深色描边，保证在亮/暗背景上都可读。
"""

import os
import subprocess

from .font import GLYPH_H, render_text_mask
from .io_utils import (
    PipelineError,
    find_tool,
    probe,
    read_exact,
)
from .model import LOST, TRACKED, load_track_json

__all__ = [
    "COLOR_GREEN",
    "COLOR_RED",
    "COLOR_YELLOW",
    "COLOR_WHITE",
    "COLOR_BLACK",
    "draw_rect",
    "draw_dashed_rect",
    "draw_text",
    "draw_frame_overlay",
    "render_video",
]

#: rgb24 三元组
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (230, 40, 40)
COLOR_YELLOW = (220, 200, 0)
COLOR_WHITE = (240, 240, 240)
COLOR_BLACK = (0, 0, 0)


def _set_pixel(frame, width, height, x, y, color):
    """写一个 rgb24 像素，越界静默跳过（画框边缘由调用方夹取）。"""
    if 0 <= x < width and 0 <= y < height:
        base = (y * width + x) * 3
        frame[base] = color[0]
        frame[base + 1] = color[1]
        frame[base + 2] = color[2]


def draw_rect(frame, width, height, x, y, rw, rh, color, thickness=2):
    """画实心描边矩形（不填充内部），坐标自动夹到帧内。"""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + rw)
    y1 = min(height, y + rh)
    for t in range(thickness):
        for px in range(x0, x1):
            _set_pixel(frame, width, height, px, y0 + t, color)
            _set_pixel(frame, width, height, px, y1 - 1 - t, color)
        for py in range(y0, y1):
            _set_pixel(frame, width, height, x0 + t, py, color)
            _set_pixel(frame, width, height, x1 - 1 - t, py, color)


def draw_dashed_rect(frame, width, height, x, y, rw, rh, color,
                     dash=4, thickness=1):
    """画虚线矩形（每 dash 像素交替点亮/留白），用于最后已知位置。"""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + rw)
    y1 = min(height, y + rh)

    def on(pos):
        return (pos // dash) % 2 == 0

    for t in range(thickness):
        for px in range(x0, x1):
            if on(px - x0):
                _set_pixel(frame, width, height, px, y0 + t, color)
                _set_pixel(frame, width, height, px, y1 - 1 - t, color)
        for py in range(y0, y1):
            if on(py - y0):
                _set_pixel(frame, width, height, x0 + t, py, color)
                _set_pixel(frame, width, height, x1 - 1 - t, py, color)


def draw_text(frame, width, height, x, y, text, color,
              font_scale=1, bg=COLOR_BLACK):
    """在 (x, y) 处画点阵文字，带 1 像素（缩放后）深色描边。

    ``font_scale`` 为整数放大倍数，掩码点放大为 scale×scale 方块。
    两遍绘制：先收集全部放大后的前景坐标，描边环只写非前景位置
    （否则后画像素的描边会覆盖相邻已画前景），最后统一写前景色。
    超出右/下边界的字形部分由 :func:`_set_pixel` 裁掉。
    """
    mw, mh, mask = render_text_mask(text)
    if mw == 0:
        return
    outline = max(1, font_scale)
    fg = set()
    for row in range(mh):
        for col in range(mw):
            if not mask[row * mw + col]:
                continue
            for sy in range(font_scale):
                for sx in range(font_scale):
                    fg.add((col * font_scale + sx,
                            row * font_scale + sy))
    # 第一遍：描边环（前景块外扩 outline），跳过前景坐标
    for fx, fy in fg:
        for ox in range(-outline, font_scale + outline):
            for oy in range(-outline, font_scale + outline):
                cx, cy = fx + ox, fy + oy
                if (cx, cy) in fg:
                    continue
                _set_pixel(frame, width, height, x + cx, y + cy, bg)
    # 第二遍：前景
    for fx, fy in fg:
        _set_pixel(frame, width, height, x + fx, y + fy, color)


def _format_score(score, method):
    """按方法习惯格式化分数文字。"""
    if method == "zncc":
        return f"{score:.3f}"
    return f"{score:.1f}"


def draw_frame_overlay(frame, width, height, record, method,
                       font_scale=1, show_last_known=False,
                       lost_count=0):
    """按一帧 :class:`~vtrack.model.FrameRecord` 画叠加层。

    - tracked：绿框 + 框上方分数；
    - lost：左上红色状态行；可选黄色虚线最后已知框。
    """
    x, y, rw, rh = record.bbox
    if record.status == TRACKED:
        draw_rect(frame, width, height, x, y, rw, rh,
                  COLOR_GREEN, thickness=2)
        label = _format_score(record.score, method)
        draw_text(frame, width, height, x,
                  max(0, y - (GLYPH_H + 2) * font_scale),
                  label, COLOR_GREEN, font_scale=font_scale)
    else:
        status = f"LOST {lost_count}" if lost_count > 0 else "LOST"
        draw_text(frame, width, height, 2, 2, status, COLOR_RED,
                  font_scale=font_scale)
        if show_last_known:
            draw_dashed_rect(frame, width, height, x, y, rw, rh,
                             COLOR_YELLOW)


def render_video(src, dst, track_path, font_scale=1,
                 show_last_known=False, progress=None):
    """按轨迹 JSON 渲染带叠加层的视频，音轨 ``-c:a copy`` 直通。

    解码固定为 rgb24（叠加层需要彩色），逐帧读 rawvideo、画框后写回
    编码器；帧数与轨迹记录数不一致时抛 :class:`ValueError`，不静默
    截断或补帧。返回渲染统计 dict。
    """
    result = load_track_json(track_path)
    records = result.frames
    ffmpeg = find_tool("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]
    frame_bytes = w * h * 3
    method = result.params.get("method", "sad")

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    enc_cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-framerate", info["fps"], "-i", "-",
    ]
    if info["has_audio"]:
        enc_cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "copy"]
    enc_cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)

    # 连续丢失计数：随帧滚动，供状态行显示
    lost_run = 0
    frames = 0
    try:
        while True:
            buf = read_exact(dec.stdout, frame_bytes)
            if buf is None:
                break
            if frames >= len(records):
                raise ValueError(
                    f"视频帧数多于轨迹记录（第 {frames} 帧起无记录），"
                    "轨迹与输入视频不匹配"
                )
            frame = bytearray(buf)
            rec = records[frames]
            if rec.status == LOST:
                lost_run += 1
            else:
                lost_run = 0
            draw_frame_overlay(
                frame, w, h, rec, method,
                font_scale=font_scale,
                show_last_known=show_last_known,
                lost_count=lost_run,
            )
            try:
                enc.stdin.write(bytes(frame))
            except (BrokenPipeError, OSError) as exc:
                raise PipelineError(f"编码器异常退出：{exc}") from exc
            frames += 1
            if progress:
                progress(frames)
    finally:
        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except OSError:
            pass
    dec.stdout.close()
    dec_rc = dec.wait()
    enc_rc = enc.wait()
    if dec_rc != 0:
        raise PipelineError(f"ffmpeg 解码失败（退出码 {dec_rc}）")
    if enc_rc != 0:
        raise PipelineError(f"ffmpeg 编码失败（退出码 {enc_rc}）")
    if frames != len(records):
        raise ValueError(
            f"轨迹记录 {len(records)} 条但视频只有 {frames} 帧，"
            "轨迹与输入视频不匹配"
        )
    return {
        "frames": frames,
        "width": w,
        "height": h,
        "track_json": os.fspath(track_path),
        "font_scale": font_scale,
        "show_last_known": show_last_known,
        "loss_events": len(result.loss_events),
    }
