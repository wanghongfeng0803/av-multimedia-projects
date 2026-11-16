"""运动标记渲染：只消费检测结果在视频帧上画叠加层，不做任何检测。

与检测严格解耦——本模块不计算掩码，输入要么是检测流水线内存中的
:class:`~vmotion.model.FrameRecord`（检测与渲染同一次 ffmpeg 解码内
完成），要么是落盘的检测 JSON（:func:`render_video` 独立重放），
同一份 JSON 可反复换样式重渲染。

绘图全部在 rgb24 打包字节平面上手工完成（零三方依赖）：

- 存活区域画绿色矩形框 + 区域编号，事件编号显示在左上角状态行，
  保证标记视频与事件列表逐帧可对照；
- 光照抑制帧画黄色 ``HOLDOFF``，空闲帧只保留灰色状态行，三种状态
  视觉上刻意区分；
- 文字用 :mod:`vmotion.font` 的内置 5×7 点阵，带深色描边，亮/暗背景
  上都可读。
"""

import subprocess

from .font import GLYPH_H, render_text_mask
from .io_utils import PipelineError, find_tool, probe, read_exact
from .model import load_detection_json

__all__ = [
    "COLOR_GREEN",
    "COLOR_YELLOW",
    "COLOR_GRAY",
    "COLOR_WHITE",
    "COLOR_BLACK",
    "draw_rect",
    "draw_text",
    "draw_frame_overlay",
    "render_video",
]

#: rgb24 三元组
COLOR_GREEN = (0, 220, 0)
COLOR_YELLOW = (220, 200, 0)
COLOR_GRAY = (170, 170, 170)
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


def draw_text(frame, width, height, x, y, text, color,
              font_scale=1, bg=COLOR_BLACK):
    """在 (x, y) 处画点阵文字，带 1 像素（缩放后）深色描边。

    两遍绘制：先收集全部放大后的前景坐标，描边环只写非前景位置
    （否则后画像素的描边会覆盖相邻已画前景），最后统一写前景色。
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
    for fx, fy in fg:
        for ox in range(-outline, font_scale + outline):
            for oy in range(-outline, font_scale + outline):
                cx, cy = fx + ox, fy + oy
                if (cx, cy) in fg:
                    continue
                _set_pixel(frame, width, height, x + cx, y + cy, bg)
    for fx, fy in fg:
        _set_pixel(frame, width, height, x + fx, y + fy, color)


def _status_line(record):
    """拼左上角状态行文本与颜色。"""
    stats = record.stats
    if stats.get("held_off"):
        return (f"F{record.frame} HOLDOFF "
                f"D{stats.get('mean_delta', 0.0):.1f}", COLOR_YELLOW)
    if record.motion:
        eid = record.event_id if record.event_id is not None else "-"
        return (f"F{record.frame} MOTION E{eid} "
                f"R{len(record.regions)}", COLOR_GREEN)
    return f"F{record.frame} IDLE", COLOR_GRAY


def draw_frame_overlay(frame, width, height, record, font_scale=1):
    """按一帧 :class:`~vmotion.model.FrameRecord` 画叠加层。

    - 运动帧：逐区域绿框 + 框上方区域编号，左上状态行标事件 id；
    - 抑制帧：黄色 HOLDOFF 与全局均值变化量；
    - 空闲帧：灰色 IDLE 状态行。
    """
    text, color = _status_line(record)
    draw_text(frame, width, height, 2, 2, text, color,
              font_scale=font_scale)
    for region in record.regions:
        x, y, rw, rh = region["x"], region["y"], region["w"], region["h"]
        draw_rect(frame, width, height, x, y, rw, rh,
                  COLOR_GREEN, thickness=2)
        label = str(region["label"])
        draw_text(frame, width, height, x,
                  max(0, y - (GLYPH_H + 2) * font_scale),
                  label, COLOR_GREEN, font_scale=font_scale)


def render_video(src, dst, detection_json, font_scale=1, progress=None):
    """按检测 JSON 渲染带叠加层的视频，音轨 ``-c:a copy`` 直通。

    解码固定 rgb24（叠加层需要彩色），逐帧读 rawvideo、画框后写回
    编码器；帧数与 JSON 记录数不一致时抛 :class:`ValueError`，不静默
    截断或补帧。JSON 文件不存在同样抛 :class:`ValueError`，与 schema
    校验失败归为同一类「数据契约问题」。返回渲染统计 dict。
    """
    try:
        result = load_detection_json(detection_json)
    except FileNotFoundError as exc:
        raise ValueError(f"检测 JSON 不存在：{detection_json}") from exc
    records = result.frames
    ffmpeg = find_tool("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]
    frame_bytes = w * h * 3

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

    frames = 0
    try:
        while True:
            buf = read_exact(dec.stdout, frame_bytes)
            if buf is None:
                break
            if frames >= len(records):
                raise ValueError(
                    f"视频帧数多于检测记录（第 {frames} 帧起无记录），"
                    "JSON 与输入视频不匹配"
                )
            frame = bytearray(buf)
            draw_frame_overlay(frame, w, h, records[frames],
                               font_scale=font_scale)
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
            f"检测记录 {len(records)} 条但视频只有 {frames} 帧，"
            "JSON 与输入视频不匹配"
        )
    return {
        "frames": frames,
        "width": w,
        "height": h,
        "detection_json": str(detection_json),
        "font_scale": font_scale,
        "events": len(result.events),
    }
