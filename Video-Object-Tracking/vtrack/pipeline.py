"""ffmpeg 追踪 / 渲染流水线：解码 rawvideo → 模板匹配 → 轨迹 JSON。

与兄弟模块同一套子进程约定：

- 解码 ``ffmpeg -i 输入 -f rawvideo -pix_fmt gray -``，追踪只用亮度
  平面（色度对模板匹配贡献低且增加 2 倍数据量）；
- 渲染在 :mod:`vtrack.render` 内单独走 rgb24 解码，追踪与渲染是两次
  独立的 ffmpeg 流水线，中间产物只有轨迹 JSON；
- 子进程 stderr 一律 DEVNULL，置 PIPE 不消费会在长视频上撑满管道
  缓冲区死锁。

初始模板三选一（优先级从高到低）：

1. ``template_image``：外部图片文件，ffmpeg 转 gray 后按其尺寸全图取模；
2. ``template_frame`` + ``init_box``：从指定视频帧裁剪；
3. 仅 ``init_box``：从第 0 帧裁剪（最常用）。
"""

import json
import subprocess

from .io_utils import (
    PipelineError,
    find_tool as _find,
    probe,
    read_exact as _read_exact,
)
from .model import (
    LOST,
    LossEvent,
    build_track_result,
    save_track_json,
)
from .template import (
    DEFAULT_THRESHOLDS,
    TEMPLATE_METHODS,
    extract_patch,
)
from .tracker import TemplateTracker

__all__ = [
    "PipelineError",
    "probe",
    "read_initial_template",
    "track_video",
    "render_video",
]

# render_video 定义在渲染模块，这里转出以保持流水线入口统一；
# render.py 只依赖 io_utils，不存在循环导入
from .render import render_video  # noqa: E402


def _decode_gray_frame(ffmpeg, src, frame_index, width, height):
    """解码指定帧的灰度平面（select 帧号从 0 起），返回紧凑 bytes。"""
    cmd = [
        ffmpeg, "-v", "error",
        "-i", str(src),
        "-vf", f"select=eq(n\\,{frame_index})",
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise PipelineError(
            f"读取第 {frame_index} 帧失败（ffmpeg 退出码 {proc.returncode}）"
        )
    need = width * height
    if len(proc.stdout) < need:
        raise PipelineError(
            f"第 {frame_index} 帧解码数据不足：{len(proc.stdout)} < {need}"
        )
    return proc.stdout[:need]


def _decode_image_gray(ffmpeg, image_path):
    """把外部图片转成灰度平面，返回 (bytes, width, height)。"""
    ffprobe = _find("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=width,height", "-of", "json", str(image_path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(out.stdout).get("streams", [])
    if not streams:
        raise PipelineError(f"模板图片 {image_path} 无法探测尺寸")
    tw, th = streams[0]["width"], streams[0]["height"]
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(image_path),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise PipelineError(
            f"模板图片解码失败（ffmpeg 退出码 {proc.returncode}）"
        )
    need = tw * th
    if len(proc.stdout) < need:
        raise PipelineError("模板图片解码数据不足")
    return proc.stdout[:need], tw, th


def read_initial_template(src, init_box=None, *, template_image=None,
                          template_frame=0):
    """提取初始模板。

    参数
    ----
    src:
        输入视频路径。
    init_box:
        ``(x, y, w, h)``。使用外部图片时可为 None（取图片全图），
        其余方式必填。
    template_image:
        外部模板图片路径，优先级最高。
    template_frame:
        从视频第几帧裁剪模板（默认第 0 帧）。

    返回 ``(template bytes, tw, th, init_x, init_y)``。
    """
    ffmpeg = _find("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]

    if template_image is not None:
        tpl, tw, th = _decode_image_gray(ffmpeg, template_image)
        if tw > w or th > h:
            raise ValueError(
                f"模板尺寸 {tw}x{th} 大于视频帧 {w}x{h}"
            )
        if init_box is not None:
            init_x, init_y = init_box[0], init_box[1]
        else:
            # 缺省锚到左上角，调用方通常会显式给出位置
            init_x, init_y = 0, 0
        if not (0 <= init_x <= w - tw and 0 <= init_y <= h - th):
            raise ValueError("初始框超出帧边界")
        return tpl, tw, th, init_x, init_y

    if init_box is None:
        raise ValueError("未提供模板图片时必须给出 init_box=(x,y,w,h)")
    init_x, init_y, tw, th = init_box
    if tw <= 0 or th <= 0:
        raise ValueError("init_box 的宽高必须为正")
    if not (0 <= init_x <= w - tw and 0 <= init_y <= h - th):
        raise ValueError("初始框超出帧边界")
    plane = _decode_gray_frame(ffmpeg, src, template_frame, w, h)
    tpl = extract_patch(plane, w, init_x, init_y, tw, th)
    return tpl, tw, th, init_x, init_y


def track_video(src, track_json, init_box, *, method="sad", threshold=None,
                search_margin=12, lost_growth=1.5, reacquire_frames=3,
                reacquire_threshold=None, update_template=False,
                update_alpha=0.2, update_score=None,
                template_image=None, template_frame=0,
                target_id="target_0", progress=None):
    """对视频逐帧做模板匹配追踪，轨迹写入 ``track_json``。

    返回 :class:`~vtrack.model.TrackResult`。丢失区间在追踪过程中按
    状态转换实时组装（进入丢失记起点，恢复确认记终点），视频结束仍
    未找回则生成 ``end_of_stream=True`` 的未闭合事件。追踪器本身
    不接触 pts，时间戳在本函数按帧号 × 帧时长统一填写。
    """
    if method not in TEMPLATE_METHODS:
        raise ValueError(
            f"未知匹配方法 {method!r}，可选：{', '.join(TEMPLATE_METHODS)}"
        )
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[method]

    ffmpeg = _find("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]
    fps_float = info["fps_float"]
    frame_bytes = w * h

    tpl, tw, th, init_x, init_y = read_initial_template(
        src, init_box, template_image=template_image,
        template_frame=template_frame,
    )

    tracker = TemplateTracker(
        width=w, height=h, template=tpl, tw=tw, th=th,
        init_x=init_x, init_y=init_y, method=method,
        threshold=threshold, search_margin=search_margin,
        lost_growth=lost_growth, reacquire_frames=reacquire_frames,
        reacquire_threshold=reacquire_threshold,
        update_template=update_template, update_alpha=update_alpha,
        update_score=update_score,
    )

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    records = []
    loss_events = []
    open_start = None       # 当前丢失区间起始帧号
    frames = 0
    try:
        while True:
            buf = _read_exact(dec.stdout, frame_bytes)
            if buf is None:
                break
            pts = frames / fps_float if fps_float > 0 else 0.0
            rec = tracker.update(buf, frames, pts_time=pts)

            # 状态转换 → 丢失事件组装
            if rec.status == LOST and open_start is None:
                open_start = frames
            recovered = tracker.take_recovery()
            if recovered is not None and open_start is not None:
                loss_events.append(LossEvent(
                    lost_frame=open_start,
                    recovered_frame=recovered,
                    lost_pts=open_start / fps_float if fps_float > 0 else 0.0,
                    recovered_pts=recovered / fps_float if fps_float > 0 else 0.0,
                    frames_lost=recovered - open_start,
                    end_of_stream=False,
                ))
                open_start = None
            records.append(rec)
            frames += 1
            if progress:
                progress(frames)
    finally:
        if dec.stdout:
            dec.stdout.close()
    dec_rc = dec.wait()
    if dec_rc != 0:
        raise PipelineError(f"ffmpeg 解码失败（退出码 {dec_rc}）")
    if frames == 0:
        raise PipelineError("输入视频没有解码出任何帧")

    # 片末仍丢失：补未闭合事件
    if open_start is not None:
        last_idx = frames - 1
        loss_events.append(LossEvent(
            lost_frame=open_start,
            recovered_frame=None,
            lost_pts=open_start / fps_float if fps_float > 0 else 0.0,
            recovered_pts=None,
            frames_lost=last_idx - open_start + 1,
            end_of_stream=True,
        ))

    meta = {
        "source": str(src),
        "target_id": target_id,
        "width": w,
        "height": h,
        "fps": info["fps"],
        "fps_float": fps_float,
        "initial_box": [init_x, init_y, tw, th],
        "template_source": (
            str(template_image) if template_image is not None
            else f"frame:{template_frame}"
        ),
    }
    params = {
        "method": method,
        "threshold": threshold,
        "reacquire_threshold": tracker.reacquire_threshold,
        "search_margin": search_margin,
        "lost_growth": lost_growth,
        "reacquire_frames": reacquire_frames,
        "update_template": update_template,
        "update_alpha": update_alpha if update_template else None,
        "update_score": update_score if update_template else None,
    }
    result = build_track_result(records, loss_events, meta, params,
                                info["fps"])
    save_track_json(result, track_json)
    return result
