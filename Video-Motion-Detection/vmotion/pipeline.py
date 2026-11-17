"""ffmpeg 检测流水线：解码 rawvideo 灰度平面 → 运动检测 → 事件 JSON。

与兄弟模块同一套子进程约定：

- 解码 ``ffmpeg -i 输入 -f rawvideo -pix_fmt gray -``，检测只在亮度
  平面上进行（色度对帧差贡献低且增加 2 倍数据量，光照变化在亮度上
  也最直接）；
- 标记渲染在 :mod:`vmotion.render` 内单独走 rgb24 解码，检测与标记
  是两次独立的 ffmpeg 流水线，中间产物只有检测 JSON，因此同一份
  JSON 可反复换样式重渲染，标记模块永远不接触检测算法；
- 子进程 stderr 一律 DEVNULL，置 PIPE 不消费会在长视频上撑满管道
  缓冲区死锁。

检测过程只保留逐帧 :class:`~vmotion.model.FrameRecord`（区域几何 +
统计，不含全分辨率掩码），内存随帧数线性但每帧开销很小；掩码仅在
检测当帧存活，随后即可被下一轮覆盖。

两种检测模式共用同一条流水线骨架：:func:`detect_video` 跑旧帧差模式
（:class:`~vmotion.detector.MotionDetector`），:func:`detect_background_video`
跑背景建模模式（:class:`~vmotion.background.BackgroundDetector`），
两者产出同构的检测 JSON，标记/事件模块不感知模式差异。
"""

import subprocess

from .background import BackgroundConfig, BackgroundDetector
from .detector import MotionConfig, MotionDetector
from .events import EventRecorder
from .io_utils import PipelineError, find_tool, probe, read_exact
from .model import FrameRecord, build_result, save_detection_json
from .render import render_video

__all__ = [
    "detect_video",
    "detect_background_video",
    "render_video",
    "PipelineError",
]


def _run_detection(src, detector, mode_name, *, progress=None):
    """共用解码骨架：喂入给定检测器，返回 (records, events, meta, 帧信息)。"""
    info = probe(src)
    w, h = info["width"], info["height"]
    fps_float = info["fps_float"]
    frame_bytes = w * h
    recorder = EventRecorder()

    dec = subprocess.Popen(
        [find_tool("ffmpeg"), "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    records = []
    frames = 0
    try:
        while True:
            buf = read_exact(dec.stdout, frame_bytes)
            if buf is None:
                break
            pts = frames / fps_float if fps_float > 0 else 0.0
            detection = detector.update(buf, frames, pts_time=pts)
            event_id = recorder.update(detection)
            records.append(FrameRecord(
                frame=detection.frame,
                pts_time=detection.pts_time,
                motion=detection.motion,
                regions=[r.to_dict() for r in detection.regions],
                stats=detection.stats,
                event_id=event_id,
            ))
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

    last_pts = (frames - 1) / fps_float if fps_float > 0 else 0.0
    events = recorder.finish(frames - 1, last_pts)
    meta = {
        "source": str(src),
        "width": w,
        "height": h,
        "fps": info["fps"],
        "fps_float": fps_float,
        "mode": mode_name,
    }
    return records, events, meta


def detect_video(src, detection_json, config=None, *, progress=None):
    """对视频逐帧做帧差模式运动检测，结果（逐帧记录 + 事件）写入 JSON。

    参数
    ----
    src:
        输入视频路径。
    detection_json:
        检测结果 JSON 输出路径。
    config:
        :class:`MotionConfig`，缺省取 :class:`MotionConfig` 默认参数。
    progress:
        可选回调 ``progress(frame_count)``，每解码一帧调用一次。

    返回 :class:`~vmotion.model.DetectionResult`。事件由
    :class:`~vmotion.events.EventRecorder` 在检测过程中沿状态边沿
    实时组装，pts 按帧号 / 帧率统一填写。
    """
    cfg = config or MotionConfig()
    cfg.validate()
    info = probe(src)
    detector = MotionDetector(info["width"], info["height"], cfg)
    records, events, meta = _run_detection(
        src, detector, "frame_diff", progress=progress)
    result = build_result(records, events, meta, detector.config_dict())
    save_detection_json(result, detection_json)
    return result


def detect_background_video(src, detection_json, config=None, *,
                            progress=None):
    """对视频逐帧做背景建模模式检测，输出与帧差模式同构的检测 JSON。

    参数
    ----
    src:
        输入视频路径。
    detection_json:
        检测结果 JSON 输出路径。
    config:
        :class:`BackgroundConfig`，缺省取 :class:`BackgroundConfig`
        默认参数（含预热帧数、学习率、吸收帧数）。
    progress:
        可选回调 ``progress(frame_count)``，每解码一帧调用一次。
    """
    cfg = config or BackgroundConfig()
    cfg.validate()
    info = probe(src)
    detector = BackgroundDetector(info["width"], info["height"], cfg)
    records, events, meta = _run_detection(
        src, detector, "background", progress=progress)
    result = build_result(records, events, meta, detector.config_dict())
    save_detection_json(result, detection_json)
    return result
