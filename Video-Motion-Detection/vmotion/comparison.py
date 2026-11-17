"""帧差模式与背景建模模式的同输入量化对比。

两种模式回答的是同一个问题（「这一帧有没有运动」），但参考模型不同：
帧差看相邻两帧的变化沿，背景建模看当前帧与运行背景的偏差。要判断
背景建模到底改变了什么，不能只看各自的事件数，必须在**完全相同的
解码帧流**上并排跑两个检测器，逐帧比对，产出可复核的不一致清单。

本模块分两层：

- :func:`compare_detections`：纯逻辑，消费两路
  :class:`~vmotion.detector.FrameDetection` 序列，不接触 ffmpeg，
  单元测试可直接用合成平面驱动；
- :func:`compare_video`：一次 ffmpeg gray 解码同时喂两个检测器，
  事件各自沿状态边沿组装，对比结果落盘为 JSON。

不一致按「运动态判定」定义（去抖/迟滞后的对外布尔状态，与事件边沿
严格对齐），分四类计数：仅帧差报警、仅背景报警、双方都报警但区域
数不同、背景处于预热/抑制的帧单列说明。掩码级逐像素差异不写入
JSON（体积不可接受），需要时按 params 重放即可。
"""

import subprocess

from .background import BackgroundConfig, BackgroundDetector
from .detector import MotionConfig, MotionDetector
from .events import EventRecorder
from .io_utils import PipelineError, find_tool, probe, read_exact

__all__ = [
    "COMPARE_SCHEMA_VERSION",
    "compare_detections",
    "compare_video",
]

#: 对比 JSON schema 版本
COMPARE_SCHEMA_VERSION = 1


def _frame_disagreement(frame, pts, fd, bd):
    """比对同一帧的两路检测结果，返回不一致描述 dict；一致返回 None。"""
    f_motion = bool(fd.motion)
    b_motion = bool(bd.motion)
    if f_motion == b_motion:
        # 运动态一致时，区域数差异只作信息项，不计入判定不一致
        if f_motion and len(fd.regions) != len(bd.regions):
            return {
                "frame": frame,
                "pts_time": round(pts, 6),
                "kind": "region_count_diff",
                "frame_diff_motion": True,
                "background_motion": True,
                "frame_diff_regions": len(fd.regions),
                "background_regions": len(bd.regions),
            }
        return None
    if f_motion and not b_motion:
        kind = "frame_diff_only"
    else:
        kind = "background_only"
    return {
        "frame": frame,
        "pts_time": round(pts, 6),
        "kind": kind,
        "frame_diff_motion": f_motion,
        "background_motion": b_motion,
        "frame_diff_regions": len(fd.regions),
        "background_regions": len(bd.regions),
        "background_stats": {
            "warmup": bool(bd.stats.get("warmup")),
            "held_off": bool(bd.stats.get("held_off")),
            "fg_blocks": bd.stats.get("fg_blocks", 0),
            "absorbed_blocks": bd.stats.get("absorbed_blocks", 0),
        },
        "frame_diff_stats": {
            "held_off": bool(fd.stats.get("held_off")),
            "gated_pixels": fd.stats.get("gated_pixels", 0),
        },
    }


def compare_detections(frame_dets, bg_dets):
    """纯逻辑对比两路等长检测结果序列。

    参数
    ----
    frame_dets, bg_dets:
        帧号严格对齐的两路 :class:`~vmotion.detector.FrameDetection`
        序列（长度必须相同，帧号逐一相等）。

    返回 dict：``agreement`` 一致率、按类别汇总的 ``counts``、
    完整 ``disagreements`` 清单与逐帧精简对照表。
    """
    if len(frame_dets) != len(bg_dets):
        raise ValueError(
            f"两路检测帧数不一致：帧差 {len(frame_dets)} 帧，"
            f"背景 {len(bg_dets)} 帧"
        )
    counts = {"agree": 0, "frame_diff_only": 0, "background_only": 0,
              "region_count_diff": 0}
    disagreements = []
    per_frame = []
    for fd, bd in zip(frame_dets, bg_dets):
        if fd.frame != bd.frame:
            raise ValueError(
                f"帧号未对齐：帧差第 {fd.frame} 帧 vs 背景第 {bd.frame} 帧"
            )
        item = _frame_disagreement(fd.frame, fd.pts_time, fd, bd)
        per_frame.append({
            "frame": fd.frame,
            "frame_diff_motion": bool(fd.motion),
            "background_motion": bool(bd.motion),
            "frame_diff_regions": len(fd.regions),
            "background_regions": len(bd.regions),
        })
        if item is None:
            counts["agree"] += 1
        else:
            counts[item["kind"]] += 1
            disagreements.append(item)
    total = len(frame_dets)
    verdict_mismatch = (counts["frame_diff_only"]
                        + counts["background_only"])
    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "frames": total,
        "agreement": round(counts["agree"] / total, 6) if total else 0.0,
        "verdict_agreement": round(
            (total - verdict_mismatch) / total, 6) if total else 0.0,
        "counts": counts,
        "disagreements": disagreements,
        "per_frame": per_frame,
    }


def compare_video(src, compare_json, frame_config=None, bg_config=None,
                  *, progress=None):
    """一次解码同时跑两种检测模式，对比结果写入 JSON。

    两路检测器各自挂一个 :class:`~vmotion.events.EventRecorder`，
    事件与逐帧对照一并落盘。返回内存 dict（与磁盘 JSON 同构）。
    """
    fcfg = frame_config or MotionConfig()
    bcfg = bg_config or BackgroundConfig()
    fcfg.validate()
    bcfg.validate()

    ffmpeg = find_tool("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]
    fps_float = info["fps_float"]
    frame_bytes = w * h

    f_det = MotionDetector(w, h, fcfg)
    b_det = BackgroundDetector(w, h, bcfg)
    f_rec = EventRecorder()
    b_rec = EventRecorder()

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    f_results = []
    b_results = []
    frames = 0
    try:
        while True:
            buf = read_exact(dec.stdout, frame_bytes)
            if buf is None:
                break
            pts = frames / fps_float if fps_float > 0 else 0.0
            fd = f_det.update(buf, frames, pts_time=pts)
            bd = b_det.update(buf, frames, pts_time=pts)
            f_rec.update(fd)
            b_rec.update(bd)
            f_results.append(fd)
            b_results.append(bd)
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
    f_events = f_rec.finish(frames - 1, last_pts)
    b_events = b_rec.finish(frames - 1, last_pts)

    comparison = compare_detections(f_results, b_results)
    import json
    payload = {
        **comparison,
        "meta": {
            "source": str(src),
            "width": w,
            "height": h,
            "fps": info["fps"],
            "fps_float": fps_float,
        },
        "params": {
            "mode": "compare",
            "frame_diff": f_det.config_dict(),
            "background": b_det.config_dict(),
        },
        "events": {
            "frame_diff": [e.to_dict() for e in f_events],
            "background": [e.to_dict() for e in b_events],
        },
        "event_counts": {
            "frame_diff": len(f_events),
            "background": len(b_events),
        },
    }
    with open(compare_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload
