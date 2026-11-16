"""ffmpeg 解码与运动估计驱动管线（估计阶段，不修改原文件）。

- 探测：ffprobe 读取宽高/fps/像素格式/音轨；
- 解码：``ffmpeg -i 输入 -f rawvideo -pix_fmt gray -`` 逐帧出灰度
  原始字节，Python 侧块平均下采样到估计分辨率，只保留相邻两帧，
  内存与视频总帧数无关；
- 估计：逐相邻帧调用 :func:`stabilize.motion.estimate_pair`，
  场景切换/低置信帧置 ``valid=False``；
- 输出：稳定 JSON 契约（``stabilize-motion`` v1），字段在本模块
  顶部文档中固定，补偿与裁剪阶段只依赖该契约。

stderr 一律 DEVNULL：PIPE 不消费会在长视频上撑满管道死锁
（与降噪模块同一坑）。
"""

import json
import os
import shutil
import subprocess

from .motion import EST_WIDTH, downsample, estimate_pair, is_scene_cut

__all__ = [
    "MOTION_FORMAT",
    "SMOOTH_FORMAT",
    "PipelineError",
    "probe",
    "read_gray_frames",
    "estimate_motion",
    "load_json",
    "dump_json",
]

MOTION_FORMAT = "stabilize-motion"
SMOOTH_FORMAT = "stabilize-smooth"


class PipelineError(RuntimeError):
    """流水线子进程失败或数据流异常。"""


def _find(tool):
    path = shutil.which(tool)
    if not path:
        raise PipelineError(f"找不到 {tool}，请确认已安装并在 PATH 中")
    return path


def probe(path):
    """ffprobe 读取首路视频流信息及是否含音轨。

    返回 ``{"width", "height", "pix_fmt", "fps", "has_audio"}``。
    """
    ffprobe = _find("ffprobe")
    out = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries",
            "stream=codec_type,width,height,pix_fmt,avg_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    video, has_audio = None, False
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and video is None:
            video = s
        elif s.get("codec_type") == "audio":
            has_audio = True
    if video is None:
        raise PipelineError(f"{path} 中没有视频流")
    fps = video.get("avg_frame_rate") or "25"
    if fps in ("0/0", ""):
        fps = "25"
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "fps": fps,
        "has_audio": has_audio,
    }


def fps_to_float(fps):
    """``"num/den"`` 形式帧率转 float；非法时回退 25.0。"""
    try:
        if "/" in fps:
            num, den = fps.split("/", 1)
            den = float(den)
            return float(num) / den if den else 25.0
        return float(fps)
    except (ValueError, ZeroDivisionError):
        return 25.0


def _read_exact(stream, size):
    """读满 size 字节；流结束时未读到任何字节返回 None，读一半报错。"""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if chunks:
                raise PipelineError("解码输出在帧中途截断")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_gray_frames(src, width, height, ffmpeg=None):
    """生成器：ffmpeg 解码出 gray 原始帧，逐帧 yield ``bytes``。

    调用方消费完即可自动回收子进程；提前 break 时由生成器关闭
    触发 ffmpeg 终止（管道关闭后其写端收到 SIGPIPE/退出）。
    """
    ffmpeg = ffmpeg or _find("ffmpeg")
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    frame_size = width * height
    try:
        while True:
            buf = _read_exact(proc.stdout, frame_size)
            if buf is None:
                break
            yield buf
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def estimate_motion(src, motion_json, target_width=EST_WIDTH, progress=None):
    """估计阶段主入口：解码 → 下采样 → 逐帧估计 → 写 motion JSON。

    原文件只读不写。返回与落盘内容相同的 dict（稳定契约）：

    ``format/version`` 固定；``downsample_scale`` 是小图→原图的
    线性缩放比（平移还原用）；``frames[0]`` 为恒等基准帧，其后
    每条目是相对上一帧的增量（``dx/dy`` 小图像素、``angle`` 度、
    ``confidence``、``valid``、``scene_cut``）。
    """
    ffmpeg = _find("ffmpeg")
    info = probe(src)
    w, h = info["width"], info["height"]

    prev_small = prev_dim = None
    entries = [{
        "index": 0, "dx": 0.0, "dy": 0.0, "angle": 0.0,
        "confidence": 1.0, "valid": True, "scene_cut": False,
    }]
    scale_used = 1
    frame_no = 0
    for raw in read_gray_frames(src, w, h, ffmpeg):
        frame_no += 1
        small, sw, sh, step = downsample(raw, w, h, target_width)
        scale_used = step
        if prev_small is not None:
            cut = is_scene_cut(prev_small, small, sw, sh)
            if cut:
                result = {"dx": 0.0, "dy": 0.0, "angle": 0.0,
                          "confidence": 0.0, "valid": False}
            else:
                result = estimate_pair(prev_small, small, sw, sh)
            entries.append({
                "index": frame_no - 1,
                "dx": round(result["dx"], 4),
                "dy": round(result["dy"], 4),
                "angle": round(result["angle"], 5),
                "confidence": round(result["confidence"], 4),
                "valid": bool(result["valid"]),
                "scene_cut": bool(cut),
            })
        prev_small, prev_dim = small, (sw, sh)
        if progress:
            progress(frame_no)

    if frame_no == 0:
        raise PipelineError(f"{src} 未解码出任何视频帧")

    doc = {
        "format": MOTION_FORMAT,
        "version": 1,
        "source": os.fspath(src),
        "width": w,
        "height": h,
        "fps": info["fps"],
        "pix_fmt": info["pix_fmt"],
        "has_audio": info["has_audio"],
        "est_width": prev_dim[0],
        "est_height": prev_dim[1],
        "downsample_scale": scale_used,
        "frame_count": frame_no,
        "frames": entries,
    }
    dump_json(doc, motion_json)
    return doc


def load_json(path):
    """读入 JSON 并做最小格式校验，返回 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or "format" not in doc:
        raise PipelineError(f"{path} 不是本工具生成的参数文件")
    return doc


def dump_json(doc, path):
    """原子写 JSON（先写 .tmp 再替换），保证中断不会留下半截参数文件。"""
    path = os.fspath(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
