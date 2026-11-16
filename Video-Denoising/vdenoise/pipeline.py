"""ffmpeg 解码 → 空间滤波（均值/中值）→ ffmpeg 编码 流水线。

- 解码：``ffmpeg -i 输入 -f rawvideo -pix_fmt <工作格式> -``，逐帧读原始字节；
- 滤波：每帧按像素格式拆平面，各平面按内存预算分带处理
  （长视频内存恒定 O(单帧)，单帧峰值 O(预算)）；
- 编码：第二路 ``-i 输入`` 供音轨 ``-c:a copy`` 直通，
  无音轨时 ``-map 1:a:0?`` 的可选标记自动忽略；
- 两个子进程的 stderr 一律 DEVNULL：置为 PIPE 而不消费会在
  长视频上撑满管道缓冲区造成死锁。

两种滤波模式：
- ``mean``（默认）：可分离盒式均值，边界窗口截断——历史行为，
  参数与结果逐字节保持不变；
- ``median``：Huang 直方图加速中值（椒盐噪声），边界镜像填充，
  可选 ``border_dir`` 把每帧边界区域（掩码外置 0）落盘供抽帧复核。
"""

import json
import os
import shutil
import subprocess
from collections import deque

from .bands import box_blur_plane_banded, plan_bands
from .median import (
    dump_border_frame,
    median_plane_banded,
    plan_median_bands,
)
from .memory import derive_budget
from .pixfmt import (
    encoder_for,
    filter_dims,
    frame_size,
    join_planes,
    split_planes,
    working_format,
)
from .temporal import (
    DEFAULT_MOTION_THRESHOLD,
    MAX_CACHE_FRAMES,
    estimate_shift,
    plan_temporal,
    temporal_blend_plane_banded,
)

__all__ = ["probe", "denoise_video", "PipelineError"]

#: 支持的滤波模式
FILTER_MODES = ("mean", "median", "temporal")


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
    video = None
    has_audio = False
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
        "width": video["width"],
        "height": video["height"],
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "fps": fps,
        "has_audio": has_audio,
    }


def _read_exact(stream, size):
    """读满 size 字节；流结束时未读到任何字节返回 None，读了一半报错。"""
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


def denoise_video(src, dst, radius=2, budget_bytes=None, progress=None,
                  mode="mean", window=None, border_dir=None,
                  temporal_radius=2, max_shift=6,
                  motion_threshold=DEFAULT_MOTION_THRESHOLD,
                  max_cache_frames=MAX_CACHE_FRAMES, band_height=None):
    """对 ``src`` 做空间/时域滤波降噪，结果写入 ``dst``，音轨原样保留。

    ``radius`` 为空间滤波半径（窗口 (2r+1)²）；``budget_bytes`` 为
    内存预算，缺省按可用内存推导；``progress`` 若给出，每处理完一帧
    以已处理帧数调用一次。返回处理统计 dict。

    ``mode``：
    - ``"mean"``（默认）：可分离盒式均值，行为与历史版本完全一致；
    - ``"median"``：直方图加速中值滤波（椒盐噪声），边界镜像填充。
      ``window`` 可直接给奇数窗口边长（按素材配置），给出时优先于
      ``radius``；不给时窗口为 (2·radius+1)²（默认 radius=2 即 5×5）。
      window=1（radius=0）时中值退化为恒等，与均值模式语义一致。
    - ``"temporal"``：帧间时域降噪。以当前帧为中心取
      [t-``temporal_radius``, t+``temporal_radius``] 时间窗，先对每个
      参考帧做全局整像素位移估计（粗-精块匹配，``max_shift`` 为搜索
      半径）对齐到当前帧，再逐像素混合；对齐残差超过
      ``motion_threshold`` 的像素判为运动区，该参考帧在该像素不参与
      混合，全部参考被拒时回退当前帧，避免运动拖影。缓存帧数受
      ``max_cache_frames`` 硬上限与内存预算双重约束，片长（每带核心
      行数）、缓存帧数、预算三者由 :func:`plan_temporal` 互推，超预算
      在启动 ffmpeg 前抛 :class:`MemoryError`。``band_height`` 可显式
      指定片长，缺省在预算内取最大片长。
    ``border_dir`` 仅在中值模式下有效：给出目录后，每帧的边界区域
    （距边缘 <= 半径的像素，掩码外置 0）以工作格式原始字节落盘，
    文件名 ``frame_XXXXXX_border.raw``，另写 ``manifest.json``
    记录工作格式、尺寸、半径与每帧文件名，供单独抽帧复核。

    无论哪种模式，输出帧数与时间戳都由同一套 rawvideo 帧循环保证，
    与输入严格对齐（一帧入、一帧出，framerate 原样透传）。
    """
    if radius < 0:
        raise ValueError("radius 不能为负")
    if mode not in FILTER_MODES:
        raise ValueError(f"未知滤波模式 {mode!r}，可选：{', '.join(FILTER_MODES)}")
    if mode == "median":
        if window is not None:
            if window < 1 or window % 2 == 0:
                raise ValueError("中值窗口边长必须为正奇数")
            radius = (window - 1) // 2
        # window 缺省时沿用 radius；radius=0 即 1x1 中值，退化为恒等
    if mode == "temporal":
        if temporal_radius < 0:
            raise ValueError("temporal_radius 不能为负")
        if max_shift < 0:
            raise ValueError("max_shift 不能为负")
        if max_cache_frames < 1:
            raise ValueError("max_cache_frames 必须 >= 1")
        if band_height is not None and band_height < 1:
            raise ValueError("band_height 必须 >= 1")
        if border_dir is not None:
            raise ValueError("border_dir 仅在中值模式下可用")
    elif border_dir is not None and mode != "median":
        raise ValueError("border_dir 仅在中值模式下可用")

    ffmpeg = _find("ffmpeg")
    info = probe(src)
    work = working_format(info["pix_fmt"])
    codec, out_pix = encoder_for(work)
    w, h = info["width"], info["height"]
    if budget_bytes is None:
        budget_bytes = derive_budget()

    dims = filter_dims(work, w, h)
    fsize = frame_size(work, w, h)

    if mode == "temporal":
        # 内存互推与提前拒绝：任何子进程启动之前完成可行性校验
        plane_sizes = [pw * ph for pw, ph in dims]
        plane_widths = [pw for pw, _ in dims]
        plane_heights = [ph for _, ph in dims]
        plan = plan_temporal(
            plane_sizes, plane_widths, plane_heights,
            temporal_radius, budget_bytes,
            max_cache_frames=max_cache_frames,
            band_height=band_height,
        )
        return _TemporalRunner(
            work, w, h, fsize, dims, plan, max_shift,
            motion_threshold, progress, str(dst),
        ).run(src, info, codec, out_pix, ffmpeg)

    if mode == "median":
        band_plans = [
            plan_median_bands(ph, pw, radius, budget_bytes)
            for pw, ph in dims
        ]
    else:
        band_plans = [plan_bands(ph, pw, radius, budget_bytes) for pw, ph in dims]

    if border_dir is not None:
        os.makedirs(border_dir, exist_ok=True)
        border_files = []

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", work, "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    enc_cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", work, "-s", f"{w}x{h}",
        "-framerate", info["fps"], "-i", "-",
    ]
    if info["has_audio"]:
        enc_cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "copy"]
    enc_cmd += ["-c:v", codec]
    if out_pix:
        enc_cmd += ["-pix_fmt", out_pix]
    enc_cmd += [str(dst)]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)

    frames = 0
    try:
        while True:
            buf = _read_exact(dec.stdout, fsize)
            if buf is None:
                break
            planes, _ = split_planes(buf, work, w, h)
            if mode == "median":
                out_planes = [
                    median_plane_banded(plane, pw, ph, radius, band_plans[i])
                    for i, (plane, (pw, ph)) in enumerate(zip(planes, dims))
                ]
                if border_dir is not None:
                    name = f"frame_{frames:06d}_border.raw"
                    dump_border_frame(
                        os.path.join(border_dir, name),
                        out_planes, dims, radius,
                    )
                    border_files.append(name)
            else:
                out_planes = [
                    box_blur_plane_banded(plane, pw, ph, radius, band_plans[i])
                    for i, (plane, (pw, ph)) in enumerate(zip(planes, dims))
                ]
            try:
                enc.stdin.write(join_planes(out_planes, work, w, h))
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
    if border_dir is not None:
        with open(os.path.join(border_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(
                {
                    "pix_fmt": work,
                    "width": w,
                    "height": h,
                    "fps": info["fps"],
                    "radius": radius,
                    "window": 2 * radius + 1,
                    "frame_bytes": fsize,
                    "files": border_files,
                },
                f, ensure_ascii=False, indent=2,
            )
    return {
        "frames": frames,
        "pix_fmt": work,
        "width": w,
        "height": h,
        "has_audio": info["has_audio"],
        "mode": mode,
        "radius": radius,
        "window": 2 * radius + 1 if mode == "median" else None,
        "border_dir": os.fspath(border_dir) if border_dir is not None else None,
    }


def _encoder_command(ffmpeg, work, w, h, fps, codec, out_pix, has_audio, src, dst):
    """组装编码器命令：rawvideo 入，可选第二路输入供音轨直通。"""
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", work, "-s", f"{w}x{h}",
        "-framerate", fps, "-i", "-",
    ]
    if has_audio:
        cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:a", "copy"]
    cmd += ["-c:v", codec]
    if out_pix:
        cmd += ["-pix_fmt", out_pix]
    cmd += [str(dst)]
    return cmd


class _TemporalRunner:
    """封装时域流水线的状态，避免长参数列表在多个私有函数间传递。

    输出顺序与输入严格一致：第 t 帧在「读到 t+half 帧」或「视频结束」
    时才发射（时间窗右边界），解码全部完成后冲刷缓存尾部。缓存容量 =
    ``plan["cache_frames"]``，由有界 deque 淘汰保证内存与总帧数无关。
    """

    def __init__(self, work, w, h, fsize, dims, plan, max_shift,
                 motion_threshold, progress, dst):
        self.work = work
        self.w, self.h = w, h
        self.fsize = fsize
        self.dims = dims
        self.plan = plan
        self.max_shift = max_shift
        self.motion_threshold = motion_threshold
        self.progress = progress
        self.dst = dst

    def run(self, src, info, codec, out_pix, ffmpeg):
        cache_cap = self.plan["cache_frames"]
        radius = self.plan["temporal_radius"]
        bands_per_plane = self.plan["bands"]
        luma_w, luma_h = self.dims[0]
        # 色度平面相对亮度的采样比（4:2:0 位移减半，4:4:4 不变）
        scale_x = [pw / luma_w for pw, _ in self.dims]
        scale_y = [ph / luma_h for _, ph in self.dims]
        # 受缓存容量约束的实际半窗：对称窗每侧最多 (cap-1)//2 个邻居
        half = min(radius, (cache_cap - 1) // 2)

        dec = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(src),
             "-f", "rawvideo", "-pix_fmt", self.work, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        enc_cmd = _encoder_command(
            ffmpeg, self.work, self.w, self.h, info["fps"], codec,
            out_pix, info["has_audio"], src, self.dst,
        )
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)

        # 有界原始帧缓存：旧 → 新，每项是一帧完整 raw 字节
        raw_cache = deque(maxlen=cache_cap)
        total = 0          # 已读入帧数
        emitted = 0        # 已发射（输出）帧数
        try:
            while True:
                buf = _read_exact(dec.stdout, self.fsize)
                if buf is None:
                    break
                raw_cache.append(buf)
                total += 1
                # 右边界 t+half 已到达即可发射目标帧 t
                ready_t = total - 1 - half
                while emitted <= ready_t:
                    self._emit(raw_cache, total, emitted, enc,
                               bands_per_plane, scale_x, scale_y,
                               luma_w, luma_h)
                    emitted += 1
                    if self.progress:
                        self.progress(emitted)
            # 冲刷尾部：剩余帧以逐渐缩小的右窗发射
            while emitted < total:
                self._emit(raw_cache, total, emitted, enc,
                           bands_per_plane, scale_x, scale_y,
                           luma_w, luma_h)
                emitted += 1
                if self.progress:
                    self.progress(emitted)
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
        return {
            "frames": emitted,
            "pix_fmt": self.work,
            "width": self.w,
            "height": self.h,
            "has_audio": info["has_audio"],
            "mode": "temporal",
            "radius": None,
            "window": None,
            "border_dir": None,
            "temporal_radius": self.plan["temporal_radius"],
            "cache_frames": self.plan["cache_frames"],
            "nominal_window": self.plan["nominal_window"],
            "band_height": self.plan["band_height"],
            "max_shift": self.max_shift,
            "motion_threshold": self.motion_threshold,
            "memory_bytes": self.plan["memory_bytes"],
        }

    def _emit(self, raw_cache, total, t, enc, bands_per_plane,
              scale_x, scale_y, luma_w, luma_h):
        """发射第 t 帧：从缓存取对称时间窗，对齐后分平面混合并写出。"""
        r = self.plan["temporal_radius"]
        lo = max(0, t - r)
        hi = min(total - 1, t + r)
        # 缓存下标：缓存保留最近 cache_cap 帧，索引 = 全局帧号 - 起点
        start = total - len(raw_cache)
        target_buf = raw_cache[t - start]
        target_planes, _ = split_planes(target_buf, self.work, self.w, self.h)
        ref_bufs = [
            raw_cache[j - start] for j in range(lo, hi + 1) if j != t
        ]
        ref_plane_sets = [
            split_planes(buf, self.work, self.w, self.h)[0]
            for buf in ref_bufs
        ]

        # 位移以亮度（第 0 平面）为准估计，再按各平面采样比缩放
        tgt_luma = target_planes[0]
        luma_shifts = [
            estimate_shift(ref_planes[0], tgt_luma, luma_w, luma_h,
                           self.max_shift)
            for ref_planes in ref_plane_sets
        ]

        out_planes = []
        for pi, (plane, (pw, ph)) in enumerate(zip(target_planes, self.dims)):
            shifts = [
                (int(round(dx * scale_x[pi])), int(round(dy * scale_y[pi])))
                for dx, dy in luma_shifts
            ]
            refs = [rps[pi] for rps in ref_plane_sets]
            blended = temporal_blend_plane_banded(
                plane, refs, pw, ph, shifts, bands_per_plane[pi],
                self.motion_threshold,
            )
            out_planes.append(blended)
        try:
            enc.stdin.write(join_planes(out_planes, self.work, self.w, self.h))
        except (BrokenPipeError, OSError) as exc:
            raise PipelineError(f"编码器异常退出：{exc}") from exc
