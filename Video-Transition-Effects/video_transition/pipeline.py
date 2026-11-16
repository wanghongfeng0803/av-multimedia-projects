"""交叉淡化主管道：双路解码 → 逐帧混合 → 编码，音轨同帧淡化。

进程/线程模型
=============

::

    解码线程 A：FrameReader(A) ──有界队列 a──┐
                                            ├─ 主线程按 frame_jobs 取帧
    解码线程 B：FrameReader(B) ──有界队列 b──┘      │
                                            纯段直通 / 重叠段整数混合
                                                    │
                                          ffmpeg 编码进程 stdin
    音频：两路 ffmpeg 解 s16le PCM → 临时文件
          → assemble_audio 整轨淡化 → PCM 临时文件 → 编码器第二输入

关键正确性约束
==============

* 转场帧**只经过** :class:`~video_transition.reader.FrameReader`，
  不存在第二条取帧路径；
* 主线程严格按 :func:`~video_transition.plan.frame_jobs` 的指令取帧，
  纯段直通、重叠段混合，取帧序号单调递增，与规划账本逐帧对应；
* 泵结束后核对两路实际解码帧数与探针帧数、实际写出帧数与规划帧数，
  任一不符即 :class:`PipelineError`，杜绝静默丢帧；
* 音频淡化在采样域复用同一张规划表（见 :mod:`video_transition.audio`），
  画面与声音的淡化边界是同一份数学，不会出现「画面淡了声音没淡」。
"""

from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .audio import S16_CHANNELS, assemble_audio, choose_sample_rate
from .blend import blend_rgb24_frame
from .errors import PipelineError
from .plan import (
    KIND_HEAD,
    KIND_OVERLAP,
    TRANSITION_SLIDE,
    TRANSITION_WIPE,
    TransitionPlan,
    build_plan,
    frame_jobs,
)
from .probe import ClipMeta, ffmpeg_bin, probe_clip
from .reader import FrameReader, load_pcm_bytes, read_pcm_track
from .slide import composite_slide_frame
from .wipe import composite_wipe_frame

__all__ = ["TransitionResult", "cross_fade", "FrameWriter"]

#: 有界预取队列容量：只缓存少量帧，长素材也不会把内存撑爆
_QUEUE_MAXSIZE = 8
#: 队列结束哨兵
_SENTINEL = None


@dataclass(frozen=True)
class TransitionResult:
    """一次转场的结构化结果，各段帧数可直接用于逐段核对。"""

    output: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    frames_written: int
    overlap_frames: int
    head_frames: int
    tail_frames: int
    duration_seconds: float
    sample_rate: int
    had_audio: tuple[bool, bool]
    encoder: str
    transition: str = "fade"
    direction: str = "right"
    easing: str = "linear"
    soft_width: int = 0

    @property
    def transition_label(self) -> str:
        """转场中文标签（淡化 / 滑动-方向 / 擦除-方向-软边 / 曲线），供账目打印。"""
        if self.transition == TRANSITION_SLIDE:
            return f"滑动({self.direction},{self.easing})"
        if self.transition == TRANSITION_WIPE:
            edge = "硬边" if self.soft_width == 0 else f"软边{self.soft_width}px"
            return f"擦除({self.direction},{edge},{self.easing})"
        return "交叉淡化"

    def describe_segments(self) -> str:
        """生成可人工核对的逐段账目字符串。"""
        return (
            f"head={self.head_frames} 帧(纯A) + "
            f"overlap={self.overlap_frames} 帧({self.transition_label}) + "
            f"tail={self.tail_frames} 帧(纯B) = {self.frames_written} 帧；"
            f"时长 {self.duration_seconds:.3f}s"
        )


class FrameWriter:
    """RGB24 裸流编码写入器（上下文管理器）。

    编码器同时吃一路视频裸流（stdin）与一路已淡化的 s16le PCM 文件，
    输出 H.264 + AAC；容器按输出扩展名选择 mp4/mkv。
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps_literal: str,
        pcm_path: str,
        sample_rate: int,
        *,
        crf: int,
        preset: str,
    ):
        cmd = [
            ffmpeg_bin(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", fps_literal, "-i", "-",
            "-f", "s16le", "-ar", str(sample_rate),
            "-ac", str(S16_CHANNELS), "-i", pcm_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-map_metadata", "-1",
        ]
        if Path(output_path).suffix.lower() == ".mkv":
            cmd += ["-f", "matroska"]
        cmd.append(output_path)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frame_size = width * height * 3

    def write_frame(self, frame: bytes | bytearray) -> None:
        """写出一帧；编码器已退出时抛 :class:`PipelineError`。"""
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame)
        except BrokenPipeError as exc:
            self.kill()
            raise PipelineError("编码进程提前退出，管道已断开") from exc

    def close(self) -> int:
        """关闭 stdin 让编码器读完 EOF 正常收尾，返回退出码。"""
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        return self._proc.wait()

    def kill(self) -> None:
        """异常路径：强杀编码器。"""
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.kill()


def _reader_worker(
    clip: ClipMeta, out_queue: "queue.Queue[bytes | None]"
) -> tuple[threading.Thread, "queue.Queue[bytes | None]", list[BaseException], list[int]]:
    """启动一个解码线程：把素材逐帧推入有界队列，EOF/异常放哨兵。

    返回值末项是单元素列表，线程结束后写入**实际解码出的整帧数**，
    供主线程与探针帧数做精确核对（多解/少解都算失败）。
    """
    errors: list[BaseException] = []
    actual_count: list[int] = [0]

    def run() -> None:
        try:
            with FrameReader(clip) as reader:
                for frame in reader:
                    out_queue.put(frame)
                actual_count[0] = reader.frames_read
        except BaseException as exc:  # 解码失败也要放哨兵，防止主线程永久阻塞
            errors.append(exc)
        finally:
            out_queue.put(_SENTINEL)

    thread = threading.Thread(
        target=run, name=f"decode-{os.path.basename(clip.path)}", daemon=True
    )
    thread.start()
    return thread, out_queue, errors, actual_count


def _next_frame(
    q: "queue.Queue[bytes | None]", label: str
) -> bytes | None:
    """阻塞取一帧；取到哨兵返回 None（调用方据此判断异常截断）。"""
    item = q.get()
    if item is _SENTINEL:
        return None
    return item


def cross_fade(
    path_a: str,
    path_b: str,
    output_path: str,
    *,
    overlap_seconds: float | None = None,
    overlap_frames: int | None = None,
    transition: str = "fade",
    direction: str = "right",
    easing: str = "linear",
    soft_width: int = 0,
    crf: int = 23,
    preset: str = "medium",
    overwrite: bool = False,
    progress_callback=None,
) -> TransitionResult:
    """在两段素材之间做转场，返回逐段可核对的结果摘要。

    默认 ``transition="fade"`` 为历史交叉淡化，行为与输出逐字节不变；
    ``transition="slide"`` 时按 ``direction``（right/left/down/up）做
    四方向滑动、``easing`` 选择速度曲线；``transition="wipe"`` 时两帧
    固定、由移动掩码擦除揭画，``soft_width`` 配置软边宽度（0=硬边，
    超过运动轴会被钳制并告警）。三种转场共用同一份帧域账本
    （成片总帧数恒为 ``Na + Nb - No``，重叠区之外一帧不丢）、同一套
    帧混合器与同一条音频进度曲线。

    :raises ProbeError: 素材无法探测；
    :raises PlanError: 帧率/尺寸不一致或转场/重叠参数非法；
    :raises PipelineError: 解码截断、帧数核对不符或编码失败。
    """
    if not os.path.isfile(path_a):
        raise FileNotFoundError(path_a)
    if not os.path.isfile(path_b):
        raise FileNotFoundError(path_b)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"输出已存在（需 overwrite=True）: {output_path}")

    meta_a = probe_clip(path_a)
    meta_b = probe_clip(path_b)
    plan = build_plan(
        meta_a, meta_b,
        overlap_seconds=overlap_seconds,
        overlap_frames=overlap_frames,
        transition=transition,
        direction=direction,
        easing=easing,
        soft_width=soft_width,
    )
    jobs = frame_jobs(plan)
    if len(jobs) != plan.output_frames:
        raise PipelineError("内部错误：帧任务数与规划帧数不一致")

    sample_rate = choose_sample_rate(plan.fps_num, plan.fps_den)

    with tempfile.TemporaryDirectory(prefix="vtransition_") as tmpdir:
        # 1) 音频先解码为整轨 PCM（无音轨则等长静音），在内存中完成淡化
        pcm_file_a = read_pcm_track(meta_a, sample_rate, S16_CHANNELS, tmpdir)
        pcm_file_b = read_pcm_track(meta_b, sample_rate, S16_CHANNELS, tmpdir)
        pcm_a = load_pcm_bytes(pcm_file_a)
        pcm_b = load_pcm_bytes(pcm_file_b)
        mixed_pcm = assemble_audio(plan, pcm_a, pcm_b, sample_rate)
        mixed_path = os.path.join(tmpdir, "mixed.s16.pcm")
        with open(mixed_path, "wb") as fh:
            fh.write(mixed_pcm)

        # 2) 双路解码线程 + 有界队列
        queue_a: "queue.Queue[bytes | None]" = queue.Queue(_QUEUE_MAXSIZE)
        queue_b: "queue.Queue[bytes | None]" = queue.Queue(_QUEUE_MAXSIZE)
        thread_a, _, errors_a, actual_a = _reader_worker(meta_a, queue_a)
        thread_b, _, errors_b, actual_b = _reader_worker(meta_b, queue_b)

        frames_written = 0
        writer_failed = False
        try:
            with FrameWriter(
                output_path, plan.width, plan.height,
                f"{plan.fps_num}/{plan.fps_den}",
                mixed_path, sample_rate,
                crf=crf, preset=preset,
            ) as writer:
                blend_buf = bytearray(plan.width * plan.height * 3)
                for job in jobs:
                    if job.kind == KIND_HEAD:
                        frame = _next_frame(queue_a, "A")
                        if frame is None:
                            raise PipelineError(
                                f"A 素材在 head 段提前耗尽（期望 {meta_a.frame_count} 帧）"
                            )
                        writer.write_frame(frame)
                    elif job.kind == KIND_OVERLAP:
                        frame_a = _next_frame(queue_a, "A")
                        frame_b = _next_frame(queue_b, "B")
                        if frame_a is None or frame_b is None:
                            raise PipelineError("素材在重叠区提前耗尽")
                        if plan.is_slide:
                            # 滑动：按帧任务里的整数位移做区域合成，
                            # 条带内部仍走 blend_rgb24_frame（端点权重）
                            composite_slide_frame(
                                blend_buf, frame_a, frame_b,
                                plan.width, plan.height,
                                plan.direction, job.shift,
                            )
                        elif plan.is_wipe:
                            # 擦除：固定两帧、按帧任务里的掩码边界与
                            # 软边宽度做分区混合（纯区端点权重、渐变带
                            # 整数权重），同样复用 blend_rgb24_frame
                            composite_wipe_frame(
                                blend_buf, frame_a, frame_b,
                                plan.width, plan.height,
                                plan.direction, job.edge, job.soft_width,
                            )
                        else:
                            blend_rgb24_frame(
                                blend_buf, frame_a, frame_b,
                                job.weight_b_num, job.weight_b_den,
                            )
                        writer.write_frame(blend_buf)
                    else:  # KIND_TAIL
                        frame = _next_frame(queue_b, "B")
                        if frame is None:
                            raise PipelineError(
                                f"B 素材在 tail 段提前耗尽（期望 {meta_b.frame_count} 帧）"
                            )
                        writer.write_frame(frame)
                    frames_written += 1
                    if progress_callback and jobs:
                        progress_callback(frames_written / len(jobs))
                enc_code = writer.close()
        except BaseException:
            writer_failed = True
            # 让解码线程脱离阻塞的 put：排空队列后它们自然走到哨兵
            _drain_queue(queue_a)
            _drain_queue(queue_b)
            thread_a.join(timeout=5)
            thread_b.join(timeout=5)
            raise
        finally:
            if not writer_failed:
                _drain_queue(queue_a)
                _drain_queue(queue_b)
                thread_a.join(timeout=5)
                thread_b.join(timeout=5)

        if enc_code != 0:
            raise PipelineError(f"编码器异常退出 (exit {enc_code})")
        if errors_a or errors_b:
            raise PipelineError(
                f"解码线程异常: A={errors_a or '无'} B={errors_b or '无'}"
            )

        # 3) 帧数三方核对：两路实际解码帧数必须与探针精确相等
        #    （少帧=截断，多帧=探针与流不一致，二者都不允许静默吞掉），
        #    成片写出帧数必须等于规划恒等式 Na + Nb - No。
        if actual_a[0] != meta_a.frame_count:
            raise PipelineError(
                f"A 实际解码帧数 {actual_a[0]} 与探针 {meta_a.frame_count} 不符"
            )
        if actual_b[0] != meta_b.frame_count:
            raise PipelineError(
                f"B 实际解码帧数 {actual_b[0]} 与探针 {meta_b.frame_count} 不符"
            )
        if frames_written != plan.output_frames:
            raise PipelineError(
                f"写出帧数 {frames_written} 与规划 {plan.output_frames} 不符"
            )

    head_len = plan.head.length
    tail_len = plan.tail.length
    return TransitionResult(
        output=output_path,
        width=plan.width,
        height=plan.height,
        fps_num=plan.fps_num,
        fps_den=plan.fps_den,
        frames_written=frames_written,
        overlap_frames=plan.overlap_frames,
        head_frames=head_len,
        tail_frames=tail_len,
        duration_seconds=plan.output_duration,
        sample_rate=sample_rate,
        had_audio=(meta_a.has_audio, meta_b.has_audio),
        encoder="libx264+aac",
        transition=plan.transition,
        direction=plan.direction,
        easing=plan.easing,
        soft_width=plan.soft_width,
    )


def _drain_queue(q: "queue.Queue[bytes | None]") -> None:
    """排空队列直到哨兵，让阻塞在 ``put`` 上的解码线程得以收尾。"""
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
