"""ffmpeg 帧管道：解码 -> 逐帧调度渲染 -> 编码，音频单独直通。

整体数据流（两个独立的 ffmpeg 子进程，经命名文件中转音频）::

    输入视频
      ├─ 进程 A：ffmpeg -i input -f rawvideo -pix_fmt rgb24 -vf fps=R -
      │           → Python 逐帧读取 RGB24 → 调度器查询字幕 → 像素混合
      ├─ 进程 B：ffmpeg -f rawvideo -pix_fmt rgb24 -i -  + 提取出的音频
      │           → H.264/MP4 输出
      └─ 音频：先用第三个 ffmpeg 解为 m4a（无音轨则跳过）

关键决策：

* **解码端加 ``fps=R`` 滤镜**：把可变帧率(VFR)输入归一化为恒定帧率，
  之后帧序号 ``i`` 与时刻 ``i/R`` 严格对应，调度器的时间轴假设成立；
* **时间戳由帧序号推导，不读容器 PTS**：rawvideo 管道本身不带时间戳，
  这迫使时间对齐完全建立在显式的帧率数学上，语义清晰且可单测；
* **音频走文件中转而非双管道**：Windows 上无法对一个子进程同时建立两条
  字节管道做选择性喂数据，文件中转跨平台且实现简单，代价是少量临时磁盘。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .fonts import Font, load_font, resolve_font
from .render import OverlayRenderer, Style
from .srt import Cue
from .style import SubtitleStyle
from .timeline import FrameTimeline, SubtitleScheduler

__all__ = ["BurnResult", "probe_video", "burn_subtitles"]

#: 进度回调，参数为 0.0~1.0 的完成比例
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class BurnResult:
    output: str
    width: int
    height: int
    fps: float
    frames_written: int
    cues: int
    encoder: str
    had_audio: bool


# ---------------------------------------------------------------------------
# ffprobe：取帧率/尺寸/时长/音轨
# ---------------------------------------------------------------------------

def _ffprobe_bin() -> str:
    return os.environ.get("FFPROBE", "ffprobe")


def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG", "ffmpeg")


def probe_video(path: str) -> FrameTimeline:
    """用 ffprobe 读取视频流参数，构造 :class:`FrameTimeline`。

    帧率优先取 avg_frame_rate（r_frame_rate 在某些容器里是保留给计时的
    怪值，如 1/90000）；总帧数取 nb_frames，缺失时用时长反推。
    """
    cmd = [
        _ffprobe_bin(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of", "json", path,
    ]
    data = json.loads(subprocess.check_output(cmd))
    stream = data["streams"][0]
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(
        stream.get("r_frame_rate")
    )
    if not fps or fps <= 0:
        raise ValueError(f"无法从 ffprobe 获得有效帧率: {path}")

    nb = int(stream["nb_frames"]) if stream.get("nb_frames", "N/A").isdigit() else 0
    if nb == 0 and stream.get("duration", "N/A").replace(".", "", 1).isdigit():
        nb = round(float(stream["duration"]) * fps)
    return FrameTimeline(
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=nb,
    )


def _parse_rate(rate: str | None) -> float:
    """把 ``num/den`` 形式的帧率字符串换算为浮点；``0/0`` 等无效值返回 0。"""
    if not rate or "/" not in rate:
        return 0.0
    num, den = rate.split("/", 1)
    den_f = float(den)
    return float(num) / den_f if den_f else 0.0


def _has_audio(path: str) -> bool:
    cmd = [
        _ffprobe_bin(), "-v", "error",
        "-select_streams", "a", "-show_entries", "stream=codec_type",
        "-of", "json", path,
    ]
    try:
        data = json.loads(subprocess.check_output(cmd))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False
    return bool(data.get("streams"))


# ---------------------------------------------------------------------------
# 子进程封装
# ---------------------------------------------------------------------------

def _run_silent(cmd: list[str]) -> None:
    """辅助型短任务（抽音频）用：失败时带上 stderr 尾部抛异常。"""
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"ffmpeg 执行失败 (exit {proc.returncode}):\n{tail}")


def _extract_audio(input_path: str, tmpdir: str) -> str | None:
    """把音轨抽成独立 m4a 文件；无音轨或抽取失败返回 None（按静默处理）。"""
    if not _has_audio(input_path):
        return None
    audio_path = os.path.join(tmpdir, "audio.m4a")
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error", "-i", input_path,
        "-vn", "-c:a", "aac", "-b:a", "192k", audio_path,
    ]
    try:
        _run_silent(cmd)
    except RuntimeError:
        return None
    return audio_path if os.path.isfile(audio_path) else None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def burn_subtitles(
    input_path: str,
    output_path: str,
    cues: list[Cue],
    *,
    font: Font | None = None,
    style: Style | SubtitleStyle | None = None,
    crf: int = 23,
    preset: str = "medium",
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> BurnResult:
    """把字幕烧录进视频，返回执行结果摘要。

    :param cues: 已解析的字幕（毫秒时间戳）
    :param font: 自定义字体；默认按样式自动选择——新版
        :class:`~subburn.style.SubtitleStyle` 走字体族回退链
        （:func:`subburn.fonts.resolve_font`），旧版 :class:`Style` 沿用
        :func:`subburn.fonts.load_font` 的系统探测；显式传入时优先级最高
    :param style: 旧版 :class:`Style` 或新版 :class:`SubtitleStyle`
    :param crf/preset: x264 质量与速度档
    :param overwrite: 输出已存在时是否覆盖
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"输出已存在（需 overwrite=True）: {output_path}")

    timeline = probe_video(input_path)
    w, h, fps = timeline.width, timeline.height, timeline.fps
    if font is None:
        if isinstance(style, SubtitleStyle):
            # 新版样式：字体族/字号/字重 -> 回退链解析（缺失自动降级）
            font = resolve_font(style.font, frame_height=h)
        else:
            font = load_font(size=max(20, h // 24))
    renderer = OverlayRenderer(font, style or Style())
    scheduler = SubtitleScheduler(cues, fps=fps)

    with tempfile.TemporaryDirectory(prefix="subburn_") as tmpdir:
        audio_path = _extract_audio(input_path, tmpdir)
        frames_written = _pump_frames(
            input_path, output_path, timeline, renderer, scheduler,
            audio_path, crf, preset, progress_callback,
        )

    return BurnResult(
        output=output_path,
        width=w, height=h, fps=fps,
        frames_written=frames_written,
        cues=len(cues),
        encoder="libx264",
        had_audio=audio_path is not None,
    )


def _pump_frames(
    input_path: str, output_path: str, timeline: FrameTimeline,
    renderer: OverlayRenderer, scheduler: SubtitleScheduler,
    audio_path: str | None, crf: int, preset: str,
    progress_callback: ProgressCallback | None,
) -> int:
    """拉起解码/编码两个进程，按帧大小在管道间搬运并就地渲染。"""
    w, h, fps = timeline.width, timeline.height, timeline.fps
    frame_bytes = w * h * 3

    # 解码端：fps 滤镜把 VFR 归一化为 CFR，rgb24 裸流从 stdout 吐出
    dec_cmd = [
        _ffmpeg_bin(), "-v", "error", "-i", input_path,
        "-vf", f"fps={fps:g}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    # 编码端：从 stdin 吃裸流；有音频则并入 m4a，MP4 无法容纳时由 mov 标签提示
    enc_cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
    ]
    if audio_path:
        enc_cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "copy"]
    enc_cmd += [
        "-map_metadata", "-1",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    if Path(output_path).suffix.lower() == ".mkv":
        enc_cmd += ["-f", "matroska"]
    enc_cmd.append(output_path)

    decoder = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    assert decoder.stdout is not None and encoder.stdin is not None

    total = timeline.frame_count
    frame_idx = 0
    try:
        while True:
            frame = decoder.stdout.read(frame_bytes)
            if len(frame) < frame_bytes:
                break  # 读到 EOF（最后一帧不足整帧说明流异常截断，丢弃残帧）
            buf = bytearray(frame)
            active = scheduler.active_cues_at(frame_idx)
            if active:
                renderer.render_frame(buf, w, h, [c.text for c in active])
            encoder.stdin.write(buf)
            frame_idx += 1
            if progress_callback and total:
                progress_callback(min(1.0, frame_idx / total))
    except BrokenPipeError:
        # 编码端提前退出：必须先杀掉解码端，否则它会阻塞在写满的 stdout
        # 管道上，导致下面的 wait() 永久死锁
        decoder.kill()
        encoder.kill()
    finally:
        # 先关 stdin 让编码器读到 EOF 正常收尾，再等待两个进程退出
        encoder.stdin.close()
        dec_wait = decoder.wait()
        enc_wait = encoder.wait()

    if enc_wait != 0 or dec_wait != 0:
        raise RuntimeError(
            f"ffmpeg 管道失败 decoder={dec_wait} encoder={enc_wait}；"
            "可用 -v error 以上日志级别手动复现命令排查"
        )
    if progress_callback:
        progress_callback(1.0)
    return frame_idx
