"""ffprobe 探针：读取素材的视频/音频参数，供转场规划使用。

设计与 ``Video-Subtitle-Burning`` 的探针保持同一套约定：

* 帧率优先取 ``avg_frame_rate`` 并**保留 num/den 有理数字面量**
  （``30000/1001`` 这类 NTSC 速率不能先转浮点再比较，否则两个素材
  之间会出现 1e-6 量级的伪差异）；
* 总帧数优先取 ``nb_frames``，缺失时用 ``duration * fps`` 反推；
* 音轨是否存在单独探测——后续音频缺失按「静音轨」处理，不影响
  视频规划，但必须在探针阶段就确定下来。
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from math import gcd

from .errors import ProbeError

__all__ = ["ClipMeta", "probe_clip", "parse_rate", "ffmpeg_bin", "ffprobe_bin"]


def ffmpeg_bin() -> str:
    """ffmpeg 可执行名，允许用环境变量覆盖（测试/多版本共存）。"""
    return os.environ.get("FFMPEG", "ffmpeg")


def ffprobe_bin() -> str:
    """ffprobe 可执行名，允许用环境变量覆盖。"""
    return os.environ.get("FFPROBE", "ffprobe")


def parse_rate(rate: str | None) -> tuple[int, int]:
    """把 ``num/den`` 帧率字符串解析为整数对并约分到最简。

    ``0/0``、``N/A`` 等无效值抛出 :class:`ProbeError`；
    不带斜杠的整数字面量按 ``N/1`` 处理。
    """
    if not rate or rate in ("N/A", "0/0"):
        raise ProbeError(f"无法解析帧率字面量: {rate!r}")
    if "/" not in rate:
        num = int(rate)
        return num, 1
    num_s, den_s = rate.split("/", 1)
    num, den = int(num_s), int(den_s)
    if den == 0 or num <= 0:
        raise ProbeError(f"非法帧率字面量: {rate!r}")
    g = gcd(num, den)
    return num // g, den // g


@dataclass(frozen=True)
class ClipMeta:
    """单段素材的探针结果（不可变）。

    :param fps_num/fps_den: 帧率的有理数分子/分母，时长与采样边界
        全部基于它做整数运算，避免浮点累积误差；
    :param frame_count: 视频总帧数（探针缺失时按时长反推的估计值）；
    :param has_audio: 是否存在音频流；缺失时管道侧用静音 PCM 补齐。
    """

    path: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    frame_count: int
    duration: float
    has_audio: bool

    @property
    def fps(self) -> float:
        """帧率浮点表示，仅用于展示与秒级估算，不参与精确对齐。"""
        return self.fps_num / self.fps_den

    @property
    def fps_literal(self) -> str:
        """传给 ffmpeg ``-r``/``fps=`` 的有理数字面量。"""
        return f"{self.fps_num}/{self.fps_den}"


def _probe_stream(path: str, select: str, entries: str) -> dict:
    cmd = [
        ffprobe_bin(), "-v", "error",
        "-select_streams", select,
        "-show_entries", entries,
        "-of", "json", path,
    ]
    try:
        out = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as exc:
        raise ProbeError(
            f"ffprobe 执行失败 (exit {exc.returncode}): {path}"
        ) from exc
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe 输出无法解析: {path}") from exc


def probe_clip(path: str) -> ClipMeta:
    """探测单个素材，返回 :class:`ClipMeta`。

    视频流缺失或参数不全时抛 :class:`ProbeError`；音频流缺失不报错，
    标记 ``has_audio=False`` 交由管道按静音处理。
    """
    if not os.path.isfile(path):
        raise ProbeError(f"素材文件不存在: {path}")

    data = _probe_stream(
        path, "v:0",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
    )
    streams = data.get("streams") or []
    if not streams:
        raise ProbeError(f"素材中没有视频流: {path}")
    st = streams[0]

    try:
        rate = st.get("avg_frame_rate")
        if rate in (None, "0/0"):
            rate = st.get("r_frame_rate")
        fps_num, fps_den = parse_rate(rate)
        width = int(st["width"])
        height = int(st["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"视频流参数不完整: {path}") from exc

    nb_raw = st.get("nb_frames", "N/A")
    if isinstance(nb_raw, str) and nb_raw.isdigit():
        frame_count = int(nb_raw)
    else:
        dur_raw = st.get("duration")
        try:
            duration = float(dur_raw)
        except (TypeError, ValueError):
            raise ProbeError(f"无法确定素材总帧数或时长: {path}")
        frame_count = round(duration * fps_num / fps_den)
    if frame_count <= 0:
        raise ProbeError(f"素材总帧数非正: {path}")

    try:
        duration = float(st.get("duration"))
    except (TypeError, ValueError):
        duration = frame_count * fps_den / fps_num

    audio_data = _probe_stream(path, "a", "stream=codec_type")
    has_audio = bool(audio_data.get("streams"))

    return ClipMeta(
        path=path,
        width=width,
        height=height,
        fps_num=fps_num,
        fps_den=fps_den,
        frame_count=frame_count,
        duration=duration,
        has_audio=has_audio,
    )
