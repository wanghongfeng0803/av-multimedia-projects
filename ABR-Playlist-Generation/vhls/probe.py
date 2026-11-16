"""探测阶段：ffprobe 读取源媒体的流信息与逐帧 PTS/时长。

时间戳一律以 ffprobe 的 ``pts_time``/``duration_time`` 为准（VFR 也成立）；
本模块对外提供的时间仍是**秒（浮点）**，上层 ``timeline`` 会在唯一入口
统一换算成整数毫秒，全链路不再做第二次浮点运算，从根上杜绝「每片四舍五入
一次、误差逐片累积」的漂移。

与兄弟项目 vgif 复用同一个 Windows 坑的解法：子进程 stderr 起守护线程
持续排空，避免管道缓冲（约 4KB）写满后 ffmpeg/ffprobe 与本进程互锁。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ProbeError",
    "FrameInfo",
    "SourceInfo",
    "probe_source",
]


class ProbeError(RuntimeError):
    """ffprobe 探测失败或源文件不含可用视频流。"""


@dataclass(frozen=True)
class FrameInfo:
    """单帧时间戳信息（来自 ffprobe，不由 ffmpeg 管线推断）。

    :ivar index: 帧序号（0 起算）。
    :ivar pts_time: 显示时间戳（秒）。
    :ivar duration_time: 该帧持续时长（秒）；探测缺失时按相邻 PTS 间隔
        或平均帧率间隔补算。
    """

    index: int
    pts_time: float
    duration_time: float


@dataclass(frozen=True)
class SourceInfo:
    """源媒体探测结果，后续时间轴规划与切割只依赖它。

    :ivar path: 源文件路径。
    :ivar width: 视频帧宽（像素）。
    :ivar height: 视频帧高（像素）。
    :ivar fps: 平均帧率（每秒帧数，浮点）。
    :ivar frames: 逐帧时间戳清单（按 PTS 严格递增）。
    :ivar has_audio: 是否存在至少一路音频流。
    """

    path: Path
    width: int
    height: int
    fps: float
    frames: tuple
    has_audio: bool

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        """源时长（秒），口径为末帧 PTS + 末帧持续时间。"""
        last = self.frames[-1]
        return last.pts_time + last.duration_time


def check_tools() -> None:
    """确认 ffmpeg/ffprobe 都在 PATH，缺失时抛 :class:`ProbeError`。"""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ProbeError("ffmpeg/ffprobe 不在 PATH，无法打包 HLS")


def run_capture(cmd: list) -> subprocess.CompletedProcess:
    """运行子进程并全程排空 stderr（规避 Windows 管道死锁）。"""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    stderr_chunks: list = []

    def _drain():
        assert proc.stderr is not None
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    stdout, _ = proc.communicate()
    t.join(timeout=5)
    if proc.returncode != 0:
        err = b"".join(stderr_chunks).decode("utf-8", "replace")
        raise ProbeError(
            f"命令失败（exit={proc.returncode}）：{' '.join(cmd)}\n{err}"
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, b"")


def _probe_streams(input_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate",
        "-of", "json", str(input_path),
    ]
    data = json.loads(run_capture(cmd).stdout.decode("utf-8"))
    return data


def _probe_frames(input_path: Path) -> list:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time,duration_time",
        "-of", "json", str(input_path),
    ]
    data = json.loads(run_capture(cmd).stdout.decode("utf-8"))
    return data.get("frames", [])


def probe_source(input_path: str | Path) -> SourceInfo:
    """探测源视频的几何信息、帧率、逐帧时间戳与音频存在性。

    逐帧 ``duration_time`` 缺失时按相邻 PTS 间隔补算，末帧回退到平均
    帧率间隔（与 vgif 同口径）。帧 PTS 必须严格递增，否则时间轴无法
    规划，直接报错而不是产出错乱的切片。
    """
    check_tools()
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ProbeError(f"输入文件不存在：{input_path}")

    streams_data = _probe_streams(input_path)
    streams = streams_data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"ffprobe 未发现视频流：{input_path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width = int(video["width"])
    height = int(video["height"])
    num, den = str(video["avg_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0
    if fps <= 0:
        raise ProbeError(
            f"无法确定源帧率：avg_frame_rate={video.get('avg_frame_rate')!r}"
        )

    raw_frames = _probe_frames(input_path)
    if not raw_frames:
        raise ProbeError(f"源视频不含任何帧：{input_path}")

    pts_list = [float(f["pts_time"]) for f in raw_frames]
    durations: list = []
    for i, frame in enumerate(raw_frames):
        if "duration_time" in frame:
            durations.append(float(frame["duration_time"]))
        elif i + 1 < len(pts_list):
            durations.append(pts_list[i + 1] - pts_list[i])
        else:
            durations.append(1.0 / fps)

    frames = tuple(
        FrameInfo(index=i, pts_time=pts_list[i], duration_time=durations[i])
        for i in range(len(pts_list))
    )
    for prev, cur in zip(frames, frames[1:]):
        if cur.pts_time <= prev.pts_time:
            raise ProbeError(
                f"帧 PTS 非严格递增：第 {prev.index} 帧 {prev.pts_time} "
                f">= 第 {cur.index} 帧 {cur.pts_time}，无法规划时间轴"
            )
    return SourceInfo(
        path=input_path, width=width, height=height, fps=fps,
        frames=frames, has_audio=has_audio,
    )
