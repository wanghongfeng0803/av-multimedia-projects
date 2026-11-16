"""抽帧阶段：ffmpeg 解码为 rawvideo rgb24，ffprobe 探测真实时间戳。

本阶段只负责「容器/编码 → 原始 RGB 帧 + 帧清单」，不做任何帧率换算、
缩放或颜色量化，因此产物可被后续任意目标参数复用（三步解耦的第一环）。

两个从 frconv 项目复用来的 ffmpeg/Windows 坑：

- **必须 ``-fps_mode passthrough``**：ffmpeg 默认可能按 CFR 重采样
  （复制/丢帧），使解码帧数与源帧不一致；passthrough 强制逐帧 1:1
  直通。ffmpeg 9 已把旧 ``-vsync`` 改名 ``-fps_mode``。
- **Windows 管道 stderr 死锁**：管道缓冲仅约 4KB，等进程结束再读
  stderr 会在 ffmpeg 输出较多时互相阻塞。这里起守护线程持续排空
  stderr，进程失败时再汇总抛出。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = [
    "ExtractError",
    "FrameInfo",
    "Manifest",
    "probe_frames",
    "extract_rgb_frames",
]


class ExtractError(RuntimeError):
    """ffprobe/ffmpeg 探测或抽帧失败。"""


@dataclass(frozen=True)
class FrameInfo:
    """单帧的时间戳信息（来自 ffprobe，不由 ffmpeg 管线推断）。

    :ivar index: 帧序号（0 起算）。
    :ivar pts_time: 显示时间戳（秒）。
    :ivar duration_time: 该帧持续时长（秒）；探测缺失时由上层按
        相邻 PTS 间隔或 CFR 帧率补算。
    """

    index: int
    pts_time: float
    duration_time: float


@dataclass(frozen=True)
class Manifest:
    """抽帧产物清单：后续量化/编码只依赖它，不再回看源视频。

    :ivar width: 帧宽（像素）。
    :ivar height: 帧高（像素）。
    :ivar pixel_format: 原始像素格式（固定 ``rgb24``，每像素 3 字节）。
    :ivar fps: 源平均帧率（每秒帧数，浮点）。
    :ivar frames: 逐帧时间戳清单。
    :ivar frame_glob: 帧文件相对工作目录的命名模板（``frame_%06d.rgb``）。
    """

    width: int
    height: int
    fps: float
    frames: tuple
    pixel_format: str = "rgb24"
    frame_glob: str = "frame_%06d.rgb"

    @property
    def frame_size(self) -> int:
        """单帧 rgb24 字节数。"""
        return self.width * self.height * 3

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        """源时长（秒），口径为末帧 PTS + 末帧持续时间。"""
        last = self.frames[-1]
        return last.pts_time + last.duration_time

    def frame_path(self, work_dir: Path, index: int) -> Path:
        """第 ``index`` 帧的原始字节文件路径。"""
        return Path(work_dir) / (self.frame_glob % (index + 1))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frames"] = [asdict(f) for f in self.frames]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        frames = tuple(FrameInfo(**f) for f in d["frames"])
        return cls(
            width=d["width"],
            height=d["height"],
            fps=d["fps"],
            frames=frames,
            pixel_format=d.get("pixel_format", "rgb24"),
            frame_glob=d.get("frame_glob", "frame_%06d.rgb"),
        )


def _check_tools() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ExtractError("ffmpeg/ffprobe 不在 PATH，无法抽帧")


def _run_capture(cmd: list) -> subprocess.CompletedProcess:
    """运行子进程并全程排空 stderr（规避 Windows 管道死锁）。"""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0,
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
    stdout, _ = proc.communicate()  # stderr 由守护线程读取
    t.join(timeout=5)
    if proc.returncode != 0:
        err = b"".join(stderr_chunks).decode("utf-8", "replace")
        raise ExtractError(
            f"命令失败（exit={proc.returncode}）：{' '.join(cmd)}\n{err}"
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, b"")


def probe_frames(input_path: str | Path) -> Manifest:
    """用 ffprobe 读取视频流几何信息与逐帧 PTS/时长。

    时间戳以 ffprobe 的 ``pts_time``/``duration_time`` 为准（VFR 也成立）；
    若流里缺 ``duration_time``，按相邻 PTS 间隔补算，末帧回退到平均
    帧率间隔。
    """
    _check_tools()
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ExtractError(f"输入文件不存在：{input_path}")

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:frame=pts_time,duration_time",
        "-of", "json", str(input_path),
    ]
    res = _run_capture(cmd)
    data = json.loads(res.stdout.decode("utf-8"))
    try:
        stream = data["streams"][0]
        raw_frames = data["frames"]
    except (KeyError, IndexError) as exc:
        raise ExtractError(f"ffprobe 未发现视频流：{input_path}") from exc

    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0
    if fps <= 0:
        raise ExtractError(f"无法确定源帧率：avg_frame_rate={stream['avg_frame_rate']!r}")

    pts_list = [float(f["pts_time"]) for f in raw_frames]
    durations: list = []
    for i, f in enumerate(raw_frames):
        if "duration_time" in f:
            durations.append(float(f["duration_time"]))
        elif i + 1 < len(pts_list):
            durations.append(pts_list[i + 1] - pts_list[i])
        else:
            durations.append(1.0 / fps)
    frames = tuple(
        FrameInfo(index=i, pts_time=pts_list[i], duration_time=durations[i])
        for i in range(len(pts_list))
    )
    if not frames:
        raise ExtractError(f"源视频不含任何帧：{input_path}")
    return Manifest(width=width, height=height, fps=fps, frames=frames)


def extract_rgb_frames(input_path: str | Path, work_dir: str | Path,
                       manifest: Manifest | None = None) -> Manifest:
    """把源视频逐帧解码为独立 ``frame_XXXXXX.rgb``（rgb24）文件。

    使用 ``-fps_mode passthrough`` 保证解码帧与 ffprobe 清单 1:1 对应；
    ``image2`` 配合 rawvideo 每帧落一个文件，便于后续按任意顺序、任意
    目标帧率重复采样而不必重新解码源视频。

    :param manifest: 已探测的清单；为 None 时先 :func:`probe_frames`。
    :return: 与落盘帧一致的清单（即传入或新探测的 manifest）。
    """
    _check_tools()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        manifest = probe_frames(input_path)

    pattern = work_dir / manifest.frame_glob
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-fps_mode", "passthrough",          # 逐帧 1:1，禁止 CFR 重采样
        "-pix_fmt", "rgb24",
        "-f", "image2",
        "-vcodec", "rawvideo",
        str(pattern),
    ]
    _run_capture(cmd)

    # 校验落盘帧数与字节数，避免 ffmpeg 静默丢帧污染后续时序。
    expected = manifest.frame_count
    actual = len(list(work_dir.glob("frame_*.rgb")))
    if actual != expected:
        raise ExtractError(
            f"抽帧数量不一致：ffprobe 报 {expected} 帧，落盘 {actual} 帧"
        )
    want_size = manifest.frame_size
    for i in range(expected):
        p = manifest.frame_path(work_dir, i)
        if p.stat().st_size != want_size:
            raise ExtractError(
                f"帧字节数异常：{p.name} 实际 {p.stat().st_size}，"
                f"期望 {want_size}（{manifest.width}x{manifest.height} rgb24）"
            )
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest
