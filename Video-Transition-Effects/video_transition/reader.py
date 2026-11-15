"""ffmpeg 帧读取器：统一的 RGB24 裸流解码与 s16le PCM 解码。

转场帧**只允许**经过 :class:`FrameReader` 进入 Python——它与
``Video-Subtitle-Burning`` 的帧管道是同一套约定：

* 解码端加 ``fps=num/den`` 滤镜把 VFR 归一化为 CFR，第 i 帧的展示
  时刻严格等于 ``i * fps_den / fps_num``，帧序号即时间轴坐标；
* ``-pix_fmt rgb24`` 裸流写到 stdout，帧大小固定为
  ``width * height * 3``，按帧大小整帧读取，残帧视为流截断；
* 上下文管理器保证进程在任何异常路径下都被回收；
* 音频同样解码为带显式采样率/声道的 s16le 裸流（经临时文件中转，
  规避 Windows 上单子进程双字节管道的选择性喂数限制）。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid

from .errors import PipelineError
from .probe import ClipMeta, ffmpeg_bin

__all__ = ["FrameReader", "read_pcm_track", "read_rgb24_frames"]


class FrameReader:
    """单段素材的逐帧 RGB24 读取器（上下文管理器）。

    用法::

        with FrameReader(meta) as reader:
            for frame in reader:
                ...  # frame 为长度 w*h*3 的 bytes，按帧序号单调吐出

    读到的帧数由调用方与 :attr:`ClipMeta.frame_count` 核对；
    提前关闭或解码失败都会在 :meth:`close`/``__exit__`` 时抛出
    :class:`PipelineError`，不会静默吞掉截断流。
    """

    def __init__(self, clip: ClipMeta):
        self.clip = clip
        self.frame_size = clip.width * clip.height * 3
        self._proc: subprocess.Popen | None = None
        self._frames_read = 0

    def __enter__(self) -> "FrameReader":
        # fps 滤镜使用有理数字面量，避免 30000/1001 被浮点截断
        cmd = [
            ffmpeg_bin(), "-v", "error", "-i", self.clip.path,
            "-vf", f"fps={self.clip.fps_literal}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        return self

    def __iter__(self) -> "FrameReader":
        return self

    def __next__(self) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise StopIteration
        frame = self._proc.stdout.read(self.frame_size)
        if len(frame) == 0:
            # 正常 EOF：回收进程并核对退出码
            self._finish()
            raise StopIteration
        if len(frame) != self.frame_size:
            # 残帧 = 流异常截断，杀掉进程后显式失败，不拿半帧凑数
            self._kill()
            raise PipelineError(
                f"{self.clip.path} 解码流在第 {self._frames_read} 帧截断："
                f"读到 {len(frame)} 字节，期望 {self.frame_size}"
            )
        self._frames_read += 1
        return frame

    def __exit__(self, exc_type, exc, tb) -> None:
        # 调用方提前 break / 异常退出时也要回收解码进程
        already_done = self._proc is None
        if not already_done:
            if exc_type is None:
                self._finish()
            else:
                self._kill()

    @property
    def frames_read(self) -> int:
        """已成功读出的整帧数量。"""
        return self._frames_read

    def _finish(self) -> None:
        """EOF 路径：关闭管道并等待退出，非零退出码转成 PipelineError。"""
        proc = self._proc
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        code = proc.wait()
        self._proc = None
        if code != 0:
            raise PipelineError(
                f"解码器异常退出 (exit {code}): {self.clip.path}"
            )

    def _kill(self) -> None:
        """异常路径：直接杀进程，避免它阻塞在写满的 stdout 管道上。"""
        proc = self._proc
        if proc is None:
            return
        proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()
        self._proc = None


def read_rgb24_frames(clip: ClipMeta) -> list[bytes]:
    """把整段素材一次性读成帧列表（供测试与小素材使用）。

    生产路径优先直接迭代 :class:`FrameReader`，避免双倍内存；
    本函数只在需要随机访问两段素材同一帧号时使用。
    """
    with FrameReader(clip) as reader:
        return list(reader)


def read_pcm_track(
    clip: ClipMeta, sample_rate: int, channels: int, tmpdir: str
) -> str:
    """把素材音轨解码为 s16le 双声道 PCM 临时文件，返回文件路径。

    无音轨（``clip.has_audio`` 为假）时生成**等视频时长的静音 PCM**，
    让后续淡化拼装无需区分「有无音轨」；有音轨但解码失败同样回退静音，
    与字幕项目「音频抽取失败按静默处理」的策略一致。

    ``-ac 2`` 强制双声道、``-ar`` 统一采样率、``-af fps 对齐用的 aresample``
    由 ffmpeg 内部完成；输出长度由调用方按帧边界重新切片，不依赖
    容器时长元数据。
    """
    expected_samples = (
        # 视频帧数对应的采样总数（上界，多出的尾部在切片时忽略）
        clip.frame_count * sample_rate * clip.fps_den // clip.fps_num
        + sample_rate  # 预留 1 秒余量，吸收解码器首尾填充
    )
    # 用 uuid 区分同名素材，避免不同目录下的 basename 冲突
    pcm_path = os.path.join(tmpdir, f"track_{uuid.uuid4().hex}_{sample_rate}.s16.pcm")

    if not clip.has_audio:
        with open(pcm_path, "wb") as fh:
            fh.write(b"\x00" * expected_samples * channels * 2)
        return pcm_path

    cmd = [
        ffmpeg_bin(), "-y", "-v", "error", "-i", clip.path,
        "-vn",
        "-af", f"aresample={sample_rate}",
        "-ac", str(channels),
        "-f", "s16le", "-acodec", "pcm_s16le", pcm_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except OSError:
        proc = None
    if proc is None or proc.returncode != 0 or not os.path.isfile(pcm_path):
        # 解码失败回退静音轨：视频转场不应该因为一条坏音轨整体失败
        with open(pcm_path, "wb") as fh:
            fh.write(b"\x00" * expected_samples * channels * 2)
        return pcm_path
    return pcm_path


def load_pcm_bytes(path: str) -> bytes:
    """读取整轨 PCM 到内存（音频拼装前的一次性加载）。"""
    with open(path, "rb") as fh:
        return fh.read()
