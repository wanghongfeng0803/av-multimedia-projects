"""音频读写: 优先 soundfile, 不支持的格式回退到 ffmpeg 解码."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# 可直接用 soundfile (libsndfile) 读取的格式
SNDFILE_EXTS = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
# 需要 ffmpeg 解封装的格式 (aac 编码 / 视频容器取音频)
FFMPEG_EXTS = {".aac", ".m4a", ".mp4", ".mov", ".mkv", ".ac3", ".ec3", ".mka"}
SUPPORTED_EXTS = SNDFILE_EXTS | FFMPEG_EXTS


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_ffmpeg(path: Path) -> tuple[int, int]:
    """ffprobe 返回 (采样率, 声道数)。"""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    sr, ch = out.stdout.strip().split(",")[:2]
    return int(sr), int(ch)


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """ffmpeg 解码为 float32 交错 PCM。"""
    sr, ch = _probe_ffmpeg(path)
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-map", "a:0", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(ch), "-ar", str(sr), "-",
        ],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch)
    else:
        x = x[:, None]
    return x, sr


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """读取音频为 (n, ch) float64 [-1, 1] 与采样率。

    常见无损/mp3 走 soundfile; aac/视频容器走 ffmpeg。
    soundfile 失败且 ffmpeg 可用时自动回退。
    """
    path = Path(path)
    ext = path.suffix.lower()
    err: Exception | None = None
    if ext in SNDFILE_EXTS or ext not in FFMPEG_EXTS:
        try:
            x, sr = sf.read(str(path), dtype="float64", always_2d=True)
            return x, sr
        except Exception as e:  # 回退 ffmpeg
            err = e
    if ffmpeg_available():
        try:
            return _decode_ffmpeg(path)
        except Exception as e:
            raise RuntimeError(f"ffmpeg 解码失败: {path.name}: {e}") from e
    if err is not None:
        raise RuntimeError(
            f"无法解码 {path.name} (soundfile: {err}); 该格式需要 ffmpeg, 未在 PATH 中找到"
        )
    raise RuntimeError(f"不支持的格式 {ext}, 且未找到 ffmpeg: {path.name}")


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """写出 24-bit PCM WAV (广播交付常用位深)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_24")
