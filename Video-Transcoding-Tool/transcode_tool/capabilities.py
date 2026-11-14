"""本机 ffmpeg 编码器能力探测。

直接解析 ``ffmpeg -hide_banner -encoders`` 的原始文本输出（不依赖第三方库），
判断各编码器是否存在且具备**编码**能力（标志位含 ``E``）。

结果按 ffmpeg 可执行文件路径缓存，避免一次转码内重复拉起进程；
探测进程本身失败时采用“乐观降级”策略：假定编码器存在，让后续真正的
转码命令给出权威结果，而不是因为一次探测抖动就拒绝工作。
"""

from __future__ import annotations

import subprocess
import threading

from .binaries import find_ffmpeg
from .codecs import CodecSpec, EncoderSpec
from .errors import EncoderUnavailableError

# ffmpeg -encoders 表头之后的数据行至少形如：
# " V....D libx264   libx264 H.264 ..."
# 第 1 列是标志位（首位 V/A/S 表示视频/音频/字幕类型，其后是能力位），
# 第 2 列才是编码器名。
# 注意：该表列出的条目本身就都是编码器（解码器在 -decoders 中），现代 ffmpeg
# （9.x）不再输出历史上的 'E' 能力位，因此只按类型位判定，'E' 仅作老版本兼容。
_ENCODER_HEADER = "Encoders:"
_lock = threading.Lock()
_cache: dict[str, frozenset[str] | None] = {}


def parse_encoders_output(text: str) -> frozenset[str]:
    """解析 ``ffmpeg -encoders`` 文本，返回可用编码器名集合。

    纯函数：不接触进程，喂入任意异常文本都不会抛出，只尽力解析。

    现代 ffmpeg（9.x）的该表条目全部是编码器且标志位不含历史上的 ``E``；
    老版本则在能力位中用 ``E`` 标注可编码。这里先扫描表头图例判断当前
    版本采用哪种体系，再决定是否强制要求 ``E`` 位。
    """
    encoders: set[str] = set()
    if not text:
        return frozenset()

    in_table = False
    in_legend = False
    requires_e_bit = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not in_table:
            if _ENCODER_HEADER in line:
                in_table = True
                in_legend = True
            continue

        stripped = line.strip()
        if in_legend:
            if set(stripped) == {"-"}:  # 图例结束分隔线 ------
                in_legend = False
                continue
            # 图例形如 " ....E. = Supports ... encoding"
            if "E" in stripped and "=" in stripped:
                requires_e_bit = True
            continue

        if not stripped:
            continue
        parts = stripped.split()
        # 数据行：标志位（如 V....D / A.....）+ 编码器名 + 描述
        if len(parts) < 3:
            continue
        flags, name = parts[0], parts[1]
        if not flags or flags[0] not in "VAS":
            continue
        if requires_e_bit and "E" not in flags[1:]:
            continue  # 老版本体系下的非编码条目
        if not _is_encoder_token(name):
            continue
        encoders.add(name)
    return frozenset(encoders)


def _is_encoder_token(token: str) -> bool:
    """编码器名只含字母/数字/下划线/连字符，排除误把描述词当名称。"""
    return bool(token) and all(ch.isalnum() or ch in "_-" for ch in token)


def available_encoders(ffmpeg_path: str | None = None) -> frozenset[str] | None:
    """返回本机可用编码器集合。

    :returns: 编码器名集合；探测进程失败时返回 None，表示“未知，按可用处理”。
    """
    ffmpeg = ffmpeg_path or find_ffmpeg()
    with _lock:
        cached = _cache.get(ffmpeg)
        if cached is not None:
            return cached
        if ffmpeg in _cache:  # 之前探测失败（显式缓存了 None）
            return None

    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        result: frozenset[str] | None = None
    else:
        if proc.returncode != 0:
            result = None
        else:
            result = parse_encoders_output(proc.stdout) or frozenset()

    with _lock:
        _cache[ffmpeg] = result
    return result


def reset_cache() -> None:
    """清空探测缓存（主要供测试使用）。"""
    with _lock:
        _cache.clear()


def is_encoder_available(encoder_name: str, ffmpeg_path: str | None = None) -> bool:
    """判断单个编码器是否可用；探测失败时乐观返回 True。"""
    available = available_encoders(ffmpeg_path)
    if available is None:
        return True
    return encoder_name in available


def select_video_encoder(
    codec: CodecSpec, ffmpeg_path: str | None = None
) -> EncoderSpec:
    """按优先级在编码的候选链中挑出第一个本机可用的视频编码器。

    :raises EncoderUnavailableError: 候选链上没有任何可用编码器
    """
    available = available_encoders(ffmpeg_path)
    candidates = list(codec.encoder_chain)

    if available is None:
        # 探测失败：乐观采用首选编码器，转码阶段若真的不存在会由 ffmpeg 报权威错误
        return candidates[0]

    for enc in candidates:
        if enc.name in available:
            return enc

    raise EncoderUnavailableError(
        f"当前 ffmpeg 不支持 {codec.label} 的任何编码器。"
        f"候选: {', '.join(e.name for e in candidates)}。"
        "请安装包含对应库的完整 FFmpeg 构建。",
        codec=codec.canonical,
        candidates=[e.name for e in candidates],
    )


def select_audio_encoder(
    preferred: str, fallbacks: tuple[str, ...] = (),
    ffmpeg_path: str | None = None,
) -> str:
    """在音频编码器候选中挑第一个可用的；探测失败时乐观返回首选。"""
    available = available_encoders(ffmpeg_path)
    if available is None:
        return preferred
    for name in (preferred, *fallbacks):
        if name in available:
            return name
    # 全部缺失：仍返回首选，让 ffmpeg 的实际报错作为权威信息
    return preferred
