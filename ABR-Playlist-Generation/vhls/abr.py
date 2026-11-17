"""多码率（ABR）HLS 打包：共享时间轴 + 逐码率切割 + 主列表原子写入。

设计总览（在单码率四步之上加一层编排）：

- **共享时间轴**：所有码率共用同一份 :class:`vhls.timeline.TimelinePlan`，
  切点/片数/序列号完全一致——「不同码率之间不能多片/少片」由构造保证，
  不靠事后对齐。
- **逐码率切割**：每路码率用各自的 :class:`vhls.cut.EncodeProfile`
  （缩放 + 目标码率）切割到独立子目录，子列表复用单码率的
  ``write_playlist``（同样的校验与原子写）。
- **主列表标注可独立核对**：
    - ``BANDWIDTH`` = 该路所有分片中「文件字节数 × 8 ÷ 实测时长」的
      峰值（RFC 8216 的峰值分片码率口径），任何核对者用磁盘上的分片
      与 EXTINF 即可重算，不依赖本项目的内部状态；
    - ``RESOLUTION`` = 该路分片视频流的 ffprobe 实测宽高，不是请求值
      （源小于目标分辨率时不放大，实测值就与请求值不同）。
- **跨码率对齐复核**：切割完成后逐片比对各码率的实测时长，偏差超容差
  即整体失败，不写主列表。

目录结构（``output_dir`` 下）::

    master.m3u8          # 主列表（EXT-X-STREAM-INF 指向各子列表）
    <name>/index.m3u8    # 每路码率一个子目录 + 子列表
    <name>/seg_00000.ts  # 分片（各码率片数/切点一致）
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .cut import (
    DEFAULT_DURATION_TOLERANCE_MS,
    CutError,
    EncodeProfile,
    SegmentResult,
    cut_segments,
    probe_segment_geometry,
)
from .pipeline import DEFAULT_TARGET_SECONDS, PackError
from .playlist import PLAYLIST_VERSION, PlaylistResult, write_playlist
from .probe import ProbeError, SourceInfo, probe_source
from .timeline import MS_PER_SEC, TimelinePlan, build_timeline

__all__ = [
    "AbrError",
    "Rendition",
    "VariantResult",
    "AbrResult",
    "default_ladder",
    "render_master",
    "write_master",
    "pack_abr",
]


class AbrError(RuntimeError):
    """ABR 打包失败（包装下层 ProbeError/CutError/PlaylistError 等）。"""


@dataclass(frozen=True)
class Rendition:
    """一路码率的规格（请求值；真实值以切割产物实测为准）。

    :ivar name: 子目录名（也是主列表里的相对 URI 前缀），如 ``"360p"``。
    :ivar height: 目标高度（像素）；源更矮时不放大，按源高度切。
    :ivar video_bitrate: 目标视频码率（bps）。
    :ivar audio_bitrate: 音频码率（bps），默认 96k。
    """

    name: str
    height: int
    video_bitrate: int
    audio_bitrate: int = 96_000


@dataclass(frozen=True)
class VariantResult:
    """一路码率的完整可观测结果，供 CLI 与测试断言。

    :ivar rendition: 该路的请求规格。
    :ivar playlist: 子列表产出摘要。
    :ivar results: 逐片切割结果（实测时长）。
    :ivar bandwidth: 实测峰值分片码率（bps），写入主列表 BANDWIDTH。
    :ivar width: 分片视频流实测宽度（像素），写入主列表 RESOLUTION。
    :ivar height: 分片视频流实测高度（像素），写入主列表 RESOLUTION。
    """

    rendition: Rendition
    playlist: PlaylistResult
    results: tuple
    bandwidth: int
    width: int
    height: int


@dataclass(frozen=True)
class AbrResult:
    """一次 ABR 打包的完整可观测结果。

    :ivar master_path: 主列表路径。
    :ivar variants: 各路码率结果（按 BANDWIDTH 升序）。
    :ivar plan: 所有码率共享的统一时间轴计划。
    :ivar source: 源探测信息。
    :ivar output_dir: 产物根目录。
    """

    master_path: Path
    variants: tuple
    plan: TimelinePlan
    source: SourceInfo
    output_dir: Path


def default_ladder(source_height: int) -> tuple:
    """按源高度生成默认码率阶梯（不放大，至少保留一路源分辨率）。

    常规阶梯 360p/480p/720p/1080p，只保留不超过源高度的档位；源比
    360p 还矮时回退到源分辨率单档（保证主列表至少有一路）。
    """
    ladder = (
        (360, 400_000),
        (480, 800_000),
        (720, 2_500_000),
        (1080, 5_000_000),
    )
    rungs = tuple(
        Rendition(name=f"{h}p", height=h, video_bitrate=br)
        for h, br in ladder if h <= source_height
    )
    if not rungs:
        rungs = (
            Rendition(
                name=f"{source_height}p", height=source_height,
                video_bitrate=400_000,
            ),
        )
    return rungs


def _measure_bandwidth(results: tuple) -> int:
    """实测峰值分片码率（bps）：max(分片字节数 × 8 ÷ 实测时长秒)。

    这是 RFC 8216 里 BANDWIDTH 的峰值分片码率口径，任何核对者用磁盘
    上的分片文件与子列表 EXTINF 即可独立重算。时长用实测毫秒换算，
    与 EXTINF 同源，不会出现「标注对不上文件」的情况。
    """
    peak = 0
    for r in results:
        size_bytes = r.path.stat().st_size
        duration_s = r.actual_duration_ms / MS_PER_SEC
        if duration_s <= 0:
            raise AbrError(f"分片 {r.path.name} 实测时长非正，无法计算码率")
        bps = int(size_bytes * 8 / duration_s + 0.5)
        peak = max(peak, bps)
    return peak


def render_master(variants: tuple) -> str:
    """渲染主列表 m3u8 文本（``\\n`` 行尾，末尾带换行）。

    按 BANDWIDTH 升序排列（播放器从第一个满足带宽的条目起选），
    RESOLUTION 写实测宽高，URI 为相对路径 ``<name>/index.m3u8``。
    """
    lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{PLAYLIST_VERSION}",
    ]
    for v in sorted(variants, key=lambda v: v.bandwidth):
        lines.append(
            f"#EXT-X-STREAM-INF:BANDWIDTH={v.bandwidth},"
            f"RESOLUTION={v.width}x{v.height}"
        )
        lines.append(f"{v.rendition.name}/index.m3u8")
    return "\n".join(lines) + "\n"


def write_master(variants: tuple, master_path: str | Path) -> Path:
    """校验 → 渲染 → 原子写入主列表，返回路径。

    校验项：至少一路码率、BANDWIDTH 为正、RESOLUTION 为正、子列表文件
    真实存在。主列表本身用「同目录临时文件 + os.replace」原子落盘。
    """
    if not variants:
        raise AbrError("主列表至少需要一路码率")
    for v in variants:
        if v.bandwidth <= 0:
            raise AbrError(f"码率 {v.rendition.name} 的 BANDWIDTH 非正")
        if v.width <= 0 or v.height <= 0:
            raise AbrError(f"码率 {v.rendition.name} 的 RESOLUTION 非正")
        sub = Path(v.playlist.path)
        if not sub.is_file():
            raise AbrError(f"子列表缺失：{sub}")

    master_path = Path(master_path)
    text = render_master(variants)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=master_path.name + ".", suffix=".tmp",
        dir=str(master_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, master_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return master_path


def _check_alignment(variants_results: tuple, tolerance_ms: int) -> None:
    """跨码率逐片对齐复核：片数一致 + 每片实测时长偏差在容差内。

    片数一致由共享时间轴构造保证，这里复核实测时长——不同码率的同序
    号分片必须切在同一组帧上，实测时长偏差超容差即判定对齐被破坏。
    """
    if len(variants_results) < 2:
        return
    ref = variants_results[0]
    for other in variants_results[1:]:
        if len(other) != len(ref):
            raise AbrError(
                f"码率间分片数不一致：{len(ref)} vs {len(other)}"
            )
        for i, (a, b) in enumerate(zip(ref, other)):
            drift = abs(a.actual_duration_ms - b.actual_duration_ms)
            if drift > tolerance_ms:
                raise AbrError(
                    f"第 {i} 片跨码率时长偏差 {drift}ms 超过容差 "
                    f"{tolerance_ms}ms，分片对齐被破坏"
                )


def pack_abr(
    input_path: str | Path,
    output_dir: str | Path,
    renditions: tuple | None = None,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS,
    source: SourceInfo | None = None,
    master_name: str = "master.m3u8",
) -> AbrResult:
    """执行完整的多码率 HLS 打包。

    :param input_path: 源媒体路径（含视频流；音频可选）。
    :param output_dir: 产物根目录，主列表与各路子目录都写在这里。
    :param renditions: 码率阶梯（:class:`Rendition` 元组）；None 时按
        源高度自动生成（不放大，至少一路源分辨率）。
    :param target_seconds: 目标分片时长（秒），默认 6。
    :param tolerance_ms: 分片实测时长与计划值的容差（毫秒）。
    :param source: 已探测的源信息；None 时内部调用 ffprobe。
    :param master_name: 主列表文件名（默认 ``master.m3u8``）。
    :raises AbrError: 任一阶段失败；主列表仅在全部校验通过后才落盘。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # ---- 阶段一：探测（所有码率共享同一份源信息）----
    try:
        if source is None:
            source = probe_source(input_path)
        elif source.path != input_path:
            raise AbrError(
                f"传入的源信息路径 {source.path} 与输入 {input_path} 不一致"
            )
    except ProbeError as exc:
        raise AbrError(f"探测失败：{exc}") from exc

    if renditions is None:
        renditions = default_ladder(source.height)
    if not renditions:
        raise AbrError("码率阶梯不能为空")

    # ---- 阶段二：共享时间轴（所有码率同一份切点/片数）----
    try:
        plan = build_timeline(source, target_seconds, segment_prefix="seg")
    except Exception as exc:
        raise AbrError(f"时间轴规划失败：{exc}") from exc

    # ---- 阶段三：逐码率切割 + 子列表（各自独立子目录）----
    variants: list = []
    all_results: list = []
    for rend in renditions:
        profile = EncodeProfile(
            height=rend.height if rend.height < source.height else None,
            video_bitrate=rend.video_bitrate,
            audio_bitrate=rend.audio_bitrate,
        )
        sub_dir = output_dir / rend.name
        try:
            results = cut_segments(
                source, plan, sub_dir, tolerance_ms, profile,
            )
        except CutError as exc:
            raise AbrError(f"码率 {rend.name} 切割失败：{exc}") from exc

        try:
            playlist = write_playlist(plan, results, sub_dir / "index.m3u8")
        except Exception as exc:
            raise AbrError(f"码率 {rend.name} 子列表写入失败：{exc}") from exc

        # 标注以产物实测为准：分辨率探测首片，码率算峰值分片码率。
        width, height = probe_segment_geometry(results[0].path)
        bandwidth = _measure_bandwidth(results)
        variants.append(VariantResult(
            rendition=rend, playlist=playlist, results=results,
            bandwidth=bandwidth, width=width, height=height,
        ))
        all_results.append(results)

    # ---- 阶段四：跨码率对齐复核 + 主列表原子写入 ----
    _check_alignment(tuple(all_results), tolerance_ms * 2)
    try:
        master_path = write_master(tuple(variants), output_dir / master_name)
    except (AbrError, OSError) as exc:
        raise AbrError(f"主列表写入失败：{exc}") from exc

    return AbrResult(
        master_path=master_path,
        variants=tuple(
            sorted(variants, key=lambda v: v.bandwidth)
        ),
        plan=plan, source=source, output_dir=output_dir,
    )
