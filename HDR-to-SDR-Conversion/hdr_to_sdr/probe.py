"""输入探测：ffprobe JSON → 结构化 HDR 信息。

**识别 PQ 还是 HLG 不靠猜**（硬性需求）：

* 仅当 ffprobe 报告 ``color_transfer=smpte2084`` 时判定 PQ；
* 仅当 ``color_transfer=arib-std-b67`` 时判定 HLG；
* 其它情况（字段缺失、bt709、unknown 等）一律抛
  :class:`~hdr_to_sdr.errors.UnsupportedHDRTypeError`，
  除非调用方通过 ``assume`` 显式指定 PQ/HLG 才放行。
"""

from __future__ import annotations

import json
import os
import subprocess

from .binaries import find_ffprobe
from .errors import (
    InputNotFoundError,
    InvalidInputError,
    UnsupportedHDRTypeError,
)
from .transfer import TRANSFER_PQ, TRANSFER_HLG
from .colorscience import RANGE_LIMITED, RANGE_FULL

__all__ = [
    "HDRType",
    "HDRInfo",
    "probe_hdr",
    "parse_hdr_info",
]


class HDRType:
    """HDR 传递函数类型常量。"""

    PQ = TRANSFER_PQ
    HLG = TRANSFER_HLG


class HDRInfo:
    """探测得到的视频流 HDR 描述。

    :ivar transfer: 内部传递函数标识（smpte2084 / arib-std-b67）。
    :ivar color_primaries: ffprobe 原色标签（如 bt2020）。
    :ivar bits_per_raw_sample: 源码深（8/10/12/16），无法判定时按 10 处理。
    :ivar pixel_range: limited / full，无法判定时默认 limited。
    :ivar is_rgb: 源像素族是否为 RGB（gbrp 等），决定解码分支。
    :ivar width / height: 帧尺寸。
    :ivar frame_rate: 平均帧率（float），无法判定时为 None。
    :ivar peak_nits: mastering display 元数据中的峰值亮度（nit），缺失为 None。
    :ivar assumed: 类型是否来自调用方 ``assume`` 而非元数据确证。
    """

    def __init__(
        self,
        transfer: str,
        color_primaries: str | None,
        bits_per_raw_sample: int,
        pixel_range: str,
        is_rgb: bool,
        width: int,
        height: int,
        frame_rate: float | None,
        peak_nits: float | None,
        assumed: bool = False,
    ):
        self.transfer = transfer
        self.color_primaries = color_primaries
        self.bits_per_raw_sample = bits_per_raw_sample
        self.pixel_range = pixel_range
        self.is_rgb = is_rgb
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.peak_nits = peak_nits
        self.assumed = assumed

    @property
    def is_pq(self) -> bool:
        return self.transfer == TRANSFER_PQ

    @property
    def is_hlg(self) -> bool:
        return self.transfer == TRANSFER_HLG

    def __repr__(self) -> str:
        kind = "PQ" if self.is_pq else "HLG"
        tag = "（assume 显式指定）" if self.assumed else ""
        return (
            f"<HDRInfo {kind}{tag} {self.width}x{self.height} "
            f"{self.bits_per_raw_sample}bit {self.pixel_range} "
            f"primaries={self.color_primaries} peak={self.peak_nits}>"
        )


def _run_ffprobe(path: str, frames: bool = False) -> dict:
    ffprobe = find_ffprobe()
    cmd = [ffprobe, "-v", "error", "-print_format", "json"]
    if frames:
        # mastering/CLL 元数据在部分版本只挂在 frame 级 side_data，
        # 只读第一帧即可（-read_intervals 限定起始区间）
        cmd += ["-select_streams", "v:0", "-show_frames",
                "-read_intervals", "%+#1"]
    else:
        cmd += ["-show_format", "-show_streams"]
    cmd.append(path)
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise InvalidInputError(
            f"ffprobe 无法解析输入: {path}\n{proc.stderr[-500:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"ffprobe 输出不是合法 JSON: {path}") from exc


def _fraction_to_float(value: object) -> float | None:
    """解析 ffprobe 分数串（如 "10000000/10000" → 1000.0）。"""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or "/" not in value:
        return None
    num, den = value.split("/", 1)
    try:
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    except ValueError:
        return None


def _first_video_stream(raw: dict) -> dict:
    for stream in raw.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise InvalidInputError("输入中不存在视频流")


def _parse_frame_rate(stream: dict) -> float | None:
    raw_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    try:
        num, den = raw_rate.split("/")
        den_f = float(den)
        if den_f == 0:
            return None
        rate = float(num) / den_f
        return rate if rate > 0 else None
    except (ValueError, AttributeError):
        return None


def _extract_peak_nits(stream: dict, frame_raw: dict | None = None) -> float | None:
    """提取峰值亮度（nit）：优先 MaxMDL，回退 MaxCLL。

    ffprobe 9.x 把 mastering display / content light level 挂在 **frame 级**
    side_data，且 ``max_luminance`` 是分数串（"10000000/10000"=1000nit）；
    旧版本可能挂在 stream 级且为数值。两种形态都兼容。
    """
    side_lists: list[list] = [stream.get("side_data_list", []) or []]
    if frame_raw and frame_raw.get("frames"):
        first_frame = frame_raw["frames"][0]
        side_lists.append(first_frame.get("side_data_list", []) or [])

    mdl: list[float] = []
    cll: list[float] = []
    for sides in side_lists:
        for side in sides:
            mdl_value = _fraction_to_float(side.get("max_luminance"))
            if mdl_value is not None and mdl_value > 0:
                mdl.append(mdl_value)
            for key in ("max_content", "max_content_light_level"):
                cll_value = _fraction_to_float(side.get(key))
                if cll_value is not None and cll_value > 0:
                    cll.append(cll_value)
    return (mdl or cll or [None])[0]


def _resolve_bits(stream: dict) -> int:
    """确定源码深：优先 bits_per_raw_sample，其次像素格式位深。"""
    raw = stream.get("bits_per_raw_sample")
    if raw and str(raw).isdigit() and int(raw) >= 8:
        return int(raw)
    pix_fmt = stream.get("pix_fmt") or ""
    for token, bits in (("p010", 10), ("yuv420p10", 10), ("yuv422p10", 10),
                        ("yuv444p10", 10), ("p012", 12), ("yuv420p12", 12),
                        ("yuv444p12", 12), ("yuv420p16", 16), ("yuv444p16", 16)):
        if token in pix_fmt:
            return bits
    return 10  # HDR 内容的事实默认码深


def _is_rgb_pixfmt(pix_fmt: str) -> bool:
    return pix_fmt.startswith(("gbrp", "rgb", "bgr", "rgba", "bgra"))


def parse_hdr_info(
    raw: dict,
    assume: str | None = None,
    frame_raw: dict | None = None,
) -> HDRInfo:
    """从 ffprobe JSON 字典构建 :class:`HDRInfo`，纯函数便于单测。

    :param raw: ``-show_streams`` 的完整 JSON。
    :param assume: ``"pq"`` / ``"hlg"`` 显式兜底；元数据已确证时该参数被忽略。
    :param frame_raw: 可选的 ``-show_frames`` JSON，用于 frame 级峰值元数据。
    :raises UnsupportedHDRTypeError: 元数据无法确证且未给 assume。
    """
    stream = _first_video_stream(raw)
    transfer = (stream.get("color_transfer") or "").lower()
    primaries = stream.get("color_primaries") or None
    pix_fmt = stream.get("pix_fmt") or ""
    pixel_range = (stream.get("color_range") or "limited").lower()
    # ffprobe 实际输出 tv/pc，兼容规范名 limited/full
    pixel_range = {"tv": RANGE_LIMITED, "pc": RANGE_FULL}.get(
        pixel_range, pixel_range
    )
    if pixel_range not in ("limited", "full"):
        pixel_range = "limited"

    assumed = False
    if transfer not in (TRANSFER_PQ, TRANSFER_HLG):
        if not assume:
            raise UnsupportedHDRTypeError(
                f"无法确证 HDR 类型：color_transfer={transfer or '<缺失>'!r}，"
                "既不是 smpte2084(PQ) 也不是 arib-std-b67(HLG)；"
                "如确认源文件类型，请用 assume='pq'/'hlg' 显式指定"
            )
        assume_norm = assume.strip().lower()
        if assume_norm in ("pq", "smpte2084", "st2084"):
            transfer = TRANSFER_PQ
        elif assume_norm in ("hlg", "arib-std-b67", "b67"):
            transfer = TRANSFER_HLG
        else:
            raise UnsupportedHDRTypeError(f"assume 取值非法: {assume}")
        assumed = True

    return HDRInfo(
        transfer=transfer,
        color_primaries=primaries,
        bits_per_raw_sample=_resolve_bits(stream),
        pixel_range=pixel_range,
        is_rgb=_is_rgb_pixfmt(pix_fmt),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        frame_rate=_parse_frame_rate(stream),
        peak_nits=_extract_peak_nits(stream, frame_raw),
        assumed=assumed,
    )


def probe_hdr(path: str, assume: str | None = None) -> HDRInfo:
    """探测媒体文件的 HDR 信息（调用 ffprobe 子进程）。"""
    if not os.path.isfile(path):
        raise InputNotFoundError(f"输入文件不存在: {path}")
    raw = _run_ffprobe(path)
    # 确证为 HDR 后再发第二次探测取 frame 级峰值，SDR/未知输入不花这个开销
    try:
        transfer = (_first_video_stream(raw).get("color_transfer") or "").lower()
    except InvalidInputError:
        transfer = ""
    frame_raw = None
    if transfer in (TRANSFER_PQ, TRANSFER_HLG):
        frame_raw = _run_ffprobe(path, frames=True)
    return parse_hdr_info(raw, assume=assume, frame_raw=frame_raw)
