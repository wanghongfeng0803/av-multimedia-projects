"""编码器规格：软编（libx264/libx265）与硬编（nvenc/qsv/amf）统一抽象。

Python 逐像素引擎在编码前已产出确定的原始帧，因此**软编与硬编的画面
数学完全一致**，差异只来自末端 ffmpeg 编码器本身的码率控制与量化取整。
本模块只负责把各编码器的差异收敛到一张规格表：

* 码流内联 VUI 参数名（x264/x265 用 ``-x26X-params``，硬编用
  ``-color_*`` 外层标签，实测 ffmpeg 9.0 下硬编外层标签会写进码流）；
* 质量控制参数（软编 CRF / nvenc CQ / qsv QPI…全局质量 / amf QPI_Q…）；
* 固定质量模式标记（qsv/amf 需要显式 global_quality / qp mode）。

硬编按厂商分三组，每组各出 H.264/HEVC 两个编码器，命名与 ffmpeg 一致：
``h264_nvenc``/``hevc_nvenc``、``h264_qsv``/``hevc_qsv``、
``h264_amf``/``hevc_amf``。
"""

from __future__ import annotations

import subprocess

from .binaries import find_ffmpeg

__all__ = [
    "SOFTWARE_ENCODERS",
    "HARDWARE_ENCODERS",
    "ALL_ENCODERS",
    "EncoderSpec",
    "encoder_spec",
    "list_available_encoders",
    "check_encoder_works",
    "is_software_encoder",
    "build_quality_args",
    "build_vui_args",
]


class EncoderSpec:
    """单个 ffmpeg 编码器的规格描述。"""

    def __init__(
        self,
        name: str,
        family: str,
        codec: str,
        inline_vui: bool,
        quality_mode: str,
    ):
        #: ffmpeg 编码器名（同时作为 -c:v 参数）
        self.name = name
        #: 厂商族：software / nvenc / qsv / amf
        self.family = family
        #: 容器语义的编码标准：h264 / hevc
        self.codec = codec
        #: True 表示必须用 -x264-params/-x265-params 内联 VUI
        self.inline_vui = inline_vui
        #: 质量控制风格：crf / cq_nvenc / qsv / amf
        self.quality_mode = quality_mode

    def __repr__(self) -> str:
        return f"<EncoderSpec {self.name} ({self.family}/{self.codec})>"


def _specs() -> dict[str, EncoderSpec]:
    return {
        # ---- 软编：VUI 必须内联（实测外层标签不进码流），CRF 控质量 ----
        "libx264": EncoderSpec("libx264", "software", "h264", True, "crf"),
        "libx265": EncoderSpec("libx265", "software", "hevc", True, "crf"),
        # ---- NVIDIA NVENC：-cq 固定质量，外层 -color_* 写 VUI ----
        "h264_nvenc": EncoderSpec("h264_nvenc", "nvenc", "h264", False, "cq_nvenc"),
        "hevc_nvenc": EncoderSpec("hevc_nvenc", "nvenc", "hevc", False, "cq_nvenc"),
        # ---- Intel QSV：global_quality + 显式 ICQ/LA-ICQ 模式 ----
        "h264_qsv": EncoderSpec("h264_qsv", "qsv", "h264", False, "qsv"),
        "hevc_qsv": EncoderSpec("hevc_qsv", "qsv", "hevc", False, "qsv"),
        # ---- AMD AMF：qp_i/p/b 三档同值的固定 QP 模式 ----
        "h264_amf": EncoderSpec("h264_amf", "amf", "h264", False, "amf"),
        "hevc_amf": EncoderSpec("hevc_amf", "amf", "hevc", False, "amf"),
    }


_ENCODER_SPECS = _specs()
SOFTWARE_ENCODERS = ("libx264", "libx265")
HARDWARE_ENCODERS = (
    "h264_nvenc",
    "hevc_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "h264_amf",
    "hevc_amf",
)
ALL_ENCODERS = SOFTWARE_ENCODERS + HARDWARE_ENCODERS


def encoder_spec(name: str) -> EncoderSpec:
    """按名取规格；未知编码器抛 ValueError（与 CLI choices 双重保险）。"""
    try:
        return _ENCODER_SPECS[name]
    except KeyError:
        raise ValueError(
            f"不支持的编码器: {name}，可选 {', '.join(ALL_ENCODERS)}"
        ) from None


def is_software_encoder(name: str) -> bool:
    """是否软编（libx264/libx265）；其余视为硬编。"""
    return encoder_spec(name).family == "software"


def build_quality_args(name: str, crf: int, preset: str) -> list[str]:
    """把统一的质量档位（crf 语义，越小越好）翻译成各编码器参数。

    各厂固定质量模式取值域不同，这里统一钳到 0..51 再按族映射，
    保证同一个 crf 数字在软/硬编下都是「固定质量」而非 VBR 码率模式。
    """
    spec = encoder_spec(name)
    q = min(max(int(crf), 0), 51)
    if spec.quality_mode == "crf":
        return ["-preset", preset, "-crf", str(q)]
    if spec.quality_mode == "cq_nvenc":
        # NVENC 固定质量（CQ），preset 用 p1..p7 数字档，medium≈p4
        return ["-preset", "p4", "-rc", "vbr", "-cq", str(q), "-b:v", "0"]
    if spec.quality_mode == "qsv":
        # QSV 固定质量：global_quality 配 ICQ 模式（H.264/HEVC 通用）
        return ["-global_quality", str(q), "-look_ahead", "0"]
    # amf：固定 QP，I/P/B 同值；quality 档直接用 preset 名
    return [
        "-quality", "quality",
        "-rc", "cqp",
        "-qp_i", str(q), "-qp_p", str(q), "-qp_b", str(q),
    ]


def build_vui_args(name: str) -> tuple[list[str], list[str]]:
    """返回码流 VUI 内联参数与外层兜底标签两组参数。

    软编（实测 ffmpeg 9.0 外层标签不进码流）走 ``-x264-params/-x265-params``
    内联；硬编外层 ``-color_*`` 会被写进码流，直接用外层标签。
    外层标签两种路径都保留，互为兜底。
    """
    spec = encoder_spec(name)
    outer = [
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
    ]
    if not spec.inline_vui:
        return [], outer
    vui = "colorprim=bt709:transfer=bt709:colormatrix=bt709"
    inline_param = "-x264-params" if spec.codec == "h264" else "-x265-params"
    return [inline_param, vui], outer


def list_available_encoders(ffmpeg: str | None = None) -> list[str]:
    """运行 ``ffmpeg -encoders`` 探测本机**编译支持**的编码器，保持规格表顺序。

    注意：出现在该列表只说明 ffmpeg 二进制编入了该编码器，硬编是否真正
    可用还取决于 GPU/驱动；运行时可用性请用 :func:`check_encoder_works`。
    探测失败（ffmpeg 异常）时返回空列表，不抛出。
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        listed = set()
        for line in proc.stdout.splitlines():
            # 行形如 " V....D h264_nvenc  NVIDIA ..."，第二列起为编码器名
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                listed.add(parts[1])
    except (OSError, subprocess.SubprocessError):
        listed = set()
    return [name for name in ALL_ENCODERS if name in listed]


def check_encoder_works(name: str, ffmpeg: str | None = None) -> bool:
    """用一帧 testsrc 真实编码，确证编码器在当前机器上**运行时可用**。

    硬编（nvenc/qsv/amf）即使被编译进 ffmpeg，缺 GPU 或驱动时也会在
    初始化阶段失败；本函数实际起一次编码进程，成功才返回 True。
    结果不缓存（调用方通常只在测试/选路时调用一次）。
    """
    if name not in _ENCODER_SPECS:
        return False
    ffmpeg = ffmpeg or find_ffmpeg()
    pix_fmt = "yuv420p"  # 全部 8 个编码器都支持的最小公分母像素格式
    try:
        proc = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi",
                "-i", "color=black:size=64x48:rate=1:duration=1",
                "-frames:v", "1",
                "-c:v", name, "-pix_fmt", pix_fmt,
                *build_quality_args(name, 23, "medium"),
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0
