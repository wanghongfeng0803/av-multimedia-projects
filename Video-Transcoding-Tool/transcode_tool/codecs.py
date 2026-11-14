"""视频编码注册表：集中描述每种编码的能力与 ffmpeg 参数构建。

设计目标：
- 数据驱动：新增一种编码只需追加一个 :class:`CodecSpec`，编排层无需改动；
- 同一套跨编码器的语义（CRF、preset）由本模块翻译成各编码器原生参数；
- 不执行任何外部进程，纯字符串/数值处理，便于单元测试。

preset 语义统一为 x264 风格的十档命名（ultrafast … placebo），
同时允许传入编码器原生的数字档位字符串（如 libvpx 的 ``"4"``），
数字合法性在具体编码器构建参数时校验。
"""

from __future__ import annotations

from dataclasses import dataclass

# 跨编码器通用的十档 preset
CANONICAL_PRESETS: tuple[str, ...] = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo",
)


@dataclass(frozen=True)
class EncoderSpec:
    """单个 ffmpeg 编码器（一种编码可能有多个可用编码器实现）。"""

    name: str                       # ffmpeg -encoders 中的名称
    preset_mode: str                # named / cpu_used / svt / rav1e
    crf_min: int
    crf_max: int
    # 数字档位的取值范围（preset_mode != named 时使用）
    numeric_min: int = 0
    numeric_max: int = 8
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodecSpec:
    """一种视频编码的完整描述。"""

    canonical: str                  # 规范名：h264 / h265 / vp9 / av1 / vp8
    aliases: tuple[str, ...]        # 用户可输入的别名
    ffprobe_name: str               # ffprobe 报告的 codec_name，用于产物复核
    label: str                      # 人类可读名称
    default_crf: int
    default_preset: str
    # 编码器候选：按优先级排列，运行时取第一个本机可用的
    encoder_chain: tuple[EncoderSpec, ...]
    # 首选容器（用于默认输出路径与容器兜底）
    preferred_containers: tuple[str, ...]

    @property
    def crf_min(self) -> int:
        return min(enc.crf_min for enc in self.encoder_chain)

    @property
    def crf_max(self) -> int:
        # 取候选编码器中的最小上限，保证 CRF 对整条链都合法
        return min(enc.crf_max for enc in self.encoder_chain)

    def resolve_encoder(self, encoder_name: str) -> EncoderSpec:
        for enc in self.encoder_chain:
            if enc.name == encoder_name:
                return enc
        raise ValueError(
            f"编码器 {encoder_name} 不属于编码 {self.canonical} 的候选: "
            f"{', '.join(e.name for e in self.encoder_chain)}"
        )


# ---------------------------------------------------------------------------
# preset 映射表：十档命名 → 各编码器原生数字档位
# ---------------------------------------------------------------------------

# libvpx / libaom 的 cpu-used（0 最慢最好，8 最快）
_CPU_USED_MAP = {
    "placebo": 0, "veryslow": 1, "slower": 2, "slow": 3, "medium": 4,
    "fast": 5, "faster": 6, "veryfast": 7, "superfast": 7, "ultrafast": 8,
}
# SVT-AV1 的数字 preset（0 最慢，13 最快）
_SVT_MAP = {
    "placebo": 0, "veryslow": 2, "slower": 4, "slow": 6, "medium": 8,
    "fast": 9, "faster": 10, "veryfast": 11, "superfast": 12, "ultrafast": 13,
}
# rav1e 的 speed（0 最慢，10 最快）
_RAV1E_MAP = {
    "placebo": 0, "veryslow": 2, "slower": 3, "slow": 4, "medium": 5,
    "fast": 6, "faster": 7, "veryfast": 8, "superfast": 9, "ultrafast": 10,
}


def _numeric_preset(preset: str, spec: EncoderSpec, mapping: dict[str, int]) -> int:
    """把命名/数字 preset 解析为编码器原生数字档位并做范围校验。"""
    if preset in mapping:
        return mapping[preset]
    if preset.lstrip("-").isdigit():
        value = int(preset)
        if not spec.numeric_min <= value <= spec.numeric_max:
            raise ValueError(
                f"编码器 {spec.name} 的数字 preset 必须在 "
                f"{spec.numeric_min}-{spec.numeric_max} 之间，收到 {value}。"
            )
        return value
    raise ValueError(
        f"编码器 {spec.name} 不支持 preset '{preset}'。"
        f"可用命名档位: {', '.join(CANONICAL_PRESETS)}；"
        f"或传入 {spec.numeric_min}-{spec.numeric_max} 的数字档位。"
    )


def build_video_encoder_args(
    codec: CodecSpec, encoder: EncoderSpec, crf: int, preset: str
) -> list[str]:
    """生成 ``-c:v <enc> ...`` 之后的质量/速度参数片段。"""
    validate_crf(codec, crf)

    if encoder.preset_mode == "named":
        if preset not in CANONICAL_PRESETS:
            raise ValueError(
                f"编码器 {encoder.name} 的 preset 必须是: "
                f"{', '.join(CANONICAL_PRESETS)}"
            )
        return ["-preset", preset, "-crf", str(crf), *encoder.extra_args]

    if encoder.preset_mode == "cpu_used":
        # libvpx / libaom 常量质量模式：-b:v 0 + -crf，cpu-used 控速度
        cpu_used = _numeric_preset(preset, encoder, _CPU_USED_MAP)
        return [
            "-b:v", "0", "-crf", str(crf),
            "-cpu-used", str(cpu_used),
            "-deadline", "good",
            *encoder.extra_args,
        ]

    if encoder.preset_mode == "svt":
        numeric = _numeric_preset(preset, encoder, _SVT_MAP)
        return ["-crf", str(crf), "-preset", str(numeric), *encoder.extra_args]

    if encoder.preset_mode == "rav1e":
        # rav1e 没有 CRF，使用 QP（0-255）：把通用 0-63 的 CRF 等比映射
        qp = round(crf * 255 / 63)
        speed = _numeric_preset(preset, encoder, _RAV1E_MAP)
        return ["-qp", str(qp), "-speed", str(speed), *encoder.extra_args]

    # 注册表内部错误：新增 preset_mode 后忘了在这里分派
    raise RuntimeError(f"未知 preset_mode: {encoder.preset_mode}")


# ---------------------------------------------------------------------------
# 注册表与查询
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, CodecSpec] = {}


def _register(spec: CodecSpec) -> CodecSpec:
    _REGISTRY[spec.canonical] = spec
    return spec


H264 = _register(CodecSpec(
    canonical="h264",
    aliases=("avc", "x264", "h.264"),
    ffprobe_name="h264",
    label="H.264 / AVC",
    default_crf=23,
    default_preset="medium",
    encoder_chain=(
        EncoderSpec("libx264", "named", crf_min=0, crf_max=51),
    ),
    preferred_containers=("mp4", "mkv"),
))

H265 = _register(CodecSpec(
    canonical="h265",
    aliases=("hevc", "x265", "h.265"),
    ffprobe_name="hevc",
    label="H.265 / HEVC",
    default_crf=28,
    default_preset="medium",
    encoder_chain=(
        # hvc1 标签让苹果生态播放器也能识别 HEVC in MP4
        EncoderSpec("libx265", "named", crf_min=0, crf_max=51,
                    extra_args=("-tag:v", "hvc1")),
    ),
    preferred_containers=("mp4", "mkv"),
))

VP9 = _register(CodecSpec(
    canonical="vp9",
    aliases=("vp09",),
    ffprobe_name="vp9",
    label="VP9",
    default_crf=31,
    default_preset="medium",
    encoder_chain=(
        EncoderSpec("libvpx-vp9", "cpu_used", crf_min=0, crf_max=63,
                    extra_args=("-row-mt", "1")),
    ),
    preferred_containers=("webm", "mkv"),
))

VP8 = _register(CodecSpec(
    canonical="vp8",
    aliases=("vp08",),
    ffprobe_name="vp8",
    label="VP8",
    default_crf=31,
    default_preset="medium",
    encoder_chain=(
        EncoderSpec("libvpx", "cpu_used", crf_min=4, crf_max=63),
    ),
    preferred_containers=("webm", "mkv"),
))

AV1 = _register(CodecSpec(
    canonical="av1",
    aliases=("av01",),
    ffprobe_name="av1",
    label="AOMedia Video 1 (AV1)",
    default_crf=30,
    default_preset="medium",
    encoder_chain=(
        # SVT-AV1 速度最快，作为首选；其次参考实现 libaom；rav1e 兜底
        EncoderSpec("libsvtav1", "svt", crf_min=0, crf_max=63,
                    numeric_min=0, numeric_max=13),
        EncoderSpec("libaom-av1", "cpu_used", crf_min=0, crf_max=63,
                    extra_args=("-row-mt", "1")),
        EncoderSpec("librav1e", "rav1e", crf_min=0, crf_max=63,
                    numeric_min=0, numeric_max=10),
    ),
    preferred_containers=("mp4", "webm", "mkv"),
))


def list_codecs() -> tuple[CodecSpec, ...]:
    """按注册顺序返回全部已注册编码。"""
    return tuple(_REGISTRY.values())


def resolve_codec(name: str | None) -> CodecSpec:
    """按规范名或别名（大小写、空格/点不敏感）解析编码。

    :raises ValueError: 未注册的编码名
    """
    if name is None:
        return H264
    key = _normalize(name)
    for spec in _REGISTRY.values():
        if key == spec.canonical or key in (_normalize(a) for a in spec.aliases):
            return spec
    known = ", ".join(
        f"{s.canonical}({'/'.join(s.aliases)})" for s in _REGISTRY.values()
    )
    raise ValueError(f"未知视频编码 '{name}'。支持: {known}")


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def validate_crf(codec: CodecSpec, crf: int) -> None:
    """按具体编码的取值范围校验 CRF。"""
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise ValueError(
            f"crf 必须是整数（编码 {codec.canonical} 允许 "
            f"{codec.crf_min}-{codec.crf_max}）。"
        )
    if not codec.crf_min <= crf <= codec.crf_max:
        raise ValueError(
            f"编码 {codec.canonical} 的 crf 必须在 "
            f"{codec.crf_min}-{codec.crf_max} 之间，收到 {crf}。"
        )


def validate_preset(codec: CodecSpec, preset: str) -> None:
    """校验 preset：命名档位对所有编码器合法；数字档位须落在链上最宽范围内。"""
    if not isinstance(preset, str) or not preset:
        raise ValueError("preset 必须是非空字符串。")
    if preset in CANONICAL_PRESETS:
        return
    if preset.lstrip("-").isdigit():
        lo = min(e.numeric_min for e in codec.encoder_chain)
        hi = max(e.numeric_max for e in codec.encoder_chain)
        value = int(preset)
        if not lo <= value <= hi:
            raise ValueError(
                f"编码 {codec.canonical} 的数字 preset 必须在 {lo}-{hi} 之间，"
                f"收到 {value}。"
            )
        return
    raise ValueError(
        f"编码 {codec.canonical} 不支持 preset '{preset}'。"
        f"命名档位: {', '.join(CANONICAL_PRESETS)}。"
    )
