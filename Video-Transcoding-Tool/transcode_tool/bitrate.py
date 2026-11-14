"""目标码率控制：码率字符串解析、边界校验与各编码器 ffmpeg 参数构建。

本模块在既有“CRF 常量质量”路线之外，新增“按目标码率编码”的路线，
支持三种速率控制模式，并把统一语义翻译成本工具支持的五族编码器
（libx264 / libx265 / libvpx(-vp9) / libsvtav1 / libaom-av1 / librav1e）
各自真实可用的参数。

为什么需要按编码器分别处理（均在本机 ffmpeg 实测确认）：
- 通用码率选项 ``-b:v/-minrate/-maxrate/-bufsize`` 对 x264/x265/libvpx/
  libaom/rav1e 都生效；
- **libsvtav1 在 ABR 模式下不接受 ``-maxrate``**（报错
  “Max Bitrate only supported with CRF mode”），它的恒定码率要通过
  私有字典 ``-svtav1-params rc=1`` 开启；
- libx264 的严格 CBR 需要 ``nal-hrd=cbr`` 注入填充 NAL，而 ffmpeg 明确
  提示 ``cbr not allowed in .mp4``，因此该私有开关仅在非 MP4 容器使用，
  MP4 退化为“紧凑 VBV 约束”的准 CBR，仍可在任何容器工作；
- rav1e 没有 CRF、用 ``-speed``，libvpx/libaom 用 ``-cpu-used``，
  码率模式下这些速度档位仍需由 preset 翻译后一并给出。

全部为纯数值/字符串处理，不启动任何外部进程，便于单元测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .codecs import CodecSpec, EncoderSpec, numeric_preset_value

__all__ = [
    "RateMode",
    "BitratePlan",
    "MIN_BITRATE_BPS",
    "MAX_BITRATE_BPS",
    "parse_bitrate",
    "format_bitrate",
    "plan_bitrate",
    "build_bitrate_encoder_args",
]

# 速率控制模式
RateMode = str
MODE_VBR: RateMode = "vbr"  # 可变码率：围绕目标浮动，带峰值约束
MODE_ABR: RateMode = "abr"  # 平均码率：长期均值贴合目标，允许短时波动
MODE_CBR: RateMode = "cbr"  # 恒定码率：min=max=目标，紧凑缓冲，波动最小
RATE_MODES: tuple[str, ...] = (MODE_VBR, MODE_ABR, MODE_CBR)

# 硬边界：低于 1kbps 对视频无意义；高于 1Gbps 几乎必然是单位笔误，
# 且会逼近部分编码器内部整型上限。命中边界一律 ValueError，明确失败。
MIN_BITRATE_BPS = 1_000
MAX_BITRATE_BPS = 1_000_000_000

# 单位后缀 → 字节/位乘数。码率单位是“比特每秒”，故 1k = 1000bps（SI 十进制，
# 与 ffmpeg 的 k/M 约定一致，而非 1024）。
_UNIT_MULTIPLIERS: dict[str, int] = {
    "": 1,
    "bps": 1,
    "k": 1_000,
    "kbps": 1_000,
    "kb/s": 1_000,
    "m": 1_000_000,
    "mbps": 1_000_000,
    "mb/s": 1_000_000,
    "g": 1_000_000_000,
    "gbps": 1_000_000_000,
    "gb/s": 1_000_000_000,
}

# 整数或小数 + 可选单位；允许中间空白，如 "2.5 Mbps"、"800k"、"1500000"
_BITRATE_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z/]*)?\s*$"
)


@dataclass(frozen=True)
class BitratePlan:
    """一次目标码率任务的归一化结果。"""

    target_bps: int          # 目标平均码率（bps，整型，可直接传给 ffmpeg）
    mode: str                 # vbr / abr / cbr
    max_bps: int             # 峰值码率（bps）
    buffer_bps: int          # VBV 缓冲大小（以“比特”计量的等效值）
    min_bps: int             # 最低码率（bps），仅 CBR 等于目标

    @property
    def target_kbps(self) -> int:
        return round(self.target_bps / 1000)


def parse_bitrate(value: str | int | float) -> int:
    """把用户码率解析为整数 bps。

    接受：
      - 纯数字（按 bps 处理）：``1500000``；
      - 带单位字符串（大小写不敏感）：``"800k"``、``"2.5Mbps"``、
        ``"5 mb/s"``、``"1Gbps"``；
      - int/float 数值（同样按 bps）。

    :raises ValueError: 空值、布尔、负数、无法识别的单位或超出硬边界。
    """
    if isinstance(value, bool):
        # bool 是 int 子类，必须显式拒绝，避免 True 被当成 1bps
        raise ValueError("码率不能是布尔值。")
    if isinstance(value, int):
        bps = float(value)
    elif isinstance(value, float):
        bps = value
    elif isinstance(value, str):
        match = _BITRATE_RE.match(value)
        if not match:
            raise ValueError(
                f"无法解析码率 '{value}'。示例：'800k'、'2.5Mbps'、'1500000'。"
            )
        unit = (match.group("unit") or "").strip().lower()
        multiplier = _UNIT_MULTIPLIERS.get(unit)
        if multiplier is None:
            raise ValueError(
                f"未知码率单位 '{unit}'。支持: bps、k/kbps、m/mbps、g/gbps。"
            )
        bps = float(match.group("num")) * multiplier
    else:
        raise ValueError(
            f"码率必须是数字或字符串，收到 {type(value).__name__}。"
        )

    if bps != bps or bps in (float("inf"), float("-inf")):  # NaN / inf 防御
        raise ValueError("码率必须是有限数值。")
    if bps <= 0:
        raise ValueError(f"码率必须是正数，收到 {value!r}。")
    if bps < MIN_BITRATE_BPS or bps > MAX_BITRATE_BPS:
        raise ValueError(
            f"码率 {_format_bps(int(round(bps)))} 超出允许范围 "
            f"{_format_bps(MIN_BITRATE_BPS)} ~ {_format_bps(MAX_BITRATE_BPS)}。"
        )
    return int(round(bps))


def _format_bps(bps: int) -> str:
    """把 bps 规整为人类可读字符串（仅用于提示信息）。"""
    if bps >= 1_000_000 and bps % 1_000_000 == 0:
        return f"{bps // 1_000_000}Mbps"
    if bps >= 1000 and bps % 1000 == 0:
        return f"{bps // 1000}kbps"
    return f"{bps}bps"


def format_bitrate(bps: int) -> str:
    """公开的码率格式化：优先 Mbps/kbps，否则回退 bps。"""
    if not isinstance(bps, int) or isinstance(bps, bool):
        raise ValueError("format_bitrate 仅接受整数 bps。")
    return _format_bps(bps)


def plan_bitrate(target: str | int | float, mode: str = MODE_VBR) -> BitratePlan:
    """根据目标码率与模式推导完整的码率控制参数。

    缓冲与峰值的取值遵循流媒体行业常用经验：
      - VBR：峰值取目标 2 倍（封顶，避免瞬时码率失控），VBV 缓冲取目标 2 倍
        （约 2 秒码量），允许编码器在复杂场景上探、简单场景下探；
      - ABR：不设硬性 VBV 峰值/缓冲，仅给 ``-b:v``，让编码器以长期平均
        码率为目标自由分配（质量比 VBR 稳、约束比 CBR 松）；
      - CBR：min=max=目标，缓冲取目标 1 倍（约 1 秒码量），把码率波动
        压到最小，适合直播/恒定带宽信道。

    :raises ValueError: 码率非法或模式未知。
    """
    target_bps = parse_bitrate(target)
    if mode not in RATE_MODES:
        raise ValueError(
            f"未知速率控制模式 '{mode}'。支持: {', '.join(RATE_MODES)}。"
        )

    if mode == MODE_VBR:
        max_bps = _cap(target_bps * 2)
        buffer_bps = _cap(target_bps * 2)
        min_bps = 0
    elif mode == MODE_ABR:
        max_bps = 0
        buffer_bps = 0
        min_bps = 0
    else:  # CBR
        max_bps = target_bps
        buffer_bps = target_bps
        min_bps = target_bps

    return BitratePlan(
        target_bps=target_bps,
        mode=mode,
        max_bps=max_bps,
        buffer_bps=buffer_bps,
        min_bps=min_bps,
    )


def _cap(bps: int) -> int:
    """把派生码率限制在硬边界内（目标本身已校验，这里只防倍增溢出上限）。"""
    return min(bps, MAX_BITRATE_BPS)


def build_bitrate_encoder_args(
    codec: CodecSpec,
    encoder: EncoderSpec,
    plan: BitratePlan,
    preset: str,
    *,
    container_canonical: str = "mp4",
) -> list[str]:
    """按目标码率模式生成各编码器真实可用的 ffmpeg 参数片段。

    与 :func:`codecs.build_video_encoder_args`（CRF 常量质量）互斥：
    调用本函数即走码率控制路线，不再传 ``-crf``。

    :param container_canonical: 容器规范名，用于决定 libx264 严格 CBR 的
        ``nal-hrd`` 开关能否使用（MP4 不允许 nal-hrd=cbr）。
    :raises ValueError: 模式/编码器/preset 不支持。
    """
    if plan.mode not in RATE_MODES:
        raise ValueError(f"未知速率控制模式: {plan.mode}")
    target = str(plan.target_bps)

    # --- libsvtav1：ABR 禁用 -maxrate，CBR 用私有 rc=1 ---
    if encoder.preset_mode == "svt":
        numeric = numeric_preset_value(encoder, preset)
        args = ["-preset", str(numeric), "-b:v", target, *encoder.extra_args]
        if plan.mode == MODE_CBR:
            args += ["-svtav1-params", "rc=1"]
        # VBR/ABR 都不传 -maxrate/-bufsize，否则 SVT 直接报错
        return args

    # --- rav1e：速度档位 -speed，码率走通用选项 ---
    if encoder.preset_mode == "rav1e":
        speed = numeric_preset_value(encoder, preset)
        args = ["-speed", str(speed), "-b:v", target, *encoder.extra_args]
        return _apply_generic_rate(args, plan)

    # --- libvpx / libaom：数字档位 -cpu-used，码率走通用选项 ---
    if encoder.preset_mode == "cpu_used":
        cpu_used = numeric_preset_value(encoder, preset)
        args = [
            "-cpu-used", str(cpu_used),
            "-deadline", "good",
            "-b:v", target,
            *encoder.extra_args,
        ]
        return _apply_generic_rate(args, plan)

    # --- 命名档位编码器：libx264 / libx265 ---
    if encoder.preset_mode == "named":
        args = ["-preset", preset, "-b:v", target, *encoder.extra_args]
        args = _apply_generic_rate(args, plan)
        if plan.mode == MODE_CBR and encoder.name == "libx264":
            if container_canonical == "mp4":
                # ffmpeg 明确禁止 MP4 内 nal-hrd=cbr；保留 min/max/buf 紧凑
                # 约束得到准 CBR，不注入会导致封装失败的 HRD 填充信号。
                pass
            else:
                # MKV 等容器：开启真正 CBR（固定帧率 + HRD 填充 NAL）
                args += ["-x264-params", "nal-hrd=cbr:force-cfr=1"]
        return args

    # 注册表内部错误：新增 preset_mode 后忘了在这里分派
    raise RuntimeError(f"未知 preset_mode: {encoder.preset_mode}")


def _apply_generic_rate(args: list[str], plan: BitratePlan) -> list[str]:
    """追加通用 VBV 码率选项；为 0 的字段表示该模式不启用对应约束。"""
    if plan.mode == MODE_ABR:
        # 仅 -b:v，长期平均码率，不加 VBV 硬约束
        return args
    if plan.min_bps > 0:
        args += ["-minrate", str(plan.min_bps)]
    if plan.max_bps > 0:
        args += ["-maxrate", str(plan.max_bps)]
    if plan.buffer_bps > 0:
        args += ["-bufsize", str(plan.buffer_bps)]
    return args
