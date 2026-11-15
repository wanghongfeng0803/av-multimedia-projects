"""由 :class:`ComposePlan` 生成 ffmpeg ``filter_complex`` 字符串。

视频链（每路）::

    setpts=PTS-STARTPTS        # 归一化起始时间戳（输入可能有非零起点）
    fps=<输出帧率>             # 统一为恒定帧率，帧数由此确定
    scale=<w>:<h>:flags=lanczos
    setsar=1                   # 缩放后像素为方形，消除 SAR 差异
    pad=<cell_w>:<cell_h>:<x>:<y>:color=black   # 仅 fit 布局需要
    format=yuv420p             # hstack 要求两路像素格式一致
    tpad=stop_mode=clone:stop_duration=<秒>     # 仅 pad 模式下较短的一路
    trim=duration=<输出时长>   # 两路帧数严格一致
    setpts=PTS-STARTPTS

然后 ``[v0][v1]hstack=inputs=2[vout]`` 左右拼接。

音频链（每路有音频的输入）::

    asetpts=PTS-STARTPTS
    aresample=48000            # 统一采样率
    aformat=sample_fmts=fltp:channel_layouts=stereo
    apad                       # 末尾补静音（配合 atrim 同时覆盖补短/截长）
    atrim=duration=<输出时长>

两路都有音频时再以 ``amix=inputs=2:duration=longest:normalize=1`` 等权混音。
"""

from __future__ import annotations

from .plan import ComposePlan

#: 时长类滤镜参数的秒数格式：固定 6 位小数，保证字符串层面也是确定性的
_SECONDS_FMT = ".6f"


def build_filter_complex(plan: ComposePlan) -> str:
    """生成完整的 filter_complex 字符串（视频拼接 + 音频对齐/混音）。"""
    chains = [
        _video_chain(0, plan),
        _video_chain(1, plan),
        "[v0][v1]hstack=inputs=2[vout]",
    ]
    chains.extend(_audio_chains(plan))
    return ";".join(chains)


def _video_chain(index: int, plan: ComposePlan) -> str:
    side = plan.left if index == 0 else plan.right
    parts = [
        "setpts=PTS-STARTPTS",
        f"fps={plan.fps_text}",
        f"scale={side.scaled_width}:{side.scaled_height}:flags=lanczos",
        "setsar=1",
    ]
    if (side.cell_width, side.cell_height) != (side.scaled_width, side.scaled_height):
        parts.append(
            f"pad={side.cell_width}:{side.cell_height}:{side.pad_x}:{side.pad_y}:color=black"
        )
    parts.append("format=yuv420p")
    if side.freeze_seconds > 0:
        parts.append(
            f"tpad=stop_mode=clone:stop_duration={side.freeze_seconds:{_SECONDS_FMT}}"
        )
    parts.append(f"trim=duration={plan.duration:{_SECONDS_FMT}}")
    parts.append("setpts=PTS-STARTPTS")
    return f"[{index}:v]" + ",".join(parts) + f"[v{index}]"


def _audio_chains(plan: ComposePlan) -> list[str]:
    """音频对齐链；无音频输入时返回空列表（输出不含音轨）。"""
    chains: list[str] = []
    for idx in plan.audio_sources:
        chains.append(
            f"[{idx}:a]asetpts=PTS-STARTPTS,"
            f"aresample={plan.sample_rate},"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"apad,atrim=duration={plan.duration:{_SECONDS_FMT}}[a{idx}]"
        )
    if len(plan.audio_sources) == 2:
        chains.append("[a0][a1]amix=inputs=2:duration=longest:normalize=1[aout]")
    elif len(plan.audio_sources) == 1:
        # 统一输出标签，调用方只需映射 [aout]
        chains.append(f"[a{plan.audio_sources[0]}]acopy[aout]")
    return chains
