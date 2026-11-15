"""音频淡化与整轨拼装：让声音淡化与画面淡化严格同帧。

采样边界数学
============

rawvideo 帧域没有时间戳，音频 PCM 同样不带时间戳，对齐只能建立在
显式数学上。输出采样率 ``sr`` 下，第 f 帧对应的采样区间用**累积
取整边界**定义（半开区间）::

    end(f)   = round(f * sr * fps_den / fps_num)
    span(f)  = [end(f), end(f+1))

每帧的采样数在 floor/ceil 之间交替（如 48kHz 配 29.97fps 时
2002/2001 个采样交替），但任意帧的边界都是单调整数，A、B 与输出
三条时间轴用同一个函数求值，重叠区每一帧三方采样数天然相等，
不会出现漂移或错位。

淡化同步
========

重叠区第 k 帧的音频增益与视频像素增益**完全相同**（``w_b = k/(No-1)``，
单帧重叠取 1/2），且按帧整块施加：画面淡到哪一帧，声音就淡到哪一帧，
从机制上排除「画面淡了声音没淡」。无音轨的素材用同长度静音 PCM 占位，
两条时间轴的形状始终一致。
"""

from __future__ import annotations

from .blend import blend_s16_samples, div_round_half_away
from .errors import PlanError
from .plan import TransitionPlan, fade_weight_b

__all__ = [
    "S16_CHANNELS",
    "choose_sample_rate",
    "frame_sample_bounds",
    "assemble_audio",
    "silence_pcm",
    "overlap_weight_fraction",
]

#: 统一输出双声道 s16le；单声道素材由解码端 ffmpeg 上混为双声道
S16_CHANNELS = 2
_FRAME_BYTES = S16_CHANNELS * 2

#: 候选输出采样率（全部为 PCM/AAC 合法速率），优先选择能让
#: 「每帧采样数 = sr * fps_den / fps_num」为整数的速率，
#: 找不到整数解时退回 48kHz 并由累积取整边界吸收零头。
_CANDIDATE_RATES = (48000, 44100, 32000, 24000, 22050, 16000)


def choose_sample_rate(fps_num: int, fps_den: int) -> int:
    """选择与帧率精确整除的输出采样率，无精确解时回退 48000。"""
    for sr in _CANDIDATE_RATES:
        if (sr * fps_den) % fps_num == 0:
            return sr
    return 48000


def frame_sample_bounds(
    frame_count: int, sample_rate: int, fps_num: int, fps_den: int
) -> list[int]:
    """返回 ``frame_count + 1`` 个累积采样边界（含 0 与末尾）。

    ``bounds[f]`` 即第 f 帧的起始采样号、``bounds[f+1]`` 为其结束，
    单调不减；相邻边界之差即该帧实际采样数。
    """
    if frame_count < 0 or sample_rate <= 0:
        raise PlanError("帧数/采样率非法")
    scale = sample_rate * fps_den
    bounds = [
        div_round_half_away(f * scale, fps_num)
        for f in range(frame_count + 1)
    ]
    return bounds


def silence_pcm(byte_length: int) -> bytearray:
    """生成长度为 ``byte_length`` 的静音 s16 PCM（全零即静音）。"""
    if byte_length % 2:
        raise PlanError("PCM 字节长度必须为偶数")
    return bytearray(byte_length)


def overlap_weight_fraction(k: int, overlap_frames: int) -> tuple[int, int]:
    """重叠区第 k 帧 B 权重的 (分子, 分母) 有理数形式。

    与 :func:`video_transition.plan.fade_weight_b` 同一语义，
    但返回整数分数供 PCM 混合直接使用，绕开浮点。
    """
    # 复用浮点版的边界校验，权重本身保持精确分数
    fade_weight_b(k, overlap_frames)
    if overlap_frames == 1:
        return 1, 2
    return k, overlap_frames - 1


def _take_frame(
    pcm: bytes | bytearray, start_sample: int, span_samples: int
) -> bytearray:
    """从整轨 PCM 截取一帧的采样块，不足部分以静音补齐。

    素材音频短于视频（容器元数据偏差）时尾部静音补；不会长于视频，
    因为调用方按该素材自己的帧边界切片。
    """
    start_byte = start_sample * _FRAME_BYTES
    needed = span_samples * _FRAME_BYTES
    available = len(pcm) - start_byte
    if available >= needed:
        return bytearray(pcm[start_byte:start_byte + needed])
    block = bytearray(needed)
    if available > 0:
        block[:available] = pcm[start_byte:start_byte + available]
    return block


def assemble_audio(
    plan: TransitionPlan,
    pcm_a: bytes | bytearray,
    pcm_b: bytes | bytearray,
    sample_rate: int,
) -> bytearray:
    """按 ``plan`` 把两段 s16le 双声道 PCM 拼成带淡化的完整成片音轨。

    :param pcm_a/pcm_b: 与各自素材**帧数等长**的整轨 PCM，无音轨时
        传静音（长度仍按帧边界算满）；
    :return: 长度为 ``bounds_out[output_frames] * 4`` 字节的成片 PCM。

    三段处理与视频帧映射共用 :class:`~video_transition.plan.Segment`
    坐标，head/tail 直接拷贝、overlap 逐帧同权重混合。
    """
    fps_num, fps_den = plan.fps_num, plan.fps_den
    na = plan.clip_a.frame_count
    nb = plan.clip_b.frame_count
    n_out = plan.output_frames

    bounds_a = frame_sample_bounds(na, sample_rate, fps_num, fps_den)
    bounds_b = frame_sample_bounds(nb, sample_rate, fps_num, fps_den)
    bounds_out = frame_sample_bounds(n_out, sample_rate, fps_num, fps_den)

    out = bytearray(bounds_out[n_out] * _FRAME_BYTES)
    head, overlap_seg, tail = plan.segments

    # 1) head：纯 A，帧号一一对应
    for f in range(head.out_end):
        s0, s1 = bounds_out[f], bounds_out[f + 1]
        block = _take_frame(pcm_a, bounds_a[f], s1 - s0)
        out[s0 * _FRAME_BYTES:s1 * _FRAME_BYTES] = block

    # 2) overlap：第 k 帧混合 A[na-No+k] 与 B[k]，增益与视频逐帧相同
    no = plan.overlap_frames
    for k in range(no):
        f_out = overlap_seg.out_start + k
        out_s0, out_s1 = bounds_out[f_out], bounds_out[f_out + 1]
        span = out_s1 - out_s0
        idx_a = overlap_seg.a_start + k
        idx_b = overlap_seg.b_start + k
        block_a = _take_frame(pcm_a, bounds_a[idx_a], span)
        block_b = _take_frame(pcm_b, bounds_b[idx_b], span)
        wnum, wden = overlap_weight_fraction(k, no)
        # 注意：bytearray 切片是副本，必须混合后再切片回写，
        # 不能直接对切片就地混合（改动会随副本一起丢弃）
        dst = bytearray(span * _FRAME_BYTES)
        blend_s16_samples(dst, block_a, block_b, wnum, wden)
        out[out_s0 * _FRAME_BYTES:out_s1 * _FRAME_BYTES] = dst

    # 3) tail：纯 B，成片帧 tail.out_start+local 对应 B 帧 No+local
    for local in range(tail.length):
        f_out = tail.out_start + local
        idx_b = tail.b_start + local
        s0, s1 = bounds_out[f_out], bounds_out[f_out + 1]
        block = _take_frame(pcm_b, bounds_b[idx_b], s1 - s0)
        out[s0 * _FRAME_BYTES:s1 * _FRAME_BYTES] = block

    return out
