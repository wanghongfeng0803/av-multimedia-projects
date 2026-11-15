"""淡化混合原语：RGB24 像素与 s16le PCM 采样的整数加权运算。

全部用**确定性整数运算**，不引入浮点中间量：

* 结果可被单测逐字节锁定，跨平台/跨解释器实现一致；
* 混合公式是标准凸组合 ``a*(1-w) + b*w``，权重以 ``num/den``
  有理数传入，``num`` 即重叠区帧号 k、``den = No-1``，
  调用方无需自己做浮点换算；
* 四舍五入采用「远离零取整」（:func:`div_round_half_away`），
  正负采样对称，避免 s16 负值区间出现系统性负偏。
"""

from __future__ import annotations

from .errors import PlanError

__all__ = [
    "blend_rgb24_frame",
    "blend_s16_samples",
    "div_round_half_away",
]


def div_round_half_away(num: int, den: int) -> int:
    """整数 ``num / den`` 四舍五入（半数远离零），``den`` 必须为正。"""
    if den <= 0:
        raise PlanError(f"分母必须为正，收到 {den}")
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def blend_rgb24_frame(
    dst: bytearray,
    frame_a: bytes | bytearray | memoryview,
    frame_b: bytes | bytearray | memoryview,
    weight_b_num: int,
    weight_den: int,
) -> None:
    """把两帧 RGB24 按 B 权重 ``weight_b_num/weight_den`` 就地混合进 ``dst``。

    逐字节独立混合（RGB 三通道互不影响），``dst`` 长度必须等于两帧
    长度；``weight_b_num`` 取 0 时结果等于 A，取 ``weight_den`` 时
    结果等于 B——重叠区首尾锚点因此与纯段逐字节相等。
    """
    size = len(dst)
    if len(frame_a) != size or len(frame_b) != size:
        raise PlanError(
            f"帧尺寸不一致: dst={size}, a={len(frame_a)}, b={len(frame_b)}"
        )
    if not 0 <= weight_b_num <= weight_den or weight_den <= 0:
        raise PlanError(f"权重越界: {weight_b_num}/{weight_den}")
    wb, wa = weight_b_num, weight_den - weight_b_num
    for i in range(size):
        dst[i] = (frame_a[i] * wa + frame_b[i] * wb + weight_den // 2) // weight_den


def blend_s16_samples(
    dst: bytearray,
    pcm_a: bytes | bytearray | memoryview,
    pcm_b: bytes | bytearray | memoryview,
    weight_b_num: int,
    weight_den: int,
) -> None:
    """把两段交错 s16le PCM 按同样的权重混合，就地写入 ``dst``。

    与 :func:`blend_rgb24_frame` 共用一套权重语义，保证「画面淡到
    哪一帧、声音就淡到哪一帧」。每 2 字节一个 little-endian 有符号
    16 位采样，先按有符号数做凸组合再夹回 [-32768, 32767]。
    """
    import array

    if len(dst) != len(pcm_a) or len(dst) != len(pcm_b):
        raise PlanError(
            f"PCM 长度不一致: dst={len(dst)}, a={len(pcm_a)}, b={len(pcm_b)}"
        )
    if len(dst) % 2:
        raise PlanError("s16 PCM 长度必须是 2 的整数倍")
    if not 0 <= weight_b_num <= weight_den or weight_den <= 0:
        raise PlanError(f"权重越界: {weight_b_num}/{weight_den}")

    samples_a = array.array("h")
    samples_b = array.array("h")
    samples_a.frombytes(bytes(pcm_a))
    samples_b.frombytes(bytes(pcm_b))
    out = array.array("h", bytes(len(dst)))
    wb, wa = weight_b_num, weight_den - weight_b_num
    for i in range(len(samples_a)):
        mixed = div_round_half_away(
            samples_a[i] * wa + samples_b[i] * wb, weight_den
        )
        # 凸组合理论上不会越界（两个输入都在 s16 范围内），
        # 夹一道仅作防御，避免异常数据产生未定义回绕
        if mixed > 32767:
            mixed = 32767
        elif mixed < -32768:
            mixed = -32768
        out[i] = mixed
    dst[:] = out.tobytes()
