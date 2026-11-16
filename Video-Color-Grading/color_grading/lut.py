"""亮度 / 对比度映射表（LUT）。

对 8bit 通道逐字节独立应用如下映射：

    offset = brightness / 100 * 255        # brightness ∈ [-100, 100]
    factor = (100 + contrast) / 100        # contrast  ∈ [-100, 100] → factor ∈ [0, 2]
    out    = clamp(round((v - 128) * factor + 128 + offset), 0, 255)

设计决策：

- 对比度以 128（中灰）为支点，中灰本身不受对比度影响；
  亮度是整体平移，在对比度之后叠加。
- 两个参数都为 0 时退化为**恒等映射**：``out == v`` 逐字节精确成立。
  因为 ``(v - 128) * 1.0 + 128.0`` 在 IEEE 双精度下对 [0, 255]
  的整数没有任何舍入误差。这是「旧无参数调用结果一致」的回归基线，
  由测试锁定。
- 参数超出合法范围时不报错，而是**钳制到边界并发出 UserWarning**
  （同时把提示文本返回给调用方），避免一次手滑毁掉整段批处理。
- 映射表是 256 字节的 ``bytes``，构建一次即可供全部帧、全部
  分块、全部 worker 进程复用（``bytes.translate`` 直接消费）。
"""

from __future__ import annotations

import warnings

BRIGHTNESS_MIN = -100.0
BRIGHTNESS_MAX = 100.0
CONTRAST_MIN = -100.0
CONTRAST_MAX = 100.0

#: 恒等映射表：lut[v] == v，无参数调用的基准。
IDENTITY_LUT = bytes(range(256))


def clamp_params(brightness: float, contrast: float) -> tuple[float, float, list[str]]:
    """把亮度 / 对比度钳制到合法范围。

    每发生一次钳制就发出一条 ``UserWarning`` 并把同样的中文提示
    收集进返回的列表，方便 CLI / 调用方展示。

    :return: ``(钳制后的亮度, 钳制后的对比度, 提示消息列表)``
    """
    messages: list[str] = []
    brightness = _clamp_one("brightness", brightness, BRIGHTNESS_MIN, BRIGHTNESS_MAX, messages)
    contrast = _clamp_one("contrast", contrast, CONTRAST_MIN, CONTRAST_MAX, messages)
    return brightness, contrast, messages


def _clamp_one(name: str, value: float, lo: float, hi: float, messages: list[str]) -> float:
    v = float(value)
    if lo <= v <= hi:
        return v
    clamped = lo if v < lo else hi
    msg = f"参数 {name}={v:g} 超出范围 [{lo:g}, {hi:g}]，已钳制为 {clamped:g}"
    warnings.warn(msg, UserWarning, stacklevel=3)
    messages.append(msg)
    return clamped


def build_lut(brightness: float = 0.0, contrast: float = 0.0) -> bytes:
    """构建 256 字节映射表。

    参数先经 :func:`clamp_params` 钳制（超范围会发出警告），
    因此直接调用本函数与走完整管道的行为一致。

    整个调色流程只应调用本函数**一次**，之后所有帧复用同一张表。
    """
    brightness, contrast, _ = clamp_params(brightness, contrast)
    offset = brightness / 100.0 * 255.0
    factor = (100.0 + contrast) / 100.0
    table = bytearray(256)
    for v in range(256):
        out = int(round((v - 128) * factor + 128.0 + offset))
        table[v] = 0 if out < 0 else 255 if out > 255 else out
    return bytes(table)


def is_identity(lut: bytes) -> bool:
    """判断映射表是否为恒等映射（无参数调用的结果）。"""
    return bytes(lut) == IDENTITY_LUT
