"""RGB 三元组的色相（hue）/ 饱和度（saturation）调整。

与 :mod:`color_grading.lut` 的逐字节 1D 映射不同，色相 / 饱和度是
**跨通道**操作，必须把每个像素的 RGB 三个字节作为一个整体，在
RGB ↔ HSV 之间换算后再写回。

通道顺序约定（全工具唯一约定，不得一处一套）
--------------------------------------------

整条像素管道统一使用 **RGB** 字节序：ffmpeg 解码侧固定
``-pix_fmt rgb24``，裸流中每个像素按 ``R, G, B`` 排列；本模块
入参 / 出参、:mod:`color_grading.chunks` 的 worker、编码侧
``-pix_fmt rgb24`` 全部遵循同一约定。代码中**不出现 BGR 命名**，
也不提供任何 BGR 入口；如未来接入 OpenCV 等 BGR 默认库，必须在
边界处显式转换后再进入本模块。

处理顺序（固定，不可调换）
--------------------------

一个像素依次经过：

1. :mod:`color_grading.lut` 的亮度 / 对比度 1D 映射（逐通道）；
2. RGB → HSV，先做**色相旋转**（``hue`` 度，逆时针为正）；
3. 再做**饱和度缩放**（``saturation`` 百分比）；
4. HSV → RGB 写回。

即顺序固定为 **先色相、后饱和度**。两者在数学上对纯 HSV 标量
可交换，但实现上固定一个顺序，保证「同时生效」时任何调用方、
任何 worker 进程得到逐字节一致的结果，该顺序同时写入 README。

恒等基线
--------

``hue == 0`` 且 ``saturation == 0`` 时本模块**不被调用**
（见 :func:`color_grading.chunks.map_stream` 的直通路径），
因此旧的无参数 / 仅亮度对比度调用仍保持逐字节恒等，不引入任何
浮点往返误差。本模块内 :func:`is_identity` 用于判断该条件。

纯标准库实现（仅 ``math``），不依赖 numpy / opencv 等第三方库。
"""

from __future__ import annotations

import math
import warnings

#: 色相旋转范围（度）：正值逆时针，负值顺时针，±180 覆盖整环。
HUE_MIN = -180.0
HUE_MAX = 180.0

#: 饱和度调整范围（百分比）：-100 完全去色，+100 饱和度翻倍。
SATURATION_MIN = -100.0
SATURATION_MAX = 100.0


def is_identity(hue: float, saturation: float) -> bool:
    """判断色相 / 饱和度参数是否为恒等（均为 0，不改变任何像素）。"""
    return float(hue) == 0.0 and float(saturation) == 0.0


def clamp_color_params(
    hue: float, saturation: float
) -> tuple[float, float, list[str]]:
    """把色相 / 饱和度钳制到合法范围。

    与 :func:`color_grading.lut.clamp_params` 同一策略：每发生一次
    钳制就发出一条 ``UserWarning`` 并把同样的中文提示收集进返回列表。

    :return: ``(钳制后的色相, 钳制后的饱和度, 提示消息列表)``
    """
    messages: list[str] = []
    hue = _clamp_one("hue", hue, HUE_MIN, HUE_MAX, messages)
    saturation = _clamp_one(
        "saturation", saturation, SATURATION_MIN, SATURATION_MAX, messages
    )
    return hue, saturation, messages


def _clamp_one(
    name: str, value: float, lo: float, hi: float, messages: list[str]
) -> float:
    v = float(value)
    if lo <= v <= hi:
        return v
    clamped = lo if v < lo else hi
    msg = f"参数 {name}={v:g} 超出范围 [{lo:g}, {hi:g}]，已钳制为 {clamped:g}"
    warnings.warn(msg, UserWarning, stacklevel=3)
    messages.append(msg)
    return clamped


def _round_byte(value: float) -> int:
    """四舍五入到最近整数并钳制到 [0, 255]。

    使用 :func:`math.floor` 实现「半值向上」的确定性舍入，避免
    内建 :func:`round` 的银行家舍入（2.5→2）造成的方向歧义。
    """
    q = math.floor(value + 0.5)
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def adjust_pixel(
    r: int, g: int, b: int, hue_shift: float, sat_factor: float
) -> tuple[int, int, int]:
    """对单个 RGB 像素（各通道 0-255）应用色相旋转与饱和度缩放。

    处理顺序固定：**先色相旋转，后饱和度缩放**（模块 docstring 所述）。
    灰度像素（``mx == mn``，色相无定义）跳过色相旋转，仅按饱和度
    向自身收缩——而灰度的收缩结果仍是同一灰度，故饱和度对纯灰无效，
    这与 HSV 中灰度像素 S=0 的定义一致。

    :param hue_shift: 色相旋转角（度），逆时针为正
    :param sat_factor: 饱和度缩放系数，1.0 为不变，0.0 为完全去色
    :return: ``(R, G, B)``，各通道为 [0, 255] 整数
    """
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    # --- RGB → HSV（H 以度表示，S/V 归一化到 [0,1]）---
    value = mx / 255.0
    if diff == 0:
        saturation = 0.0
        hue = 0.0  # 灰度：色相无定义，置 0 且下方旋转会被跳过
    else:
        saturation = diff / mx
        if mx == r:
            hue = (g - b) / diff % 6.0
        elif mx == g:
            hue = (b - r) / diff + 2.0
        else:
            hue = (r - g) / diff + 4.0
        hue *= 60.0  # 六分区分组值 → 角度

    # --- 固定顺序第 1 步：色相旋转（灰度无色相，跳过）---
    if diff != 0 and hue_shift != 0.0:
        hue = (hue + hue_shift) % 360.0

    # --- 固定顺序第 2 步：饱和度缩放 ---
    if sat_factor != 1.0:
        saturation = 0.0 if saturation * sat_factor < 0.0 else saturation * sat_factor
        if saturation > 1.0:
            saturation = 1.0

    # --- HSV → RGB（标准六分区扇形模型）---
    if saturation == 0.0:
        gray = _round_byte(value * 255.0)
        return gray, gray, gray

    h = hue / 60.0
    sector = int(h)
    frac = h - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))

    if sector == 0:
        rf, gf, bf = value, t, p
    elif sector == 1:
        rf, gf, bf = q, value, p
    elif sector == 2:
        rf, gf, bf = p, value, t
    elif sector == 3:
        rf, gf, bf = p, q, value
    elif sector == 4:
        rf, gf, bf = t, p, value
    else:  # sector == 5
        rf, gf, bf = value, p, q

    return _round_byte(rf * 255.0), _round_byte(gf * 255.0), _round_byte(bf * 255.0)


def map_chunk(
    data: bytes, hue: float = 0.0, saturation: float = 0.0
) -> bytes:
    """对一块 rgb24 裸流（长度为 3 的倍数）应用色相 / 饱和度。

    入参与出参都是 RGB 字节序（见模块 docstring 的统一约定）。
    本函数是 worker 入口，必须为模块级函数才能被 ``spawn`` pickle。

    为控制大分辨率下的纯 Python 开销，热点循环在此**内联**了
    :func:`adjust_pixel` 的同一套数学（含「先色相、后饱和度」顺序），
    两者逐像素等价，由 ``test_colorspace.py`` 的等价性测试锁定；
    :func:`adjust_pixel` 保留为可读、可单独调用的参考实现。
    """
    n = len(data)
    if n % 3:
        raise ValueError("rgb24 数据长度必须是 3 的倍数")
    if is_identity(hue, saturation):
        # 恒等参数直接返回原字节，避免无意义的逐像素浮点往返。
        return bytes(data)

    hue_shift = float(hue)
    sat_factor = (100.0 + float(saturation)) / 100.0
    rotate = hue_shift != 0.0
    scale_sat = sat_factor != 1.0
    out = bytearray(n)
    floor = math.floor
    for i in range(0, n, 3):
        r = data[i]
        g = data[i + 1]
        b = data[i + 2]

        mx = r if r >= g else g
        if b > mx:
            mx = b
        mn = r if r <= g else g
        if b < mn:
            mn = b
        diff = mx - mn

        value = mx / 255.0
        if diff == 0:
            sat = 0.0
            ang = 0.0
        else:
            sat = diff / mx
            if mx == r:
                ang = (g - b) / diff % 6.0
            elif mx == g:
                ang = (b - r) / diff + 2.0
            else:
                ang = (r - g) / diff + 4.0
            ang *= 60.0
            if rotate:  # 固定顺序第 1 步：色相旋转（灰度无色相，跳过）
                ang = (ang + hue_shift) % 360.0

        if scale_sat:  # 固定顺序第 2 步：饱和度缩放
            sat *= sat_factor
            if sat < 0.0:
                sat = 0.0
            elif sat > 1.0:
                sat = 1.0

        if sat == 0.0:
            gray = floor(value * 255.0 + 0.5)
            if gray > 255:
                gray = 255
            out[i] = out[i + 1] = out[i + 2] = gray
            continue

        h6 = ang / 60.0
        sector = int(h6)
        frac = h6 - sector
        p = value * (1.0 - sat)
        q = value * (1.0 - sat * frac)
        t = value * (1.0 - sat * (1.0 - frac))

        if sector == 0:
            rf, gf, bf = value, t, p
        elif sector == 1:
            rf, gf, bf = q, value, p
        elif sector == 2:
            rf, gf, bf = p, value, t
        elif sector == 3:
            rf, gf, bf = p, q, value
        elif sector == 4:
            rf, gf, bf = t, p, value
        else:
            rf, gf, bf = value, p, q

        for channel, cv in ((i, rf), (i + 1, gf), (i + 2, bf)):
            qv = floor(cv * 255.0 + 0.5)
            out[channel] = 0 if qv < 0 else 255 if qv > 255 else qv
    return bytes(out)
