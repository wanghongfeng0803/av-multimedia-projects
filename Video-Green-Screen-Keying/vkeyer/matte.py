"""第二层：alpha 遮罩后处理（matte）。

第一层产出的软 alpha 在真实素材上常带压缩噪点与半透明毛刺，本层
提供三个**可独立调用、也可按固定顺序串联**的算子，供后续羽化
与背景替换复用：

1. 阈值硬化（:func:`threshold_alpha`）：把接近纯透明/纯不透明的
   alpha 钳到 0/255，消灭幕布区残留的稀疏噪点；
2. choke 收缩/扩张（:func:`choke_alpha`）：3×3 序统计滤波，
   正值取局部最小值收缩前景边缘（去绿边最直接的手段），负值取
   局部最大值扩张；边界按 REFLECT_101 镜像，不补黑边；
3. 盒式羽化（:func:`feather_alpha`）：可分离滑动窗均值，边界
   截断到有效邻域（与 vdenoise 均值滤波同一套语义），把硬边
   变成平滑过渡带。

固定处理顺序（:class:`MatteConfig` / :func:`postprocess_alpha`）：
**先硬化、再 choke、最后羽化**。羽化必须在最后，否则它新造出
的中间灰会被阈值硬化再次吃掉；choke 在羽化前作用于干净的
二值/近二值遮罩，边缘位移量可预期。

本层只读写 alpha 平面（``width*height`` 字节），不触碰 RGB，
与键控层、编码层彻底解耦。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

from .keyer import despill_rgb, dominant_channel

__all__ = [
    "MatteConfig",
    "threshold_alpha",
    "reflect101",
    "choke_alpha",
    "feather_alpha_row",
    "feather_alpha",
    "postprocess_alpha",
    "apply_matte_to_frame",
]


def _check_plane(alpha, width, height):
    if len(alpha) != width * height:
        raise ValueError(
            f"alpha 平面长度 {len(alpha)} 与 {width}x{height} 不符"
        )


def threshold_alpha(alpha, width, height, low=8, high=247):
    """阈值硬化：``a<=low`` 钳 0、``a>=high`` 钳 255，中间值保留。

    默认阈值围绕双阈值键控的天然分布设置：幕布核心区 alpha=0、
    前景核心区 alpha=255，只有软化带与压缩噪点落在中间。
    """
    _check_plane(alpha, width, height)
    if not 0 <= low <= high <= 255:
        raise ValueError("需满足 0 <= low <= high <= 255")
    out = bytearray(width * height)
    for i, a in enumerate(alpha):
        if a <= low:
            out[i] = 0
        elif a >= high:
            out[i] = 255
        else:
            out[i] = a
    return out


def reflect101(index, length):
    """REFLECT_101 镜像下标（与 numpy ``mode='reflect'`` 同语义）。

    边界外的索引按 ``... 2 1 0 0 1 2 ...`` 反射，端点不重复。
    先对周期 ``2*(length-1)`` 取模折回一个周期，再把后半段
    折返——对任意越界深度（含多层嵌套反射）都收敛回有效范围，
    单条 ``last - abs(last - idx)`` 公式只在单层越界时成立。
    """
    if length == 1:
        return 0
    period = 2 * (length - 1)
    idx = index % period
    last = length - 1
    return idx if idx <= last else period - idx


def choke_alpha(alpha, width, height, radius=1):
    """3×3（或更大奇数窗）序统计滤波收缩/扩张遮罩。

    - ``radius > 0``：窗口取**最小值**，前景边缘向内收缩 radius
      像素，连带吃掉边缘一圈幕布色（去残绿的几何手段）；
    - ``radius < 0``：取**最大值**，遮罩向外扩张 |radius| 像素；
    - ``radius = 0``：原样拷贝。

    边界按 :func:`reflect101` 镜像填充，输出尺寸不变、无黑边。
    实现为逐像素窗口扫描，choke 半径实际素材几乎只用 1–2。
    """
    _check_plane(alpha, width, height)
    if radius == 0:
        return bytes(alpha)
    r = abs(radius)
    take_min = radius > 0
    out = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            best = 255 if take_min else 0
            for dy in range(-r, r + 1):
                sy = reflect101(y + dy, height)
                base = sy * width
                for dx in range(-r, r + 1):
                    sx = reflect101(x + dx, width)
                    v = alpha[base + sx]
                    if take_min:
                        if v < best:
                            best = v
                    elif v > best:
                        best = v
            out[y * width + x] = best
    return out


def feather_alpha_row(row, radius):
    """单行水平滑动窗均值（截断边界，定点四舍五入）。"""
    width = len(row)
    if radius <= 0 or width == 0:
        return list(row)
    prefix = list(accumulate(row, initial=0))
    out = [0] * width
    for x in range(width):
        lo = x - radius
        if lo < 0:
            lo = 0
        hi = x + radius
        if hi >= width:
            hi = width - 1
        cnt = hi - lo + 1
        out[x] = (prefix[hi + 1] - prefix[lo] + cnt // 2) // cnt
    return out


def feather_alpha(alpha, width, height, radius=2):
    """可分离盒式羽化：水平均值 + 垂直均值，边界截断。

    与 vdenoise 盒式滤波同一设计：两遍各自四舍五入，全覆盖窗口
    与直算均值最多 ±1 误差；``radius <= 0`` 时原样返回。
    """
    _check_plane(alpha, width, height)
    if radius <= 0:
        return bytes(alpha)
    hrows = [
        feather_alpha_row(alpha[y * width:(y + 1) * width], radius)
        for y in range(height)
    ]
    out = bytearray(width * height)
    for y in range(height):
        lo = y - radius
        if lo < 0:
            lo = 0
        hi = y + radius
        if hi >= height:
            hi = height - 1
        cnt = hi - lo + 1
        half = cnt // 2
        for x in range(width):
            s = 0
            for j in range(lo, hi + 1):
                s += hrows[j][x]
            out[y * width + x] = (s + half) // cnt
    return out


@dataclass(frozen=True)
class MatteConfig:
    """alpha 后处理参数（不可变）；全默认值为不改变遮罩的恒等基线。

    - ``threshold_low/high``：硬化阈值，``low=0 且 high=255`` 时关闭；
    - ``choke``：收缩（正）/扩张（负）半径，0 关闭；
    - ``feather``：羽化半径，0 关闭。
    """

    threshold_low: int = 0
    threshold_high: int = 255
    choke: int = 0
    feather: int = 0

    def __post_init__(self):
        if not 0 <= self.threshold_low <= self.threshold_high <= 255:
            raise ValueError("需满足 0 <= threshold_low <= threshold_high <= 255")
        if self.choke < 0:
            raise ValueError("choke 暂只支持非负收缩半径")
        if self.feather < 0:
            raise ValueError("feather 不能为负")

    @property
    def is_identity(self) -> bool:
        """三项全关闭时为恒等映射，流水线可跳过整个后处理遍。"""
        return (
            self.threshold_low == 0
            and self.threshold_high == 255
            and self.choke == 0
            and self.feather == 0
        )


def postprocess_alpha(alpha, width, height, config: MatteConfig):
    """按固定顺序（硬化 → choke → 羽化）应用后处理，返回新 alpha。"""
    _check_plane(alpha, width, height)
    if config.is_identity:
        return bytes(alpha)
    a = threshold_alpha(
        alpha, width, height, config.threshold_low, config.threshold_high
    )
    if config.choke > 0:
        a = choke_alpha(a, width, height, config.choke)
    if config.feather > 0:
        a = feather_alpha(a, width, height, config.feather)
    return bytes(a)


def apply_matte_to_frame(rgba, width, height, alpha, key_rgb,
                         despill=True):
    """把后处理后的 alpha 平面写回 RGBA 帧，并对新变半透明的边缘
    重新做一次 despill。

    第一层的 despill 依据**原始** alpha 加权；硬化/choke/羽化之后
    边缘像素的 alpha 发生变化，若不同步刷新 RGB，被 choke 救回的
    前景像素仍带着旧的抑制色。这里按新 alpha 统一重算一遍，
    保证「RGB 去色强度」与「最终 alpha」严格一致。
    """
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA 帧长度与尺寸不符")
    if len(alpha) != width * height:
        raise ValueError("alpha 平面长度与尺寸不符")
    out = bytearray(rgba)
    dom = dominant_channel(key_rgb) if despill else -1
    for i, a in enumerate(alpha):
        p = i * 4
        out[p + 3] = a
        if dom >= 0 and a < 255:
            r, g, b = despill_rgb(out[p], out[p + 1], out[p + 2],
                                  dom, 255 - a)
            out[p], out[p + 1], out[p + 2] = r, g, b
    return out
