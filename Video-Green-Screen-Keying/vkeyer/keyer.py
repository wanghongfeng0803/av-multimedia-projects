"""第一层：色度键控（keying）。

本层只做两件事——由键色与相似度/软化度计算逐像素 alpha，以及对
半透明像素做溢色抑制（despill，去残绿）。键色一律由外部传入
（来自 :mod:`vkeyer.sampler` 的画面采样或调用方显式指定），
**算法里不存在任何写死的绿色**，蓝幕、红幕、任意纯色幕同理可用。

距离度量
========
统一在 rgb24 字节域上用欧氏距离::

    d = sqrt((R-Kr)^2 + (G-Kg)^2 + (B-Kb)^2)

最大可能距离为 sqrt(3)*255 ≈ 441.67。选择 RGB 欧氏距离而非色相角，
是因为它对任意键色语义相同、无需约定特殊颜色空间，且与后续
despill 的强度权重天然同尺度。

alpha 软边（双阈值，线性过渡）
=============================
- ``d <= similarity``        → alpha = 0（完全抠除，幕布核心区）
- ``d >= similarity+softness`` → alpha = 255（完全保留，前景核心区）
- 两阈值之间线性插值，定点四舍五入到整数；
- ``softness=0`` 退化为硬键（硬切边），阈值上的像素归入前景。

溢色抑制（despill，去残绿/残蓝）
================================
半透明边缘像素的颜色被幕布反射光污染（典型为绿边）。只在
``alpha < 255`` 的像素上、按 ``1 - alpha/255`` 的权重把「键色的
主导通道」压到另外两通道的最大值：

- 键色 G 为严格最大通道时压 G（绿幕去残绿）；B 严格最大时压 B
  （蓝幕去残蓝）；三通道无严格主导（灰/白幕）时不做抑制；
- 完全不透明像素（alpha=255）一律不动，前景主体颜色零损伤；
- 全透明像素权重为 1，RGB 被中和为不含幕布色调的中性值，保证
  后续背景替换在任何 alpha 预乘约定下都不翻出幕布色。

本层输出固定为 RGBA 交错字节（每像素 R, G, B, A 四字节），
供第二层 :mod:`vkeyer.matte` 与第三层 :mod:`vkeyer.encoder` 复用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MAX_DISTANCE",
    "KeyerConfig",
    "Keyer",
    "dominant_channel",
    "despill_rgb",
]

#: rgb24 字节域上离任意键色的最大欧氏距离 sqrt(3)*255
MAX_DISTANCE = math.sqrt(3.0) * 255.0


def _check_channel(value, name) -> int:
    if not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"{name} 必须是 [0, 255] 整数，得到 {value!r}")
    return value


@dataclass(frozen=True)
class KeyerConfig:
    """键控参数（不可变，便于整次处理复用与哈希对照）。

    - ``key_rgb``：键色三通道元组，来自画面采样或调用方指定；
    - ``similarity``：内阈值（0–255 量纲的距离），距离不超过它的
      像素完全透明；
    - ``softness``：软化带宽度，距离在
      ``[similarity, similarity+softness]`` 内线性过渡到不透明；
    - ``despill``：是否启用溢色抑制。
    """

    key_rgb: tuple[int, int, int]
    similarity: float = 32.0
    softness: float = 64.0
    despill: bool = True

    def __post_init__(self):
        if len(self.key_rgb) != 3:
            raise ValueError("key_rgb 必须是 (R, G, B) 三元组")
        for i, c in enumerate(self.key_rgb):
            _check_channel(c, f"key_rgb[{i}]")
        if self.similarity < 0:
            raise ValueError("similarity 不能为负")
        if self.softness < 0:
            raise ValueError("softness 不能为负")
        if self.similarity > MAX_DISTANCE:
            raise ValueError(f"similarity 超过最大距离 {MAX_DISTANCE:.2f}")


def dominant_channel(key_rgb) -> int:
    """返回键色的严格主导通道下标（0=R, 1=G, 2=B），无严格主导返回 -1。

    绿幕返回 1、蓝幕返回 2；灰/白幕三通道并列或无唯一最大值时
    返回 -1，此时不做单通道 despill。
    """
    r, g, b = key_rgb
    hi = max(key_rgb)
    if key_rgb.count(hi) != 1:
        return -1
    return key_rgb.index(hi)


def despill_rgb(r, g, b, dom, weight):
    """对单个像素做溢色抑制，返回新的 ``(r, g, b)``。

    ``dom`` 为 :func:`dominant_channel` 的结果；``weight`` 为抑制
    权重（0 不动、1 全量），实现上取 0–255 的定点权重。
    仅当主导通道确实高于另外两通道最大值时才下压，避免误伤
    前景中天然偏亮的同通道像素。
    """
    if dom < 0 or weight <= 0:
        return r, g, b
    px = [r, g, b]
    other_max = max(px[c] for c in range(3) if c != dom)
    if px[dom] <= other_max:
        return r, g, b
    # 定点混合：new = old - w*(old-other_max)，w = weight/255
    excess = px[dom] - other_max
    px[dom] = px[dom] - (weight * excess + 127) // 255
    return px[0], px[1], px[2]


class Keyer:
    """逐帧色度键控器：rgb24 裸帧 → RGBA 裸帧（含 despill）。

    线程安全：构造后只读，可在多进程分块 worker 中各自持有副本。
    """

    def __init__(self, config: KeyerConfig):
        self.config = config
        kr, kg, kb = config.key_rgb
        self._kr, self._kg, self._kb = kr, kg, kb
        self._sim = float(config.similarity)
        self._soft = float(config.softness)
        # 外阈值与倒数一次算好，热循环里只做乘法不做除法
        self._outer = self._sim + self._soft
        self._inv_soft = 1.0 / self._soft if self._soft > 0 else 0.0
        self._dom = dominant_channel(config.key_rgb) if config.despill else -1

    def key_frame(self, rgb_bytes: bytes, width: int, height: int) -> bytearray:
        """把一帧 rgb24 交错字节键控为 rgba 交错字节。

        输入长度必须恰为 ``width*height*3``，输出长度为其 4/3 倍。
        alpha 与 despill 均在同一遍扫描内完成，避免二次遍历。
        """
        expected = width * height * 3
        if len(rgb_bytes) != expected:
            raise ValueError(
                f"rgb24 帧长度 {len(rgb_bytes)} 与 {width}x{height} 不符"
                f"（应为 {expected}）"
            )
        kr, kg, kb = self._kr, self._kg, self._kb
        sim, outer, inv_soft = self._sim, self._outer, self._inv_soft
        dom = self._dom
        out = bytearray(width * height * 4)
        oi = 0
        for i in range(0, expected, 3):
            r = rgb_bytes[i]
            g = rgb_bytes[i + 1]
            b = rgb_bytes[i + 2]
            d = math.sqrt(
                (r - kr) * (r - kr)
                + (g - kg) * (g - kg)
                + (b - kb) * (b - kb)
            )
            if d <= sim:
                alpha = 0
            elif d >= outer:
                alpha = 255
            elif inv_soft > 0.0:
                # 线性过渡，定点四舍五入并钳制到 [1, 254]
                a = int((d - sim) * inv_soft * 255.0 + 0.5)
                alpha = 1 if a < 1 else (254 if a > 254 else a)
            else:
                # softness=0：硬键，阈值之外全部不透明
                alpha = 255
            if dom >= 0 and alpha < 255:
                # 半透明/全透明像素按不透明度的补数加权抑制溢色
                weight = 255 - alpha
                r, g, b = despill_rgb(r, g, b, dom, weight)
            out[oi] = r
            out[oi + 1] = g
            out[oi + 2] = b
            out[oi + 3] = alpha
            oi += 4
        return out
