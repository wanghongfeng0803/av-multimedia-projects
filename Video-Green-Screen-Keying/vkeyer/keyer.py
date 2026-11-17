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

容差（tolerance，调容差不切发丝）
=================================
``tolerance ∈ [0, 1]`` 以**比例缩放**方式放宽距离判定::

    d_eff = d * (1 - tolerance)

距离映射处处连续：增大容差时，软化带内像素的 alpha 沿既有线性
斜坡**平滑下移**，不会被硬切成 0——发丝、薄纱等半透明边缘只是
变得更透，而不是被齐根裁掉。只有极端容差（趋近 1）才会把软带
中段像素压到全透明。默认 ``tolerance=0`` 时 ``d_eff ≡ d``，
与旧版本逐像素一致。

溢色抑制（despill，去残绿/残蓝）
================================
半透明边缘像素的颜色被幕布反射光污染（典型为绿边）。只在
``alpha < 255`` 的像素上、按 ``(1 - alpha/255) * despill_strength``
的权重把「键色的主导通道」压到另外两通道的最大值：

- 键色 G 为严格最大通道时压 G（绿幕去残绿）；B 严格最大时压 B
  （蓝幕去残蓝）；三通道无严格主导（灰/白幕）时不做抑制；
- 完全不透明像素（alpha=255）一律不动，前景主体颜色零损伤；
- 仅当像素的主导通道**确实高于**另两通道最大值时才下压——肤色
  （R 主导）、灰衣（无主导通道）等非绿前景即使落在半透明边缘
  上也不会被污染；
- ``despill_strength ∈ [0, 1]`` 线性缩放抑制力度，默认 1.0 与
  旧版本逐像素一致，0 等价于关闭 despill；
- 全透明像素权重最大，RGB 被中和为不含幕布色调的中性值，保证
  后续背景替换在任何 alpha 预乘约定下都不翻出幕布色。

运行中改参数（无中间脏帧）
==========================
:class:`Keyer` 的全部派生量（阈值、倒数、主导通道、定点强度）
打包在一个不可变对象里，:meth:`Keyer.update_config` 在锁内**整体
替换**；:meth:`Keyer.key_frame` 只在帧首取一次快照。因此任意一帧
要么完全按旧参数、要么完全按新参数处理，绝不出现半帧旧半帧新的
中间脏帧。

本层输出固定为 RGBA 交错字节（每像素 R, G, B, A 四字节），
供第二层 :mod:`vkeyer.matte` 与第三层 :mod:`vkeyer.encoder` 复用。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

__all__ = [
    "MAX_DISTANCE",
    "KeyerConfig",
    "Keyer",
    "EdgeStats",
    "dominant_channel",
    "despill_rgb",
    "despill_weight",
]

#: rgb24 字节域上离任意键色的最大欧氏距离 sqrt(3)*255
MAX_DISTANCE = math.sqrt(3.0) * 255.0


def _check_channel(value, name) -> int:
    if not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"{name} 必须是 [0, 255] 整数，得到 {value!r}")
    return value


def _check_unit_interval(value, name) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1]，得到 {value!r}")
    return v


@dataclass(frozen=True)
class KeyerConfig:
    """键控参数（不可变，便于整次处理复用与哈希对照）。

    - ``key_rgb``：键色三通道元组，来自画面采样或调用方指定；
    - ``similarity``：内阈值（0–255 量纲的距离），距离不超过它的
      像素完全透明；
    - ``softness``：软化带宽度，距离在
      ``[similarity, similarity+softness]`` 内线性过渡到不透明；
    - ``despill``：是否启用溢色抑制；
    - ``tolerance``：容差 [0, 1]，按比例 ``(1-t)`` 缩放距离判定，
      默认 0 与旧版一致；连续映射，不会把半透明边缘硬切掉；
    - ``despill_strength``：溢色抑制强度 [0, 1]，默认 1 与旧版
      一致，0 等价于关闭 despill。
    """

    key_rgb: tuple[int, int, int]
    similarity: float = 32.0
    softness: float = 64.0
    despill: bool = True
    tolerance: float = 0.0
    despill_strength: float = 1.0

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
        _check_unit_interval(self.tolerance, "tolerance")
        _check_unit_interval(self.despill_strength, "despill_strength")


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


def despill_weight(alpha, strength_q) -> int:
    """由 alpha 与定点强度（0–255）计算 0–255 的溢色抑制权重。

    ``weight = round((255 - alpha) * strength_q / 255)``。
    ``strength_q=255``（即强度 1.0）时精确退化为旧版的
    ``255 - alpha``；``strength_q=0`` 时恒为 0（不抑制）。
    """
    return ((255 - alpha) * strength_q + 127) // 255


def despill_rgb(r, g, b, dom, weight):
    """对单个像素做溢色抑制，返回新的 ``(r, g, b)``。

    ``dom`` 为 :func:`dominant_channel` 的结果；``weight`` 为抑制
    权重（0 不动、255 全量），实现上取 0–255 的定点权重。
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


def _alpha_of(dist, sim, outer, inv_soft) -> int:
    """由（已缩放）距离查双阈值软 alpha，键控与统计共用同一语义。"""
    if dist <= sim:
        return 0
    if dist >= outer:
        return 255
    if inv_soft > 0.0:
        # 线性过渡，定点四舍五入并钳制到 [1, 254]
        a = int((dist - sim) * inv_soft * 255.0 + 0.5)
        return 1 if a < 1 else (254 if a > 254 else a)
    # softness=0：硬键，阈值之外全部不透明
    return 255


@dataclass(frozen=True)
class _Derived:
    """由 KeyerConfig 一次性算好的派生量（不可变，整体替换用）。"""

    kr: int
    kg: int
    kb: int
    sim: float
    outer: float
    inv_soft: float
    dist_scale: float   # 1 - tolerance
    dom: int            # 键色主导通道；despill 关闭时为 -1
    strength_q: int     # despill_strength 的 0–255 定点


@dataclass(frozen=True)
class EdgeStats:
    """一帧在当前配置下的边缘像素统计（:meth:`Keyer.analyze_frame`）。

    - ``pixels``：总像素数；
    - ``edge_pixels``：半透明边缘像素数（0 < alpha < 255）；
    - ``transparent_pixels`` / ``opaque_pixels``：全透明/全不透明数；
    - ``tolerance_affected``：与 ``tolerance=0`` 基线相比 alpha
      发生变化的像素数——量化「调容差」影响了多少边缘像素；
    - ``despill_affected``：RGB 被溢色抑制实际改写的像素数
      （相对 ``despill_strength=0`` 基线）——量化「溢出抑制」
      影响了多少边缘像素。

    两个计数各自独立可比：对同一输入分别调整容差与抑制强度，
    用本结构即可量化二者各自触及的边缘像素规模差异。
    """

    pixels: int
    edge_pixels: int
    transparent_pixels: int
    opaque_pixels: int
    tolerance_affected: int
    despill_affected: int


class Keyer:
    """逐帧色度键控器：rgb24 裸帧 → RGBA 裸帧（含 despill）。

    线程安全：派生量整体不可变，:meth:`update_config` 原子换参，
    可在多进程分块 worker 中各自持有副本，也可在运行中安全调参。
    """

    def __init__(self, config: KeyerConfig):
        self._lock = threading.Lock()
        self._config = config
        self._derived = self._derive(config)

    @staticmethod
    def _derive(config: KeyerConfig) -> _Derived:
        kr, kg, kb = config.key_rgb
        sim = float(config.similarity)
        soft = float(config.softness)
        return _Derived(
            kr=kr, kg=kg, kb=kb,
            sim=sim,
            outer=sim + soft,
            inv_soft=1.0 / soft if soft > 0 else 0.0,
            dist_scale=1.0 - float(config.tolerance),
            dom=dominant_channel(config.key_rgb) if config.despill else -1,
            strength_q=int(float(config.despill_strength) * 255.0 + 0.5),
        )

    @property
    def config(self) -> KeyerConfig:
        """当前生效的配置（随 update_config 原子更新）。"""
        with self._lock:
            return self._config

    def update_config(self, config: KeyerConfig) -> None:
        """运行中原子换参：下一帧起整体按新配置处理。

        派生量先在锁外算好，锁内只做一次引用替换；正在执行的
        :meth:`key_frame` 已持有旧快照，会完整跑完旧参数的一帧，
        不会产生半新半旧的中间脏帧。``config`` 非法时
        :class:`KeyerConfig` 构造即抛错，当前配置不受影响。
        """
        derived = self._derive(config)
        with self._lock:
            self._config = config
            self._derived = derived

    def _snapshot(self) -> _Derived:
        with self._lock:
            return self._derived

    def key_frame(self, rgb_bytes: bytes, width: int, height: int) -> bytearray:
        """把一帧 rgb24 交错字节键控为 rgba 交错字节。

        输入长度必须恰为 ``width*height*3``，输出长度为其 4/3 倍。
        alpha 与 despill 均在同一遍扫描内完成，避免二次遍历。
        整帧使用帧首快照的同一份派生参数，帧内绝不混参。
        """
        expected = width * height * 3
        if len(rgb_bytes) != expected:
            raise ValueError(
                f"rgb24 帧长度 {len(rgb_bytes)} 与 {width}x{height} 不符"
                f"（应为 {expected}）"
            )
        d = self._snapshot()
        kr, kg, kb = d.kr, d.kg, d.kb
        sim, outer, inv_soft = d.sim, d.outer, d.inv_soft
        scale, dom, strength_q = d.dist_scale, d.dom, d.strength_q
        out = bytearray(width * height * 4)
        oi = 0
        for i in range(0, expected, 3):
            r = rgb_bytes[i]
            g = rgb_bytes[i + 1]
            b = rgb_bytes[i + 2]
            dist = math.sqrt(
                (r - kr) * (r - kr)
                + (g - kg) * (g - kg)
                + (b - kb) * (b - kb)
            ) * scale
            alpha = _alpha_of(dist, sim, outer, inv_soft)
            if dom >= 0 and alpha < 255:
                # 半透明/全透明像素按不透明度补数×强度加权抑制溢色
                r, g, b = despill_rgb(
                    r, g, b, dom, despill_weight(alpha, strength_q)
                )
            out[oi] = r
            out[oi + 1] = g
            out[oi + 2] = b
            out[oi + 3] = alpha
            oi += 4
        return out

    def analyze_frame(self, rgb_bytes: bytes, width: int,
                      height: int) -> EdgeStats:
        """统计当前配置下的边缘像素与两参数各自受影响的像素数。

        对同一输入，``tolerance_affected`` 量化调容差触及的边缘
        像素（alpha 相对零容差基线变化），``despill_affected``
        量化溢出抑制触及的边缘像素（RGB 相对零强度基线被改写），
        两个计数独立可比。只统计、不产出图像，与
        :meth:`key_frame` 共用同一份派生参数与 alpha 语义。
        """
        expected = width * height * 3
        if len(rgb_bytes) != expected:
            raise ValueError(
                f"rgb24 帧长度 {len(rgb_bytes)} 与 {width}x{height} 不符"
                f"（应为 {expected}）"
            )
        d = self._snapshot()
        kr, kg, kb = d.kr, d.kg, d.kb
        sim, outer, inv_soft = d.sim, d.outer, d.inv_soft
        scale, dom, strength_q = d.dist_scale, d.dom, d.strength_q
        edge = transparent = opaque = 0
        tol_affected = dsp_affected = 0
        for i in range(0, expected, 3):
            r = rgb_bytes[i]
            g = rgb_bytes[i + 1]
            b = rgb_bytes[i + 2]
            dist = math.sqrt(
                (r - kr) * (r - kr)
                + (g - kg) * (g - kg)
                + (b - kb) * (b - kb)
            )
            alpha = _alpha_of(dist * scale, sim, outer, inv_soft)
            if alpha != _alpha_of(dist, sim, outer, inv_soft):
                tol_affected += 1
            if alpha == 0:
                transparent += 1
            elif alpha == 255:
                opaque += 1
            else:
                edge += 1
            if dom >= 0 and alpha < 255:
                weight = despill_weight(alpha, strength_q)
                if weight > 0:
                    px = (r, g, b)
                    other_max = max(px[c] for c in range(3) if c != dom)
                    if px[dom] > other_max:
                        dsp_affected += 1
        return EdgeStats(
            pixels=width * height,
            edge_pixels=edge,
            transparent_pixels=transparent,
            opaque_pixels=opaque,
            tolerance_affected=tol_affected,
            despill_affected=dsp_affected,
        )
