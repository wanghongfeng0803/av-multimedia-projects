"""调色处理节点与有序节点链。

3D LUT 不是一次性脚本参数，而是调色链路里的一个**独立节点**。本模块
把链路中每个处理步骤统一成 :class:`GradeNode`：消费 rgb24 裸流分块、
输出同尺寸 rgb24 分块，自带「是否恒等」判断。节点放进有序列表即构成
处理链，**列表顺序就是像素被处理的顺序**，因此调换节点位置会得到
确定且可预期的不同结果（顺序本身由调用方显式控制，节点间不隐式排序）。

内置三个节点（覆盖现有全部能力）：

- :class:`BrightnessContrastNode`：逐通道 1D 映射表（亮度/对比度）；
- :class:`HueSaturationNode`：跨通道 HSV 色相旋转 + 饱和度缩放；
- :class:`LUT3DNode`：3D LUT 三线性插值 + [0,1] 强度混合。

设计约束
--------

- 节点必须可 pickle（Windows ``spawn``）：只持有 ``bytes`` / ``float`` /
  :class:`~color_grading.lut3d.CubeLUT` 等可 pickle 的纯数据，节点方法
  与 worker 入口都是模块级 / 类级可导入符号。
- 恒等节点在链路里**零开销跳过**（见 :func:`is_identity_chain` 与
  :mod:`color_grading.chunks` 的直通路径），保证无参数调用逐字节恒等。
- 同一节点链对同一输入的结果逐字节确定、与进程数 / 分块大小无关。
"""

from __future__ import annotations

from dataclasses import dataclass

from .colorspace import is_identity as _hsv_is_identity
from .colorspace import map_chunk as _hsv_map_chunk
from .lut import IDENTITY_LUT
from .lut3d import CubeLUT, apply_chunk as _lut3d_apply_chunk


class GradeNode:
    """调色节点基类。

    子类用 :class:`dataclasses.dataclass` 实现为持有纯数据的值对象，
    并实现下面两个方法。节点本身不做 I/O，只做纯像素变换。
    """

    def is_identity(self) -> bool:
        """该节点是否对任何像素都不产生改变（恒等）。"""
        raise NotImplementedError

    def apply(self, data: bytes) -> bytes:
        """对一块 rgb24 裸流（长度为 3 的倍数）做本节点变换。"""
        raise NotImplementedError


@dataclass(frozen=True)
class BrightnessContrastNode(GradeNode):
    """亮度 / 对比度 1D LUT 节点，持有构建好的 256 字节映射表。"""

    table: bytes = IDENTITY_LUT

    def is_identity(self) -> bool:
        return self.table == IDENTITY_LUT

    def apply(self, data: bytes) -> bytes:
        if self.is_identity():
            return bytes(data)
        return data.translate(self.table)


@dataclass(frozen=True)
class HueSaturationNode(GradeNode):
    """色相旋转 + 饱和度缩放节点（固定先色相、后饱和度）。"""

    hue: float = 0.0
    saturation: float = 0.0

    def is_identity(self) -> bool:
        return _hsv_is_identity(self.hue, self.saturation)

    def apply(self, data: bytes) -> bytes:
        if self.is_identity():
            return bytes(data)
        return _hsv_map_chunk(data, self.hue, self.saturation)


@dataclass(frozen=True)
class LUT3DNode(GradeNode):
    """3D LUT 节点：三线性插值 + 强度混合。

    强度为 0 时恒等（:func:`apply` 原样返回）；强度为 1 时等价于
    全量 LUT。``lut`` 是已解析校验的 :class:`CubeLUT`，因此本节点
    不会在 worker 内再触发任何解析 / 域校验失败——所有「会报错」的
    环节都在进入像素管道之前完成。
    """

    lut: CubeLUT
    strength: float = 1.0

    def is_identity(self) -> bool:
        return float(self.strength) == 0.0

    def apply(self, data: bytes) -> bytes:
        if self.is_identity():
            return bytes(data)
        return _lut3d_apply_chunk(data, self.lut, float(self.strength))


def is_identity_chain(nodes: list[GradeNode]) -> bool:
    """整条节点链是否为恒等（所有节点都恒等，含空链）。"""
    return all(node.is_identity() for node in nodes)


def apply_chain(data: bytes, nodes: list[GradeNode]) -> bytes:
    """按列表顺序对一块像素依次应用各节点（worker 入口，模块级可 pickle）。

    恒等节点逐个跳过；空链 / 全恒等链返回输入的拷贝，语义即恒等。
    """
    for node in nodes:
        if node.is_identity():
            continue
        data = node.apply(data)
    return bytes(data)
