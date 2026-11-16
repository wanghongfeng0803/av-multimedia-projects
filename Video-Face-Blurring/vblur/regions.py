"""区域管理：钳制、运行时增删、块表快照。

钳制规则（:func:`clamp_region`，顺序固定）：

1. **负尺寸归一化**：``w < 0`` 视为「终点坐标」写法，``x += w,
   w = -w``；``h`` 同理。归一化后 ``w, h >= 0``。
2. **零面积**：``w == 0`` 或 ``h == 0`` → ``None``（无像素可打）。
3. **完全越界**：与帧矩形无交集 → ``None``。
4. **部分越界**：裁剪到 ``[0, width) × [0, height)``。

:class:`RegionManager` 支持在处理循环运行期间从任意线程增删区域：
增删用锁串行化，快照（:meth:`RegionManager.regions` /
:meth:`RegionManager.block_table`）拿到的是不可变结果，之后的增删
不影响已拿到的快照；:meth:`RegionManager.as_source` 返回的逐帧块表
源按代际缓存——无增删时逐帧返回同一对象（已稳定帧零开销、零变化，
即「无闪烁」的实现基础）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .mosaic import BlockTable, build_block_table

__all__ = ["Region", "RegionManager", "clamp_region"]


@dataclass(frozen=True)
class Region:
    """已钳制的帧内矩形区域（像素坐标，``w``/``h`` >= 1）。

    :ivar x: 左边界（含），``0 <= x < x + w <= 帧宽``。
    :ivar y: 上边界（含）。
    :ivar w: 宽（像素）。
    :ivar h: 高（像素）。
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        """右边界（不含）。"""
        return self.x + self.w

    @property
    def bottom(self) -> int:
        """下边界（不含）。"""
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h


def clamp_region(x, y, w, h, width, height):
    """把任意矩形钳制到帧内，返回 :class:`Region` 或 ``None``。

    钳制顺序见模块 docstring；完全越界 / 零面积返回 ``None``。
    """
    if width < 1 or height < 1:
        raise ValueError("帧尺寸必须为正")
    if w < 0:
        x += w
        w = -w
    if h < 0:
        y += h
        h = -h
    if w == 0 or h == 0:
        return None
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, width)
    y1 = min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Region(x0, y0, x1 - x0, y1 - y0)


class RegionManager:
    """线程安全的运行时区域表。

    完全越界 / 零面积的区域 :meth:`add` 直接丢弃并返回 ``None``
    （不计入表、不占用 id）。其余区域分配单调递增 id，凭 id 删除。
    """

    def __init__(self, width, height):
        if width < 1 or height < 1:
            raise ValueError("帧尺寸必须为正")
        self._width = width
        self._height = height
        self._lock = threading.Lock()
        self._regions = {}  # id -> Region
        self._next_id = 1
        self._generation = 0

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def add(self, x, y, w, h):
        """添加区域，返回区域 id；钳制后为空（完全越界/零面积）返回 ``None``。"""
        region = clamp_region(x, y, w, h, self._width, self._height)
        if region is None:
            return None
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            self._regions[rid] = region
            self._generation += 1
        return rid

    def remove(self, region_id):
        """按 id 删除区域；不存在返回 ``False``。"""
        with self._lock:
            if region_id not in self._regions:
                return False
            del self._regions[region_id]
            self._generation += 1
        return True

    def clear(self):
        """清空全部区域。"""
        with self._lock:
            if self._regions:
                self._regions.clear()
                self._generation += 1

    def regions(self):
        """当前有效区域的不可变快照（之后的增删不影响它）。"""
        with self._lock:
            return tuple(self._regions.values())

    def block_table(self, block):
        """当前区域并集命中的全局网格块表（一次性快照）。"""
        with self._lock:
            return build_block_table(
                tuple(self._regions.values()), self._width, self._height,
                block,
            )

    def as_source(self, block):
        """返回 ``frame_index -> BlockTable`` 契约可调用对象。

        供编码模块逐帧消费；内部按代际缓存——无增删时每次调用返回
        同一个 :class:`BlockTable` 对象（不可变，可安全共享），有增删
        时重建一次。``frame_index`` 参数仅为契约形状，静态区域不使用。
        """
        if block < 1:
            raise ValueError("块边长必须 >= 1")
        cache = {}

        def source(frame_index):
            with self._lock:
                gen = self._generation
                if cache.get("gen") != gen:
                    cache["table"] = build_block_table(
                        tuple(self._regions.values()),
                        self._width, self._height, block,
                    )
                    cache["gen"] = gen
                return cache["table"]

        return source
