"""打码模块：全局块网格上的一次性马赛克。

对应需求的三条硬约束：

1. **重叠只打一次、无拼接缝**：所有区域共用一套原点固定在 (0,0)、
   边长固定的全局块网格。区域先各自映射为命中块，再求并集（set
   去重），每块只对原图取一次均值、回填一次——重叠区不存在
   「二次打码」，也不存在两套网格相位错开产生的拼接缝。
2. **运行时增删不闪烁**：块均值只取决于块自身像素与全局网格相位，
   与其他区域无关；增删区域只改变命中块集合，不动既有块的网格
   相位与均值口径，已稳定画面逐字节不变。
3. **契约自包含**：:class:`BlockTable` 是打码 / 区域管理 / 编码
   三模块之间的唯一数据契约。编码侧只消费它；后续自动检测
   （人脸 / 隐私区域识别）只要产出同一契约即可无缝接入。

打码语义：命中块**整块**以块内逐通道均值回填（区域边缘按块对齐，
向外取整到块边界）。均值舍入为 ``(sum + n//2) // n``（四舍五入、
定点可复现）。``block=1`` 时退化为逐像素恒等，是恒等基线。
"""

from dataclasses import dataclass

__all__ = ["BlockTable", "build_block_table", "apply_mosaic"]


@dataclass(frozen=True)
class BlockTable:
    """一帧的打码契约：全局网格参数 + 命中块清单（像素坐标）。

    :ivar width: 帧宽（像素）。
    :ivar height: 帧高（像素）。
    :ivar block: 全局网格块边长（像素，>= 1）。
    :ivar blocks: 命中块 ``((x, y, w, h), ...)``，按 (y, x) 排序；
        每块原点对齐全局网格（``x % block == y % block == 0``），
        帧右/下边缘的块被裁小（``w``/``h`` < ``block``）。
    """

    width: int
    height: int
    block: int
    blocks: tuple

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError("帧尺寸必须为正")
        if self.block < 1:
            raise ValueError("块边长必须 >= 1")
        for b in self.blocks:
            if len(b) != 4:
                raise ValueError(f"非法块条目 {b!r}")
            x, y, w, h = b
            if x % self.block or y % self.block:
                raise ValueError(f"块原点未对齐全局网格：{b!r}")
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"块原点越界：{b!r}")
            if not (1 <= w <= self.block and 1 <= h <= self.block):
                raise ValueError(f"块尺寸非法：{b!r}")
            if x + w > self.width or y + h > self.height:
                raise ValueError(f"块越出帧边界：{b!r}")

    @property
    def is_empty(self) -> bool:
        """无命中块（打码退化为恒等）。"""
        return not self.blocks


def build_block_table(regions, width, height, block):
    """由已钳制区域集合构建块表（并集去重，每块只出现一次）。

    ``regions`` 为若干具有 ``x/y/w/h`` 属性的对象（通常来自
    :mod:`vblur.regions` 的钳制输出，已保证在帧内且宽高 >= 1）。
    覆盖判定：与区域有任意像素相交的网格块即命中，整块打码。
    """
    if width < 1 or height < 1:
        raise ValueError("帧尺寸必须为正")
    if block < 1:
        raise ValueError("块边长必须 >= 1")
    hits = set()
    for r in regions:
        bx0 = r.x // block
        by0 = r.y // block
        bx1 = (r.x + r.w - 1) // block
        by1 = (r.y + r.h - 1) // block
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                hits.add((bx, by))
    blocks = []
    for bx, by in sorted(hits, key=lambda t: (t[1], t[0])):
        x = bx * block
        y = by * block
        blocks.append((x, y, min(block, width - x), min(block, height - y)))
    return BlockTable(width, height, block, tuple(blocks))


def apply_mosaic(frame, table, channels=3):
    """按块表对一帧 packed 字节就地打码，返回同一 ``bytearray``。

    每块逐通道求均值后整块回填。逐通道求和用步进切片（C 速度），
    避免逐像素 Python 循环；空表为恒等基线（帧逐字节不变）。
    """
    width = table.width
    stride = width * channels
    for x, y, w, h in table.blocks:
        n = w * h
        sums = [0] * channels
        for row in range(y, y + h):
            base = row * stride + x * channels
            stop = base + w * channels
            for c in range(channels):
                sums[c] += sum(frame[base + c:stop:channels])
        fill = bytes((s + n // 2) // n for s in sums) * w
        row_bytes = w * channels
        for row in range(y, y + h):
            base = row * stride + x * channels
            frame[base:base + row_bytes] = fill
    return frame
