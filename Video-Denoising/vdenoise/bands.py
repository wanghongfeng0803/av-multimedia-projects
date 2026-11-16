"""横向分带与接缝加权合并。

为什么上下文必须是 2r 而不是 r：
每带除核心行外还向外输出 r 行扩展区，相邻两带的扩展区重叠 2r 行，
重叠区用线性加权合并。若每带只读 r 行上下文，扩展区边缘的输出是
「窗口被截断」的近似值，而邻带同一行是精确值，加权合并后仍残留
最大约 0.5×截断误差的接缝。改为读 2r 行上下文后，重叠区两带都是
精确值，加权合并退化为恒等操作，分带结果与整帧滤波逐像素相等
（测试以零容差锁定这一性质）。

代价：每带多驻留 2r 行上下文，:func:`plan_bands` 按核心行 + 4r 行
（上下各 2r）扣预算。
"""

from .filter import horizontal_pass_row, vertical_pass

__all__ = ["plan_bands", "box_blur_plane_banded", "merge_rows"]


def plan_bands(height, width, radius, budget_bytes, bytes_per_pixel=1, safety=2):
    """按内存预算把 ``height`` 行切成核心行区间列表 ``[(s, e), ...]``。

    每带实际驻留 核心行 + 4r 行（上下各 2r 上下文），故可负担行数
    先扣 4r 再作核心行高；``safety`` 放大每行估算，覆盖水平遍 int
    列表等额外开销。预算再小也保证核心行高 >= 1。
    """
    if height <= 0:
        return []
    if radius <= 0:
        return [(0, height)]
    per_row = max(1, width * bytes_per_pixel * safety)
    afford = budget_bytes // per_row
    core = afford - 4 * radius
    if core < 1:
        core = 1
    if core >= height:
        return [(0, height)]
    bands = []
    y = 0
    while y < height:
        e = y + core
        if e > height:
            e = height
        bands.append((y, e))
        y = e
    return bands


def merge_rows(old, new, length, t):
    """重叠区线性加权合并：``old`` 权重 ``length-1-t``，``new`` 权重 ``t``。

    重叠区两带均为精确值时 ``old == new``，合并退化为恒等操作；
    权重仅在输入有差异（理论上不应发生）时平滑过渡。
    """
    if length <= 1:
        return bytes(new)
    den = length - 1
    w_old = den - t
    half = den // 2
    return bytes((o * w_old + n * t + half) // den for o, n in zip(old, new))


def box_blur_plane_banded(data, width, height, radius, bands):
    """分带均值滤波，结果与 :func:`vdenoise.filter.box_blur_plane` 逐像素相等。

    ``bands`` 为 :func:`plan_bands` 的输出。每带读入核心行上下各 2r
    行上下文，输出核心行加各 r 行扩展区；相邻带重叠的 2r 行用
    :func:`merge_rows` 线性加权合并。
    """
    if radius <= 0:
        return bytes(data)
    out = [None] * height
    for s, e in bands:
        ctx_lo = s - 2 * radius
        if ctx_lo < 0:
            ctx_lo = 0
        ctx_hi = e + 2 * radius
        if ctx_hi > height:
            ctx_hi = height
        hrows = [
            horizontal_pass_row(data[y * width:(y + 1) * width], radius)
            for y in range(ctx_lo, ctx_hi)
        ]
        out_lo = s - radius
        if out_lo < 0:
            out_lo = 0
        out_hi = e + radius
        if out_hi > height:
            out_hi = height
        vrows = vertical_pass(hrows, radius, out_lo - ctx_lo, out_hi - ctx_lo)
        # 与上一带的重叠只可能出现在本带核心起点 s 附近：
        # 上一带输出到下界 min(height, s+r)，本带输出从上界 max(0, s-r) 开始。
        ov_lo = s - radius
        if ov_lo < 0:
            ov_lo = 0
        ov_hi = s + radius
        if ov_hi > height:
            ov_hi = height
        ov_len = ov_hi - ov_lo
        for i, row in enumerate(vrows):
            y = out_lo + i
            prev = out[y]
            if prev is None:
                out[y] = row
            else:
                out[y] = merge_rows(prev, row, ov_len, y - ov_lo)
    return b"".join(out)
