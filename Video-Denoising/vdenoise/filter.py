"""可分离滑动窗盒式（均值）滤波。

设计决策：
- 二维均值滤波拆成水平、垂直两个一维滑动窗，复杂度与半径无关：
  水平遍用 itertools.accumulate 前缀和，垂直遍维护整行宽度的列运行和。
- 边界不越界：窗口截断到有效范围，除以实际参与的像素数，
  不做边缘复制，边缘像素的均值仍是其真实邻域的均值。
- 两遍各自四舍五入 ``(sum + count//2) // count``：全覆盖窗口的
  结果与「全局均值直算」最多相差 ±1，对照测试须留此容差。
- 性能：垂直遍滑动时用 ``[a+b for a,b in zip(colsum, row)]``
  整体加/减一行（C 层循环），比逐像素 strided 访问快一个量级。
"""

from itertools import accumulate

__all__ = ["horizontal_pass_row", "vertical_pass", "box_blur_plane"]


def horizontal_pass_row(row, radius):
    """单行水平滑动窗均值，返回 int 列表（供垂直遍继续求和）。

    ``row`` 为 bytes 或等长 int 序列；半径 <= 0 时原样返回列表。
    """
    width = len(row)
    if radius <= 0 or width == 0:
        return list(row)
    prefix = list(accumulate(row, initial=0))
    out = []
    for x in range(width):
        lo = x - radius
        if lo < 0:
            lo = 0
        hi = x + radius
        if hi >= width:
            hi = width - 1
        cnt = hi - lo + 1
        out.append((prefix[hi + 1] - prefix[lo] + cnt // 2) // cnt)
    return out


def vertical_pass(rows, radius, out_lo=0, out_hi=None):
    """对行序列做垂直滑动窗均值，返回 bytes 行列表。

    ``rows`` 每项是等长 int 序列（通常是 :func:`horizontal_pass_row`
    的输出）。只输出 ``[out_lo, out_hi)`` 行（相对 ``rows`` 的索引），
    分带处理借此复用同一实现：窗口截断到 ``rows`` 首尾，调用方保证
    提供的上下文足够，截断行为即与整帧滤波一致。
    """
    height = len(rows)
    if height == 0:
        return []
    if out_hi is None or out_hi > height:
        out_hi = height
    if out_lo < 0:
        out_lo = 0
    width = len(rows[0])
    result = []
    colsum = None
    cur_lo = cur_hi = 0
    for y in range(out_lo, out_hi):
        lo = y - radius
        if lo < 0:
            lo = 0
        hi = y + radius
        if hi >= height:
            hi = height - 1
        if colsum is None:
            colsum = [0] * width
            for j in range(lo, hi + 1):
                row = rows[j]
                colsum = [a + b for a, b in zip(colsum, row)]
        else:
            while cur_lo < lo:
                row = rows[cur_lo]
                colsum = [a - b for a, b in zip(colsum, row)]
                cur_lo += 1
            while cur_hi < hi:
                cur_hi += 1
                row = rows[cur_hi]
                colsum = [a + b for a, b in zip(colsum, row)]
        cur_lo, cur_hi = lo, hi
        cnt = hi - lo + 1
        half = cnt // 2
        result.append(bytes((s + half) // cnt for s in colsum))
    return result


def box_blur_plane(data, width, height, radius):
    """整帧单平面均值滤波，输入输出均为 ``width*height`` 字节的 bytes。"""
    if radius <= 0 or width <= 0 or height <= 0:
        return bytes(data)
    hrows = [
        horizontal_pass_row(data[y * width:(y + 1) * width], radius)
        for y in range(height)
    ]
    return b"".join(vertical_pass(hrows, radius))
