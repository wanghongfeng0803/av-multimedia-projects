"""中值滤波（椒盐噪声专用），直方图加速，镜像边界。

设计决策：
- **边界采用镜像填充（REFLECT_101，不复制边缘）**：长度 n 的轴上，
  填充后的位置 p 映射到源位置 ``n-1 - abs((n-1) - abs(p) % (2*(n-1)))``
  （n=1 退化为恒等）。每个输出窗口恒为 (2r+1)² 个有效像素，不存在
  均值模式那种「窗口截断、除以实际计数」的语义——两种模式边界行为
  刻意不同，互不影响。
- **Huang 直方图加速**：窗口为奇数边长 side=2r+1，中值即排序后第
  side²//2 位。逐行维护一个 256 桶直方图：
    1. 每行左端用 side 个竖条（每竖条 side 像素）建一次直方图；
    2. 横向滑动时移出左竖条、移入右竖条，每像素只动 2*side 个桶；
    3. 中值用累计计数定位，并用 ``<= 中值计数`` 增量行走，不做排序。
  每像素代价 O(r + 256)，与窗口面积 O(r²) 的暴力排序相比，窗口
  放大时耗时只随 r 近线性增长而非成倍恶化。
- **分带复用同一内核**：每带只构造核心行上下各 r 行上下文的镜像
  子网格，直接调整帧内核，带间无重叠输出、无加权合并——中值滤波
  不存在均值模式的接缝问题，分带结果与整帧逐像素相等。
- 中值滤波不可分离（中值不满足结合律），因此没有水平/垂直两遍
  写法，直方图是标准的单遍二维加速方案。
"""

import os

__all__ = [
    "mirror_index",
    "mirror_indices",
    "median_plane",
    "plan_median_bands",
    "median_plane_banded",
    "border_mask",
    "dump_border_frame",
]


def mirror_index(p, n):
    """镜像填充（REFLECT_101）下填充位置 p 对应的源轴索引。

    效果：源 [0,1,2,3]（n=4）在 p=-1 处取 1、p=-2 处取 2、
    p=4 处取 2、p=5 处取 1，不复制边缘像素。n=1 时恒返回 0。
    """
    if n <= 1:
        return 0
    period = 2 * (n - 1)
    q = abs(p) % period
    return n - 1 - abs((n - 1) - q)


def mirror_indices(n, radius):
    """返回长度 n+2r 的填充索引序列：前 r + 源 n + 后 r 的源索引。"""
    return [mirror_index(p, n) for p in range(-radius, n + radius)]


def _median_from_hist(hist, target):
    """按累计计数定位直方图中排序第 target 位（0 基）的值。"""
    acc = 0
    for v, c in enumerate(hist):
        acc += c
        if acc > target:
            return v
    return 255  # 理论不可达：target < 直方图元素总数


def _huang_median_grid(grid, width, height, side):
    """对已镜像填充的网格跑 Huang 直方图中值，返回 width*height 字节。

    ``grid`` 为 (height+side-1) 行、每行 (width+side-1) 个 int 的
    填充网格；输出取每个 (side x side) 窗口的中值，恰好对应源图
    width*height 个像素。
    """
    r = side // 2
    target = side * side // 2
    out = bytearray(width * height)
    for y in range(height):
        hist = [0] * 256
        # 行左端：列窗口 [0, side) 的 side 个竖条，每竖条 side 像素
        for dy in range(side):
            row = grid[y + dy]
            for dx in range(side):
                hist[row[dx]] += 1
        med = _median_from_hist(hist, target)
        le = sum(hist[:med + 1])
        out[y * width] = med
        for x in range(1, width):
            old_col, new_col = x - 1, x + side - 1
            for dy in range(side):
                row = grid[y + dy]
                ov, nv = row[old_col], row[new_col]
                if ov == nv:
                    continue
                hist[ov] -= 1
                if ov <= med:
                    le -= 1
                hist[nv] += 1
                if nv <= med:
                    le += 1
            # 行走恢复 le > target：le 为 <= med 的元素个数
            while le <= target:
                med += 1
                le += hist[med]
            while le - hist[med] > target:
                le -= hist[med]
                med -= 1
            out[y * width + x] = med
    return bytes(out)


def median_plane(data, width, height, radius):
    """整帧单平面中值滤波，输入输出均为 width*height 字节的 bytes。

    边界镜像填充，窗口 (2r+1)²。radius <= 0 时原样返回。
    """
    if radius <= 0 or width <= 0 or height <= 0:
        return bytes(data)
    side = 2 * radius + 1
    col_idx = mirror_indices(width, radius)
    row_idx = mirror_indices(height, radius)
    grid = [
        [data[sy * width + sx] for sx in col_idx]
        for sy in row_idx
    ]
    return _huang_median_grid(grid, width, height, side)


def plan_median_bands(height, width, radius, budget_bytes,
                      bytes_per_pixel=1, safety=2):
    """按内存预算规划中值滤波分带，返回核心行区间 ``[(s, e), ...]``。

    每带驻留 核心行 + 2r 行上下文（中值窗口只需 r 行上下文，区别于
    均值分带的 2r），另有镜像行与水平填充的额外驻留，``safety``
    放大每行估算覆盖它们。预算再小也保证核心行高 >= 1。
    """
    if height <= 0:
        return []
    if radius <= 0:
        return [(0, height)]
    per_row = max(1, width * bytes_per_pixel * safety)
    afford = budget_bytes // per_row
    core = afford - 2 * radius
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


def median_plane_banded(data, width, height, radius, bands):
    """分带中值滤波，结果与 :func:`median_plane` 逐像素相等。

    每带按镜像规则只构造核心行上下各 r 行上下文的子网格，直接跑
    Huang 内核；各带输出行区间互不重叠，无需接缝合并。
    """
    if radius <= 0 or width <= 0 or height <= 0:
        return bytes(data)
    side = 2 * radius + 1
    col_idx = mirror_indices(width, radius)
    out = [None] * height
    for s, e in bands:
        # 输出行 [s, e) 所需源行窗口覆盖 [s-r, e+r)，按镜像取源索引
        grid_src_rows = [
            mirror_index(p, height)
            for p in range(s - radius, e + radius)
        ]
        grid = [
            [data[sy * width + sx] for sx in col_idx]
            for sy in grid_src_rows
        ]
        band_out = _huang_median_grid(grid, width, e - s, side)
        out[s:e] = [
            band_out[k * width:(k + 1) * width]
            for k in range(e - s)
        ]
    return b"".join(out)


def border_mask(width, height, radius):
    """构造边界复核掩码（bytes，0/1）：距边缘 <= radius 的像素为 1。

    掩码落在各平面自己的像素网格上；镜像填充影响的恰好是这个区域，
    抽帧复核时只看该区域即可核对边界行为。
    """
    mask = bytearray(width * height)
    for y in range(height):
        near_horizontal_edge = y <= radius or y >= height - 1 - radius
        row_base = y * width
        for x in range(width):
            if near_horizontal_edge or x <= radius or x >= width - 1 - radius:
                mask[row_base + x] = 1
    return bytes(mask)


def dump_border_frame(path, planes, dims, radius):
    """把一帧的边界区域（掩码外置 0）写成工作格式原始字节文件。

    ``planes`` 为 :func:`vdenoise.pixfmt.split_planes` 拆出的平面
    列表，``dims`` 为对应的 (宽, 高) 列表；packed RGB 的三个通道
    共用同尺寸掩码。可用 ffmpeg 按工作格式读回抽帧复核：
    ``ffmpeg -f rawvideo -pix_fmt <fmt> -s WxH -i frame.raw ...``
    """
    masked = []
    for plane, (pw, ph) in zip(planes, dims):
        mask = border_mask(pw, ph, radius)
        masked.append(bytes(p if m else 0 for p, m in zip(plane, mask)))
    with open(path, "wb") as f:
        f.write(b"".join(masked))
    return os.fspath(path)
