"""二值形态学与连通域分析：检测后处理的纯 Python 实现。

检测链路只产出逐像素二值掩码，掩码里的椒盐噪声、断裂孔洞与相邻目标
粘连都在本模块处理，分三层、每层职责单一：

1. :func:`erode` / :func:`dilate`——方形结构元（2r+1）² 的腐蚀与膨胀，
   用行游程表做 O(WH·r) 实现而非 O(WH·r²) 暴力窗口；
2. :func:`open_mask` / :func:`close_mask`——腐蚀+膨胀的标准组合，
   开运算去孤立小点，闭运算填洞接裂；
3. :func:`connected_components`——两遍式并查集连通域（4 邻接），
   返回标签平面与逐区域外接框/面积，区域结构即后续背景建模复用的
   最小信息单元（:class:`Region`）。

坐标约定：掩码为紧凑 ``bytearray``/``bytes``，长度 ``width*height``，
非零即前景；标签平面 0 为背景，区域标签从 1 起连续编号。
"""

__all__ = [
    "Region",
    "erode",
    "dilate",
    "open_mask",
    "close_mask",
    "connected_components",
]


class Region:
    """一个连通域的几何信息（不可变语义，供检测/事件/背景建模复用）。

    属性
    ----
    label:
        从 1 起的连通域标签（0 保留给背景）。
    bbox:
        ``(x, y, w, h)`` 外接矩形，左上坐标 + 宽高。
    area:
        前景像素数（不是外接框面积）。
    centroid:
        ``(cx, cy)`` 质心，按前景像素一阶矩计算，浮点。
    """

    __slots__ = ("label", "bbox", "area", "centroid")

    def __init__(self, label, bbox, area, centroid):
        self.label = label
        self.bbox = tuple(bbox)
        self.area = area
        self.centroid = (float(centroid[0]), float(centroid[1]))

    def to_dict(self):
        """转为事件 JSON 使用的扁平 dict。"""
        x, y, w, h = self.bbox
        return {
            "label": self.label,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": self.area,
            "cx": round(self.centroid[0], 3),
            "cy": round(self.centroid[1], 3),
        }

    def __repr__(self):
        return (f"Region(label={self.label}, bbox={self.bbox}, "
                f"area={self.area})")


# --------------------------------------------------------------- 行游程表
def _run_lengths(mask, width, height):
    """把二值平面转成逐行 ``(起列, 止列排他)`` 前景游程列表。

    游程表把逐像素窗口运算变成区间运算：方形结构元的行方向影响只与
    同行游程有关，列方向影响只需跨行查表。
    """
    rows = [None] * height
    for y in range(height):
        base = y * width
        runs = []
        x = 0
        while x < width:
            if mask[base + x]:
                x0 = x
                x += 1
                while x < width and mask[base + x]:
                    x += 1
                runs.append((x0, x))
            else:
                x += 1
        rows[y] = runs
    return rows


def _runs_to_mask(rows, width, height):
    """游程表还原为紧凑二值 bytearray。"""
    out = bytearray(width * height)
    for y, runs in enumerate(rows):
        base = y * width
        for x0, x1 in runs:
            for x in range(x0, x1):
                out[base + x] = 1
    return out


def _intersect_runs(a, b):
    """两个升序、互不重叠的游程列表求交集（双指针线性合并）。"""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def erode(mask, width, height, radius):
    """半径 ``radius`` 的方形结构元腐蚀（输出前景 ⊆ 输入前景）。

    帧外像素一律按背景处理（OpenCV 常量零边框语义），因此贴边前景
    也会被正常剥掉，两遍实现与之严格等价：

    1. 行方向：每个游程两端各内缩 ``radius``；
    2. 列方向：输出行 y 的游程 = 窗口内各行（y-r..y+r）行腐蚀结果
       的交集——某列在窗口内任一行缺失即被剥掉。
    """
    if radius <= 0:
        return bytearray(mask)
    in_rows = _run_lengths(mask, width, height)
    # 第一遍：行方向腐蚀
    h_rows = []
    for runs in in_rows:
        shrunk = []
        for x0, x1 in runs:
            nx0, nx1 = x0 + radius, x1 - radius
            if nx0 < nx1:
                shrunk.append((nx0, nx1))
        h_rows.append(shrunk)
    # 第二遍：列方向腐蚀（跨行游程交集）
    out_rows = [[] for _ in range(height)]
    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        common = h_rows[y]
        for ry in range(y0, y1):
            if ry == y or not common:
                continue
            common = _intersect_runs(common, h_rows[ry])
        out_rows[y] = common
    return _runs_to_mask(out_rows, width, height)


def dilate(mask, width, height, radius):
    """半径 ``radius`` 的方形结构元膨胀。

    实现走「前景游程向外扩 r，再做跨行区间并集」：每个输入游程在
    ±r 行内贡献 [x0-r, x1+r) 区间，同行区间合并后即为该行膨胀结果。
    """
    if radius <= 0:
        return bytearray(mask)
    in_rows = _run_lengths(mask, width, height)
    out_rows = [[] for _ in range(height)]
    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        intervals = []
        for ry in range(y0, y1):
            for x0, x1 in in_rows[ry]:
                intervals.append((max(0, x0 - radius),
                                  min(width, x1 + radius)))
        if not intervals:
            continue
        intervals.sort()
        merged = [intervals[0]]
        for x0, x1 in intervals[1:]:
            lx0, lx1 = merged[-1]
            if x0 <= lx1:
                if x1 > lx1:
                    merged[-1] = (lx0, x1)
            else:
                merged.append((x0, x1))
        out_rows[y] = merged
    return _runs_to_mask(out_rows, width, height)


def open_mask(mask, width, height, radius):
    """开运算 = 先腐蚀后膨胀，去除小于结构元的孤立噪点。"""
    if radius <= 0:
        return bytearray(mask)
    eroded = erode(mask, width, height, radius)
    return dilate(eroded, width, height, radius)


def close_mask(mask, width, height, radius):
    """闭运算 = 先膨胀后腐蚀，填补小空洞、连接邻近断裂。"""
    if radius <= 0:
        return bytearray(mask)
    dilated = dilate(mask, width, height, radius)
    return erode(dilated, width, height, radius)


# --------------------------------------------------------------- 连通域
class _DSU:
    """最小并查集：路径压缩 + 按秩合并。"""

    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent = []
        self.rank = []

    def new(self):
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def connected_components(mask, width, height):
    """4 邻接两遍式连通域分析。

    返回 ``(labels, regions)``：``labels`` 为长度 WH 的 ``bytearray``
    （区域数 ≤255 时）或 ``array('I')`` 标签平面，0 为背景；
    ``regions`` 为按标签升序排列的 :class:`Region` 列表。

    第一遍只扫上、左两个邻居做等价合并（光栅序下其余邻居尚未出现），
    第二遍压实标签并一次性累计面积/外接框/质心矩。
    """
    dsu = _DSU()
    # 第一遍：临时标签平面
    prelim = [0] * (width * height)
    for y in range(height):
        base = y * width
        for x in range(width):
            if not mask[base + x]:
                continue
            up = prelim[base - width + x] if y > 0 else 0
            left = prelim[base + x - 1] if x > 0 else 0
            if up == 0 and left == 0:
                prelim[base + x] = dsu.new() + 1
            elif up != 0 and left == 0:
                prelim[base + x] = up
            elif up == 0 and left != 0:
                prelim[base + x] = left
            else:
                tag = min(up, left)
                prelim[base + x] = tag
                dsu.union(up - 1, left - 1)

    # 根标签 → 压实编号（1 起）
    root_to_final = {}

    def final_of(raw_tag):
        root = dsu.find(raw_tag - 1)
        tag = root_to_final.get(root)
        if tag is None:
            tag = len(root_to_final) + 1
            root_to_final[root] = tag
        return tag

    n = width * height
    try:
        labels = bytearray(n)
        compact_fit = True
    except (MemoryError, OverflowError):  # pragma: no cover - 极小概率
        labels = None
        compact_fit = False
    min_x = []
    min_y = []
    max_x = []
    max_y = []
    area = []
    sum_x = []
    sum_y = []

    def ensure(tag):
        while len(area) < tag:
            min_x.append(width)
            min_y.append(height)
            max_x.append(-1)
            max_y.append(-1)
            area.append(0)
            sum_x.append(0)
            sum_y.append(0)

    for i, raw in enumerate(prelim):
        if raw == 0:
            continue
        tag = final_of(raw)
        ensure(tag)
        idx = tag - 1
        x = i % width
        y = i // width
        if compact_fit and tag <= 255:
            labels[i] = tag
        elif compact_fit:
            # 超出 byte 范围：升级为 32 位无符号数组
            from array import array
            labels = array("I", labels)
            compact_fit = False
            labels[i] = tag
        else:
            labels[i] = tag
        if x < min_x[idx]:
            min_x[idx] = x
        if x > max_x[idx]:
            max_x[idx] = x
        if y < min_y[idx]:
            min_y[idx] = y
        if y > max_y[idx]:
            max_y[idx] = y
        area[idx] += 1
        sum_x[idx] += x
        sum_y[idx] += y

    regions = []
    for tag in range(1, len(area) + 1):
        idx = tag - 1
        if area[idx] == 0:
            continue
        cx = sum_x[idx] / area[idx]
        cy = sum_y[idx] / area[idx]
        regions.append(Region(
            label=tag,
            bbox=(min_x[idx], min_y[idx],
                  max_x[idx] - min_x[idx] + 1,
                  max_y[idx] - min_y[idx] + 1),
            area=area[idx],
            centroid=(cx, cy),
        ))
    return labels, regions
