"""逐帧模板匹配：SAD / SSD / ZNCC 三种分数，纯标准库字节平面实现。

平面约定与兄弟模块一致：单字节 ``bytes``/``bytearray``，行主序、
每行 ``width`` 字节，像素值 0..255。

三种方法与分数方向（阈值语义见 :func:`higher_is_better`）：

- ``sad``：绝对差之和，越小越好，窗口归一化后阈值即「平均每像素灰度
  差」，默认 30；
- ``ssd``：平方差之和，越小越好，窗口归一化后阈值即「平均每像素平方
  差」，默认 900（≈ 灰度差 30 的平方量级）；
- ``zncc``：零均值归一化互相关，越大越好，取值理论区间 [-1, 1]，
  对线性光照变化鲁棒，默认阈值 0.7。模板或候选窗零方差（常数块）时
  相关性无定义，按 -1 处理（必然低于阈值），避免除零并判为不匹配。

复杂度控制：

- SAD/SSD 为逐窗绝对差/平方差穷举，窗内用行偏移切片式循环，
  积分图工具（:func:`integral_image` / :func:`rect_sum`）随包提供，
  供后续盒式加速与其它算法复用；
- ZNCC 无法用单张积分图表达，走粗-精两级：先在块平均下采样平面上
  穷举（面积极小），取前若干粗候选，再回原分辨率在 ±2 像素邻域精搜。
"""

__all__ = [
    "METHOD_SAD",
    "METHOD_SSD",
    "METHOD_ZNCC",
    "TEMPLATE_METHODS",
    "DEFAULT_THRESHOLDS",
    "REFINE_RADIUS",
    "higher_is_better",
    "score_accepted",
    "extract_patch",
    "block_downsample",
    "integral_image",
    "rect_sum",
    "window_sums",
    "zncc_at",
    "best_match",
    "best_match_zncc_coarse_fine",
    "best_match_coarse",
    "match_template",
]

METHOD_SAD = "sad"
METHOD_SSD = "ssd"
METHOD_ZNCC = "zncc"

#: 支持的模板匹配方法
TEMPLATE_METHODS = (METHOD_SAD, METHOD_SSD, METHOD_ZNCC)

#: 各方法的默认接受阈值（SAD/SSD 为窗口归一化后的每像素均值）
DEFAULT_THRESHOLDS = {
    METHOD_SAD: 30.0,
    METHOD_SSD: 900.0,
    METHOD_ZNCC: 0.7,
}

#: ZNCC 粗-精搜索的下采样因子候选与精搜半径、粗候选保留数
_DOWNSAMPLE_FACTORS = (4, 2)
REFINE_RADIUS = 2
_COARSE_KEEP = 5


def higher_is_better(method):
    """该方法的分数是否越大越好（ZNCC 为 True，SAD/SSD 为 False）。"""
    return method == METHOD_ZNCC


def score_accepted(method, score, threshold):
    """按方法的分数方向判断 ``score`` 是否达到 ``threshold``。"""
    if higher_is_better(method):
        return score >= threshold
    return score <= threshold


def extract_patch(plane, width, x, y, tw, th):
    """从平面抠出 (x, y) 起、tw×th 的模板，返回紧凑 bytes。

    调用方须保证窗口完全落在平面内（越界由上层搜索区夹取保证）。
    """
    rows = []
    for r in range(y, y + th):
        base = r * width + x
        rows.append(plane[base:base + tw])
    return b"".join(rows)


def block_downsample(plane, width, height, factor):
    """块平均下采样：每个 factor×factor 块取整数均值。

    返回 ``(小平面 bytes, 小宽, 小高)``；边缘不足整块的部分截断丢弃，
    因此小图尺寸为 floor(宽/factor) × floor(高/factor)。factor=1 时
    原样返回（不拷贝）。
    """
    if factor <= 1:
        return plane, width, height
    sw, sh = width // factor, height // factor
    out = bytearray(sw * sh)
    f2 = factor * factor
    for sy in range(sh):
        src_y = sy * factor
        for sx in range(sw):
            src_x = sx * factor
            total = 0
            for dy in range(factor):
                row = src_y + dy
                base = row * width + src_x
                for dx in range(factor):
                    total += plane[base + dx]
            out[sy * sw + sx] = total // f2
    return bytes(out), sw, sh


def integral_image(plane, width, height, square=False):
    """二维前缀和（积分图），尺寸 (width+1)×(height+1)，行主序。

    ``square=True`` 时对像素平方求和。窗 (x,y,w,h) 的和用
    :func:`rect_sum` 以 4 次查表得到，与窗口大小无关。
    """
    stride = width + 1
    integ = [0] * (stride * (height + 1))
    if square:
        for r in range(height):
            src_base = r * width
            dst_base = (r + 1) * stride
            row_sum = 0
            for c in range(width):
                v = plane[src_base + c]
                row_sum += v * v
                integ[dst_base + c + 1] = (
                    integ[dst_base - stride + c + 1] + row_sum
                )
    else:
        for r in range(height):
            src_base = r * width
            dst_base = (r + 1) * stride
            row_sum = 0
            for c in range(width):
                row_sum += plane[src_base + c]
                integ[dst_base + c + 1] = (
                    integ[dst_base - stride + c + 1] + row_sum
                )
    return integ


def rect_sum(integ, stride, x, y, w, h):
    """积分图 4 点查表求矩形窗之和。"""
    return (
        integ[(y + h) * stride + (x + w)]
        - integ[y * stride + (x + w)]
        - integ[(y + h) * stride + x]
        + integ[y * stride + x]
    )


def window_sums(integ, stride, x, y, w, h):
    """:func:`rect_sum` 的语义别名，供调用方按「窗和」含义阅读。"""
    return rect_sum(integ, stride, x, y, w, h)


def _sad_window(frame, width, x, y, template, tw, th):
    """单个候选窗的 SAD 原始和（未归一化）。"""
    total = 0
    for r in range(th):
        fb = (y + r) * width + x
        tb = r * tw
        for c in range(tw):
            d = frame[fb + c] - template[tb + c]
            total += d if d >= 0 else -d
    return total


def _ssd_window(frame, width, x, y, template, tw, th):
    """单个候选窗的 SSD 原始和（未归一化）。"""
    total = 0
    for r in range(th):
        fb = (y + r) * width + x
        tb = r * tw
        for c in range(tw):
            d = frame[fb + c] - template[tb + c]
            total += d * d
    return total


def zncc_at(frame, width, x, y, template, tw, th,
            t_sum, t_sum2, n=None):
    """候选窗 (x,y) 与模板的 ZNCC，零方差窗返回 -1。

    模板的 ``t_sum``/``t_sum2`` 由调用方预算一次复用；帧窗的两个和
    在本函数内累加。公式：

        N·ΣIT − ΣI·ΣT
        ──────────────────────────────────────────
        sqrt(N·ΣI²−(ΣI)²) · sqrt(N·ΣT²−(ΣT)²)
    """
    if n is None:
        n = tw * th
    i_sum = 0
    i_sum2 = 0
    cross = 0
    for r in range(th):
        fb = (y + r) * width + x
        tb = r * tw
        for c in range(tw):
            iv = frame[fb + c]
            tv = template[tb + c]
            i_sum += iv
            i_sum2 += iv * iv
            cross += iv * tv
    var_i = n * i_sum2 - i_sum * i_sum
    var_t = n * t_sum2 - t_sum * t_sum
    if var_i <= 0 or var_t <= 0:
        return -1.0
    numerator = n * cross - i_sum * t_sum
    # 浮点舍入可能使 |值| 略微越界，夹回 [-1, 1]
    val = numerator / ((var_i * var_t) ** 0.5)
    if val > 1.0:
        return 1.0
    if val < -1.0:
        return -1.0
    return val


def best_match(frame, width, height, template, tw, th,
               method=METHOD_SAD, search=None):
    """在搜索区内穷举最佳候选，返回 ``(x, y, 归一化分数)``。

    参数
    ----
    frame:
        当前帧亮度平面。
    template:
        模板字节（tw×th 紧凑排列）。
    search:
        ``(x0, y0, x1, y1)`` 候选左上角的闭区间，缺省为全帧可容纳范围，
        自动夹到合法边界。

    归一化分数：SAD/SSD 除以像素数 n，阈值按「每像素」解释；
    ZNCC 本身即归一化值。同分按先上后下、先左后右确定（穷举序天然
    保证，首个最优不被后续同分替换）。ZNCC 请改用
    :func:`best_match_zncc_coarse_fine` 走粗-精两级控成本。
    """
    if method not in (METHOD_SAD, METHOD_SSD):
        raise ValueError(
            f"best_match 仅支持 SAD/SSD 穷举，ZNCC 请用粗-精搜索，"
            f"收到 {method!r}"
        )
    if tw <= 0 or th <= 0 or tw > width or th > height:
        raise ValueError("模板尺寸非法或大于帧")
    x0, y0, x1, y1 = _clip_search(search, width, height, tw, th)
    n = tw * th

    if method == METHOD_SAD:
        score_fn = _sad_window
    else:
        score_fn = _ssd_window

    best = None
    best_score = None
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            raw = score_fn(frame, width, x, y, template, tw, th)
            s = raw / n
            if best_score is None or s < best_score:
                best_score = s
                best = (x, y, s)
    return best


def best_match_zncc_coarse_fine(frame, width, height, template, tw, th,
                                search=None, factors=_DOWNSAMPLE_FACTORS,
                                keep=_COARSE_KEEP, refine=REFINE_RADIUS):
    """ZNCC 粗-精两级搜索，返回 ``(x, y, 分数, 实际搜索区)``。

    1. 选首个使下采样模板仍 >= 3×3 的因子做块平均下采样，在小图上
       穷举，保留 ``keep`` 个粗候选；
    2. 每个粗候选映射回原分辨率坐标，在 ±``refine`` 邻域精搜取最优。

    若没有可用因子（模板太小），退化为全搜索区单级穷举。
    搜索区坐标为候选左上角闭区间，返回值第 4 项为实际使用的
    ``(x0, y0, x1, y1)``，供轨迹记录。
    """
    x0, y0, x1, y1 = _clip_search(search, width, height, tw, th)
    factor = 0
    for f in factors:
        if tw // f >= 3 and th // f >= 3:
            factor = f
            break
    if factor == 0:
        x, y, s = _best_zncc(frame, width, height, template, tw, th,
                             x0, y0, x1, y1)
        return x, y, s, (x0, y0, x1, y1)

    # 下采样搜索区：候选窗左上角在小图上的合法范围
    sx0, sy0 = x0 // factor, y0 // factor
    sx1 = min(x1 // factor, width // factor - tw // factor)
    sy1 = min(y1 // factor, height // factor - th // factor)
    small_frame, sw, sh = block_downsample(frame, width, height, factor)
    small_tpl, stw, sth = block_downsample(template, tw, th, factor)
    if stw < 1 or sth < 1 or sx1 < sx0 or sy1 < sy0:
        x, y, s = _best_zncc(frame, width, height, template, tw, th,
                             x0, y0, x1, y1)
        return x, y, s, (x0, y0, x1, y1)

    n = stw * sth
    t_sum = sum(small_tpl)
    t_sum2 = sum(v * v for v in small_tpl)
    coarse = []
    for sy in range(sy0, sy1 + 1):
        for sx in range(sx0, sx1 + 1):
            s = zncc_at(small_frame, sw, sx, sy, small_tpl, stw, sth,
                        t_sum, t_sum2, n)
            coarse.append((s, sx, sy))
    coarse.sort(reverse=True)
    candidates = coarse[:keep]

    # 精搜：粗坐标 × factor 还原，在原分辨率邻域穷举
    best = None
    best_score = -2.0
    n_full = tw * th
    tf_sum = sum(template)
    tf_sum2 = sum(v * v for v in template)
    for _, csx, csy in candidates:
        cx, cy = csx * factor, csy * factor
        rx0 = max(x0, cx - refine)
        ry0 = max(y0, cy - refine)
        rx1 = min(x1, cx + refine)
        ry1 = min(y1, cy + refine)
        for y in range(ry0, ry1 + 1):
            for x in range(rx0, rx1 + 1):
                s = zncc_at(frame, width, x, y, template, tw, th,
                            tf_sum, tf_sum2, n_full)
                if s > best_score:
                    best_score = s
                    best = (x, y)
    return best[0], best[1], best_score, (x0, y0, x1, y1)


def best_match_coarse(frame, width, height, template, tw, th, search=None):
    """:func:`best_match_zncc_coarse_fine` 的对外简写（ZNCC 粗精搜索）。"""
    x, y, s, _ = best_match_zncc_coarse_fine(
        frame, width, height, template, tw, th, search=search
    )
    return x, y, s


def _best_zncc(frame, width, height, template, tw, th, x0, y0, x1, y1):
    """ZNCC 全搜索区穷举（小模板/小搜索区路径）。"""
    n = tw * th
    t_sum = sum(template)
    t_sum2 = sum(v * v for v in template)
    best = None
    best_score = -2.0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            s = zncc_at(frame, width, x, y, template, tw, th,
                        t_sum, t_sum2, n)
            if s > best_score:
                best_score = s
                best = (x, y, s)
    return best


def _clip_search(search, width, height, tw, th):
    """把搜索区夹到候选左上角的合法闭区间 [0, W-tw] × [0, H-th]。"""
    max_x = width - tw
    max_y = height - th
    if search is None:
        return 0, 0, max_x, max_y
    x0, y0, x1, y1 = search
    x0 = max(0, min(x0, max_x))
    y0 = max(0, min(y0, max_y))
    x1 = max(0, min(x1, max_x))
    y1 = max(0, min(y1, max_y))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def match_template(frame, width, height, template, tw, th,
                   method=METHOD_SAD, search=None):
    """统一入口：按 ``method`` 分派匹配算法。

    SAD/SSD 走全搜索区穷举（归一化分数）；ZNCC 走粗-精两级搜索。
    返回 ``(x, y, 分数)``。
    """
    if method not in TEMPLATE_METHODS:
        raise ValueError(f"未知匹配方法 {method!r}，可选：{', '.join(TEMPLATE_METHODS)}")
    if method == METHOD_ZNCC:
        return best_match_coarse(frame, width, height,
                                 template, tw, th, search)
    return best_match(frame, width, height, template, tw, th,
                      method=method, search=search)
