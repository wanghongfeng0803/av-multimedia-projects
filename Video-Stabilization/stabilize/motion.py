"""全局运动估计：相邻帧的平移与旋转分别估计、分别输出。

设计要点（与需求逐条对应）：

- **只在缩小的灰度帧上工作**：解码统一走 ffmpeg 出 gray 原始帧，
  Python 侧再做块平均下采样（宽 <= ``EST_WIDTH``），估计在小图上
  完成，平移结果按缩放比还原到原分辨率（浮点亚像素），旋转角度
  与尺度无关直接使用。内存与单帧小图成正比，长视频只保留相邻两帧。
- **平移与旋转分两套独立机制**：
  平移 = 粗-精两级块匹配（SAD 穷举 + 抛物线三点亚像素拟合）；
  旋转 = 去均值环形径向签名的循环互相关（FFT，见 :mod:`stabilize.fft_mod`），
  角度按环周长与 bin 数换算，再用相邻 bin 二次插值取亚 bin 精度。
- **交替精修一次**：先平移后旋转，平移补偿后再估旋转，旋转补偿后
  再估一次平移，抑制两者耦合（旋转会污染块匹配，平移会污染环签名）。
- **场景切换显式标记**：帧间全局 SAD 超阈值（相对平均亮度归一）
  判为切点，该帧对的位移不可信，``valid=False`` 且分量置 0，
  交由平滑阶段插值，绝不把切点误当巨幅抖动。

坐标约定（全模块统一）：位移 ``(dx, dy)`` 表示 **ref 帧内容对齐到
cur 帧所需的平移**，即 ref 像素 (x,y) 出现在 cur 的 (x+dx, y+dy)；
``angle`` 为同样语义下 ref 到 cur 的顺时针旋转角（度）。
"""

import math

from .fft_mod import circular_correlation, peak_lag

__all__ = [
    "EST_WIDTH",
    "RING_COUNT",
    "ANGLE_BINS",
    "SCENE_CUT_RATIO",
    "ROTATION_PEAK_RATIO",
    "MAX_REL_SHIFT",
    "downsample",
    "block_match",
    "parabolic_refine",
    "bilinear_sample",
    "ring_signature",
    "estimate_rotation",
    "estimate_pair",
    "is_scene_cut",
]

#: 估计用小图的目标宽度（像素）；过小丢细节，过大拖慢 Python 匹配
EST_WIDTH = 96
#: 环形签名的同心环数（半径从 0.15R 到 0.9R 等分）
RING_COUNT = 12
#: 每个环的角度采样 bin 数（循环互相关长度，取 2 的幂便于 FFT）
ANGLE_BINS = 128
#: 全局平均绝对差 / 255 超过该比例判场景切换
SCENE_CUT_RATIO = 0.35
#: 旋转互相关峰均比置信门槛
ROTATION_PEAK_RATIO = 1.2
#: 平移搜索半径相对小图宽度的比例（±15% 画幅）
MAX_REL_SHIFT = 0.15


# ------------------------------------------------------------ 基础工具

def downsample(plane, width, height, target_width=EST_WIDTH):
    """step×step 块平均下采样，返回 ``(bytes, 新宽, 新高, step)``。

    step 用 **上取整** 保证 ``width > target_width`` 时 step>=2，
    小视频也能真正缩小（整除会让 160→目标 96 时退化成 step=1）；
    ``target_width >= width`` 时 step=1 原样返回。
    """
    if not target_width or target_width >= width:
        return bytes(plane), width, height, 1
    step = (width + target_width - 1) // target_width
    nw, nh = width // step, height // step
    inv = step * step
    out = bytearray(nw * nh)
    for ny in range(nh):
        sy = ny * step
        base = ny * nw
        for nx in range(nw):
            sx = nx * step
            total = 0
            for dy in range(step):
                row = plane[(sy + dy) * width:]
                for dx in range(step):
                    total += row[sx + dx]
            out[base + nx] = (total + inv // 2) // inv
    return bytes(out), nw, nh, step


def _sad_overlap(ref, cur, w, h, dx, dy, stride=1):
    """cur 与平移 (dx,dy) 后的 ref 在重叠区的采样平均绝对差。

    与降噪模块同约定：ref 像素 (x,y) 对齐到 cur 的 (x+dx, y+dy)。
    无重叠返回 None。
    """
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    if x1 <= x0 or y1 <= y0:
        return None
    total, count = 0, 0
    for cy in range(y0, y1, stride):
        ry = cy - dy
        cbase, rbase = cy * w, ry * w
        for cx in range(x0, x1, stride):
            total += abs(cur[cbase + cx] - ref[rbase + cx - dx])
            count += 1
    return total / count if count else None


def block_match(ref, cur, w, h, max_shift, stride=1):
    """整像素穷举块匹配，返回 ``(dx, dy, 最优SAD, 次优SAD)``。

    在 ``[-max_shift, max_shift]²`` 内最小化重叠区 SAD。平局取位移
    幅度最小、再按 (dy, dx) 字典序最小，保证确定性。次优 SAD 用于
    置信度判断（尖锐最优 vs 平坦响应）。
    """
    best = None      # (sad, |dx|+|dy|, dy, dx)
    second = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            sad = _sad_overlap(ref, cur, w, h, dx, dy, stride)
            if sad is None:
                continue
            key = (sad, abs(dx) + abs(dy), dy, dx)
            if best is None or key < best:
                second = best[0] if best else None
                best = key
            elif second is None or sad < second:
                second = sad
    if best is None:
        return 0, 0, 0.0, 0.0
    return best[3], best[2], best[0], (second if second is not None else best[0])


def parabolic_refine(c0, c_m, c_p):
    """三点抛物线亚像素拟合，返回相对中心的偏移量 δ∈[-1, 1]。

    输入是 **代价**（SAD，越小越优，拟合的是谷底），故顶点公式
    δ = (c_m − c_p) / (2·(c_m + c_p − 2·c0))：右侧代价更低时
    c_m > c_p、δ 为正，顶点偏向 +1。退化（共线或顶点超出范围）
    时夹回 0。
    """
    denom = c_m + c_p - 2 * c0
    if denom == 0:
        return 0.0
    delta = (c_m - c_p) / (2.0 * denom)
    if not (-1.0 <= delta <= 1.0):
        return 0.0
    return delta


def bilinear_sample(plane, w, h, x, y):
    """双线性采样（坐标越界夹到边缘），用于平移补偿后的环签名提取。"""
    if x < 0:
        x = 0.0
    elif x > w - 1:
        x = float(w - 1)
    if y < 0:
        y = 0.0
    elif y > h - 1:
        y = float(h - 1)
    x0, y0 = int(x), int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    v00 = plane[y0 * w + x0]
    v10 = plane[y0 * w + x1]
    v01 = plane[y1 * w + x0]
    v11 = plane[y1 * w + x1]
    top = v00 * (1 - fx) + v10 * fx
    bot = v01 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


# ------------------------------------------------------------ 旋转

def ring_signature(plane, w, h, shift=(0.0, 0.0),
                   rings=RING_COUNT, bins=ANGLE_BINS):
    """提取去均值的同心环角度签名，返回 ``rings × bins`` 的二维 list。

    每个环在半径 r 处按角度等分 ``bins`` 个采样点（双线性取值），
    再减去该环均值消除整体亮度与平移后残留的径向偏置；旋转在签名上
    表现为循环平移，平移只造成轻微形变（已先用块匹配补偿）。

    ``shift=(dx,dy)`` 给出先验平移：采样点先减去它再取 ref 的值，
    即把 ref 平移到 cur 坐标系后提取签名。
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_max = 0.9 * min(cx, cy)
    r_min = 0.15 * min(cx, cy)
    dx, dy = shift
    sig = []
    for ri in range(rings):
        r = r_min + (r_max - r_min) * ri / max(1, rings - 1)
        row = [0.0] * bins
        for k in range(bins):
            theta = 2.0 * math.pi * k / bins
            # 采样 ref 在 cur 坐标系对应位置：cur 点 = ref 点 + shift
            x = cx + r * math.cos(theta) - dx
            y = cy + r * math.sin(theta) - dy
            row[k] = bilinear_sample(plane, w, h, x, y)
        mean = sum(row) / bins
        sig.append([v - mean for v in row])
    return sig


def _correlate_signatures(sig_ref, sig_cur, bins):
    """所有环的循环互相关逐 bin 求和，强化共用角度峰、压掉随机噪声。"""
    total = [0.0] * bins
    for row_r, row_c in zip(sig_ref, sig_cur):
        corr = circular_correlation(row_r, row_c)
        for k in range(bins):
            total[k] += corr[k]
    return total


def estimate_rotation(ref, cur, w, h, shift=(0.0, 0.0),
                      rings=RING_COUNT, bins=ANGLE_BINS):
    """环形签名循环互相关估计旋转角，返回 ``(角度度, 峰均比)``。

    角度方向：cur 相对 ref 顺时针转 α 时，cur 环上同一图案出现在
    更大的角度 bin，互相关主峰 lag 为正，故角度 = lag·(360/bins)。
    亚 bin 精度由主峰与相邻 bin 的抛物线插值给出。
    """
    sig_r = ring_signature(ref, w, h, shift, rings, bins)
    # cur 自身不做平移补偿：shift 描述的是 ref→cur，签名直接在
    # cur 原坐标系提取，两者的循环相位差即旋转角。
    sig_c = ring_signature(cur, w, h, (0.0, 0.0), rings, bins)
    corr = _correlate_signatures(sig_r, sig_c, bins)
    # 真实抖动角度有限：限制 ±45°，避免半圈折返误判
    max_lag = max(1, int(bins * 45.0 / 360.0))
    lag, peak, ratio = peak_lag(corr, max_lag)
    # 抛物线亚 bin（循环下标取模）
    c_m = corr[(lag - 1) % bins]
    c_p = corr[(lag + 1) % bins]
    sub = parabolic_refine(peak, c_m, c_p)
    angle = (lag + sub) * 360.0 / bins
    return angle, ratio


# ------------------------------------------------------------ 帧对联合估计

def is_scene_cut(ref, cur, w, h, stride=4):
    """整帧（粗抽样）平均绝对差归一化判场景切换。"""
    total, count = 0, 0
    for y in range(0, h, stride):
        base = y * w
        for x in range(0, w, stride):
            total += abs(cur[base + x] - ref[base + x])
            count += 1
    return count > 0 and (total / count) / 255.0 >= SCENE_CUT_RATIO


def estimate_pair(ref, cur, w, h):
    """估计相邻帧 ref→cur 的全局运动，返回结果 dict。

    返回字段（稳定输出契约的单帧条目，见模块包文档）：
    ``dx/dy``（小图坐标像素，浮点）、``angle``（度，顺时针）、
    ``confidence``（0~1，平移与旋转两路置信的较小者）、``valid``。

    流程：粗平移 → 亚像素精修 → 平移补偿后估旋转 → 旋转补偿后
    再精修一次平移。任一路置信不足或场景切换时 ``valid=False``，
    分量保留估计值但平滑阶段会按无效帧处理。
    """
    max_shift = max(1, int(w * MAX_REL_SHIFT))
    dx, dy, sad_best, _ = block_match(ref, cur, w, h, max_shift)

    # 亚像素：在整像素最优点邻域取三点 SAD 做抛物线拟合
    sx = sy = 0.0
    if w > 2 and h > 2:
        c0 = _sad_overlap(ref, cur, w, h, dx, dy) or 0.0
        cxm = _sad_overlap(ref, cur, w, h, dx - 1, dy)
        cxp = _sad_overlap(ref, cur, w, h, dx + 1, dy)
        cym = _sad_overlap(ref, cur, w, h, dx, dy - 1)
        cyp = _sad_overlap(ref, cur, w, h, dx, dy + 1)
        if cxm is not None and cxp is not None:
            sx = parabolic_refine(c0, cxm, cxp)
        if cym is not None and cyp is not None:
            sy = parabolic_refine(c0, cym, cyp)
    fx, fy = dx + sx, dy + sy

    # 平移置信：以最优 SAD 的绝对残差水平为准（亚像素平移下
    # 次优/最优比值天然平坦，比值法会误杀；残差小才是真对齐）。
    # 场景切换阈值 0.35·255 作为「完全对不上」的锚点线性映射。
    sad_anchor = SCENE_CUT_RATIO * 255.0
    trans_conf = max(0.0, min(1.0, 1.0 - sad_best / sad_anchor))

    # 旋转（在平移补偿后的坐标系）
    angle, rot_ratio = estimate_rotation(ref, cur, w, h, shift=(fx, fy))
    rot_conf = min(1.0, max(0.0, (rot_ratio - 1.0)
                            / (ROTATION_PEAK_RATIO - 1.0)))

    # 旋转补偿后再精修平移：仅当角度超过死区（0.5°）才值得付出
    # 一次全帧旋转；小角度下旋转对平移的耦合可以忽略，直接采信首估。
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    if abs(angle) >= 0.5:
        # 反向旋转 cur 到 ref 朝向，再与 ref 块匹配（邻域 ±3 小图像素）
        unrot = _rotate_plane(cur, w, h, -angle)
        dx2, dy2, _, _ = block_match(ref, unrot, w, h, min(3, max_shift))
        # 第二次位移是 ref→去旋转 cur 的平移；旋转换基的原点平移项
        # （中心 (cx,cy)）：dx2 = dx + (1-cos)·cx - sin·cy，
        #                  dy2 = dy + sin·cx + (1-cos)·cy
        cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
        fx2 = dx2 + (1 - cos_a) * cx0 - sin_a * cy0
        fy2 = dy2 + sin_a * cx0 + (1 - cos_a) * cy0
        # 与首估加权融合：置信高的精修结果权重更大，避免二次估计噪声翻转
        weight = 0.5 if trans_conf >= 0.5 else 0.25
        fx = (1 - weight) * fx + weight * fx2
        fy = (1 - weight) * fy + weight * fy2

    valid = (trans_conf >= 0.5 and rot_conf >= 0.5)
    return {
        "dx": fx,
        "dy": fy,
        "angle": angle,
        "confidence": min(trans_conf, rot_conf),
        "valid": valid,
    }


def _rotate_plane(plane, w, h, angle_deg):
    """绕中心旋转整帧（正向采样，越界夹边），仅用于精修阶段的小图。"""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    out = bytearray(w * h)
    for y in range(h):
        base = y * w
        dy0 = y - cy
        for x in range(w):
            dx0 = x - cx
            sx = cx + cos_a * dx0 - sin_a * dy0
            sy = cy + sin_a * dx0 + cos_a * dy0
            out[base + x] = int(round(bilinear_sample(plane, w, h, sx, sy)))
    return bytes(out)
