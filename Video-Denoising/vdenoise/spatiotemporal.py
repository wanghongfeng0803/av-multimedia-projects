"""空时联合降噪：时间窗软权重融合 + 空间均值联合，掩码复用前级估计。

在 :mod:`vdenoise.temporal` 的「全局对齐 + 运动掩码」基础上做三件事：

1. **硬掩码升级为按运动掩码分配的软权重**（:func:`motion_weight_map`）：
   对齐残差 ``d`` 不超过 ``motion_threshold`` 时满权（255），在
   ``[threshold, 2·threshold]`` 之间线性降到 0，超过 2 倍阈值拒绝。
   当前帧恒为满权；某像素全部参考权重为 0 时回退当前帧原值，绝不
   平均错位内容（与时域模式同一条防拖影底线）。
2. **掩码复用前级估计，边缘平滑避免块状**：权重图只在亮度平面、对
   每个参考帧构造一次（残差本身就是前级位移估计的对齐副产物，不做
   第二次估计），再用小半径盒式均值平滑权重图（:func:`smooth_weight_map`），
   让运动边界两侧的权重渐变而非硬切；色度平面不另估掩码，按其相对
   亮度的采样比做块平均重采样（:func:`resample_weight_map`，
   4:2:0 即 2×2 平均，天然抗块状）。
3. **空时联合**：分带内先做时间窗加权融合（:func:`fuse_band`），
   紧接空间盒式均值（复用 :mod:`vdenoise.bands` 的 2r 上下文 + 接缝
   合并实现）。运动区参考权重被压到 0、时域退化为近恒等时，空间级
   仍在该带内压制噪声，两级互补。

内存固定不随片长涨：帧缓存是容量 K 的有界 deque，权重图每参考帧
一张、共 K-1 张且只存亮度平面，长视频下旧帧/旧图随 deque 淘汰，
驻留量与总帧数无关。预算口径与 :mod:`vdenoise.temporal` 一致——
``(缓存与权重图 + 带工作区原始字节) × safety``，超预算在任何
ffmpeg 子进程启动之前抛 :class:`MemoryError`。

全部运算为定点整数（权重以 0..255 定点），无浮点、无随机源，
同输入同参数输出逐字节可复现。纯标准库实现，平面均为单字节 bytes，
行主序、宽 ``width``。
"""

from array import array
from collections import deque

from .bands import box_blur_plane_banded, plan_bands

__all__ = [
    "MAX_CACHE_FRAMES",
    "DEFAULT_MOTION_THRESHOLD",
    "DEFAULT_SPATIAL_RADIUS",
    "DEFAULT_MASK_RADIUS",
    "WEIGHT_SCALE",
    "SAFETY",
    "spatiotemporal_memory_bytes",
    "derive_cache_frames",
    "derive_band_height",
    "plan_spatiotemporal",
    "motion_weight_map",
    "smooth_weight_map",
    "resample_weight_map",
    "fuse_band",
    "process_plane",
    "FrameCache",
    "planar_mad",
    "temporal_mad",
]

#: 缓存帧数的硬上限（与时域模式一致，防止半径吃光内存）
MAX_CACHE_FRAMES = 15
#: 对齐残差满权/开始降权的分界灰度差（与时域模式默认阈值一致）
DEFAULT_MOTION_THRESHOLD = 24
#: 空间级盒式均值的默认半径（3×3，轻空间降噪，避免画面糊化）
DEFAULT_SPATIAL_RADIUS = 1
#: 权重图边缘平滑的默认半径（3×3，消除逐像素硬切块状）
DEFAULT_MASK_RADIUS = 1
#: 权重定点刻度：满权 = 255
WEIGHT_SCALE = 255
#: 内存估算安全系数，覆盖 Python bytes/list 等额外驻留，与前级模式一致
SAFETY = 2


# ---------------------------------------------------------------- 内存互推

def _work_raw(plane_row_bytes, plane_heights, band_rows, spatial_radius):
    """带工作区原始字节：各平面实际核心行 + 空间级 4r 上下文 × 行字节。

    时间融合在带内逐行进行（紧凑定长数组，不计入 bytes 口径，由
    safety 覆盖）；空间级盒式均值需要上下各 2r 行上下文（见
    :mod:`vdenoise.bands` 的 2r 结论），与均值模式同口径。
    """
    ctx = 4 * spatial_radius
    return sum(
        min(band_rows + ctx, ph + ctx) * row
        for row, ph in zip(plane_row_bytes, plane_heights)
    )


def _fixed_raw(plane_sizes, luma_size, cache_frames):
    """与片长无关的固定驻留：K 帧缓存 + (K-1) 张亮度权重图（原始字节）。"""
    frame_bytes = sum(plane_sizes)
    weight_bytes = max(0, cache_frames - 1) * luma_size
    return frame_bytes * cache_frames + weight_bytes


def spatiotemporal_memory_bytes(plane_sizes, luma_size, cache_frames,
                                band_rows, plane_row_bytes, plane_heights,
                                spatial_radius, safety=SAFETY):
    """空时联合处理的峰值内存估算（字节）。

    内存分两部分：

    - 固定项：``cache_frames`` 帧全帧缓存 + ``cache_frames-1`` 张只存
      亮度平面的权重图，与总帧数、片长都无关；
    - 带工作区：最大带 ``band_rows`` 核心行加空间级 4r 上下文在全部
      平面上的驻留（融合累加用紧凑 ``array``/list 临时量由 safety 放大）。
    """
    fixed = _fixed_raw(plane_sizes, luma_size, cache_frames)
    work_raw = _work_raw(plane_row_bytes, plane_heights,
                         band_rows, spatial_radius)
    return (fixed + work_raw) * safety


def derive_cache_frames(plane_sizes, luma_size, budget_bytes, band_rows,
                        plane_row_bytes, plane_heights, spatial_radius,
                        max_cache_frames=MAX_CACHE_FRAMES, safety=SAFETY):
    """由预算 B 与片长 L 反推可负担的缓存帧数（口径同内存估算）。"""
    frame_bytes = sum(plane_sizes)
    work_raw = _work_raw(plane_row_bytes, plane_heights,
                         band_rows, spatial_radius)
    # B/safety >= work_raw + F·k + Luma·(k-1)
    #         = work_raw - Luma + k·(F + Luma)
    spare = budget_bytes // safety - work_raw + luma_size
    per_frame = frame_bytes + luma_size
    frames = spare // max(1, per_frame)
    return max(1, min(max_cache_frames, int(frames)))


def derive_band_height(plane_sizes, luma_size, cache_frames, budget_bytes,
                       plane_row_bytes, plane_heights, spatial_radius,
                       safety=SAFETY):
    """由预算 B 与缓存帧数 K 反推每带可负担的核心行数（按最宽平面折算）。

    预算不足以容纳固定项时返回 0，由 :func:`plan_spatiotemporal` 拒绝。
    """
    fixed = _fixed_raw(plane_sizes, luma_size, cache_frames)
    work_budget = budget_bytes // safety - fixed
    if work_budget < 0:
        return 0
    widest = max(plane_row_bytes)
    ctx = 4 * spatial_radius
    afford_rows = work_budget // max(1, widest)
    return max(0, afford_rows - ctx)


def plan_spatiotemporal(plane_sizes, plane_widths, plane_heights,
                        temporal_radius, budget_bytes, spatial_radius,
                        max_cache_frames=MAX_CACHE_FRAMES,
                        band_height=None, safety=SAFETY):
    """缓存帧数 K、片长 L、内存 B 的统一互推与可行性校验。

    参数
    ----
    plane_sizes / plane_widths / plane_heights:
        各平面单帧字节数、宽、高（三个列表等长，按平面顺序对齐）。
    temporal_radius:
        时域半径 r，目标窗 [t-r, t+r]，名义窗 2r+1 帧。
    budget_bytes:
        峰值内存预算（字节）。
    spatial_radius:
        空间级盒式均值半径，决定每带额外的 4r 上下文驻留。
    max_cache_frames:
        缓存帧数硬上限。
    band_height:
        显式片长（每带核心行数）。``None`` 时在预算内自动取最大片长；
        给出时按该片长校验预算。

    返回
    ----
    dict，含 ``cache_frames``、``nominal_window``、``band_height``、
    ``bands``（各平面的核心行区间列表）、``spatial_bands``（各平面
    空间级分带，含 2r 上下文口径）、``memory_bytes``、``frame_bytes``。

    异常
    ----
    MemoryError: 预算放不下「硬上限内至少 1 帧缓存 + 每平面至少 1 行
        核心带」，或显式 ``band_height`` 超预算。在任何 ffmpeg 进程
        启动之前抛出。
    ValueError: 非法参数。
    """
    if temporal_radius < 0:
        raise ValueError("temporal_radius 不能为负")
    if spatial_radius < 0:
        raise ValueError("spatial_radius 不能为负")
    if budget_bytes is None or budget_bytes <= 0:
        raise ValueError("budget_bytes 必须为正")
    if not plane_sizes:
        raise ValueError("至少需要一个平面")

    frame_bytes = sum(plane_sizes)
    luma_size = plane_sizes[0]  # 权重图只存第 0 平面（亮度/首通道）
    nominal = 2 * temporal_radius + 1
    cache_cap = min(nominal, max_cache_frames)
    plane_row_bytes = list(plane_widths)

    def mem_of(k, l):
        return spatiotemporal_memory_bytes(
            plane_sizes, luma_size, k, l, plane_row_bytes, plane_heights,
            spatial_radius, safety)

    # 下限场景：1 帧缓存（无权重图）+ 每平面 1 行核心带都放不下 → 拒绝
    min_need = mem_of(1, 1)
    if min_need > budget_bytes:
        raise MemoryError(
            f"内存预算 {budget_bytes} 字节不足：最少需要 {min_need} 字节"
            f"（1 帧缓存 + 最小带工作区）"
        )

    if band_height is not None:
        if band_height < 1:
            raise ValueError("band_height 必须 >= 1")
        k = min(cache_cap,
                derive_cache_frames(plane_sizes, luma_size, budget_bytes,
                                    band_height, plane_row_bytes,
                                    plane_heights, spatial_radius,
                                    max_cache_frames, safety))
        need = mem_of(k, band_height)
        if need > budget_bytes:
            raise MemoryError(
                f"内存预算 {budget_bytes} 字节不足：片长 {band_height} 行、"
                f"缓存 {k} 帧需 {need} 字节"
            )
        l = band_height
    else:
        # 自动模式：争取整片一带，缓存帧数从名义值逐档下调到预算内；
        # 整片仍放不下才接受被预算切碎的更小带（与时域规划同策略）。
        target_band = max(plane_heights)
        k = cache_cap
        while k > 1 and mem_of(k, target_band) > budget_bytes:
            k -= 1
        l = derive_band_height(plane_sizes, luma_size, k, budget_bytes,
                               plane_row_bytes, plane_heights,
                               spatial_radius, safety)
        l = min(l, max(plane_heights))
        while l < 1 and k > 1:
            k -= 1
            l = derive_band_height(plane_sizes, luma_size, k, budget_bytes,
                                   plane_row_bytes, plane_heights,
                                   spatial_radius, safety)
            l = min(l, max(plane_heights))
        if l < 1:
            raise MemoryError("预算不足以容纳任何可行的缓存/片长组合")

    # 时间融合带：无空间邻域，带间行不重叠，各平面按统一核心行高切
    bands = [_uniform_bands(ph, l) for ph in plane_heights]
    # 空间级分带：复用均值模式的预算/上下文口径（每带 4r 上下文），
    # 各平面按自身行字节在同一总预算内独立规划。
    spatial_bands = [
        plan_bands(ph, pw, spatial_radius,
                   max(1, budget_bytes // safety // max(1, len(plane_sizes))))
        for pw, ph in zip(plane_widths, plane_heights)
    ]

    return {
        "cache_frames": k,
        "nominal_window": nominal,
        "band_height": l,
        "bands": bands,
        "spatial_bands": spatial_bands,
        "memory_bytes": mem_of(k, l),
        "frame_bytes": frame_bytes,
        "luma_size": luma_size,
        "temporal_radius": temporal_radius,
        "spatial_radius": spatial_radius,
    }


def _uniform_bands(height, core):
    """把 height 行按固定核心行高 core 切成区间（最后一带吃余数）。"""
    if core >= height:
        return [(0, height)]
    out = []
    y = 0
    while y < height:
        e = min(height, y + core)
        out.append((y, e))
        y = e
    return out


# ---------------------------------------------------------------- 帧缓存

class FrameCache:
    """有界帧缓存：按到达顺序保留最近 ``maxlen`` 帧的全部平面。

    与 :class:`vdenoise.temporal.FrameCache` 同语义：长视频下旧帧从
    右端自动淘汰，驻留帧数恒 <= ``maxlen``，内存与总帧数无关。
    每帧存为 ``(planes, dims)``，planes 为各平面 bytes。
    """

    def __init__(self, maxlen):
        if maxlen < 1:
            raise ValueError("缓存容量必须 >= 1")
        self._dq = deque(maxlen=maxlen)

    def append(self, planes, dims=None):
        self._dq.append((list(planes), dims))

    def frames(self):
        """返回缓存帧列表（旧 → 新），每项 ``(planes, dims)``。"""
        return list(self._dq)

    def __len__(self):
        return len(self._dq)

    def planes_at(self, i):
        return self._dq[i][0]


# ------------------------------------------------- 权重图：构造 / 平滑 / 重采样

def motion_weight_map(target, ref, width, height, shift,
                      motion_threshold=DEFAULT_MOTION_THRESHOLD):
    """按前级对齐残差构造单个参考帧的逐像素软权重图（只在亮度平面调）。

    权重三档（定点 0..:data:`WEIGHT_SCALE`）：

    - ``d <= threshold``：满权 255（静态/匹配，参考帧充分参与）；
    - ``threshold < d < 2·threshold``：线性降权
      ``255·(2·threshold - d) / threshold``，运动边界处平滑过渡；
    - ``d >= 2·threshold``：权重 0（明确运动，拒绝该参考此像素贡献）。

    平移后落在帧外的像素无对应源值，权重 0。残差 ``d`` 复用前级
    全局位移估计对齐后的同坐标比较，不做第二次运动估计。
    """
    dx, dy = shift
    t2 = 2 * motion_threshold
    out = bytearray(width * height)
    for y in range(height):
        base = y * width
        ry = y - dy
        row_ok = 0 <= ry < height
        if row_ok:
            rbase = ry * width
        for x in range(width):
            idx = base + x
            if not row_ok:
                continue
            rx = x - dx
            if not (0 <= rx < width):
                continue
            d = abs(target[idx] - ref[rbase + rx])
            if d <= motion_threshold:
                out[idx] = WEIGHT_SCALE
            elif d < t2:
                out[idx] = WEIGHT_SCALE * (t2 - d) // motion_threshold
    return bytes(out)


def smooth_weight_map(weights, width, height, mask_radius):
    """对权重图做小半径盒式均值平滑，消除运动边界的逐像素硬切块状。

    边界窗口截断、除以实际像素数，四舍五入；``mask_radius <= 0`` 时
    原样返回。均值对象是权重而非像素，故平滑只让边界权重渐变，不会
    把运动内容本身平均进静态区。
    """
    if mask_radius <= 0:
        return bytes(weights)
    out = bytearray(width * height)
    for y in range(height):
        y0 = max(0, y - mask_radius)
        y1 = min(height, y + mask_radius + 1)
        base = y * width
        for x in range(width):
            x0 = max(0, x - mask_radius)
            x1 = min(width, x + mask_radius + 1)
            acc = 0
            cnt = 0
            for yy in range(y0, y1):
                rb = yy * width
                for xx in range(x0, x1):
                    acc += weights[rb + xx]
                    cnt += 1
            out[base + x] = (acc + cnt // 2) // cnt
    return bytes(out)


def resample_weight_map(weights, src_w, src_h, dst_w, dst_h):
    """把亮度权重图块平均重采样到色度平面尺寸（掩码复用，不重新估计）。

    支持的采样比与 :mod:`vdenoise.pixfmt` 的布局一一对应：1:1 原样、
    2:1 / 1:2 / 2:2 块平均（4:2:0 即 2×2）。非整数比由最近网格兜底，
    当前支持的像素格式不会触发。边界不足整块时按实际覆盖像素平均。
    """
    if dst_w == src_w and dst_h == src_h:
        return bytes(weights)
    out = bytearray(dst_w * dst_h)
    for dy in range(dst_h):
        sy0 = dy * src_h // dst_h
        sy1 = (dy + 1) * src_h // dst_h
        db = dy * dst_w
        for dx in range(dst_w):
            sx0 = dx * src_w // dst_w
            sx1 = (dx + 1) * src_w // dst_w
            if sx1 <= sx0:
                sx1 = sx0 + 1
            if sy1 <= sy0:
                sy1 = sy0 + 1
            acc = 0
            cnt = 0
            for yy in range(sy0, min(sy1, src_h)):
                rb = yy * src_w
                for xx in range(sx0, min(sx1, src_w)):
                    acc += weights[rb + xx]
                    cnt += 1
            out[db + dx] = (acc + cnt // 2) // cnt if cnt else 0
    return bytes(out)


# ------------------------------------------------------- 带内时间加权融合

def fuse_band(target, refs, weights, shifts, width, height, band):
    """对单个平面的一个带做时间窗软权重融合，返回该带行区间的 bytes。

    - ``target``：当前帧平面（融合中心，恒以满权 :data:`WEIGHT_SCALE`
      参与）；
    - ``refs``：参考帧平面列表（同尺寸 bytes）；
    - ``weights``：与 refs 对齐的权重图列表（本平面坐标系，色度平面
      由调用方完成重采样），元素为 0..255 定点 bytes；
    - ``shifts``：与 refs 对齐的 ``(dx, dy)`` 列表；
    - ``band``：核心行区间 ``(s, e)``。

    融合为定点加权均值：``out = (Σ value·w + 半刻度) // Σw``，
    当前帧权重固定 255。某像素全部参考权重为 0（运动/越界）时
    ``Σw`` 只剩当前帧的 255，结果严格等于当前帧原值——软权重版本的
    「全拒回退」防拖影保证。累加器用无符号 32 位紧凑数组：最坏
    Σ(value·w) = 255·255·15 ≈ 97.5 万，远低于 2³²。
    """
    s, e = band
    n_rows = e - s
    n = n_rows * width
    acc = array("I", [0]) * n
    wsum = array("I", [0]) * n
    # 当前帧满权
    for i in range(n):
        tv = target[s * width + i]
        acc[i] = tv * WEIGHT_SCALE
        wsum[i] = WEIGHT_SCALE
    for ref, wmap, (dx, dy) in zip(refs, weights, shifts):
        for ry_index in range(n_rows):
            y = s + ry_index
            src_y = y - dy
            row_base = ry_index * width
            if not (0 <= src_y < height):
                continue
            rbase = src_y * width
            tbase = y * width
            for x in range(width):
                sx = x - dx
                if not (0 <= sx < width):
                    continue
                w = wmap[tbase + x]
                if w == 0:
                    continue
                k = row_base + x
                acc[k] += ref[rbase + sx] * w
                wsum[k] += w
    half = WEIGHT_SCALE // 2
    return bytes((acc[i] + half) // wsum[i] for i in range(n))


def process_plane(target, refs, luma_weights, luma_w, luma_h, width, height,
                  shifts, bands, spatial_bands, spatial_radius, is_luma):
    """单平面完整空时联合处理：带内时间融合 → 空间盒式均值。

    亮度平面（``is_luma`` 真）直接使用已平滑的亮度权重图；色度平面
    先把每张亮度权重图 :func:`resample_weight_map` 到本平面尺寸
    （掩码复用，``luma_w/luma_h`` 为亮度平面尺寸）。权重图的边缘
    平滑由调用方在构造阶段统一完成（:func:`smooth_weight_map`），
    此处不再重复。时间融合按 ``bands`` 逐带产出并拼成整帧，再交给
    :func:`vdenoise.bands.box_blur_plane_banded` 做空间级滤波；
    ``spatial_radius=0`` 时空间级为恒等，输出即时间融合结果。
    """
    if is_luma:
        weights = luma_weights
    else:
        weights = [
            resample_weight_map(wm, luma_w, luma_h, width, height)
            for wm in luma_weights
        ]
    fused = bytearray(width * height)
    for s, e in bands:
        fused[s * width:e * width] = fuse_band(
            target, refs, weights, shifts, width, height, (s, e))
    fused = bytes(fused)
    if spatial_radius <= 0:
        return fused
    return box_blur_plane_banded(fused, width, height,
                                 spatial_radius, spatial_bands)


# ---------------------------------------------------------------- 降噪指标

def planar_mad(plane_a, plane_b):
    """两平面逐像素平均绝对差（MAD，定点整数，无浮点）。

    用于前后指标对比：空间级 MAD 对照同一帧的空间平滑结果，量化
    被压制的高频噪声能量；返回整数（灰度差均值，四舍五入）。
    """
    if not plane_a:
        return 0
    total = sum(abs(a - b) for a, b in zip(plane_a, plane_b))
    return (total + len(plane_a) // 2) // len(plane_a)


def temporal_mad(planes_a, planes_b):
    """两组平面对应帧的逐像素平均绝对差（时域噪声/帧间跳动能量）。

    两组平面列表等长、逐平面同尺寸；返回各平面 MAD 按面积加权后的
    单帧均值（整数）。静态场景下该值主要由帧间独立噪声贡献，降噪
    后应显著下降。
    """
    total = 0
    count = 0
    for a, b in zip(planes_a, planes_b):
        total += sum(abs(x - y) for x, y in zip(a, b))
        count += len(a)
    if count == 0:
        return 0
    return (total + count // 2) // count
