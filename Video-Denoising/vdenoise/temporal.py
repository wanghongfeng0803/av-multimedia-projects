"""时域（帧间）降噪：全局位移估计对齐后再做时间窗混合。

核心设计（需求逐条对应）：

- **缓存帧数有上限**：时间窗为以当前帧为中心的 [t-r, t+r]，实际
  缓存帧数 K=min(2r+1, 帧数, ``max_cache_frames``)。缓存用有界 deque，
  超出窗口的旧帧立即弹出，长视频内存恒定，与总帧数无关。
- **帧间对齐用全局位移估计，不按同坐标混合**：先在（下采样后的）
  参考平面上做粗-精两级块匹配，求整像素全局位移 (dx, dy)，再把
  每个参考帧按该位移平移到当前帧坐标系。位移以亮度平面为准，色度
  平面按其相对亮度的采样比缩放位移（4:2:0 位移减半）。
- **运动区避免拖影**：对齐后逐像素比较，差的绝对值超过
  ``motion_threshold`` 即判为运动区，该参考帧在该像素不参与混合；
  所有参考帧都不匹配时回退当前帧原值（绝不平均错位内容，否则就是
  鬼影/拖影）。

片长（每带核心行数 L）、缓存帧数 K、内存预算 B 三者互推，
见 :func:`plan_temporal`；超预算在开工前由 :class:`MemoryError`
提前拒绝，不启动任何 ffmpeg 子进程。

纯标准库实现，平面均为单字节 bytes，行主序、宽 ``width``。
"""

from collections import deque

__all__ = [
    "MAX_CACHE_FRAMES",
    "DEFAULT_MOTION_THRESHOLD",
    "SAFETY",
    "temporal_memory_bytes",
    "derive_cache_frames",
    "derive_band_height",
    "plan_temporal",
    "FrameCache",
    "estimate_shift",
    "temporal_blend_plane_banded",
]

#: 缓存帧数的硬上限（防止用户把时域半径开到把内存吃光）
MAX_CACHE_FRAMES = 15
#: 对齐残差超过该灰度差即判运动（单字节平面的像素级阈值）
DEFAULT_MOTION_THRESHOLD = 24
#: 内存估算安全系数，覆盖 Python bytes/int 等额外驻留，与空间分带一致
SAFETY = 2


# ---------------------------------------------------------------- 内存互推

def _work_raw(plane_row_bytes, plane_heights, band_rows):
    """带工作区原始字节：各平面实际核心行数（不超平面高）× 行字节。"""
    return sum(
        min(band_rows, ph) * row
        for row, ph in zip(plane_row_bytes, plane_heights)
    )


def temporal_memory_bytes(plane_sizes, cache_frames, band_rows,
                          plane_row_bytes, plane_heights, safety=SAFETY):
    """时域处理的峰值内存估算（字节）。

    内存分两部分：

    - 缓存：``cache_frames`` 帧 × 全帧各平面字节（``sum(plane_sizes)``）；
    - 带工作区：最大带 ``band_rows`` 核心行在全部平面上的驻留，
      混合用累加/计数等临时结构按 ``safety`` 放大。

    时域混合无空间邻域，带间行不重叠，故没有空间滤波那种上下文驻留。
    """
    frame_bytes = sum(plane_sizes)
    cache = frame_bytes * cache_frames
    work_raw = _work_raw(plane_row_bytes, plane_heights, band_rows)
    return (cache + work_raw) * safety


def derive_cache_frames(plane_sizes, budget_bytes, band_rows,
                        plane_row_bytes, plane_heights,
                        max_cache_frames=MAX_CACHE_FRAMES,
                        safety=SAFETY):
    """由预算 B 与片长 L（``band_rows`` 核心行）反推可负担的缓存帧数。

    内存口径与 :func:`temporal_memory_bytes` 完全一致：
    ``B >= (F·k + work_raw(L))·safety``。解出 k 后夹到 [1, 硬上限]。
    """
    frame_bytes = sum(plane_sizes)
    work_raw = _work_raw(plane_row_bytes, plane_heights, band_rows)
    frames = (budget_bytes // safety - work_raw) // max(1, frame_bytes)
    return max(1, min(max_cache_frames, int(frames)))


def derive_band_height(plane_sizes, cache_frames, budget_bytes,
                       plane_row_bytes, plane_heights,
                       safety=SAFETY):
    """由预算 B 与缓存帧数 K 反推每带可负担的核心行数（取最紧平面）。

    同口径 ``B >= (F·k + work_raw(L))·safety``：先扣缓存项，剩余额度
    除以 safety 得带工作区原始字节上限，再按最贵平面折算行数。
    预算不足以容纳 K 帧缓存时返回 0，由 :func:`plan_temporal` 提前拒绝。
    """
    frame_bytes = sum(plane_sizes)
    work_budget = budget_bytes // safety - frame_bytes * cache_frames
    if work_budget < 0:
        return 0
    # 片长是统一的「行数」预算：按每行最贵（最宽）的平面折算可负担行数，
    # 各平面真正切带时再各自夹到自身高度（_uniform_bands / _work_raw）。
    # 不能在这里 min(ph)，否则色度平面的较小高度会被误当成全局片长上限。
    widest = max(plane_row_bytes)
    return max(0, work_budget // max(1, widest))


def plan_temporal(plane_sizes, plane_widths, plane_heights,
                  temporal_radius, budget_bytes,
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
    max_cache_frames:
        缓存帧数硬上限。
    band_height:
        显式片长（每带核心行数）。``None`` 时在预算内自动取最大片长
        （带数最少）；给出时按该片长校验预算。

    返回
    ----
    dict，含 ``cache_frames``、``band_height``、``bands``（各平面的
    核心行区间列表）、``memory_bytes``（估算峰值）、``frame_bytes``。

    异常
    ----
    MemoryError: 预算放不下「硬上限内至少 1 帧缓存 + 每平面至少 1 行
        核心带」，或显式 ``band_height`` 超预算。在任何 ffmpeg 进程
        启动之前抛出，实现「超预算提前拒绝」。
    ValueError: 时域半径为负等非法参数。
    """
    if temporal_radius < 0:
        raise ValueError("temporal_radius 不能为负")
    if budget_bytes is None or budget_bytes <= 0:
        raise ValueError("budget_bytes 必须为正")
    if not plane_sizes:
        raise ValueError("至少需要一个平面")

    frame_bytes = sum(plane_sizes)
    nominal = 2 * temporal_radius + 1
    cache_cap = min(nominal, max_cache_frames)
    plane_row_bytes = list(plane_widths)  # 单字节平面：每行字节数 = 宽

    def mem_of(k, l):
        # 与 temporal_memory_bytes 同口径：(F·k + work_raw)·safety
        work_raw = _work_raw(plane_row_bytes, plane_heights, l)
        return (frame_bytes * k + work_raw) * safety

    # 下限场景：1 帧缓存 + 每平面 1 行核心带都放不下 → 直接拒绝
    min_need = mem_of(1, 1)
    if min_need > budget_bytes:
        raise MemoryError(
            f"内存预算 {budget_bytes} 字节不足：最少需要 {min_need} 字节"
            f"（1 帧缓存 + 最小带工作区）"
        )

    if band_height is not None:
        if band_height < 1:
            raise ValueError("band_height 必须 >= 1")
        # 显式片长：缓存帧数在预算内尽量接近名义窗
        k = min(cache_cap,
                derive_cache_frames(plane_sizes, budget_bytes, band_height,
                                    plane_row_bytes, plane_heights,
                                    max_cache_frames, safety))
        need = mem_of(k, band_height)
        if need > budget_bytes:
            raise MemoryError(
                f"内存预算 {budget_bytes} 字节不足：片长 {band_height} 行、"
                f"缓存 {k} 帧需 {need} 字节"
            )
        l = band_height
    else:
        # 自动模式：在预算内寻找「缓存帧数尽量大、同时片长不被压碎」的
        # 组合。优先满足片长目标（默认取最大平面全高，即争取整片一带），
        # 从名义缓存帧数逐档下调，直到该片长下的内存落在预算内；
        # 即使降到 1 帧也放不下整片，才接受被预算切碎的更小带。
        target_band = max(plane_heights)
        k = cache_cap
        while k > 1 and mem_of(k, target_band) > budget_bytes:
            k -= 1
        l = derive_band_height(plane_sizes, k, budget_bytes,
                               plane_row_bytes, plane_heights, safety)
        l = min(l, max(plane_heights))  # 片长不超过最高平面，整片即一带
        while l < 1 and k > 1:  # 兜底：理论上被 min_need 校验拦截
            k -= 1
            l = derive_band_height(plane_sizes, k, budget_bytes,
                                   plane_row_bytes, plane_heights, safety)
            l = min(l, max(plane_heights))
        if l < 1:
            raise MemoryError("预算不足以容纳任何可行的缓存/片长组合")

    # 时域混合无空间邻域、带间行不重叠，各平面直接按统一核心行高 l 切带，
    # 峰值内存与 mem_of(k, l) 的估算严格对应。
    bands = [_uniform_bands(ph, l) for ph in plane_heights]

    return {
        "cache_frames": k,
        "nominal_window": nominal,
        "band_height": l,
        "bands": bands,
        "memory_bytes": mem_of(k, l),
        "frame_bytes": frame_bytes,
        "temporal_radius": temporal_radius,
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

    长视频下旧帧从右端自动淘汰，驻留帧数恒 <= ``maxlen``，内存与总
    帧数无关。每帧存为 ``(planes, dims)``，planes 为各平面 bytes。
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


# -------------------------------------------------------- 全局位移估计

def _downsample(plane, width, height, step):
    """step×step 块平均下采样，返回 (字节, 新宽, 新高)。step<=1 原样。"""
    if step <= 1:
        return bytes(plane), width, height
    nw, nh = width // step, height // step
    out = bytearray(nw * nh)
    for ny in range(nh):
        src_y0 = ny * step
        base = ny * nw
        row_acc = [0] * nw
        for dy in range(step):
            row = plane[(src_y0 + dy) * width:]
            for nx in range(nw):
                src_x0 = nx * step
                s = 0
                for dx in range(step):
                    s += row[src_x0 + dx]
                row_acc[nx] += s
        inv = step * step
        for nx in range(nw):
            out[base + nx] = (row_acc[nx] + inv // 2) // inv
    return bytes(out), nw, nh


def _sad(ref, cur, rw, rh, dx, dy, stride):
    """cur 与平移 (dx,dy) 后的 ref 在重叠区的采样平均绝对差。

    坐标约定：ref 像素 (x,y) 对齐到 cur 的 (x+dx, y+dy)。只统计两帧
    都覆盖的重叠区，按 ``stride`` 等距抽样控成本；无重叠返回 None。
    """
    x0 = max(0, dx)
    x1 = min(rw, rw + dx)  # cur 列区间上界：ref 列 [0,rw) → cur [dx, rw+dx)
    y0 = max(0, dy)
    y1 = min(rh, rh + dy)
    if x1 <= x0 or y1 <= y0:
        return None
    total = 0
    count = 0
    for cy in range(y0, y1, stride):
        ry = cy - dy
        cbase = cy * rw
        rbase = ry * rw
        for cx in range(x0, x1, stride):
            rx = cx - dx
            total += abs(cur[cbase + cx] - ref[rbase + rx])
            count += 1
    if count == 0:
        return None
    return total / count


def estimate_shift(ref, cur, width, height, max_shift,
                   coarse_max_width=32, fine_stride=2):
    """全局整像素位移估计：粗-精两级块匹配，最小化采样 SAD。

    返回 ``(dx, dy)``：ref 对齐到 cur 的位移（同 :func:`_sad` 约定）。

    - 粗级：块平均下采样到宽 <= ``coarse_max_width``，在下采样网格的
      [-max_shift, max_shift] 全域穷举，SAD 用步长 1；
    - 精级：回到原分辨率，以粗解 × 粗步长为中心、±粗步长 的邻域穷举，
      SAD 按 ``fine_stride`` 抽样。
    平局取位移幅度最小、再按 (dy, dx) 字典序最小，保证确定性。
    平面太小（宽 <= coarse_max_width）时跳过下采样直接全域穷举。
    """
    if max_shift <= 0 or width <= 1 or height <= 1:
        return 0, 0

    step = max(1, width // coarse_max_width)
    if step > 1:
        dref, dw, dh = _downsample(ref, width, height, step)
        dcur, _, _ = _downsample(cur, width, height, step)
        dmax = max(1, max_shift // step)
    else:
        dref, dcur, dw, dh = bytes(ref), bytes(cur), width, height
        dmax = max_shift

    coarse_score = None
    coarse_best = (0, 0)
    for dy in range(-dmax, dmax + 1):
        for dx in range(-dmax, dmax + 1):
            score = _sad(dref, dcur, dw, dh, dx, dy, 1)
            if score is None:
                continue
            if (coarse_score is None or score < coarse_score
                    or (score == coarse_score
                        and _shift_order(dx, dy) < _shift_order(*coarse_best))):
                coarse_score = score
                coarse_best = (dx, dy)

    if step <= 1:
        return coarse_best

    cx, cy = coarse_best[0] * step, coarse_best[1] * step
    fine_score = None
    fine_best = (cx, cy)
    for dy in range(cy - step, cy + step + 1):
        if abs(dy) > max_shift:
            continue
        for dx in range(cx - step, cx + step + 1):
            if abs(dx) > max_shift:
                continue
            score = _sad(ref, cur, width, height, dx, dy, fine_stride)
            if score is None:
                continue
            if (fine_score is None or score < fine_score
                    or (score == fine_score
                        and _shift_order(dx, dy) < _shift_order(*fine_best))):
                fine_score = score
                fine_best = (dx, dy)
    return fine_best


def _shift_order(dx, dy):
    """平局排序键：幅度小者优先，再按 (dy, dx) 字典序。"""
    return (abs(dx) + abs(dy), dy, dx)


# --------------------------------------------------- 对齐 + 运动掩码 + 混合

def temporal_blend_plane_banded(target, refs, width, height, shifts,
                                bands, motion_threshold=DEFAULT_MOTION_THRESHOLD):
    """对单个平面做时间窗对齐混合，分带处理，返回 width*height 字节。

    - ``target``：当前帧平面（混合中心，永远参与，且在全部参考被运动
      掩码拒绝时作为回退）；
    - ``refs``：参考帧平面列表（同尺寸 bytes）；
    - ``shifts``：与 refs 对齐的 ``(dx, dy)`` 列表，位移已是本平面
      坐标系（色度平面由调用方按采样比缩放好）；
    - ``bands``：核心行区间 ``[(s, e), ...]``，时域混合无空间邻域，
      带间行不重叠，无需上下文与接缝合并；
    - ``motion_threshold``：对齐残差超过该值的像素，该参考帧在此像素
      不参与累加（运动区避免拖影）。

    混合为均匀权重的整数均值（含当前帧），四舍五入；参考帧在窗口
    边界天然变少，截断语义与空间均值模式一致。
    """
    if not refs or not bands:
        return bytes(target)
    out = bytearray(width * height)
    for s, e in bands:
        for y in range(s, e):
            base = y * width
            for x in range(width):
                acc = target[base + x]
                cnt = 1
                tv = acc
                for ref, (dx, dy) in zip(refs, shifts):
                    rx, ry = x - dx, y - dy
                    if not (0 <= rx < width and 0 <= ry < height):
                        continue  # 平移后无对应源像素，该参考放弃此点
                    rv = ref[ry * width + rx]
                    if abs(rv - tv) > motion_threshold:
                        continue  # 运动区：不混入，防止拖影
                    acc += rv
                    cnt += 1
                out[base + x] = (acc + cnt // 2) // cnt
    return bytes(out)
