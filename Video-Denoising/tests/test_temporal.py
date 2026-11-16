"""时域降噪纯逻辑测试：位移估计、运动掩码、混合、缓存、内存三互推。

不依赖 ffmpeg，全部用合成 bytes 平面验证算法契约：
- 全局位移必须被块匹配找回（不是按同坐标混合）；
- 运动区残差超阈值时拒绝参考帧，回退当前帧（无拖影）；
- 静态噪声被多帧平均压制、静止画面恒等；
- 缓存有界；缓存帧数 / 片长 / 预算三者互推自洽，超预算提前拒绝。
"""

import pytest

from vdenoise.temporal import (
    DEFAULT_MOTION_THRESHOLD,
    MAX_CACHE_FRAMES,
    SAFETY,
    FrameCache,
    derive_band_height,
    derive_cache_frames,
    estimate_shift,
    plan_temporal,
    temporal_blend_plane_banded,
    temporal_memory_bytes,
)

W, H = 24, 16


def flat_plane(value, w=W, h=H):
    return bytes([value]) * (w * h)


def shift_plane(plane, dx, dy, w=W, h=H, fill=0):
    """把 plane 平移 (dx, dy)：输出 (x,y) 取源 (x-dx, y-dy)，越界填 fill。

    与 estimate_shift 的坐标约定一致：ref 平移 (dx,dy) 后对齐 cur。
    """
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            sx, sy = x - dx, y - dy
            out[y * w + x] = plane[sy * w + sx] if 0 <= sx < w and 0 <= sy < h else fill
    return bytes(out)


def textured_plane(seed=1, w=W, h=H):
    """确定性高纹理平面：多频率网格，保证 SAD 曲面有唯一极小值。"""
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            v = (x * 13 + y * 7 + seed * 5 + (x * y) % 11 + (x // 3) * 17) & 255
            out[y * w + x] = v
    return bytes(out)


# ---------- 全局位移估计 ----------

@pytest.mark.parametrize("dx,dy", [(2, 0), (-3, 1), (0, 2), (4, -3), (-2, -2)])
def test_estimate_shift_recovers_known_translation(dx, dy):
    ref = textured_plane()
    cur = shift_plane(ref, dx, dy, fill=127)
    got_dx, got_dy = estimate_shift(ref, cur, W, H, max_shift=6)
    assert (got_dx, got_dy) == (dx, dy)


def test_estimate_shift_identical_frames_zero_shift():
    plane = textured_plane()
    assert estimate_shift(plane, plane, W, H, max_shift=6) == (0, 0)


def test_estimate_shift_zero_max_shift_returns_zero():
    ref = textured_plane()
    cur = shift_plane(ref, 3, 3)
    assert estimate_shift(ref, cur, W, H, max_shift=0) == (0, 0)


def test_estimate_shift_respects_max_shift_bound():
    ref = textured_plane()
    cur = shift_plane(ref, 5, 0)
    # 搜索半径只给 3：真值 5 在界外，估计结果必须落在 [-3, 3] 内
    dx, dy = estimate_shift(ref, cur, W, H, max_shift=3)
    assert abs(dx) <= 3 and abs(dy) <= 3


def test_estimate_shift_deterministic_tie_break():
    # 完全平坦平面：所有位移 SAD 相同，必须确定性地回落到 (0,0)
    plane = flat_plane(100)
    assert estimate_shift(plane, plane, W, H, max_shift=4) == (0, 0)


# ---------- 对齐混合与运动掩码 ----------

def test_blend_aligned_static_frames_averages_noise():
    # 同一静态场景 + 独立噪声：对齐后多帧平均应把均值拉回真值附近
    target = bytes([100 + ((i * 37) % 11) - 5 for i in range(W * H)])
    refs = [
        bytes([100 + ((i * 53 + k * 7) % 11) - 5 for i in range(W * H)])
        for k in range(4)
    ]
    shifts = [(0, 0)] * len(refs)
    bands = [(0, H)]
    out = temporal_blend_plane_banded(target, refs, W, H, shifts, bands,
                                      motion_threshold=40)
    # 所有像素都在阈值内 → 5 帧均匀平均，结果围绕 100，误差比单帧小
    assert all(95 <= v <= 105 for v in out)
    dev_before = sum(abs(v - 100) for v in target) / len(target)
    dev_after = sum(abs(v - 100) for v in out) / len(out)
    assert dev_after < dev_before


def test_blend_flat_identical_frames_is_identity():
    plane = flat_plane(120)
    refs = [bytes(plane), bytes(plane)]
    out = temporal_blend_plane_banded(
        plane, refs, W, H, [(0, 0), (0, 0)], [(0, H)])
    assert out == plane


def test_blend_motion_region_falls_back_to_target():
    # 目标帧左半 = 60，右半 = 200；参考帧整体 60（右半是突变运动区）
    target = bytearray(W * H)
    for y in range(H):
        for x in range(W):
            target[y * W + x] = 60 if x < W // 2 else 200
    target = bytes(target)
    ref = flat_plane(60)
    # 不给位移（同坐标），阈值 24：右半差 140 必判运动，左半差 0 接受
    out = temporal_blend_plane_banded(
        target, [ref], W, H, [(0, 0)], [(0, H)],
        motion_threshold=DEFAULT_MOTION_THRESHOLD)
    for y in range(H):
        for x in range(W):
            idx = y * W + x
            if x >= W // 2:
                assert out[idx] == 200  # 运动区回退当前帧，绝不混入 60
            else:
                assert out[idx] == 60   # 静态区两帧一致


def test_blend_shifted_reference_is_aligned_before_mixing():
    # 参考帧是目标帧平移 (2,0)：若按同坐标混合会整片误判/模糊，
    # 传入正确位移后应几乎逐点对齐，输出与目标一致。
    target = textured_plane()
    ref = shift_plane(target, 2, 0, fill=target[0])
    out = temporal_blend_plane_banded(
        target, [ref], W, H, [(2, 0)], [(0, H)],
        motion_threshold=DEFAULT_MOTION_THRESHOLD)
    # 内部区域（去掉平移产生的边缘列）必须与目标相等
    inside = out[2:W - 2]
    expect = target[2:W - 2]
    # 逐行比较内部列
    mismatch = 0
    for y in range(H):
        base = y * W
        for x in range(2, W - 2):
            if out[base + x] != target[base + x]:
                mismatch += 1
    assert mismatch == 0
    assert len(inside) == len(expect)


def test_blend_banded_equals_whole_frame():
    # 时域混合无空间邻域，分带拼接必须与整帧一次处理逐字节相等
    target = textured_plane(seed=3)
    refs = [textured_plane(seed=3 + k) for k in range(1, 4)]
    shifts = [(1, 0), (0, 1), (-1, -1)]
    whole = temporal_blend_plane_banded(
        target, refs, W, H, shifts, [(0, H)], motion_threshold=30)
    bands = [(0, 5), (5, 11), (11, H)]
    banded = temporal_blend_plane_banded(
        target, refs, W, H, shifts, bands, motion_threshold=30)
    assert banded == whole


def test_blend_no_refs_is_identity():
    plane = textured_plane()
    assert temporal_blend_plane_banded(
        plane, [], W, H, [], [(0, H)]) == plane


def test_blend_out_of_bounds_shift_skips_pixel():
    # 大位移使参考帧在目标边缘无对应源像素：这些点只由当前帧贡献
    target = flat_plane(90)
    ref = flat_plane(90)
    out = temporal_blend_plane_banded(
        target, [ref], W, H, [(W + 2, 0)], [(0, H)])  # 完全错开
    assert out == target


# ---------- 有界帧缓存 ----------

def test_frame_cache_bounded_eviction():
    cache = FrameCache(maxlen=3)
    for i in range(5):
        cache.append([bytes([i])], dims=[(1, 1)])
    assert len(cache) == 3
    # 只保留最近 3 帧（值 2,3,4）
    assert [f[0][0][0] for f in cache.frames()] == [2, 3, 4]


def test_frame_cache_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        FrameCache(0)


def test_frame_cache_planes_at():
    cache = FrameCache(maxlen=2)
    cache.append([b"a"], None)
    cache.append([b"b"], None)
    assert cache.planes_at(0) == [b"a"]
    assert cache.planes_at(1) == [b"b"]


# ---------- 内存三互推 ----------

def _one_plane_dims(w=W, h=H):
    return [w * h], [w], [h]


def test_plan_temporal_auto_uses_full_nominal_window_when_budget_ample():
    sizes, widths, heights = _one_plane_dims()
    plan = plan_temporal(sizes, widths, heights,
                         temporal_radius=2, budget_bytes=10 << 20)
    assert plan["cache_frames"] == 5          # 名义窗 2r+1
    assert plan["nominal_window"] == 5
    assert plan["band_height"] == H           # 预算充足 → 整片一带
    assert plan["bands"] == [[(0, H)]]
    assert plan["memory_bytes"] <= 10 << 20


def test_plan_temporal_hard_cache_cap():
    sizes, widths, heights = _one_plane_dims()
    plan = plan_temporal(sizes, widths, heights,
                         temporal_radius=10, budget_bytes=10 << 20,
                         max_cache_frames=7)
    assert plan["cache_frames"] == 7          # 名义 21 被硬上限夹到 7
    assert plan["nominal_window"] == 21


def test_plan_temporal_rejects_infeasible_budget_before_work():
    sizes, widths, heights = _one_plane_dims()
    # 预算连 1 帧缓存 + 1 行带工作区（再乘 safety）都不够
    with pytest.raises(MemoryError):
        plan_temporal(sizes, widths, heights,
                      temporal_radius=2, budget_bytes=10)


def test_plan_temporal_explicit_band_height_over_budget_rejected():
    sizes, widths, heights = _one_plane_dims()
    frame_bytes = sizes[0]
    # 给一个只够 1 帧缓存的极小预算，再强行要求大片长
    tiny = frame_bytes * SAFETY + 64
    with pytest.raises(MemoryError):
        plan_temporal(sizes, widths, heights,
                      temporal_radius=1, budget_bytes=tiny,
                      band_height=H)


def test_plan_temporal_three_way_consistency():
    # 由 B、L 推 K，再用同一组 K、L 算内存，必须落在预算内且自洽
    sizes, widths, heights = _one_plane_dims()
    budget = 200_000
    plan = plan_temporal(sizes, widths, heights,
                         temporal_radius=4, budget_bytes=budget)
    k, l = plan["cache_frames"], plan["band_height"]
    mem = temporal_memory_bytes(sizes, k, l, widths, heights)
    assert mem == plan["memory_bytes"]
    assert mem <= budget
    # 反推函数与规划结果一致（同样的输入口径）
    assert derive_cache_frames(sizes, budget, l, widths, heights) >= k
    assert derive_band_height(sizes, k, budget, widths, heights) >= l


def test_plan_temporal_shrinks_cache_when_budget_tight():
    sizes, widths, heights = _one_plane_dims()
    frame_bytes = sizes[0]
    # 用规划器同一口径构造：够 1 帧缓存 + 2 行带工作区，不够 2 帧。
    # 预算够 1 帧全帧 + 一点带工作区，但容不下第二帧缓存。
    budget = (frame_bytes * 1 + widths[0] * 2) * SAFETY
    assert budget < frame_bytes * SAFETY * 2  # 确实不够 2 帧缓存
    plan = plan_temporal(sizes, widths, heights,
                         temporal_radius=4, budget_bytes=budget)
    assert plan["cache_frames"] == 1
    assert plan["band_height"] >= 1
    assert plan["memory_bytes"] <= budget


def test_plan_temporal_multiplane_yuv420_bands():
    # yuv420p：Y 64x48，U/V 各 32x24
    w, h = 64, 48
    sizes = [w * h, (w // 2) * (h // 2), (w // 2) * (h // 2)]
    widths = [w, w // 2, w // 2]
    heights = [h, h // 2, h // 2]
    plan = plan_temporal(sizes, widths, heights,
                         temporal_radius=1, budget_bytes=1 << 20,
                         band_height=8)
    assert len(plan["bands"]) == 3
    assert plan["bands"][0] == [(0, 8), (8, 16), (16, 24),
                                (24, 32), (32, 40), (40, 48)]
    # 色度平面高 24，同样按 8 行切
    assert plan["bands"][1] == [(0, 8), (8, 16), (16, 24)]


def test_plan_temporal_validates_arguments():
    sizes, widths, heights = _one_plane_dims()
    with pytest.raises(ValueError):
        plan_temporal(sizes, widths, heights, temporal_radius=-1,
                      budget_bytes=1 << 20)
    with pytest.raises(ValueError):
        plan_temporal(sizes, widths, heights, temporal_radius=1,
                      budget_bytes=0)
    with pytest.raises(ValueError):
        plan_temporal(sizes, widths, heights, temporal_radius=1,
                      budget_bytes=1 << 20, band_height=0)


def test_default_constants_are_sane():
    assert MAX_CACHE_FRAMES >= 3
    assert 1 <= DEFAULT_MOTION_THRESHOLD <= 255
    assert SAFETY >= 1
