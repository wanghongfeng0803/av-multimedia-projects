"""空时联合降噪纯逻辑测试：软权重、平滑/重采样、融合、内存互推、指标。

不依赖 ffmpeg，全部用合成 bytes 平面验证算法契约：
- 软权重三档：阈值内满权、[T,2T) 线性降权、>=2T 拒绝、越界零权；
- 权重图平滑只让边界渐变、不改变内部满权/零权区域；
- 色度重采样为亮度块平均（掩码复用，不二次估计）；
- 融合在全匹配时等价均匀平均、全拒时严格回退当前帧（防拖影）；
- 空间半径 0 时输出即时间融合；分带拼接与整带逐字节相等；
- 缓存有界；缓存帧数 / 片长 / 预算三互推自洽，超预算提前拒绝；
- 指标为定点整数，同输入两次运行逐字节可复现。
"""

import pytest

from vdenoise.spatiotemporal import (
    DEFAULT_MASK_RADIUS,
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_SPATIAL_RADIUS,
    MAX_CACHE_FRAMES,
    SAFETY,
    WEIGHT_SCALE,
    FrameCache,
    derive_band_height,
    derive_cache_frames,
    fuse_band,
    motion_weight_map,
    planar_mad,
    plan_spatiotemporal,
    process_plane,
    resample_weight_map,
    smooth_weight_map,
    spatiotemporal_memory_bytes,
    temporal_mad,
)

W, H = 24, 16


def flat_plane(value, w=W, h=H):
    return bytes([value]) * (w * h)


def textured_plane(seed=1, w=W, h=H):
    """确定性高纹理平面（与时域测试同构造，保证可复现）。"""
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            v = (x * 13 + y * 7 + seed * 5 + (x * y) % 11 + (x // 3) * 17) & 255
            out[y * w + x] = v
    return bytes(out)


# ---------- 软权重三档 ----------

def test_weight_map_full_weight_within_threshold():
    target = flat_plane(100)
    ref = flat_plane(100 + DEFAULT_MOTION_THRESHOLD)  # 残差恰好 = T
    wmap = motion_weight_map(target, ref, W, H, (0, 0),
                             DEFAULT_MOTION_THRESHOLD)
    assert all(w == WEIGHT_SCALE for w in wmap)


def test_weight_map_zero_at_double_threshold():
    target = flat_plane(100)
    ref = flat_plane(100 + 2 * DEFAULT_MOTION_THRESHOLD)  # 残差 = 2T
    wmap = motion_weight_map(target, ref, W, H, (0, 0),
                             DEFAULT_MOTION_THRESHOLD)
    assert all(w == 0 for w in wmap)


def test_weight_map_linear_ramp_between_thresholds():
    t = DEFAULT_MOTION_THRESHOLD
    target = flat_plane(100)
    # 残差 T + T//2，位于斜坡中点附近
    delta = t + t // 2
    ref = flat_plane(100 + delta)
    wmap = motion_weight_map(target, ref, W, H, (0, 0), t)
    expect = WEIGHT_SCALE * (2 * t - delta) // t
    assert all(w == expect for w in wmap)
    assert 0 < expect < WEIGHT_SCALE
    # 单调性：残差越大权重越小
    ref_near = flat_plane(100 + t + 2)
    ref_far = flat_plane(100 + 2 * t - 2)
    w_near = motion_weight_map(target, ref_near, W, H, (0, 0), t)[0]
    w_far = motion_weight_map(target, ref_far, W, H, (0, 0), t)[0]
    assert w_near > w_far > 0


def test_weight_map_out_of_bounds_is_zero():
    target = flat_plane(90)
    ref = flat_plane(90)
    # 整体平移到帧外：所有点无对应源像素，权重全 0
    wmap = motion_weight_map(target, ref, W, H, (W + 2, 0),
                             DEFAULT_MOTION_THRESHOLD)
    assert all(w == 0 for w in wmap)


def test_weight_map_aligned_shift_uses_mapped_coordinate():
    # 位移约定：函数内 target(y,x) 取 ref(y-dy, x-dx)。要让两者对齐，
    # 构造 ref 时须满足 ref(y, x-dx) = target(y, x)，即 ref 在 sx 列
    # 存放 target 的 sx+dx 列（ref 内容相对 target 左移 dx）。
    target = textured_plane()
    dx = 2
    ref = bytearray(W * H)
    for y in range(H):
        for sx in range(W):
            tx = sx + dx
            ref[y * W + sx] = target[y * W + tx] if tx < W else 0
    wmap = motion_weight_map(target, bytes(ref), W, H, (dx, 0),
                             DEFAULT_MOTION_THRESHOLD)
    # x>=dx 时 rx=x-dx 落在 ref 有效列且内容对齐 → 满权
    for y in range(H):
        for x in range(dx, W):
            assert wmap[y * W + x] == WEIGHT_SCALE
    # 最左 dx 列 rx=x-dx<0 越界，无对应源像素 → 零权
    for y in range(H):
        for x in range(dx):
            assert wmap[y * W + x] == 0


# ---------- 权重图平滑 ----------

def test_smooth_weight_uniform_map_unchanged():
    wmap = bytes([WEIGHT_SCALE]) * (W * H)
    assert smooth_weight_map(wmap, W, H, DEFAULT_MASK_RADIUS) == wmap
    zero = bytes(W * H)
    assert smooth_weight_map(zero, W, H, DEFAULT_MASK_RADIUS) == zero


def test_smooth_weight_radius_zero_is_identity():
    wmap = bytes((i * 37) % 256 for i in range(W * H))
    assert smooth_weight_map(wmap, W, H, 0) == wmap


def test_smooth_weight_graduates_motion_boundary():
    # 左半满权、右半零权的硬边界，平滑后边界列必须出现中间值（无硬切）
    wmap = bytearray(W * H)
    for y in range(H):
        for x in range(W):
            wmap[y * W + x] = WEIGHT_SCALE if x < W // 2 else 0
    sm = smooth_weight_map(bytes(wmap), W, H, 1)
    boundary_vals = {sm[y * W + W // 2] for y in range(1, H - 1)}
    assert all(0 < v < WEIGHT_SCALE for v in boundary_vals)
    # 远离边界的两侧保持原值
    assert sm[H // 2 * W + 1] == WEIGHT_SCALE
    assert sm[H // 2 * W + W - 1] == 0


# ---------- 色度重采样（掩码复用） ----------

def test_resample_identity_when_same_size():
    wmap = textured_plane(seed=7)
    assert resample_weight_map(wmap, W, H, W, H) == wmap


def test_resample_2x2_block_average_for_yuv420():
    lw, lh = 8, 8
    wmap = bytes((i * 13) % 256 for i in range(lw * lh))
    cw, ch = 4, 4
    out = resample_weight_map(wmap, lw, lh, cw, ch)
    assert len(out) == cw * ch
    for cy in range(ch):
        for cx in range(cw):
            vals = [
                wmap[(2 * cy + dy) * lw + 2 * cx + dx]
                for dy in range(2) for dx in range(2)
            ]
            expect = (sum(vals) + 2) // 4
            assert out[cy * cw + cx] == expect


def test_resample_2x1_for_yuv422():
    lw, lh = 8, 4
    wmap = bytes((i * 7) % 256 for i in range(lw * lh))
    out = resample_weight_map(wmap, lw, lh, 4, 4)
    assert len(out) == 16
    for cy in range(4):
        for cx in range(4):
            vals = [wmap[cy * lw + 2 * cx], wmap[cy * lw + 2 * cx + 1]]
            assert out[cy * 4 + cx] == (sum(vals) + 1) // 2


# ---------- 带内时间加权融合 ----------

def test_fuse_full_matching_refs_equals_uniform_average():
    # 全部满权（255）时加权融合退化为均匀均值。取能被 3 整除的构造，
    # 避开定点半刻度（127）与整数均值半刻度（1）的 ±1 舍入差异。
    target = bytes([99] * (W * H))
    refs = [bytes([100] * (W * H)), bytes([101] * (W * H))]
    weights = [bytes([WEIGHT_SCALE]) * (W * H) for _ in refs]
    shifts = [(0, 0), (0, 0)]
    out = fuse_band(target, refs, weights, shifts, W, H, (0, H))
    expect = (99 + 100 + 101) // 3  # = 100，整除，两种舍入口径一致
    assert all(v == expect for v in out)
    # 再用通用定点公式核对一遍
    for i in range(W * H):
        acc = target[i] + refs[0][i] + refs[1][i]
        formula = (WEIGHT_SCALE * acc + WEIGHT_SCALE // 2) // (WEIGHT_SCALE * 3)
        assert out[i] == formula


def test_fuse_all_rejected_falls_back_to_target():
    target = flat_plane(120)
    refs = [flat_plane(200), flat_plane(210)]  # 内容完全不同
    weights = [bytes(W * H), bytes(W * H)]     # 全零权
    shifts = [(0, 0), (0, 0)]
    out = fuse_band(target, refs, weights, shifts, W, H, (0, H))
    assert out == target


def test_fuse_soft_weight_mixes_proportionally():
    # 单参考、固定权重 w：out = (t·255 + r·w + 半刻度) // (255+w)
    target = flat_plane(100)
    ref = flat_plane(200)
    w = 128
    weights = [bytes([w]) * (W * H)]
    out = fuse_band(target, [ref], weights, [(0, 0)], W, H, (0, H))
    expect = (100 * 255 + 200 * w + 127) // (255 + w)
    assert all(v == expect for v in out)
    assert 100 < expect < 200


def test_fuse_banded_concatenation_covers_frame():
    # 多个带的拼接必须覆盖整帧且与整带一次融合逐字节相等
    target = textured_plane(seed=3)
    refs = [textured_plane(seed=4)]
    weights = [bytes([WEIGHT_SCALE - 10]) * (W * H)]
    shifts = [(1, 0)]
    whole = fuse_band(target, refs, weights, shifts, W, H, (0, H))
    pieces = b"".join(
        fuse_band(target, refs, weights, shifts, W, H, band)
        for band in ((0, 5), (5, 11), (11, H))
    )
    assert pieces == whole


def test_fuse_zero_weight_skips_out_of_bounds_reference():
    target = flat_plane(90)
    ref = flat_plane(90)
    # 即使参考同值，位移使其完全越界（权重图也对应全 0）→ 恒等
    weights = [bytes(W * H)]
    out = fuse_band(target, [ref], weights, [(W + 2, 0)], W, H, (0, H))
    assert out == target


# ---------- process_plane：时间融合 + 空间级 ----------

def _full_bands(h):
    return [(0, h)]


def test_process_plane_spatial_radius_zero_is_temporal_only():
    target = textured_plane(seed=2)
    refs = [textured_plane(seed=3)]
    lw = smooth_weight_map(
        motion_weight_map(target, refs[0], W, H, (0, 0), 40), W, H, 1)
    out = process_plane(
        target, refs, [lw], W, H, W, H, [(0, 0)], _full_bands(H),
        _full_bands(H), spatial_radius=0, is_luma=True)
    expect = fuse_band(target, refs, [lw], [(0, 0)], W, H, (0, H))
    assert out == expect


def test_process_plane_chroma_resamples_luma_weights():
    # 4:2:0 色度平面：传入亮度尺寸权重图，结果必须等于显式重采样后的融合
    lw, lh = 8, 8
    cw, ch = 4, 4
    target_l = bytes((i * 11) % 256 for i in range(lw * lh))
    ref_l = bytes((i * 13 + 5) % 256 for i in range(lw * lh))
    wmap = smooth_weight_map(
        motion_weight_map(target_l, ref_l, lw, lh, (0, 0), 30), lw, lh, 1)
    target_c = flat_plane(128, cw, ch)
    ref_c = flat_plane(132, cw, ch)
    out = process_plane(
        target_c, [ref_c], [wmap], lw, lh, cw, ch, [(0, 0)],
        _full_bands(ch), _full_bands(ch), spatial_radius=0,
        is_luma=False)
    cw_weights = [resample_weight_map(wmap, lw, lh, cw, ch)]
    expect = fuse_band(target_c, [ref_c], cw_weights, [(0, 0)],
                       cw, ch, (0, ch))
    assert out == expect


def test_process_plane_spatial_blur_smooths_fused_result():
    # 空间级开启后，椒盐式离群点应被邻域平均拉回（输出极差收窄）
    target = bytearray([100] * (W * H))
    target[H // 2 * W + W // 2] = 250  # 单个亮点
    target = bytes(target)
    refs = [bytes(target)]  # 参考与目标完全一致 → 时间融合恒等
    wmap = bytes([WEIGHT_SCALE]) * (W * H)
    out = process_plane(
        target, refs, [wmap], W, H, W, H, [(0, 0)], _full_bands(H),
        _full_bands(H), spatial_radius=1, is_luma=True)
    assert max(out) - min(out) < 250 - 100  # 亮点被空间均值扩散


# ---------- 有界帧缓存 ----------

def test_frame_cache_bounded_eviction():
    cache = FrameCache(maxlen=3)
    for i in range(5):
        cache.append([bytes([i])], dims=[(1, 1)])
    assert len(cache) == 3
    assert [f[0][0][0] for f in cache.frames()] == [2, 3, 4]


def test_frame_cache_rejects_bad_capacity():
    with pytest.raises(ValueError):
        FrameCache(0)


# ---------- 内存三互推 ----------

def _one_plane_dims(w=W, h=H):
    return [w * h], [w], [h]


def test_plan_ample_budget_full_window_single_band():
    sizes, widths, heights = _one_plane_dims()
    plan = plan_spatiotemporal(sizes, widths, heights,
                               temporal_radius=2, budget_bytes=10 << 20,
                               spatial_radius=1)
    assert plan["cache_frames"] == 5
    assert plan["nominal_window"] == 5
    assert plan["band_height"] == H
    assert plan["bands"] == [[(0, H)]]
    assert plan["memory_bytes"] <= 10 << 20


def test_plan_hard_cache_cap():
    sizes, widths, heights = _one_plane_dims()
    plan = plan_spatiotemporal(sizes, widths, heights,
                               temporal_radius=10, budget_bytes=10 << 20,
                               spatial_radius=1, max_cache_frames=7)
    assert plan["cache_frames"] == 7
    assert plan["nominal_window"] == 21


def test_plan_rejects_infeasible_budget_before_work():
    sizes, widths, heights = _one_plane_dims()
    with pytest.raises(MemoryError):
        plan_spatiotemporal(sizes, widths, heights,
                            temporal_radius=2, budget_bytes=10,
                            spatial_radius=1)


def test_plan_explicit_band_height_over_budget_rejected():
    sizes, widths, heights = _one_plane_dims()
    frame_bytes = sizes[0]
    tiny = frame_bytes * SAFETY + 64
    with pytest.raises(MemoryError):
        plan_spatiotemporal(sizes, widths, heights,
                            temporal_radius=1, budget_bytes=tiny,
                            spatial_radius=1, band_height=H)


def test_plan_three_way_consistency():
    sizes, widths, heights = _one_plane_dims()
    budget = 200_000
    plan = plan_spatiotemporal(sizes, widths, heights,
                               temporal_radius=4, budget_bytes=budget,
                               spatial_radius=1)
    k, l = plan["cache_frames"], plan["band_height"]
    mem = spatiotemporal_memory_bytes(
        sizes, sizes[0], k, l, widths, heights, 1)
    assert mem == plan["memory_bytes"]
    assert mem <= budget
    assert derive_cache_frames(sizes, sizes[0], budget, l, widths, heights,
                               1) >= k
    assert derive_band_height(sizes, sizes[0], k, budget, widths, heights,
                              1) >= l


def test_plan_tight_budget_shrinks_cache():
    sizes, widths, heights = _one_plane_dims()
    frame_bytes = sizes[0]
    # 仅够 1 帧缓存 + 少量带工作区（含 4r 上下文），不够 2 帧
    budget = (frame_bytes + widths[0] * (2 + 4)) * SAFETY
    plan = plan_spatiotemporal(sizes, widths, heights,
                               temporal_radius=4, budget_bytes=budget,
                               spatial_radius=1)
    assert plan["cache_frames"] == 1
    assert plan["band_height"] >= 1
    assert plan["memory_bytes"] <= budget


def test_plan_yuv420_multiplane_bands():
    w, h = 64, 48
    sizes = [w * h, (w // 2) * (h // 2), (w // 2) * (h // 2)]
    widths = [w, w // 2, w // 2]
    heights = [h, h // 2, h // 2]
    plan = plan_spatiotemporal(sizes, widths, heights,
                               temporal_radius=1, budget_bytes=1 << 20,
                               spatial_radius=1, band_height=8)
    assert len(plan["bands"]) == 3
    assert plan["bands"][0] == [(0, 8), (8, 16), (16, 24),
                                (24, 32), (32, 40), (40, 48)]
    assert plan["bands"][1] == [(0, 8), (8, 16), (16, 24)]
    # 空间级分带逐平面存在且覆盖全高
    for sb, ph in zip(plan["spatial_bands"], heights):
        assert sb[0][0] == 0 and sb[-1][1] == ph


def test_plan_validates_arguments():
    sizes, widths, heights = _one_plane_dims()
    with pytest.raises(ValueError):
        plan_spatiotemporal(sizes, widths, heights, temporal_radius=-1,
                            budget_bytes=1 << 20, spatial_radius=1)
    with pytest.raises(ValueError):
        plan_spatiotemporal(sizes, widths, heights, temporal_radius=1,
                            budget_bytes=1 << 20, spatial_radius=-1)
    with pytest.raises(ValueError):
        plan_spatiotemporal(sizes, widths, heights, temporal_radius=1,
                            budget_bytes=0, spatial_radius=1)
    with pytest.raises(ValueError):
        plan_spatiotemporal(sizes, widths, heights, temporal_radius=1,
                            budget_bytes=1 << 20, spatial_radius=1,
                            band_height=0)


# ---------- 指标与可复现 ----------

def test_planar_mad_zero_for_identical_planes():
    plane = textured_plane()
    assert planar_mad(plane, plane) == 0


def test_planar_mad_positive_and_symmetric():
    a = flat_plane(100)
    b = flat_plane(110)
    assert planar_mad(a, b) == 10
    assert planar_mad(a, b) == planar_mad(b, a)


def test_temporal_mad_area_weighted():
    a = [flat_plane(100), flat_plane(100)]
    b = [flat_plane(104), flat_plane(106)]
    # 两组等面积平面：总绝对差 / 总像素 = (4+6)/2 = 5
    assert temporal_mad(a, b) == 5


def test_temporal_mad_empty_returns_zero():
    assert temporal_mad([], []) == 0


def test_full_pipeline_deterministic_bit_exact():
    # 同样的输入与参数跑两遍，process_plane 输出必须逐字节相等
    target = textured_plane(seed=5)
    refs = [textured_plane(seed=5 + k) for k in (1, 2, 3)]
    shifts = [(0, 0), (1, 0), (0, 1)]
    wmaps = [
        smooth_weight_map(
            motion_weight_map(target, r, W, H, s, 30), W, H, 1)
        for r, s in zip(refs, shifts)
    ]
    bands = _full_bands(H)
    kwargs = dict(refs=refs, luma_weights=wmaps, luma_w=W, luma_h=H,
                  width=W, height=H, shifts=shifts, bands=bands,
                  spatial_bands=bands, spatial_radius=1,
                  is_luma=True)
    out1 = process_plane(target, **kwargs)
    out2 = process_plane(target, **kwargs)
    assert out1 == out2


def test_default_constants_sane():
    assert MAX_CACHE_FRAMES >= 3
    assert 1 <= DEFAULT_MOTION_THRESHOLD <= 255
    assert DEFAULT_SPATIAL_RADIUS >= 1
    assert DEFAULT_MASK_RADIUS >= 1
    assert WEIGHT_SCALE == 255
    assert SAFETY >= 1
