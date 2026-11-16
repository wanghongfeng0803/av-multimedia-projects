"""运动轨迹平滑：把逐帧抖动变成补偿用的平滑轨迹，并推导裁剪量。

三步管线的第二步，输入是估计阶段写出的 motion JSON，输出
smooth JSON（补偿与裁剪两个后续步骤共用，裁剪不再重新估计）。

处理顺序（每一步都有确定性的边界规则）：

1. **无效帧插值**：场景切换或低置信帧 ``valid=False``，对三个分量
   分别做线性插值；端点处的无效段向最近有效值外推（平延，不编造
   趋势）；全片无效则退化为全零轨迹。
2. **累积成轨迹**：逐帧运动是相邻帧增量，按刚体变换合成累积轨迹
   （旋转角直接累加；平移按旋转坐标系递推），同时保留增量视角，
   补偿量 = 平滑轨迹 − 原始轨迹。
3. **时域平滑**：默认居中均匀滑动平均（窗口 2r+1，边界截断），
   可选高斯窗（σ 与窗口半径挂钩）。平滑只作用于累积轨迹。
4. **裁剪量推导**：逐帧补偿位移取各方向最大值，旋转角取全片
   最大绝对值按角点轨迹折算，加 1px 安全余量，统一四边 inset；
   同时做下限（0）与上限（不超过短边 45%）钳制。
"""

import math

__all__ = [
    "DEFAULT_RADIUS",
    "GAUSS_SIGMA_RATIO",
    "MAX_CROP_RATIO",
    "CROP_SAFETY_PX",
    "interpolate_invalid",
    "accumulate_trajectory",
    "moving_average",
    "gaussian_weights",
    "smooth_trajectory",
    "rotation_inset",
    "compute_crop",
    "smooth_motion",
]

#: 默认滑动平均半径（帧）；窗口 = 2r+1
DEFAULT_RADIUS = 15
#: 高斯窗 σ = 半径 / 该比值（r=15 时 σ≈5）
GAUSS_SIGMA_RATIO = 3.0
#: 单边裁剪量不允许超过短边的该比例（防止极端素材裁没画面）
MAX_CROP_RATIO = 0.45
#: 裁剪安全余量（原分辨率像素），吸收亚像素与重采样误差
CROP_SAFETY_PX = 1.0


def interpolate_invalid(values, valid):
    """对无效位置做线性插值，端点无效段平延到最近有效值。

    ``values``/``valid`` 等长。返回新 list（float），不改输入。
    全部无效时返回全零。
    """
    n = len(values)
    if n == 0:
        return []
    good = [i for i in range(n) if valid[i]]
    if not good:
        return [0.0] * n

    out = [float(v) for v in values]
    # 端点平延
    first, last = good[0], good[-1]
    for i in range(0, first):
        out[i] = float(values[first])
    for i in range(last + 1, n):
        out[i] = float(values[last])
    # 内部无效段：夹在两个有效值之间线性插值
    for a, b in zip(good, good[1:]):
        if b == a + 1:
            continue
        va, vb = float(values[a]), float(values[b])
        span = b - a
        for i in range(a + 1, b):
            out[i] = va + (vb - va) * (i - a) / span
    return out


def accumulate_trajectory(dx, dy, angle):
    """相邻帧增量 → 累积刚体轨迹。

    返回 ``(tra_x, tra_y, tra_a)`` 三个等长 list，第 0 帧轨迹为 0。
    平移递推按当前累积角旋转：``pₜ = pₜ₋₁ + R(aₜ₋₁)·(dx,dy)``。
    屏幕坐标 y 轴向下、正角为顺时针，对应旋转矩阵
    ``R=[c s; -s c]``（数学逆时针矩阵的 y 翻转形式）；角度直接累加。
    """
    n = len(angle)
    tx, ty, ta = [0.0] * n, [0.0] * n, [0.0] * n
    ax = ay = aa = 0.0
    for t in range(1, n):
        # 先用当前帧增量角更新朝向，再把同帧平移换至世界系，
        # 保证第一段非零旋转后的平移立即按新朝向合成。
        aa += angle[t]
        rad = math.radians(aa)
        c, s = math.cos(rad), math.sin(rad)
        # 屏幕坐标顺时针旋转矩阵：x' = c·x + s·y，y' = -s·x + c·y
        ax += c * dx[t] + s * dy[t]
        ay += -s * dx[t] + c * dy[t]
        tx[t], ty[t], ta[t] = ax, ay, aa
    return tx, ty, ta


def gaussian_weights(radius, sigma):
    """半宽 ``radius`` 的归一化高斯窗权重（居中，长度 2r+1）。"""
    raw = [
        math.exp(-(k * k) / (2.0 * sigma * sigma))
        for k in range(-radius, radius + 1)
    ]
    total = sum(raw)
    return [w / total for w in raw]


def moving_average(seq, radius, weights=None):
    """居中滑动平均；``weights`` 给定时做加权平均（如高斯窗）。

    边界截断：窗口超出序列时只对实际覆盖的权重重新归一化，
    因此首尾帧不会被拉向零。
    """
    n = len(seq)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0 / (2 * radius + 1)] * (2 * radius + 1)
    out = [0.0] * n
    for i in range(n):
        acc, wsum = 0.0, 0.0
        for k, w in enumerate(weights):
            j = i + k - radius
            if 0 <= j < n:
                acc += seq[j] * w
                wsum += w
        out[i] = acc / wsum if wsum else 0.0
    return out


def smooth_trajectory(tra_x, tra_y, tra_a, radius=DEFAULT_RADIUS,
                      method="uniform"):
    """对三条累积轨迹分别做时域平滑，返回平滑后的三元组。

    ``method``：``"uniform"`` 均匀滑动平均（默认）或
    ``"gaussian"`` 高斯加权（σ = radius/3）。
    """
    if radius < 0:
        raise ValueError("平滑半径不能为负")
    if method not in ("uniform", "gaussian"):
        raise ValueError(f"未知平滑方法 {method!r}")
    weights = None
    if method == "gaussian":
        sigma = max(1e-6, radius / GAUSS_SIGMA_RATIO)
        weights = gaussian_weights(radius, sigma)
    return (
        moving_average(tra_x, radius, weights),
        moving_average(tra_y, radius, weights),
        moving_average(tra_a, radius, weights),
    )


def rotation_inset(width, height, max_angle_deg):
    """绕中心旋转最大角度时，四边保证不出现黑边的统一 inset（像素）。

    取角点旋转后在两轴上的最大伸出量：
    ``inset = max(|sin a|·半宽 + (1-cos a)·半高, 交换项)``，
    两个方向取最大保证四边同值。
    """
    a = math.radians(abs(max_angle_deg))
    hw, hh = width / 2.0, height / 2.0
    s, c = abs(math.sin(a)), 1.0 - math.cos(a)
    return max(s * hw + c * hh, s * hh + c * hw)


def compute_crop(width, height, comp_x, comp_y, comp_a,
                 scale=1.0):
    """由逐帧补偿量推导统一四边裁剪 inset（原分辨率像素）。

    - 平移：各帧补偿位移在左右/上下方向的最大伸出量；
    - 旋转：全片最大补偿角按 :func:`rotation_inset` 折算；
    - 余量：``CROP_SAFETY_PX`` 像素，吸收亚像素/重采样误差；
    - 钳制：``[0, 短边·MAX_CROP_RATIO]``，超上限说明补偿量异常，
      保留上限并由调用方在 JSON 中告警。
    """
    max_x = max((abs(v) for v in comp_x), default=0.0)
    max_y = max((abs(v) for v in comp_y), default=0.0)
    max_a = max((abs(v) for v in comp_a), default=0.0)
    inset = max(max_x, max_y) + rotation_inset(width, height, max_a)
    inset = inset * scale + CROP_SAFETY_PX
    cap = min(width, height) * MAX_CROP_RATIO
    clamped = min(inset, cap)
    return {
        "inset": round(clamped, 3),
        "raw_inset": round(inset, 3),
        "clamped": inset > cap,
        "max_shift_x": round(max_x, 3),
        "max_shift_y": round(max_y, 3),
        "max_angle": round(max_a, 4),
    }


def smooth_motion(motion_doc, radius=DEFAULT_RADIUS, method="uniform",
                  scale=1.0):
    """估计 JSON → 平滑 JSON 的完整第二步（纯函数，便于单测）。

    输入为 :func:`stabilize.pipeline.estimate_motion` 写出的 dict，
    返回可直接 JSON 序列化的平滑结果 dict，含：

    - ``frames``：每帧平滑轨迹与补偿量（原分辨率，已乘缩放比）；
    - ``crop``：统一四边 inset 与诊断字段；
    - ``params``：平滑参数与无效帧统计。
    """
    src = motion_doc
    frames_in = src["frames"]
    n = len(frames_in)
    width, height = src["width"], src["height"]
    est_scale = src.get("downsample_scale", 1.0)

    valid = [bool(f.get("valid", True)) for f in frames_in]
    dx_small = [f["dx"] for f in frames_in]
    dy_small = [f["dy"] for f in frames_in]
    ang = [f["angle"] for f in frames_in]

    dx_s = interpolate_invalid(dx_small, valid)
    dy_s = interpolate_invalid(dy_small, valid)
    ang_s = interpolate_invalid(ang, valid)

    # 小图位移还原到原分辨率（角度与尺度无关）
    dx = [v * est_scale for v in dx_s]
    dy = [v * est_scale for v in dy_s]

    tra_x, tra_y, tra_a = accumulate_trajectory(dx, dy, ang_s)
    sm_x, sm_y, sm_a = smooth_trajectory(
        tra_x, tra_y, tra_a, radius, method
    )
    # 补偿量 = 平滑轨迹 − 原始轨迹：把画面反向推回平滑路径
    comp_x = [s - r for s, r in zip(sm_x, tra_x)]
    comp_y = [s - r for s, r in zip(sm_y, tra_y)]
    comp_a = [s - r for s, r in zip(sm_a, tra_a)]

    out_frames = [
        {
            "index": t,
            "trajectory": {
                "x": round(tra_x[t], 4),
                "y": round(tra_y[t], 4),
                "angle": round(tra_a[t], 5),
            },
            "smoothed": {
                "x": round(sm_x[t], 4),
                "y": round(sm_y[t], 4),
                "angle": round(sm_a[t], 5),
            },
            "compensation": {
                "dx": round(comp_x[t], 4),
                "dy": round(comp_y[t], 4),
                "angle": round(comp_a[t], 5),
            },
            "valid": valid[t],
        }
        for t in range(n)
    ]
    crop = compute_crop(width, height, comp_x, comp_y, comp_a, scale)
    return {
        "format": "stabilize-smooth",
        "version": 1,
        "source": src.get("source"),
        "width": width,
        "height": height,
        "fps": src.get("fps"),
        "frames": out_frames,
        "crop": crop,
        "params": {
            "radius": radius,
            "method": method,
            "frame_count": n,
            "invalid_frames": n - sum(valid),
        },
    }
