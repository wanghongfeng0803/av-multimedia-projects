"""运动估计纯逻辑测试：合成已知平移/旋转的灰度帧验证估计精度。

不依赖 ffmpeg：测试帧在 Python 内用显式公式生成（棋盘 + 径向
亮斑，纹理足够支撑块匹配与环形签名）。
"""

import math

import pytest

from stabilize.motion import (
    bilinear_sample,
    block_match,
    downsample,
    estimate_pair,
    estimate_rotation,
    is_scene_cut,
    parabolic_refine,
    ring_signature,
)


def _pattern(x, y):
    """合成纹理：棋盘格 + 两个偏置径向亮斑，值夹到 [0,255]。"""
    v = 110.0
    v += 45.0 * ((int(x // 8) + int(y // 8)) % 2 * 2 - 1)
    v += 60.0 * math.exp(-(((x - 20) ** 2 + (y - 16) ** 2) / 80.0))
    v -= 50.0 * math.exp(-(((x - 44) ** 2 + (y - 36) ** 2) / 60.0))
    return max(0, min(255, int(round(v))))


def _base_frame(w=64, h=48):
    return bytearray(_pattern(x, y) for y in range(h) for x in range(w))


def _shift_frame(base, w, h, dx, dy):
    """按 ref→cur 平移 (dx,dy) 生成 cur：cur(x,y)=ref(x-dx,y-dy)。"""
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            out[y * w + x] = int(round(
                bilinear_sample(base, w, h, x - dx, y - dy)
            ))
    return bytes(out)


def _rotate_frame(base, w, h, angle_deg, dx=0.0, dy=0.0):
    """绕中心顺时针旋转 angle 度并平移，生成 cur 帧。"""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            # cur 点 (x,y) 对应 ref 的逆变换坐标（标准逆旋转矩阵）
            dxp, dyp = x - cx - dx, y - cy - dy
            xr = cx + c * dxp + s * dyp
            yr = cy - s * dxp + c * dyp
            out[y * w + x] = int(round(
                bilinear_sample(base, w, h, xr, yr)
            ))
    return bytes(out)


# ---------------------------------------------------------- 基础工具

def test_downsample_shape_and_scale():
    frame = bytes(range(256)) * 16  # 64x64
    small, w, h, step = downsample(frame, 64, 64, target_width=16)
    assert step == 4 and (w, h) == (16, 16)
    assert len(small) == 256
    small2, w2, h2, step2 = downsample(frame, 64, 64, target_width=64)
    assert step2 == 1 and (w2, h2) == (64, 64) and small2 == frame


def test_parabolic_refine_symmetry():
    assert parabolic_refine(1.0, 2.0, 2.0) == pytest.approx(0.0)
    # 右侧代价更低 → 顶点偏正方向
    delta = parabolic_refine(1.0, 2.0, 1.5)
    assert 0 < delta < 1
    # 退化共线 → 夹回 0
    assert parabolic_refine(1.0, 1.0, 1.0) == 0.0


def test_bilinear_edges_clamp():
    frame = bytes([10, 20, 30, 40])  # 2x2
    assert bilinear_sample(frame, 2, 2, -5, -5) == 10.0
    assert bilinear_sample(frame, 2, 2, 5, 5) == 40.0
    assert bilinear_sample(frame, 2, 2, 0, 0) == 10.0


# ---------------------------------------------------------- 平移

def test_block_match_integer_shift():
    base = _base_frame()
    cur = _shift_frame(base, 64, 48, 3, -2)
    dx, dy, sad, second = block_match(base, cur, 64, 48, max_shift=8)
    assert (dx, dy) == (3, -2)
    assert sad < 2.0  # 整数位移在纹理区几乎完全对齐


def test_block_match_zero_shift_identity():
    base = _base_frame()
    dx, dy, sad, _ = block_match(base, bytes(base), 64, 48, max_shift=5)
    assert (dx, dy) == (0, 0)
    assert sad == pytest.approx(0.0)


def test_estimate_pair_subpixel_translation():
    base = _base_frame()
    cur = _shift_frame(base, 64, 48, 2.5, -1.5)
    r = estimate_pair(base, cur, 64, 48)
    assert r["valid"]
    assert r["dx"] == pytest.approx(2.5, abs=0.35)
    assert r["dy"] == pytest.approx(-1.5, abs=0.35)
    assert abs(r["angle"]) < 1.5  # 无旋转时角度应接近 0


# ---------------------------------------------------------- 旋转

def test_ring_signature_rotation_invariance_shape():
    base = _base_frame()
    sig = ring_signature(base, 64, 48)
    assert len(sig) == 12 and all(len(row) == 128 for row in sig)
    # 去均值：每环均值≈0
    for row in sig:
        assert abs(sum(row) / len(row)) < 1e-9


def test_estimate_rotation_positive_angle():
    base = _base_frame()
    rotated = _rotate_frame(base, 64, 48, 6.0)
    angle, ratio = estimate_rotation(base, rotated, 64, 48)
    assert ratio > 1.3
    assert angle == pytest.approx(6.0, abs=1.2)


def test_estimate_rotation_negative_angle():
    base = _base_frame()
    rotated = _rotate_frame(base, 64, 48, -5.0)
    angle, ratio = estimate_rotation(base, rotated, 64, 48)
    assert ratio > 1.3
    assert angle == pytest.approx(-5.0, abs=1.2)


def test_estimate_pair_combined_motion():
    base = _base_frame()
    # 平移 2 像素 + 顺时针 4 度
    cur = _rotate_frame(base, 64, 48, 4.0, dx=2.0, dy=-1.5)
    r = estimate_pair(base, cur, 64, 48)
    assert r["valid"]
    assert r["angle"] == pytest.approx(4.0, abs=1.5)
    assert r["dx"] == pytest.approx(2.0, abs=1.0)
    assert r["dy"] == pytest.approx(-1.5, abs=1.0)


# ---------------------------------------------------------- 场景切换

def test_scene_cut_detection():
    base = _base_frame()
    same = bytes(base)
    # 反相画面：每点灰度差都很大，必然超过场景切换阈值
    inverted = bytes(255 - v for v in base)
    assert is_scene_cut(base, same, 64, 48) is False
    assert is_scene_cut(base, inverted, 64, 48) is True


def test_estimate_pair_marks_low_confidence_on_noise():
    # 两帧独立随机噪声：任何位移都没有可信度
    import random
    random.seed(7)
    a = bytes(random.randrange(256) for _ in range(64 * 48))
    b = bytes(random.randrange(256) for _ in range(64 * 48))
    r = estimate_pair(a, b, 64, 48)
    assert r["valid"] is False
    assert 0.0 <= r["confidence"] <= 1.0
