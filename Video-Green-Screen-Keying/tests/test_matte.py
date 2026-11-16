"""层 2 alpha 后处理纯逻辑测试：硬化、choke、羽化、顺序与 despill 刷新。"""

import pytest

from vkeyer.matte import (
    MatteConfig,
    apply_matte_to_frame,
    choke_alpha,
    feather_alpha,
    feather_alpha_row,
    postprocess_alpha,
    reflect101,
    threshold_alpha,
)


# ---------- 阈值硬化 ----------

def test_threshold_snaps_low_and_high():
    alpha = bytes([0, 5, 8, 100, 247, 250, 255])
    out = threshold_alpha(alpha, 7, 1, low=8, high=247)
    assert list(out) == [0, 0, 0, 100, 255, 255, 255]


def test_threshold_validation():
    with pytest.raises(ValueError):
        threshold_alpha(bytes(4), 2, 2, low=200, high=100)


# ---------- REFLECT_101 镜像 ----------

def test_reflect101_basic():
    # 长度 4：有效下标 0..3，反射序列 ... 1 0 0 1 2 3 3 2 ...
    assert reflect101(0, 4) == 0
    assert reflect101(3, 4) == 3
    assert reflect101(-1, 4) == 1
    assert reflect101(4, 4) == 2
    assert reflect101(-2, 4) == 2
    assert reflect101(5, 4) == 1
    # 任意深度都收敛回有效范围
    for idx in range(-20, 24):
        assert 0 <= reflect101(idx, 4) < 4


# ---------- choke ----------

def test_choke_shrinks_foreground_block():
    # 5x5 中心 3x3 为前景，其余透明
    alpha = bytearray(25)
    for y in range(1, 4):
        for x in range(1, 4):
            alpha[y * 5 + x] = 255
    out = choke_alpha(bytes(alpha), 5, 5, radius=1)
    # 收缩一圈后只剩中心 1x1
    assert out[2 * 5 + 2] == 255
    assert out[1 * 5 + 2] == 0
    assert out[2 * 5 + 1] == 0
    # 原本透明的角落不会被扩张（min 滤波保持 0）
    assert out[0] == 0


def test_negative_choke_expands():
    alpha = bytearray(25)
    alpha[2 * 5 + 2] = 255
    out = choke_alpha(bytes(alpha), 5, 5, radius=-1)
    # 扩张一圈：中心 3x3 全亮
    for y in range(1, 4):
        for x in range(1, 4):
            assert out[y * 5 + x] == 255
    assert out[0] == 0


def test_choke_zero_is_copy():
    alpha = bytes([1, 2, 3, 4])
    assert choke_alpha(alpha, 2, 2, radius=0) == alpha


# ---------- 羽化 ----------

def test_feather_row_constant_plane_unchanged():
    row = [100] * 8
    assert feather_alpha_row(row, 2) == row


def test_feather_row_matches_manual_average():
    row = [0, 0, 100, 100, 0, 0]
    out = feather_alpha_row(row, 1)
    # x=0 邻域 [0,1]：均值 0
    assert out[0] == 0
    # x=2 邻域 [1,3]：(0+100+100)/3 = 67（四舍五入 66.67→67）
    assert out[2] == 67


def test_feather_full_constant_identity():
    alpha = bytes([128] * 36)
    out = feather_alpha(alpha, 6, 6, radius=2)
    assert out == alpha


def test_feather_smoothes_block_edge():
    # 左半 255 右半 0，羽化后边界出现中间灰且两侧核心不变
    alpha = bytearray(8 * 8)
    for y in range(8):
        for x in range(4):
            alpha[y * 8 + x] = 255
    out = feather_alpha(bytes(alpha), 8, 8, radius=1)
    assert out[0] == 255          # 左核心
    assert out[7] == 0            # 右核心
    edge_values = {out[y * 8 + 3] for y in range(1, 7)}
    assert edge_values and all(0 < v < 255 for v in edge_values)


def test_feather_zero_radius_passthrough():
    alpha = bytes([10, 20, 30, 40])
    assert feather_alpha(alpha, 2, 2, radius=0) == alpha


# ---------- 配置与固定顺序 ----------

def test_matte_config_validation():
    MatteConfig()
    with pytest.raises(ValueError):
        MatteConfig(threshold_low=200, threshold_high=100)
    with pytest.raises(ValueError):
        MatteConfig(choke=-1)
    with pytest.raises(ValueError):
        MatteConfig(feather=-1)


def test_matte_identity_skips_processing():
    cfg = MatteConfig()
    assert cfg.is_identity
    alpha = bytes([0, 128, 255, 64])
    assert postprocess_alpha(alpha, 2, 2, cfg) == alpha


def test_postprocess_order_threshold_then_choke_then_feather():
    """顺序锁定：先硬化（产生二值）→ choke 收缩 → 羽化抹边。

    用 255 的实心块验证：硬化后为二值，choke 收缩一圈，羽化再把
    收缩后的硬边抹成渐变；中心核心保持不透明、外缘出现中间灰。
    """
    alpha = bytearray(7 * 7)
    for y in range(1, 6):
        for x in range(1, 6):
            alpha[y * 7 + x] = 255
    cfg = MatteConfig(threshold_low=8, threshold_high=247,
                      choke=1, feather=1)
    out = postprocess_alpha(bytes(alpha), 7, 7, cfg)
    assert out[3 * 7 + 3] == 255          # 中心核心不透明
    assert 0 < out[1 * 7 + 1] < 255       # 外缘被羽化出中间灰
    assert out[0] == 0                    # 角落始终透明


def test_postprocess_threshold_snaps_before_choke():
    """硬化先于 choke：247 的近不透明噪点先被钳成 255，choke 才不会
    因 min 窗把它当成半透明弱点侵蚀掉。"""
    # 5x5：中心一个 5（真透明洞），其余全是 247（应硬化为 255）
    alpha = bytearray(25)
    for i in range(25):
        alpha[i] = 247
    alpha[12] = 5
    cfg = MatteConfig(threshold_low=8, threshold_high=247, choke=1)
    out = postprocess_alpha(bytes(alpha), 5, 5, cfg)
    # 角落的 3x3 邻域不接触中心洞，硬化后 choke 仍保持 255
    assert out[0] == 255 and out[4] == 255
    assert out[20] == 255 and out[24] == 255
    # 中心透明洞被 min 滤波保持
    assert out[12] == 0


# ---------- 遮罩写回与 despill 刷新 ----------

def test_apply_matte_updates_alpha_and_rgb_together():
    # RGBA 帧：一个带绿色调的半透明像素
    rgba = bytearray([60, 150, 60, 128])
    alpha = bytes([40])  # 后处理把 alpha 改得更透明
    out = apply_matte_to_frame(bytes(rgba), 1, 1, alpha, (0, 200, 0))
    assert out[3] == 40
    # alpha 变小 → despill 权重变大 → G 被压得比原来更低
    original_g = 150
    assert out[1] < original_g


def test_apply_matte_opaque_pixel_rgb_untouched():
    rgba = bytearray([200, 30, 30, 255])
    out = apply_matte_to_frame(bytes(rgba), 1, 1, bytes([255]),
                               (0, 200, 0))
    assert bytes(out[:3]) == bytes([200, 30, 30])
