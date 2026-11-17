"""层 1 新增参数测试：容差（tolerance）与溢色抑制强度
（despill_strength）的独立性、边缘语义、默认值逐像素恒等、
运行中原子换参与受影响边缘像素量化。"""

import threading

import pytest

from vkeyer.keyer import (
    EdgeStats,
    Keyer,
    KeyerConfig,
    despill_weight,
)
from vkeyer.matte import apply_matte_to_frame
from tests.conftest import make_rgb_frame

GREEN = (0, 200, 0)

#: 黄金帧：幕布纯色 / 远处前景 / 软带绿边 / 软带外前景 各一像素
GOLDEN_PIXELS = [(0, 200, 0), (200, 30, 30), (40, 180, 40), (90, 80, 40)]
GOLDEN_FRAME = bytes(c for px in GOLDEN_PIXELS for c in px)
#: 旧版本（无 tolerance / despill_strength 概念时）的正确输出，
#: 由旧公式手工核定：sim=32, soft=64, despill 权重 255-alpha
GOLDEN_RGBA = bytes([
    0, 0, 0, 0,          # 幕布色：全透明，RGB 被中和
    200, 30, 30, 255,    # 远处前景：不透明，RGB 不动
    40, 101, 40, 112,    # 软带绿边：alpha=112，G 被压到 101
    90, 80, 40, 255,     # 软带外：不透明，RGB 不动
])


# ---------- 默认值逐像素恒等 ----------

def test_default_params_match_old_version_golden_bytes():
    """默认配置（tolerance=0, despill_strength=1）输出钉死在旧版黄金字节上。"""
    keyer = Keyer(KeyerConfig(GREEN, similarity=32, softness=64))
    assert bytes(keyer.key_frame(GOLDEN_FRAME, 4, 1)) == GOLDEN_RGBA


def test_explicit_neutral_params_equal_defaults():
    """显式给中性值（t=0, s=1）与纯默认构造输出完全一致。"""
    a = Keyer(KeyerConfig(GREEN, similarity=32, softness=64))
    b = Keyer(KeyerConfig(GREEN, similarity=32, softness=64,
                          tolerance=0.0, despill_strength=1.0))
    assert a.key_frame(GOLDEN_FRAME, 4, 1) == b.key_frame(GOLDEN_FRAME, 4, 1)


def test_despill_weight_neutral_strength_is_legacy():
    """strength_q=255（强度 1.0）时权重精确退化为旧版 255-alpha。"""
    for alpha in (0, 1, 100, 254, 255):
        assert despill_weight(alpha, 255) == 255 - alpha
    assert despill_weight(100, 0) == 0


def test_new_param_validation():
    with pytest.raises(ValueError):
        KeyerConfig(GREEN, tolerance=-0.1)
    with pytest.raises(ValueError):
        KeyerConfig(GREEN, tolerance=1.1)
    with pytest.raises(ValueError):
        KeyerConfig(GREEN, despill_strength=-0.1)
    with pytest.raises(ValueError):
        KeyerConfig(GREEN, despill_strength=1.1)


# ---------- 容差：连续映射，不硬切半透明边缘 ----------

def test_tolerance_scales_alpha_continuously_no_hard_cut():
    """软带中段像素（发丝级边缘）：容差增大时 alpha 平滑单调下移，
    中等容差内始终保持半透明，不会被齐根切成 0。"""
    frame = make_rgb_frame(1, 1, lambda x, y: (64, 200, 0))  # d=64，带中央
    alphas = []
    for t in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4):
        keyer = Keyer(KeyerConfig(GREEN, similarity=32, softness=64,
                                  tolerance=t))
        a = keyer.key_frame(frame, 1, 1)[3]
        assert 0 < a < 255, f"tolerance={t} 不应硬切半透明边缘"
        alphas.append(a)
    assert alphas == sorted(alphas, reverse=True)
    assert len(set(alphas)) == len(alphas)  # 连续变化，无平台/跳变


def test_tolerance_extreme_values():
    """tolerance=1 时所有距离归零、全帧透明；tolerance=0 即旧版。"""
    frame = make_rgb_frame(2, 1, lambda x, y: (200, 30, 30) if x else GREEN)
    keyer = Keyer(KeyerConfig(GREEN, tolerance=1.0))
    out = keyer.key_frame(frame, 2, 1)
    assert out[3] == 0 and out[7] == 0


def test_tolerance_does_not_change_despill_strength_semantics():
    """两参数独立：只调容差时，同一像素上强度 0/1 的相对效果不变。"""
    px = (60, 150, 60)
    frame = make_rgb_frame(1, 1, lambda x, y: px)
    for t in (0.0, 0.2):
        off = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                                tolerance=t, despill_strength=0.0))
        full = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                                 tolerance=t, despill_strength=1.0))
        o_off = off.key_frame(frame, 1, 1)
        o_full = full.key_frame(frame, 1, 1)
        assert o_off[3] == o_full[3]          # 强度不动 alpha
        assert o_off[1] == 150 > o_full[1]    # 强度只动 RGB


# ---------- 溢色抑制强度：独立可调，不污染非绿前景 ----------

def test_despill_strength_zero_equals_disabled():
    """strength=0 与 despill=False 逐像素一致。"""
    frame = make_rgb_frame(1, 1, lambda x, y: (60, 150, 60))
    a = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                          despill_strength=0.0))
    b = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                          despill=False))
    assert a.key_frame(frame, 1, 1) == b.key_frame(frame, 1, 1)


def test_despill_strength_scales_suppression_linearly():
    """同一绿边像素：G 的下压量随强度分级（150→120→90），alpha 不动。"""
    frame = make_rgb_frame(1, 1, lambda x, y: (60, 150, 60))
    results = {}
    for s in (0.0, 0.5, 1.0):
        keyer = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                                  despill_strength=s))
        results[s] = keyer.key_frame(frame, 1, 1)
    assert results[0.0][1] == 150
    assert results[0.5][1] == 120
    assert results[1.0][1] == 90
    assert results[0.0][3] == results[0.5][3] == results[1.0][3] == 85


def test_despill_never_pollutes_skin_or_gray_foreground():
    """肤色（R 主导）与灰衣（无主导）即使落在半透明边缘上，
    任意强度下 RGB 都零改动——溢出抑制只压键色主导通道。"""
    keyer = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                              despill_strength=1.0))
    for px in ((180, 150, 110), (120, 120, 120)):
        out = keyer.key_frame(make_rgb_frame(1, 1, lambda x, y: px), 1, 1)
        assert 0 < out[3] < 255            # 确为半透明边缘像素
        assert (out[0], out[1], out[2]) == px


# ---------- 受影响边缘像素量化 ----------

def test_analyze_frame_default_baseline():
    """默认配置：容差影响恒为 0；抑制只数实际被改写的像素。"""
    keyer = Keyer(KeyerConfig(GREEN, similarity=32, softness=64))
    stats = keyer.analyze_frame(GOLDEN_FRAME, 4, 1)
    assert stats == EdgeStats(
        pixels=4, edge_pixels=1, transparent_pixels=1, opaque_pixels=2,
        tolerance_affected=0, despill_affected=2,
    )


def test_analyze_frame_quantifies_each_param_independently():
    """同一输入上，两参数各自的受影响边缘像素数独立可量化。"""
    # 肤色边缘 + 灰边缘 + 绿边：只有绿边被抑制改写
    frame = bytes((180, 150, 110) + (120, 120, 120) + (60, 150, 60))
    base = Keyer(KeyerConfig(GREEN, similarity=32, softness=200))
    stats = base.analyze_frame(frame, 3, 1)
    assert stats.edge_pixels == 3
    assert stats.despill_affected == 1      # 只有绿边 (60,150,60)
    assert stats.tolerance_affected == 0

    # 只调强度：容差计数不变，抑制计数随强度 0 归零
    weak = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                             despill_strength=0.0))
    stats_w = weak.analyze_frame(frame, 3, 1)
    assert stats_w.tolerance_affected == 0
    assert stats_w.despill_affected == 0

    # 只调容差：alpha 变化被单独计数
    tol = Keyer(KeyerConfig(GREEN, similarity=32, softness=200,
                            tolerance=0.3))
    stats_t = tol.analyze_frame(frame, 3, 1)
    assert stats_t.tolerance_affected > 0
    assert stats_t.despill_affected >= 0  # 两计数各自独立可读


def test_analyze_frame_validates_length():
    keyer = Keyer(KeyerConfig(GREEN))
    with pytest.raises(ValueError):
        keyer.analyze_frame(b"\x00" * 10, 4, 4)


# ---------- 运行中原子换参 ----------

def _varied_frame(width=32, height=32):
    """含幕布、前景、软带、肤色、灰色的确定性混合帧。"""
    palette = [GREEN, (200, 30, 30), (40, 180, 40), (64, 200, 0),
               (180, 150, 110), (120, 120, 120), (60, 150, 60)]
    return make_rgb_frame(width, height,
                          lambda x, y: palette[(x + y * width) % len(palette)])


def test_update_config_takes_effect_next_frame():
    frame = _varied_frame()
    cfg_a = KeyerConfig(GREEN, similarity=32, softness=64)
    cfg_b = KeyerConfig(GREEN, similarity=32, softness=64,
                        tolerance=0.4, despill_strength=0.5)
    keyer = Keyer(cfg_a)
    assert keyer.config is cfg_a
    assert keyer.key_frame(frame, 32, 32) == Keyer(cfg_a).key_frame(frame, 32, 32)
    keyer.update_config(cfg_b)
    assert keyer.config is cfg_b
    assert keyer.key_frame(frame, 32, 32) == Keyer(cfg_b).key_frame(frame, 32, 32)


def test_update_config_atomic_no_torn_frames():
    """换参与键控并发：每一帧输出要么全旧、要么全新，绝无中间脏帧。"""
    frame = _varied_frame()
    cfg_a = KeyerConfig(GREEN, similarity=32, softness=64)
    cfg_b = KeyerConfig(GREEN, similarity=48, softness=32,
                        tolerance=0.3, despill_strength=0.5)
    keyer = Keyer(cfg_a)
    out_a = Keyer(cfg_a).key_frame(frame, 32, 32)
    out_b = Keyer(cfg_b).key_frame(frame, 32, 32)
    assert out_a != out_b  # 两份配置确实产出不同帧，检验才有意义

    torn = []
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            out = keyer.key_frame(frame, 32, 32)
            if out != out_a and out != out_b:
                torn.append(bytes(out))

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(300):
        keyer.update_config(cfg_b)
        keyer.update_config(cfg_a)
    stop.set()
    t.join()
    assert not torn


def test_update_config_invalid_leaves_old_config_active():
    """非法配置在 KeyerConfig 构造期即抛错，当前配置不被污染。"""
    keyer = Keyer(KeyerConfig(GREEN, similarity=32, softness=64))
    with pytest.raises(ValueError):
        KeyerConfig(GREEN, tolerance=2.0)
    assert keyer.config.tolerance == 0.0


# ---------- 层 2 刷新与层 1 同强度 ----------

def test_matte_refresh_respects_despill_strength():
    """apply_matte_to_frame 的强度语义与层 1 一致：默认 1.0 与旧版
    逐像素相同，0 不改 RGB，中间强度下压量减半。"""
    key_rgb = GREEN
    # 一个软带绿边像素：层 1 输出 + 层 2 新 alpha=128
    rgba = bytearray([60, 150, 60, 85])
    alpha = bytes([128])

    legacy = apply_matte_to_frame(rgba, 1, 1, alpha, key_rgb)  # 默认强度
    assert legacy[3] == 128
    # 旧版权重 255-128=127：G 下压 (127*90+127)//255=45 → 105
    assert legacy[1] == 105

    off = apply_matte_to_frame(rgba, 1, 1, alpha, key_rgb,
                               despill_strength=0.0)
    assert (off[0], off[1], off[2]) == (60, 150, 60)

    half = apply_matte_to_frame(rgba, 1, 1, alpha, key_rgb,
                                despill_strength=0.5)
    assert legacy[1] < half[1] < off[1]

    with pytest.raises(ValueError):
        apply_matte_to_frame(rgba, 1, 1, alpha, key_rgb,
                             despill_strength=1.5)
