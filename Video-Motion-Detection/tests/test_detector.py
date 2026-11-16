"""运动检测器测试：局部自适应门控、光照突变抑制、去抖迟滞。

合成平面直接喂 :class:`MotionDetector`，不依赖 ffmpeg。
"""

import pytest

from vmotion.detector import MotionConfig, MotionDetector
from vmotion.morphology import Region


def _plane(w, h, fill):
    return bytearray([fill]) * (w * h)


def _rect(plane, w, x, y, rw, rh, value):
    for row in range(y, y + rh):
        base = row * w + x
        for col in range(rw):
            plane[base + col] = value


def _cfg(**over):
    """小帧测试专用默认：8x8 块、关掉形态学与面积门槛，先验纯门控。"""
    base = dict(block=8, local_radius=2, pixel_threshold=10.0, gain=3.0,
                min_area=1, open_radius=0, close_radius=0,
                trigger_frames=2, release_frames=2,
                illum_change_threshold=25.0, illum_hold_frames=2)
    base.update(over)
    return MotionConfig(**base)


# --------------------------------------------------------------- 基础运动
def test_first_frame_establishes_reference_without_motion():
    det = MotionDetector(32, 32, _cfg())
    out = det.update(_plane(32, 32, 40), 0, 0.0)
    assert out.motion is False
    assert out.regions == [] and sum(out.mask) == 0


def test_static_scene_never_triggers():
    det = MotionDetector(32, 32, _cfg())
    det.update(_plane(32, 32, 40), 0)
    for i in range(1, 8):
        out = det.update(_plane(32, 32, 40), i)
        assert out.motion is False
        assert out.stats["gated_pixels"] == 0


def test_moving_block_detected_with_region_bbox():
    cfg = _cfg()
    det = MotionDetector(64, 64, cfg)
    det.update(_plane(64, 64, 30), 0)
    # 第 1、2 帧：方块出现在不同位置（16x16 亮块，跨 2x2 个块）
    f1 = _plane(64, 64, 30)
    _rect(f1, 64, 8, 8, 16, 16, 200)
    r1 = det.update(f1, 1, 0.1)
    assert r1.motion is False          # trigger_frames=2，首帧去抖
    f2 = _plane(64, 64, 30)
    _rect(f2, 64, 24, 8, 16, 16, 200)
    r2 = det.update(f2, 2, 0.2)
    assert r2.motion is True
    assert len(r2.regions) >= 1
    region = r2.regions[0]
    assert isinstance(region, Region)
    # 区域落在方块活动带内（外接框覆盖新位置附近）
    x, y, rw, rh = region.bbox
    assert x >= 8 and y <= 16 and rw > 0 and rh > 0
    assert region.area >= cfg.min_area


def test_trigger_debounce_ignores_single_frame_spike():
    det = MotionDetector(32, 32, _cfg(trigger_frames=3))
    det.update(_plane(32, 32, 30), 0)
    spike = _plane(32, 32, 30)
    _rect(spike, 32, 0, 0, 16, 16, 220)
    r1 = det.update(spike, 1)
    r2 = det.update(_plane(32, 32, 30), 2)
    assert r1.motion is False and r2.motion is False


def test_release_hysteresis_keeps_state_through_short_pause():
    det = MotionDetector(64, 64, _cfg(release_frames=3))
    det.update(_plane(64, 64, 30), 0)
    positions = [8, 24, 40]
    for k, px in enumerate(positions, start=1):
        f = _plane(64, 64, 30)
        _rect(f, 64, px, 8, 16, 16, 200)
        det.update(f, k)
    assert det._motion is True
    # 帧 4、5：miss_run 1、2 < release_frames=3，迟滞保持运动态
    for k in (4, 5):
        out = det.update(_plane(64, 64, 30), k)
        assert out.motion is True
    # 帧 6：第三个无命中帧，miss_run 到 3 触发清位（本帧状态机末尾生效）
    det.update(_plane(64, 64, 30), 6)
    # 帧 7：对外稳定为非运动态
    out = det.update(_plane(64, 64, 30), 7)
    assert out.motion is False


# --------------------------------------------------------------- 光照处理
def test_global_illumination_change_is_fully_suppressed():
    """整帧提亮 80：holdoff 帧空掩码，解除后静态画面也不报警。"""
    cfg = _cfg(illum_change_threshold=25.0, illum_hold_frames=2)
    det = MotionDetector(32, 32, cfg)
    det.update(_plane(32, 32, 40), 0)
    bright = _plane(32, 32, 120)       # 全局 +80，无任何物体移动
    r1 = det.update(bright, 1)
    assert r1.stats["held_off"] is True
    assert r1.motion is False and sum(r1.mask) == 0
    r2 = det.update(bright, 2)         # holdoff 第 2 帧
    assert r2.stats["held_off"] is True
    r3 = det.update(bright, 3)         # 解除后第一帧，参考已冻结在暗帧
    assert r3.stats["held_off"] is False
    # 局部 MAD 在全帧均匀变化下处处很大 → 门控后无命中
    assert r3.stats["gated_pixels"] == 0
    assert r3.motion is False
    r4 = det.update(bright, 4)
    assert r4.motion is False


def test_repeated_illumination_changes_rearm_holdoff():
    det = MotionDetector(32, 32, _cfg(illum_hold_frames=1))
    det.update(_plane(32, 32, 40), 0)
    det.update(_plane(32, 32, 120), 1)   # 突变，hold 1 帧
    mid = det.update(_plane(32, 32, 120), 2)
    assert mid.stats["held_off"] is False
    again = det.update(_plane(32, 32, 40), 3)  # 再次突变（-80）
    assert again.stats["held_off"] is True


def test_local_illumination_change_is_detected():
    """局部提亮（车灯模型）：只有少数块变化，邻域外阈值不抬升 → 检出。

    运动检测响应的是「变化沿」，因此光斑逐帧移动；同样幅度的全局
    变化会被 MAD 门控拦死，而局部变化的受影响块能越过自适应阈值。
    """
    cfg = _cfg(illum_change_threshold=25.0, local_radius=1, gain=2.0)
    det = MotionDetector(64, 64, cfg)
    det.update(_plane(64, 64, 30), 0)
    # 仅左上 16x16（2x2 块）提亮 90，其余区域不动
    local1 = _plane(64, 64, 30)
    _rect(local1, 64, 0, 0, 16, 16, 120)
    r1 = det.update(local1, 1)
    local2 = _plane(64, 64, 30)
    _rect(local2, 64, 16, 0, 16, 16, 120)   # 光斑右移一格
    r2 = det.update(local2, 2)
    assert r2.motion is True
    # 全局均值变化 = 90 * (16*16)/(64*64) ≈ 5.6，不应触发全局 holdoff
    assert r1.stats["held_off"] is False
    assert abs(r1.stats["mean_delta"]) < cfg.illum_change_threshold
    # 命中块集中在左侧活动带内
    hit_xs = [rg.bbox[0] for rg in r2.regions]
    assert all(x < 40 for x in hit_xs)


def test_local_change_detectable_while_equivalent_global_change_blocked():
    """同幅度变化的对照实验：局部能检出，全局被门控，证明非全局归一化。"""
    cfg = _cfg(illum_change_threshold=255.0, local_radius=1, gain=2.0)

    def run(first, second):
        det = MotionDetector(64, 64, cfg)
        det.update(first, 0)
        det.update(second, 1)
        return det.update(second, 2)

    base = _plane(64, 64, 30)
    # 全局 +30：每个邻域 MAD 都抬高，门控后零命中
    g2 = _plane(64, 64, 60)
    g_out = run(base, g2)
    assert g_out.stats["gated_pixels"] == 0

    # 局部 +90（远强于全局那 30）：受影响块邻域外阈值不动，应能命中
    l1 = _plane(64, 64, 30)
    _rect(l1, 64, 0, 0, 16, 16, 120)
    l2 = _plane(64, 64, 30)
    _rect(l2, 64, 16, 0, 16, 16, 120)
    det = MotionDetector(64, 64, cfg)
    det.update(base, 0)
    det.update(l1, 1)
    l_out = det.update(l2, 2)
    assert l_out.stats["gated_pixels"] > 0


def test_local_mad_gate_blocks_uniform_global_shift_without_holdoff():
    """把 holdoff 阈值调到不可能触发，单独验证 MAD 门控对全局变化有效。"""
    cfg = _cfg(illum_change_threshold=255.0, gain=3.0)
    det = MotionDetector(32, 32, cfg)
    det.update(_plane(32, 32, 40), 0)
    shifted = _plane(32, 32, 70)       # 全局 +30，holdoff 不会触发
    out = det.update(shifted, 1)
    assert out.stats["held_off"] is False
    assert out.stats["raw_pixels"] > 0          # 硬阈值确实被越过
    assert out.stats["gated_pixels"] == 0       # 但自适应门控全部拦下
    assert out.motion is False


# --------------------------------------------------------------- 参数与契约
def test_thresholds_are_configurable():
    strict = _cfg(pixel_threshold=200.0)
    det = MotionDetector(32, 32, strict)
    det.update(_plane(32, 32, 30), 0)
    f = _plane(32, 32, 30)
    _rect(f, 32, 0, 0, 16, 16, 120)   # 差值 90 < 200
    out = det.update(f, 1)
    assert out.stats["gated_pixels"] == 0

    loose = _cfg(pixel_threshold=20.0)
    det2 = MotionDetector(32, 32, loose)
    det2.update(_plane(32, 32, 30), 0)
    out2 = det2.update(f, 1)
    assert out2.stats["raw_pixels"] > 0


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        MotionConfig(block=0).validate()
    with pytest.raises(ValueError):
        MotionConfig(pixel_threshold=300).validate()
    with pytest.raises(ValueError):
        MotionConfig(gain=-1).validate()
    with pytest.raises(ValueError):
        MotionConfig(illum_change_threshold=-5).validate()
    with pytest.raises(ValueError):
        MotionConfig(trigger_frames=0).validate()


def test_plane_size_mismatch_raises():
    det = MotionDetector(32, 32, _cfg())
    det.update(_plane(32, 32, 30), 0)
    with pytest.raises(ValueError):
        det.update(_plane(16, 16, 30), 1)


def test_min_area_filters_small_regions():
    # 门槛高于运动块实际面积（外扩后约 320 像素）：区域被丢弃
    cfg = _cfg(min_area=500)
    det = MotionDetector(64, 64, cfg)
    det.update(_plane(64, 64, 30), 0)
    f1 = _plane(64, 64, 30)
    _rect(f1, 64, 8, 8, 16, 16, 200)
    det.update(f1, 1)
    f2 = _plane(64, 64, 30)
    _rect(f2, 64, 24, 8, 16, 16, 200)
    out = det.update(f2, 2)
    # 区域被面积门槛丢弃，命中不成立，状态机不进入运动态
    assert out.regions == []
    assert sum(out.mask) == 0
    assert out.motion is False
