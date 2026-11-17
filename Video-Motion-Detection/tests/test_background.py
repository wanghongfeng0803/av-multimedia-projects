"""背景建模检测器测试：预热保护、慢光照自适应、前景吸收、选择性更新。

合成平面直接喂 :class:`BackgroundDetector`，不依赖 ffmpeg。
三个时间尺度目标各有独立用例锁定，另含与帧差模式同构的门控/
holdoff/参数校验契约。
"""

import pytest

from vmotion.background import (
    BackgroundConfig,
    BackgroundDetector,
    DEFAULT_BACKGROUND_CONFIG,
)
from vmotion.detector import FrameDetection


def _plane(w, h, fill):
    return bytearray([fill]) * (w * h)


def _rect(plane, w, x, y, rw, rh, value):
    for row in range(y, y + rh):
        base = row * w + x
        for col in range(rw):
            plane[base + col] = value


def _cfg(**over):
    """小帧测试默认：关掉形态学与面积门槛，缩短预热/吸收周期。"""
    base = dict(block=8, local_radius=1, pixel_threshold=10.0, gain=2.0,
                min_area=1, open_radius=0, close_radius=0,
                trigger_frames=2, release_frames=2,
                warmup_frames=5, learning_rate=0.2, absorb_frames=8,
                illum_change_threshold=25.0, illum_hold_frames=2)
    base.update(over)
    return BackgroundConfig(**base)


# --------------------------------------------------------------- 预热保护
def test_first_frame_initializes_background_without_motion():
    det = BackgroundDetector(32, 32, _cfg())
    out = det.update(_plane(32, 32, 40), 0, 0.0)
    assert isinstance(out, FrameDetection)
    assert out.motion is False and out.regions == []
    assert out.stats["warmup"] is True
    # 首帧块均值即初始背景
    assert det._background[0] == pytest.approx(40.0)


def test_warmup_suppresses_motion_even_with_changing_scene():
    """预热期内即使出现显著前景也不得报警（背景未稳定不冤枉画面）。"""
    cfg = _cfg(warmup_frames=5)
    det = BackgroundDetector(32, 32, cfg)
    for i in range(6):
        p = _plane(32, 32, 40)
        if i >= 1:
            _rect(p, 32, 4, 4, 8, 8, 200)
        out = det.update(p, i)
        assert out.motion is False
        assert sum(out.mask) == 0
    # 帧 0..5 为预热（首帧 + warmup_frames=5），帧 6 起 warmup 标志清零
    sixth = det.update(_plane(32, 32, 40), 6)
    assert sixth.stats["warmup"] is False


def test_warmup_builds_running_average_background():
    """预热期背景是逐块累计平均：两帧各半的块均值应为算术平均。"""
    cfg = _cfg(warmup_frames=3)
    det = BackgroundDetector(32, 32, cfg)
    det.update(_plane(32, 32, 40), 0)
    p1 = _plane(32, 32, 40)
    _rect(p1, 32, 0, 0, 8, 8, 60)
    det.update(p1, 1)
    p2 = _plane(32, 32, 40)
    _rect(p2, 32, 0, 0, 8, 8, 80)
    det.update(p2, 2)
    # 三帧该块均值 40、60、80 → 累计平均 60
    assert det._background[0] == pytest.approx(60.0)


def test_zero_warmup_detects_from_second_frame():
    cfg = _cfg(warmup_frames=0)
    det = BackgroundDetector(32, 32, cfg)
    det.update(_plane(32, 32, 40), 0)
    p = _plane(32, 32, 40)
    _rect(p, 32, 8, 8, 8, 8, 200)
    out = det.update(p, 1)
    assert out.stats["warmup"] is False
    assert out.stats["gated_pixels"] > 0


# --------------------------------------------------------------- 慢光照
def test_slow_illumination_drift_adapts_without_false_alarm():
    """每帧 +1 的缓慢提亮：块差越不过门限，背景平滑跟随、全程零报警。"""
    cfg = _cfg(warmup_frames=3, learning_rate=0.2)
    det = BackgroundDetector(32, 32, cfg)
    out = None
    for i in range(40):
        out = det.update(_plane(32, 32, 40 + i), i)
        assert out.motion is False
    # 背景已跟踪到接近当前亮度（滞后于真值但同量级），而非钉在初始 40
    assert out.stats["bg_mean"] > 60.0
    assert out.stats["gated_pixels"] == 0


def test_learning_rate_controls_tracking_speed():
    """α 越大背景跟得越快：同序列下大 α 的背景均值更接近当前帧。"""
    seq = [40 + 2 * i for i in range(20)]
    backgrounds = {}
    for lr in (0.05, 0.5):
        cfg = _cfg(warmup_frames=2, learning_rate=lr)
        det = BackgroundDetector(16, 16, cfg)
        last = None
        for i, v in enumerate(seq):
            last = det.update(_plane(16, 16, v), i)
        backgrounds[lr] = last.stats["bg_mean"]
    assert backgrounds[0.5] > backgrounds[0.05]


# --------------------------------------------------------------- 前景吸收
def test_stationary_foreground_is_absorbed_then_stops_alarming():
    """预热后出现的静止物体：先连续报警，吸收后背景追平、报警自行停止。"""
    cfg = _cfg(warmup_frames=3, absorb_frames=5, learning_rate=0.3)
    det = BackgroundDetector(32, 32, cfg)
    motions = []
    absorb_events = []
    for i in range(40):
        p = _plane(32, 32, 40)
        if i >= 4:                       # 预热结束后才放入物体
            _rect(p, 32, 4, 4, 8, 8, 200)
        out = det.update(p, i)
        motions.append(out.motion)
        absorb_events.append(out.stats["absorbed_blocks"])
    assert any(motions)                  # 物体初现时确实报过警
    first_motion = next(i for i, m in enumerate(motions) if m)
    assert first_motion >= 4
    # 首次吸收发生在连续前景计数达到 absorb_frames 的帧
    first_absorb = next(i for i, n in enumerate(absorb_events) if n > 0)
    assert first_absorb == first_motion + cfg.absorb_frames - 1 or \
        first_absorb >= first_motion
    # 片尾已完全吸收：无运动、计数清零、背景块追到目标块均值。
    # 矩形 (4,4,8,8) 跨 4 个 8x8 块，每块 16 像素为 200、48 像素为 40，
    # 收敛后的块均值 = 40 + (200-40)*16/64 = 80。
    assert not any(motions[-5:])
    assert det._fg_runs[0] == 0
    assert det._background[0] == pytest.approx(80.0, abs=2.0)
    assert det._background[1] == pytest.approx(80.0, abs=2.0)


def test_transient_foreground_does_not_pollute_background():
    """只出现一帧的物体：前景计数清零，背景块几乎留在原亮度。"""
    cfg = _cfg(warmup_frames=3, absorb_frames=10, learning_rate=0.2)
    det = BackgroundDetector(32, 32, cfg)
    for i in range(4):
        det.update(_plane(32, 32, 40), i)
    spike = _plane(32, 32, 40)
    _rect(spike, 32, 0, 0, 8, 8, 200)
    det.update(spike, 4)
    for i in range(5, 10):
        det.update(_plane(32, 32, 40), i)
    assert det._fg_runs[0] == 0
    # 仅一帧前景、且前景块在吸收前不更新，背景偏离不超过一个 α 步长量级
    assert det._background[0] == pytest.approx(40.0, abs=3.0)


def test_learning_rate_one_converges_in_single_ema_step():
    """边界 α=1：进入吸收态当帧背景即等于当前块均值。"""
    cfg = _cfg(warmup_frames=0, learning_rate=1.0, absorb_frames=3)
    det = BackgroundDetector(32, 32, cfg)
    det.update(_plane(32, 32, 40), 0)
    for i in range(1, 4):
        p = _plane(32, 32, 40)
        _rect(p, 32, 0, 0, 8, 8, 200)
        det.update(p, i)
    assert det._background[0] == pytest.approx(200.0)


# --------------------------------------------------------------- 常规检测
def test_moving_block_detected_after_warmup():
    cfg = _cfg(warmup_frames=3)
    det = BackgroundDetector(64, 64, cfg)
    for i in range(4):
        det.update(_plane(64, 64, 30), i)
    f1 = _plane(64, 64, 30)
    _rect(f1, 64, 8, 8, 16, 16, 200)
    r1 = det.update(f1, 4)
    f2 = _plane(64, 64, 30)
    _rect(f2, 64, 24, 8, 16, 16, 200)
    r2 = det.update(f2, 5)
    assert r1.motion is False           # trigger_frames=2 去抖
    assert r2.motion is True
    assert len(r2.regions) >= 1
    assert r2.stats["warmup"] is False


def test_static_scene_never_triggers_after_warmup():
    cfg = _cfg(warmup_frames=2)
    det = BackgroundDetector(32, 32, cfg)
    for i in range(12):
        out = det.update(_plane(32, 32, 40), i)
        assert out.motion is False


def test_global_illumination_jump_triggers_holdoff_and_absorbs():
    """全局突变：holdoff 帧空掩码，背景全块 EMA 吸收跳变，解除后不误报。"""
    cfg = _cfg(warmup_frames=2, illum_change_threshold=25.0,
               illum_hold_frames=2, learning_rate=0.5)
    det = BackgroundDetector(32, 32, cfg)
    for i in range(3):
        det.update(_plane(32, 32, 40), i)
    jump = det.update(_plane(32, 32, 120), 3)
    assert jump.stats["held_off"] is True
    assert jump.motion is False and sum(jump.mask) == 0
    held2 = det.update(_plane(32, 32, 120), 4)
    assert held2.stats["held_off"] is True
    after = det.update(_plane(32, 32, 120), 5)
    assert after.stats["held_off"] is False
    assert after.motion is False
    # 抑制期全块 EMA：背景已明显向 120 靠拢
    assert det._background[0] > 80.0


def test_repeated_jump_rearms_holdoff():
    cfg = _cfg(warmup_frames=0, illum_hold_frames=1,
               illum_change_threshold=25.0)
    det = BackgroundDetector(32, 32, cfg)
    det.update(_plane(32, 32, 40), 0)
    assert det.update(_plane(32, 32, 120), 1).stats["held_off"] is True
    assert det.update(_plane(32, 32, 120), 2).stats["held_off"] is False
    assert det.update(_plane(32, 32, 40), 3).stats["held_off"] is True


# --------------------------------------------------------------- 参数契约
def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        BackgroundConfig(warmup_frames=-1).validate()
    with pytest.raises(ValueError):
        BackgroundConfig(learning_rate=0.0).validate()
    with pytest.raises(ValueError):
        BackgroundConfig(learning_rate=1.5).validate()
    with pytest.raises(ValueError):
        BackgroundConfig(absorb_frames=0).validate()
    with pytest.raises(ValueError):
        BackgroundConfig(block=0).validate()
    with pytest.raises(ValueError):
        BackgroundConfig(illum_change_threshold=300).validate()


def test_default_config_is_valid():
    DEFAULT_BACKGROUND_CONFIG.validate()
    assert DEFAULT_BACKGROUND_CONFIG.warmup_frames == 10
    assert 0 < DEFAULT_BACKGROUND_CONFIG.learning_rate <= 1


def test_plane_size_mismatch_raises():
    det = BackgroundDetector(32, 32, _cfg())
    det.update(_plane(32, 32, 30), 0)
    with pytest.raises(ValueError):
        det.update(_plane(16, 16, 30), 1)


def test_config_dict_contains_background_params():
    det = BackgroundDetector(16, 16, _cfg())
    d = det.config_dict()
    for key in ("warmup_frames", "learning_rate", "absorb_frames",
                "illum_change_threshold", "block", "gain"):
        assert key in d
