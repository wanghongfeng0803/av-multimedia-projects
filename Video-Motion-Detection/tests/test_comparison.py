"""双模式量化对比测试：不一致分类、契约校验、同输入行为差异。

前半部分直接构造 :class:`FrameDetection` 序列驱动纯逻辑对比，不依赖
ffmpeg；后半部分把同一组合成平面喂进两个真实检测器，验证预热期与
前景吸收确实产生可量化、可解释的判定分歧。
"""

import pytest

from vmotion.background import BackgroundConfig, BackgroundDetector
from vmotion.comparison import (
    COMPARE_SCHEMA_VERSION,
    compare_detections,
)
from vmotion.detector import FrameDetection, MotionConfig, MotionDetector


def _det(frame, motion, regions=0, held_off=False, warmup=False):
    """构造最小可用检测结果（regions 用个数占位，对比只看长度）。"""
    return FrameDetection(
        frame=frame,
        pts_time=frame * 0.1,
        mask=b"",
        regions=[{"x": i} for i in range(regions)],
        motion=motion,
        stats={"held_off": held_off, "warmup": warmup,
               "gated_pixels": 1 if motion else 0,
               "fg_blocks": 0, "absorbed_blocks": 0},
    )


# --------------------------------------------------------------- 纯逻辑
def test_identical_verdicts_full_agreement():
    seq = [_det(i, motion=(i in (2, 3, 4))) for i in range(6)]
    out = compare_detections(seq, [_det(i, motion=(i in (2, 3, 4)))
                                   for i in range(6)])
    assert out["schema_version"] == COMPARE_SCHEMA_VERSION
    assert out["frames"] == 6
    assert out["agreement"] == pytest.approx(1.0)
    assert out["verdict_agreement"] == pytest.approx(1.0)
    assert out["disagreements"] == []
    assert out["counts"]["agree"] == 6
    assert len(out["per_frame"]) == 6


def test_frame_diff_only_and_background_only_classification():
    f_seq = [_det(0, True), _det(1, False), _det(2, True), _det(3, False)]
    b_seq = [_det(0, False), _det(1, True), _det(2, True), _det(3, False)]
    out = compare_detections(f_seq, b_seq)
    assert out["counts"]["frame_diff_only"] == 1
    assert out["counts"]["background_only"] == 1
    assert out["counts"]["agree"] == 2
    kinds = {d["frame"]: d["kind"] for d in out["disagreements"]}
    assert kinds[0] == "frame_diff_only"
    assert kinds[1] == "background_only"
    # 背景侧预热/抑制状态被带进不一致明细，便于解释分歧来源
    detail = next(d for d in out["disagreements"] if d["frame"] == 1)
    assert "background_stats" in detail


def test_region_count_diff_reported_when_both_in_motion():
    f_seq = [_det(0, True, regions=3)]
    b_seq = [_det(0, True, regions=1)]
    out = compare_detections(f_seq, b_seq)
    # 运动态一致：不计入判定不一致，但区域数差异单列
    assert out["verdict_agreement"] == pytest.approx(1.0)
    assert out["counts"]["region_count_diff"] == 1
    assert out["counts"]["frame_diff_only"] == 0
    item = out["disagreements"][0]
    assert item["frame_diff_regions"] == 3
    assert item["background_regions"] == 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="帧数不一致"):
        compare_detections([_det(0, False)],
                           [_det(0, False), _det(1, False)])


def test_frame_misalignment_raises():
    with pytest.raises(ValueError, match="帧号未对齐"):
        compare_detections([_det(0, False), _det(2, False)],
                           [_det(0, False), _det(1, False)])


# ----------------------------------------------------- 同输入真实双检测器
def _plane(w, h, fill):
    return bytearray([fill]) * (w * h)


def _rect(plane, w, x, y, rw, rh, value):
    for row in range(y, y + rh):
        base = row * w + x
        for col in range(rw):
            plane[base + col] = value


def _shared_gate():
    return dict(block=8, local_radius=1, pixel_threshold=10.0, gain=2.0,
                min_area=1, open_radius=0, close_radius=0,
                trigger_frames=2, release_frames=2,
                illum_change_threshold=255.0, illum_hold_frames=1)


def test_warmup_creates_quantifiable_disagreement():
    """预热期帧差已能报警、背景模式刻意不报警 → frame_diff_only 分歧。

    物体必须持续移动：帧差只响应变化沿，静止物体会被迟滞释放，无法
    在预热窗口内形成稳定分歧。
    """
    f_det = MotionDetector(32, 32, MotionConfig(**_shared_gate()))
    b_det = BackgroundDetector(32, 32, BackgroundConfig(
        **_shared_gate(), warmup_frames=6, learning_rate=0.2,
        absorb_frames=20))
    f_outs, b_outs = [], []
    for i in range(14):
        p = _plane(32, 32, 40)
        if i >= 1:
            x = 4 + 8 * ((i - 1) % 2)      # 两位置来回跳，持续有变化沿
            _rect(p, 32, x, 4, 8, 8, 200)
        f_outs.append(f_det.update(p, i))
        b_outs.append(b_det.update(p, i))
    out = compare_detections(f_outs, b_outs)
    # 背景前 7 帧（首帧 + warmup=6）恒非运动；帧差在去抖后持续运动
    warmup_disagree = [
        d for d in out["disagreements"]
        if d["kind"] == "frame_diff_only" and d["frame"] <= 6
    ]
    assert warmup_disagree
    assert all(not b.motion for b in b_outs[:7])
    # 预热结束后两模式对同一运动序列重新收敛为一致
    assert all(f.motion and b.motion
               for f, b in zip(f_outs[8:], b_outs[8:]))


def test_stationary_object_absorption_changes_verdict_then_converges():
    """物体预热后入场并停留：两模式先一致报警，吸收后背景停报、
    帧差也因无变化沿停报，片尾两模式重新一致。"""
    f_det = MotionDetector(32, 32, MotionConfig(**_shared_gate()))
    b_det = BackgroundDetector(32, 32, BackgroundConfig(
        **_shared_gate(), warmup_frames=2, learning_rate=0.3,
        absorb_frames=5))
    f_outs, b_outs = [], []
    for i in range(40):
        p = _plane(32, 32, 40)
        if i >= 3:
            _rect(p, 32, 4, 4, 8, 8, 200)
        f_outs.append(f_det.update(p, i))
        b_outs.append(b_det.update(p, i))
    out = compare_detections(f_outs, b_outs)
    # 背景模式确实经历过吸收事件
    assert any(o.stats["absorbed_blocks"] > 0 for o in b_outs)
    # 片尾 5 帧两模式都不再判运动 → 重新一致
    tail = out["per_frame"][-5:]
    assert all(not row["frame_diff_motion"] and not row["background_motion"]
               for row in tail)
    # 全程分歧清单里每一条都能在 per_frame 对照中找到对应帧
    disagree_frames = {d["frame"] for d in out["disagreements"]}
    per_frames = {row["frame"] for row in out["per_frame"]}
    assert disagree_frames <= per_frames
