"""追踪状态机纯逻辑测试：跟丢冻结、不乱跟、扩区重搜与恢复确认。"""

import pytest

from conftest import extract, make_plane, put_pattern, put_rect
from vtrack.model import LOST, TRACKED
from vtrack.tracker import (
    STATE_LOST,
    STATE_REACQUIRE,
    STATE_TRACKED,
    TemplateTracker,
)
from vtrack.template import METHOD_SAD, METHOD_ZNCC

W, H = 64, 48
TW, TH = 10, 8
X0, Y0 = 12, 10


def _scene(x, y, bg=20, seed=3):
    """带纹理目标块的合成帧，返回 (帧, 模板)。"""
    frame = make_plane(W, H, fill=bg)
    put_pattern(frame, W, x, y, TW, TH, seed=seed)
    return frame, extract(frame, W, x, y, TW, TH)


def _tracker(tpl, **kw):
    kwargs = dict(method=METHOD_SAD, threshold=30.0, search_margin=8,
                  reacquire_frames=3)
    kwargs.update(kw)
    return TemplateTracker(W, H, tpl, TW, TH, X0, Y0, **kwargs)


def _drive(tracker, frames):
    recs = []
    for i, frame in enumerate(frames):
        recs.append(tracker.update(frame, i, pts_time=i * 0.1))
    return recs


def test_first_frame_confirms_initial_box_with_ideal_score():
    frame0, tpl = _scene(X0, Y0)
    tr = _tracker(tpl)
    rec = tr.update(frame0, 0)
    assert rec.status == TRACKED and rec.confirmed
    assert rec.bbox == (X0, Y0, TW, TH)
    assert rec.score == 0.0


def test_tracks_smooth_translation_within_margin():
    frames = []
    tpl = None
    positions = [(X0 + 2 * i, Y0 + i) for i in range(5)]
    for i, (x, y) in enumerate(positions):
        f, t = _scene(x, y)
        frames.append(f)
        if i == 0:
            tpl = t
    tr = _tracker(tpl)
    recs = _drive(tr, frames)
    for rec, (x, y) in zip(recs, positions):
        assert rec.status == TRACKED
        assert (rec.bbox[0], rec.bbox[1]) == (x, y)


def test_low_score_freezes_box_and_reports_lost():
    frame0, tpl = _scene(X0, Y0)
    # 纯噪声帧：任何窗口与纹理模板差异都很大
    import random
    random.seed(42)
    noise = bytearray(random.randrange(0, 256) for _ in range(W * H))
    tr = _tracker(tpl, threshold=30.0)
    tr.update(frame0, 0)
    rec = tr.update(bytes(noise), 1)
    assert rec.status == LOST
    assert not rec.confirmed
    # 框冻结在最后已知位置，没有跳到噪声里的任何候选
    assert (rec.bbox[0], rec.bbox[1]) == (X0, Y0)
    assert tr.state == STATE_LOST
    assert tr.loss_started == 1


def test_lost_search_region_grows_and_eventually_covers_frame():
    frame0, tpl = _scene(X0, Y0)
    tr = _tracker(tpl, search_margin=4, lost_growth=2.0)
    tr.update(frame0, 0)
    tr.state = STATE_LOST
    tr.loss_started = 1
    tr.lost_frames = 1
    first = tr.lost_search()
    tr.lost_frames = 4
    later = tr.lost_search()
    # 边距复利增长：4*2^0=4 → 4*2^3=32
    assert (later[2] - later[0]) > (first[2] - first[0])
    tr.lost_frames = 20
    full = tr.lost_search()
    assert full == (0, 0, W - TW, H - TH)


def test_reacquire_needs_consecutive_confirmations():
    # margin=8：丢失第 2 帧搜索边距 16，覆盖距最后已知位置约 14px 的新目标
    frame0, tpl = _scene(X0, Y0)
    occluded = make_plane(W, H, fill=20)
    nx, ny = X0 + 14, Y0 + 8
    appear, _ = _scene(nx, ny)
    tr = _tracker(tpl, search_margin=8, reacquire_frames=3,
                  lost_growth=3.0, threshold=30.0)
    tr.update(frame0, 0)
    r1 = tr.update(occluded, 1)
    assert r1.status == LOST
    r2 = tr.update(appear, 2)
    # 首次达标只进入重确认，对外仍是 lost
    assert tr.state == STATE_REACQUIRE
    assert r2.status == LOST and not r2.confirmed
    r3 = tr.update(appear, 3)
    assert r3.status == LOST
    r4 = tr.update(appear, 4)
    # 连续 3 帧达标后恢复
    assert r4.status == TRACKED and r4.confirmed
    assert (r4.bbox[0], r4.bbox[1]) == (nx, ny)
    assert tr.take_recovery() == 4
    assert tr.take_recovery() is None


def test_reacquire_falls_back_to_lost_on_bad_frame():
    frame0, tpl = _scene(X0, Y0)
    occluded = make_plane(W, H, fill=20)
    nx, ny = X0 + 14, Y0 + 8
    appear, _ = _scene(nx, ny)
    import random
    random.seed(7)
    noise = bytearray(random.randrange(0, 256) for _ in range(W * H))
    tr = _tracker(tpl, search_margin=8, reacquire_frames=3,
                  lost_growth=3.0)
    tr.update(frame0, 0)
    tr.update(occluded, 1)
    tr.update(appear, 2)          # 进入 REACQUIRE
    assert tr.state == STATE_REACQUIRE
    tr.update(bytes(noise), 3)    # 打断确认
    assert tr.state == STATE_LOST
    # 之后需要重新累计 3 帧
    tr.update(appear, 4)
    assert tr.state == STATE_REACQUIRE
    tr.update(appear, 5)
    assert tr.state == STATE_REACQUIRE
    rec = tr.update(appear, 6)
    assert rec.status == TRACKED


def test_reacquire_frames_one_confirms_immediately():
    frame0, tpl = _scene(X0, Y0)
    occluded = make_plane(W, H, fill=20)
    nx, ny = X0 + 14, Y0 + 8
    appear, _ = _scene(nx, ny)
    tr = _tracker(tpl, search_margin=8, reacquire_frames=1,
                  lost_growth=3.0)
    tr.update(frame0, 0)
    tr.update(occluded, 1)
    rec = tr.update(appear, 2)
    assert rec.status == TRACKED
    assert tr.take_recovery() == 2


def test_no_false_recovery_on_distractor_below_threshold():
    """遮挡后出现的是别的东西（分数不达标），必须继续报丢失。"""
    frame0, tpl = _scene(X0, Y0)
    occluded = make_plane(W, H, fill=20)
    # 不同纹理的块：与模板 SAD 很大
    other = make_plane(W, H, fill=20)
    put_pattern(other, W, 40, 30, TW, TH, seed=99)
    tr = _tracker(tpl, search_margin=6, reacquire_frames=2,
                  lost_growth=4.0, threshold=30.0)
    tr.update(frame0, 0)
    tr.update(occluded, 1)
    rec = tr.update(other, 2)
    assert rec.status == LOST
    assert tr.state == STATE_LOST


def test_bbox_never_moves_while_lost():
    frame0, tpl = _scene(X0, Y0)
    occluded = make_plane(W, H, fill=20)
    tr = _tracker(tpl, search_margin=4, lost_growth=2.0)
    tr.update(frame0, 0)
    for i in range(1, 6):
        rec = tr.update(occluded, i)
        assert (rec.bbox[0], rec.bbox[1]) == (X0, Y0)


def test_adaptive_template_update_toggle():
    frame0, tpl = _scene(X0, Y0)
    # 同位置、纹理轻微变化（换 seed）：SAD 非零但仍达标，模拟目标外观渐变
    f1, _ = _scene(X0, Y0, seed=9)
    # 默认关闭：work 恒等于 anchor
    tr = _tracker(tpl, threshold=40.0)
    tr.update(frame0, 0)
    rec = tr.update(f1, 1)
    assert rec.status == TRACKED
    assert tr.work == tr.anchor
    # 开启后：达标的外观变化会改变 work，但 anchor 不动
    tr2 = _tracker(tpl, update_template=True, update_alpha=0.5,
                   threshold=40.0)
    tr2.update(frame0, 0)
    tr2.update(f1, 1)
    assert tr2.work != tr2.anchor
    assert tr2.anchor == tpl


def test_template_resets_to_anchor_after_recovery():
    frame0, tpl = _scene(X0, Y0)
    f1, _ = _scene(X0, Y0, seed=9)
    occluded = make_plane(W, H, fill=20)
    nx, ny = X0 + 14, Y0 + 8
    appear, _ = _scene(nx, ny)
    tr = _tracker(tpl, search_margin=8, reacquire_frames=2,
                  lost_growth=3.0, update_template=True, update_alpha=0.5,
                  threshold=40.0)
    tr.update(frame0, 0)
    tr.update(f1, 1)
    assert tr.work != tpl
    tr.update(occluded, 2)
    tr.update(appear, 3)
    rec = tr.update(appear, 4)
    assert rec.status == TRACKED
    assert tr.work == tpl  # 恢复即重置，杜绝漂移累积


def test_zncc_tracker_reacquires_textured_target():
    frame0, tpl = _scene(X0, Y0)
    occluded = make_plane(W, H, fill=20)
    nx, ny = X0 + 14, Y0 + 8
    appear, _ = _scene(nx, ny)
    tr = TemplateTracker(W, H, tpl, TW, TH, X0, Y0,
                         method=METHOD_ZNCC, threshold=0.7,
                         search_margin=8, reacquire_frames=2,
                         lost_growth=3.0)
    r0 = tr.update(frame0, 0)
    assert r0.score == 1.0
    tr.update(occluded, 1)
    tr.update(appear, 2)
    rec = tr.update(appear, 3)
    assert rec.status == TRACKED
    assert (rec.bbox[0], rec.bbox[1]) == (nx, ny)


def test_invalid_construction_arguments():
    tpl = bytes(TW * TH)
    with pytest.raises(ValueError):
        TemplateTracker(W, H, tpl, TW, TH, W, 0)          # 越界
    with pytest.raises(ValueError):
        TemplateTracker(W, H, tpl, 0, TH, 0, 0)           # 坏尺寸
    with pytest.raises(ValueError):
        TemplateTracker(W, H, tpl, TW, TH, 0, 0,
                        method="ncc")                      # 坏方法
    with pytest.raises(ValueError):
        TemplateTracker(W, H, tpl, TW, TH, 0, 0,
                        search_margin=-1)
    with pytest.raises(ValueError):
        TemplateTracker(W, H, tpl, TW, TH, 0, 0,
                        lost_growth=0.5)
    with pytest.raises(ValueError):
        TemplateTracker(W, H, tpl, TW, TH, 0, 0,
                        update_alpha=0.0)
