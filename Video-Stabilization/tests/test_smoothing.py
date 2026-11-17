"""轨迹平滑与裁剪量推导的纯逻辑测试（不依赖 ffmpeg）。"""

import math

import pytest

from stabilize.smoothing import (
    DEFAULT_RADIUS,
    accumulate_trajectory,
    clamp_compensation,
    compute_crop,
    dynamic_insets,
    gaussian_weights,
    interpolate_invalid,
    moving_average,
    rotation_inset,
    smooth_motion,
    smooth_trajectory,
)


# ----------------------------------------------------- 无效帧插值

def test_interpolate_middle_gap():
    values = [0.0, 10.0, -999.0, -999.0, 40.0]
    valid = [True, True, False, False, True]
    out = interpolate_invalid(values, valid)
    assert out == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])


def test_interpolate_endpoint_extrapolation():
    out = interpolate_invalid([5.0, 5.0, 8.0], [False, True, True])
    assert out == pytest.approx([5.0, 5.0, 8.0])
    out2 = interpolate_invalid([5.0, 8.0, 8.0], [True, True, False])
    assert out2 == pytest.approx([5.0, 8.0, 8.0])


def test_interpolate_all_invalid_is_zero():
    assert interpolate_invalid([1.0, 2.0], [False, False]) == [0.0, 0.0]
    assert interpolate_invalid([], []) == []


# ----------------------------------------------------- 轨迹累积

def test_accumulate_pure_translation():
    # 零旋转时累积轨迹就是增量前缀和
    dx = [0.0, 1.0, 2.0, -1.0]
    dy = [0.0, 0.0, 1.0, 1.0]
    ang = [0.0, 0.0, 0.0, 0.0]
    tx, ty, ta = accumulate_trajectory(dx, dy, ang)
    assert tx == pytest.approx([0.0, 1.0, 3.0, 2.0])
    assert ty == pytest.approx([0.0, 0.0, 1.0, 2.0])
    assert ta == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_accumulate_rotated_translation():
    # 屏幕坐标 90° 顺时针：x 方向帧体增量 (1,0) 落到世界系 (0,-1)
    tx, ty, ta = accumulate_trajectory(
        [0.0, 1.0], [0.0, 0.0], [0.0, 90.0]
    )
    assert ta == pytest.approx([0.0, 90.0])
    assert tx[1] == pytest.approx(0.0, abs=1e-9)
    assert ty[1] == pytest.approx(-1.0, abs=1e-9)


# ----------------------------------------------------- 平滑窗

def test_moving_average_constant_sequence():
    seq = [3.0] * 10
    assert moving_average(seq, 2) == pytest.approx(seq)


def test_moving_average_spike_suppressed():
    seq = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    out = moving_average(seq, 3)
    assert max(out) < 10.0
    # 边界截断不归零：首帧窗口只覆盖 [0,3]，均值非零但有限
    assert out[0] == pytest.approx(10.0 / 4)
    assert out[-1] == pytest.approx(10.0 / 4)


def test_gaussian_weights_normalized_and_symmetric():
    w = gaussian_weights(3, 1.0)
    assert sum(w) == pytest.approx(1.0)
    assert w == pytest.approx(list(reversed(w)))
    assert w[3] == max(w)  # 中心权重最大


def test_smooth_trajectory_rejects_bad_method():
    with pytest.raises(ValueError):
        smooth_trajectory([0.0], [0.0], [0.0], radius=1, method="bogus")
    with pytest.raises(ValueError):
        smooth_trajectory([0.0], [0.0], [0.0], radius=-1)


# ----------------------------------------------------- 裁剪几何

def test_rotation_inset_zero_angle():
    assert rotation_inset(100, 80, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_rotation_inset_monotone():
    a = rotation_inset(200, 100, 3.0)
    b = rotation_inset(200, 100, 8.0)
    assert 0 < a < b
    # 90° 时画面角点必然转出，inset 接近半幅
    assert rotation_inset(200, 100, 90.0) > 40.0


def test_compute_crop_uses_compensation_extremes():
    crop = compute_crop(
        200, 100,
        comp_x=[0.0, 4.0, -3.0],
        comp_y=[0.0, 2.0, -5.0],
        comp_a=[0.0, 1.0, 0.0],
    )
    # 平移主导项 max(4,5)=5，加上 1° 旋转伸出量与 1px 安全余量
    assert crop["inset"] > 5.0
    assert crop["clamped"] is False
    assert crop["max_shift_x"] == 4.0
    assert crop["max_shift_y"] == 5.0


def test_compute_crop_clamps_to_short_side_ratio():
    crop = compute_crop(
        100, 100, [60.0], [60.0], [0.0]
    )
    assert crop["clamped"] is True
    assert crop["inset"] == pytest.approx(45.0, abs=0.01)


# ----------------------------------------------------- 补偿量限幅

def test_clamp_compensation_hard_limits():
    cx, cy, ca, stats = clamp_compensation(
        [0.0, 5.0, -7.0], [1.0, -1.0, 0.5], [0.0, 3.0, -3.0],
        max_shift=4.0, max_angle=2.0,
    )
    assert cx == [0.0, 4.0, -4.0]
    assert cy == [1.0, -1.0, 0.5]          # 未超限时逐值保留
    assert ca == [0.0, 2.0, -2.0]
    assert stats["clamped_frames"] == 2    # 第 1、2 帧任一分量被夹
    assert stats["shift_frames"] == 2
    assert stats["angle_frames"] == 2
    assert stats["raw_peak"]["dx"] == 7.0  # 限幅前峰值保留供对比
    assert stats["peak"]["dx"] == 4.0
    assert stats["peak"]["angle"] == 2.0


def test_clamp_compensation_none_passthrough():
    cx, cy, ca, stats = clamp_compensation([9.0], [-9.0], [9.0])
    assert cx == [9.0] and cy == [-9.0] and ca == [9.0]
    assert stats["clamped_frames"] == 0
    assert stats["peak"]["dx"] == 9.0


def test_clamp_compensation_rejects_negative_limit():
    with pytest.raises(ValueError):
        clamp_compensation([0.0], [0.0], [0.0], max_shift=-1.0)
    with pytest.raises(ValueError):
        clamp_compensation([0.0], [0.0], [0.0], max_angle=-0.5)


def test_smooth_motion_limits_leave_smoothing_untouched():
    # 限幅与平滑不打架：smoothed/trajectory 与不限幅逐帧一致，
    # 限幅只截断最终补偿信号，平滑器看不到被夹的值
    doc = _motion_doc()
    plain = smooth_motion(doc, radius=3)
    limited = smooth_motion(doc, radius=3, max_shift=2.0, max_angle=0.5)
    assert "limits" not in plain           # 旧模式 JSON 无 limits 段
    assert "limits" in limited
    for fp, fl in zip(plain["frames"], limited["frames"]):
        assert fp["smoothed"] == fl["smoothed"]
        assert fp["trajectory"] == fl["trajectory"]
        # 帧条目契约不变（与旧模式同 schema）
        assert set(fl) == {"index", "trajectory", "smoothed",
                           "compensation", "valid"}
    lim = limited["limits"]
    assert lim["max_shift"] == 2.0 and lim["max_angle"] == 0.5
    assert lim["clamped_frames"] > 0
    for f in limited["frames"]:
        assert abs(f["compensation"]["dx"]) <= 2.0 + 1e-4
        assert abs(f["compensation"]["dy"]) <= 2.0 + 1e-4
        assert abs(f["compensation"]["angle"]) <= 0.5 + 1e-4
    # 裁剪量由限幅后补偿推导：只减不增
    assert limited["crop"]["inset"] <= plain["crop"]["inset"] + 1e-9
    # 限幅前后峰值都记录，可直接对比两种模式
    assert lim["raw_peak"]["dx"] >= lim["peak"]["dx"]


# ----------------------------------------------------- 动态裁剪 inset

def test_dynamic_insets_constant_compensation():
    ins = dynamic_insets(200, 100, [3.0] * 9, [0.0] * 9, [0.0] * 9,
                         radius=2)
    assert len(ins) == 9
    # 常数补偿 → 常数 inset（边界截断重新归一化，不向零塌陷）
    assert all(v == ins[0] for v in ins)
    assert ins[0] >= 3 + 1                 # 补偿 + 安全余量


def test_dynamic_insets_smoothes_breathing():
    # 单帧尖峰被滑动平均摊薄：峰值 inset 远小于瞬时需求（防呼吸）
    comp = [0.0] * 5 + [20.0] + [0.0] * 5
    ins = dynamic_insets(400, 300, comp, [0.0] * 11, [0.0] * 11, radius=2)
    assert 0 < max(ins) < 20 + 1


def test_dynamic_insets_cover_rotation():
    no_rot = dynamic_insets(200, 100, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    with_rot = dynamic_insets(200, 100, [0.0] * 5, [0.0] * 5, [5.0] * 5)
    assert all(r > n for r, n in zip(with_rot, no_rot))


# ----------------------------------------------------- 端到端纯函数

def _motion_doc(n=21, w=320, h=240, scale=4):
    """造一份带抖动、含两个无效帧的 motion 文档。"""
    frames = [{"index": 0, "dx": 0.0, "dy": 0.0, "angle": 0.0,
               "confidence": 1.0, "valid": True, "scene_cut": False}]
    for i in range(1, n):
        frames.append({
            "index": i,
            "dx": 1.0 + (0.8 if i % 2 else -0.8),
            "dy": 0.5 * math.sin(i),
            "angle": 0.3 * math.cos(i * 0.7),
            "confidence": 0.9,
            "valid": i not in (5, 6),
            "scene_cut": i == 5,
        })
    return {
        "format": "stabilize-motion", "version": 1,
        "source": "synthetic.mp4", "width": w, "height": h,
        "fps": "10/1", "pix_fmt": "yuv420p", "has_audio": False,
        "est_width": w // scale, "est_height": h // scale,
        "downsample_scale": scale, "frame_count": n, "frames": frames,
    }


def test_smooth_motion_contract_and_scaling():
    doc = _motion_doc()
    out = smooth_motion(doc, radius=3, method="uniform")
    assert out["format"] == "stabilize-smooth" and out["version"] == 1
    assert len(out["frames"]) == 21
    # 小图位移按 scale=4 还原：平滑后累积量在原分辨率量级
    assert out["frames"][-1]["trajectory"]["x"] > 10.0
    # 补偿量 = 平滑 − 原始，字段齐全且无效帧标记透传
    f5 = out["frames"][5]
    assert f5["valid"] is False
    for f in out["frames"]:
        assert set(f) == {"index", "trajectory", "smoothed",
                          "compensation", "valid"}
    assert out["params"]["invalid_frames"] == 2
    assert "inset" in out["crop"]


def test_smooth_motion_gaussian_runs_and_jsonable():
    import json
    out = smooth_motion(_motion_doc(), radius=4, method="gaussian")
    # 全部字段可 JSON 序列化（稳定契约要求）
    json.dumps(out, ensure_ascii=False)
    assert out["params"]["method"] == "gaussian"


def test_smoothed_trajectory_reduces_jitter():
    # 高频抖动经平滑后，逐帧差分的方差应显著小于原始轨迹
    doc = _motion_doc(n=41)
    out = smooth_motion(doc, radius=DEFAULT_RADIUS, method="uniform")
    raw = [f["trajectory"]["x"] for f in out["frames"]]
    sm = [f["smoothed"]["x"] for f in out["frames"]]

    def diff_var(seq):
        d = [b - a for a, b in zip(seq, seq[1:])]
        m = sum(d) / len(d)
        return sum((v - m) ** 2 for v in d) / len(d)

    assert diff_var(sm) < diff_var(raw) * 0.5
