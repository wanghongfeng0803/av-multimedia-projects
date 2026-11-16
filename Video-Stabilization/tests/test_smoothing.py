"""轨迹平滑与裁剪量推导的纯逻辑测试（不依赖 ffmpeg）。"""

import math

import pytest

from stabilize.smoothing import (
    DEFAULT_RADIUS,
    accumulate_trajectory,
    compute_crop,
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
