"""sendcmd 生成与参数钳制的纯逻辑测试（不跑 ffmpeg）。"""

import math

import pytest

from stabilize.compensate import build_sendcmd, crop_video
from stabilize.pipeline import PipelineError


def _smooth_doc(n=5, w=120, h=80, inset=6.0, fps="10/1"):
    return {
        "format": "stabilize-smooth", "version": 1,
        "source": "in.mp4", "width": w, "height": h, "fps": fps,
        "has_audio": False,
        "frames": [
            {
                "index": i,
                "trajectory": {"x": 0.0, "y": 0.0, "angle": 0.0},
                "smoothed": {"x": 0.0, "y": 0.0, "angle": 0.0},
                "compensation": {
                    "dx": float(i - 2), "dy": float(2 - i),
                    "angle": 0.5 * i,
                },
                "valid": True,
            }
            for i in range(n)
        ],
        "crop": {"inset": inset, "raw_inset": inset, "clamped": False,
                 "max_shift_x": 2.0, "max_shift_y": 2.0, "max_angle": 2.0},
        "params": {"radius": 2, "method": "uniform",
                   "frame_count": n, "invalid_frames": 0},
    }


def test_build_sendcmd_structure(tmp_path):
    doc = _smooth_doc()
    path = build_sendcmd(doc, str(tmp_path / "s.cmd"))
    text = open(path, encoding="utf-8").read()
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 5
    # 每帧一条 [enter] 命令，同时驱动 rotate 与两个 crop 参数
    first = lines[0]
    assert "0.000000 [enter]" in first
    assert "rotate@st angle" in first
    assert "crop@cp x" in first and "crop@cp y" in first
    # PTS 按帧率推导：第 3 帧 = 3/10 秒
    assert "0.300000 [enter]" in lines[3]


def test_sendcmd_angle_sign_is_negative(tmp_path):
    # 补偿角必须取负（反向旋转），且单位为弧度
    doc = _smooth_doc()
    path = build_sendcmd(doc, str(tmp_path / "s.cmd"))
    text = open(path, encoding="utf-8").read()
    # 第 2 帧补偿角 +1.0° → 下发 -1.0° 的弧度值
    expect = -math.radians(1.0)
    assert f"angle {expect:.8f}" in text


def test_sendcmd_origin_clamped_to_frame(tmp_path):
    # 补偿位移超过 inset 时，原点必须钳在 [0, w-cw] 内
    doc = _smooth_doc(inset=2.0)
    path = build_sendcmd(doc, str(tmp_path / "s.cmd"))
    text = open(path, encoding="utf-8").read()
    # 取样框宽 = 120-4=116，原点上界 4；任何下发表达式都带 min/max 钳制
    assert "min(max(" in text
    assert ",4)" in text


def test_build_sendcmd_rejects_wrong_format(tmp_path):
    doc = _smooth_doc()
    doc["format"] = "stabilize-motion"
    with pytest.raises(PipelineError):
        build_sendcmd(doc, str(tmp_path / "s.cmd"))


def test_build_sendcmd_rejects_oversized_inset(tmp_path):
    doc = _smooth_doc(inset=59.0)  # 短边 80，剩余高度 < 2
    with pytest.raises(PipelineError):
        build_sendcmd(doc, str(tmp_path / "s.cmd"))


def test_crop_video_clamps_inset_before_ffmpeg(tmp_path, monkeypatch):
    # 超画幅 inset 在调用 ffmpeg 之前被夹到合法上界：ffmpeg 不会
    # 收到非法裁剪尺寸。这里让 _find 直接失败以证明未走到子进程。
    doc = _smooth_doc(w=100, h=50)
    monkeypatch.setattr("stabilize.compensate._find",
                        lambda tool: pytest.fail("不应启动 ffmpeg"))
    # 合法上界 = (短边-2)//2 = 24；输入 1000 先被夹回，随后因
    # _find 被替换而失败——证明尺寸校验/钳制全部发生在其之前。
    with pytest.raises(pytest.fail.Exception):
        crop_video(doc, tmp_path / "in.mp4", tmp_path / "out.mp4",
                   inset=1000.0)
