"""sendcmd 生成与参数钳制的纯逻辑测试（不跑 ffmpeg）。"""

import math
import subprocess
import types

import pytest

from stabilize.compensate import (
    BORDER_MODES,
    build_sendcmd,
    build_sendcmd_dynamic,
    build_sendcmd_mirror,
    compensate_video,
    crop_video,
    mirror_margin,
)
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


# ----------------------------------------------------- 动态裁剪模式

def test_dynamic_sendcmd_has_per_frame_window(tmp_path):
    doc = _smooth_doc()  # 120x80，5 帧
    path = build_sendcmd_dynamic(doc, str(tmp_path / "d.cmd"),
                                 [4, 5, 6, 6, 5])
    text = open(path, encoding="utf-8").read()
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 5
    # 第 2 行（inset=5）：窗口 110x70，逐帧下发 w/h/x/y 四参数
    assert "crop@cp w 110" in lines[1]
    assert "crop@cp h 70" in lines[1]
    assert "crop@cp x min(max(" in lines[1]
    assert "rotate@st angle" in lines[1]
    # 第 3 行（inset=6）：窗口 108x68
    assert "crop@cp w 108" in lines[2]


def test_dynamic_sendcmd_rejects_length_mismatch(tmp_path):
    doc = _smooth_doc()
    with pytest.raises(PipelineError):
        build_sendcmd_dynamic(doc, str(tmp_path / "d.cmd"), [4, 4])


def test_dynamic_sendcmd_rejects_oversized_inset(tmp_path):
    doc = _smooth_doc()
    with pytest.raises(PipelineError):
        build_sendcmd_dynamic(doc, str(tmp_path / "d.cmd"),
                              [4, 4, 50, 4, 4])  # 短边 80，剩余高度 < 2


# ----------------------------------------------------- 镜像填充模式

def test_mirror_margin_covers_raw_inset():
    doc = _smooth_doc()  # crop: raw_inset=6, max_angle=2.0
    m = mirror_margin(doc)
    # 2° 时放大系数 1/(1-s-c)≈1.037：ceil(6×1.037)+1 = 8
    assert m == 8
    assert m >= doc["crop"]["raw_inset"]


def test_mirror_margin_rejects_extreme_angle():
    doc = _smooth_doc()
    doc["crop"]["max_angle"] = 60.0
    with pytest.raises(PipelineError):
        mirror_margin(doc)


def test_mirror_sendcmd_full_frame_window(tmp_path):
    doc = _smooth_doc()
    path = build_sendcmd_mirror(doc, str(tmp_path / "m.cmd"), margin=10)
    text = open(path, encoding="utf-8").read()
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 5
    # 窗口恒为原画幅：不下发 w/h，只下发原点
    assert "crop@cp w" not in text
    # 帧 0 补偿 dx=-2 → 原点 x = 10-(-2) = 12；dy=2 → 原点 y = 8
    assert "crop@cp x min(max(12.000000" in lines[0]
    assert "crop@cp y min(max(8.000000" in lines[0]
    # 原点钳制上界 = 2×margin
    assert "\\,20)" in lines[0]


# ----------------------------------------------------- 滤镜链分发

def _capture_ffmpeg(monkeypatch):
    """替换 _find 与 subprocess.run，返回捕获命令行的 dict。"""
    captured = {}
    monkeypatch.setattr("stabilize.compensate._find", lambda tool: tool)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def _vf_of(captured):
    cmd = captured["cmd"]
    return cmd[cmd.index("-vf") + 1]


def test_compensate_fixed_mode_chain_unchanged(tmp_path, monkeypatch):
    captured = _capture_ffmpeg(monkeypatch)
    doc = _smooth_doc()
    info = compensate_video(doc, "in.mp4", tmp_path / "o.mp4",
                            work_dir=str(tmp_path / "w"))
    vf = _vf_of(captured)
    # 旧行为回归：rotate + 小框 crop + pad 回原画幅
    assert "rotate@st=" in vf and "crop@cp=108:68" in vf
    assert "pad=120:80" in vf
    assert "fillborders" not in vf and "scale=" not in vf
    assert info["border_mode"] == "fixed" and info["final"] is False


def test_compensate_dynamic_mode_chain(tmp_path, monkeypatch):
    captured = _capture_ffmpeg(monkeypatch)
    doc = _smooth_doc()
    info = compensate_video(doc, "in.mp4", tmp_path / "o.mp4",
                            work_dir=str(tmp_path / "w"),
                            border_mode="dynamic")
    vf = _vf_of(captured)
    # 逐帧窗口 + 帧内缩回原尺寸，无 pad（无黑边）
    assert "crop@cp=" in vf
    assert "scale=120:80" in vf and "eval=frame" in vf
    assert "pad=" not in vf and "fillborders" not in vf
    assert info["final"] is True
    # sendcmd 副本含逐帧 w/h 命令
    text = open(info["sendcmd"], encoding="utf-8").read()
    assert "crop@cp w " in text


def test_compensate_mirror_mode_chain(tmp_path, monkeypatch):
    captured = _capture_ffmpeg(monkeypatch)
    doc = _smooth_doc()
    info = compensate_video(doc, "in.mp4", tmp_path / "o.mp4",
                            work_dir=str(tmp_path / "w"),
                            border_mode="mirror")
    vf = _vf_of(captured)
    # 先镜像扩边再补偿，窗口恒为原画幅，无 pad/scale
    assert vf.startswith("fillborders=8:8:8:8:mode=mirror")
    assert "crop@cp=120:80" in vf
    assert "pad=" not in vf and "scale=" not in vf
    assert info["final"] is True


def test_compensate_rejects_unknown_mode(tmp_path):
    doc = _smooth_doc()
    with pytest.raises(PipelineError):
        compensate_video(doc, "in.mp4", tmp_path / "o.mp4",
                         work_dir=str(tmp_path / "w"),
                         border_mode="bogus")
    assert "bogus" not in BORDER_MODES
