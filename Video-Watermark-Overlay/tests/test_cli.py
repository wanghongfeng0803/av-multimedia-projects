"""命令行入口测试：成功路径、参数校验、错误退出码与无交集告警。"""

import pytest

from watermark_overlay.cli import main

from conftest import make_png


def _prepare_inputs(tmp_path):
    frame = tmp_path / "frame.png"
    wm = tmp_path / "wm.png"
    make_png(str(frame), 8, 6, channels=4, fill=(40, 80, 120, 255))
    make_png(str(wm), 2, 2, channels=4, fill=(255, 0, 0, 200))
    return frame, wm


def test_cli_success(tmp_path, capsys):
    frame, wm = _prepare_inputs(tmp_path)
    out = tmp_path / "out.png"
    code = main(
        [str(frame), str(wm), "-o", str(out), "--anchor", "bottom-right", "--opacity", "0.5"]
    )
    assert code == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "完成" in printed


def test_cli_adaptive_flags(tmp_path):
    frame, wm = _prepare_inputs(tmp_path)
    out = tmp_path / "out.png"
    code = main(
        [
            str(frame), str(wm), "-o", str(out),
            "--adaptive", "--mask-strength", "0.8",
            "--detail-low", "10", "--detail-high", "120", "--mask-radius", "3",
            "--width", "4",
        ]
    )
    assert code == 0
    assert out.exists()


def test_cli_invalid_anchor_returns_error(tmp_path, capsys):
    frame, wm = _prepare_inputs(tmp_path)
    out = tmp_path / "out.png"
    # argparse choices 直接拦截，SystemExit 且退出码为 2
    with pytest.raises(SystemExit) as exc:
        main([str(frame), str(wm), "-o", str(out), "--anchor", "nonsense"])
    assert exc.value.code == 2


def test_cli_invalid_opacity_returns_exit_2(tmp_path, capsys):
    frame, wm = _prepare_inputs(tmp_path)
    out = tmp_path / "out.png"
    code = main([str(frame), str(wm), "-o", str(out), "--opacity", "1.5"])
    assert code == 2
    assert "错误" in capsys.readouterr().err
    assert not out.exists()


def test_cli_missing_input_file_returns_exit_2(tmp_path, capsys):
    out = tmp_path / "out.png"
    code = main([str(tmp_path / "missing.png"), str(tmp_path / "wm.png"), "-o", str(out)])
    assert code == 2
    assert "错误" in capsys.readouterr().err


def test_cli_no_overlap_warns_but_succeeds(tmp_path, capsys):
    frame, wm = _prepare_inputs(tmp_path)
    out = tmp_path / "out.png"
    code = main(
        [str(frame), str(wm), "-o", str(out), "--anchor", "top-left",
         "--offset-x", "100", "--offset-y", "100"]
    )
    assert code == 0
    assert "警告" in capsys.readouterr().err
    assert out.exists()


def test_cli_bad_detail_thresholds(tmp_path, capsys):
    frame, wm = _prepare_inputs(tmp_path)
    out = tmp_path / "out.png"
    code = main(
        [str(frame), str(wm), "-o", str(out), "--detail-low", "200", "--detail-high", "50"]
    )
    assert code == 2


def test_cli_motion_json_success(tmp_path, capsys):
    import json

    frame, wm = _prepare_inputs(tmp_path)
    motion = tmp_path / "motion.json"
    motion.write_text(
        json.dumps(
            {
                "keyframes": [
                    {"time": 0, "x": 0, "y": 0, "angle": 0},
                    {"time": 1, "x": 2, "y": 1, "angle": 90},
                ],
                "easing": "ease-in-out",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.png"
    code = main(
        [
            str(frame), str(wm), "-o", str(out),
            "--motion", str(motion), "--frame-index", "12", "--fps", "24",
        ]
    )
    assert code == 0
    assert out.exists()
    assert "角度" in capsys.readouterr().out


def test_cli_motion_bad_json_returns_exit_2(tmp_path, capsys):
    frame, wm = _prepare_inputs(tmp_path)
    motion = tmp_path / "motion.json"
    motion.write_text("{not json", encoding="utf-8")
    out = tmp_path / "out.png"
    code = main([str(frame), str(wm), "-o", str(out), "--motion", str(motion)])
    assert code == 2
    assert "错误" in capsys.readouterr().err
    assert not out.exists()


def test_cli_motion_invalid_config_returns_exit_2(tmp_path, capsys):
    import json

    frame, wm = _prepare_inputs(tmp_path)
    motion = tmp_path / "motion.json"
    # 关键帧时间未递增，validate 应拒绝
    motion.write_text(
        json.dumps({"keyframes": [{"time": 2}, {"time": 1}]}), encoding="utf-8"
    )
    out = tmp_path / "out.png"
    code = main([str(frame), str(wm), "-o", str(out), "--motion", str(motion)])
    assert code == 2
    assert not out.exists()
