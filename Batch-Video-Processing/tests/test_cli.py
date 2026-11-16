"""CLI 冒烟测试：退出码与 verify 子命令（不依赖 ffmpeg，注入假核心）。"""

from __future__ import annotations

import argparse
import json
import os

from bvp import cli
from bvp.errors import FFmpegRunError
from bvp.planning import EXIST_SKIP


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\x00")


class _FakeResult:
    def __init__(self, output_path):
        self.output_path = output_path


def _success_core(input_path, output_path=None, **kwargs):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(b"x")
    return _FakeResult(output_path)


def _failing_core(input_path, output_path=None, **kwargs):
    raise FFmpegRunError("boom")


def _batch_namespace(tmp_path, out_dir):
    """构造与 argparse 解析结果等价的 Namespace，直接驱动 _run_batch。"""
    return argparse.Namespace(
        inputs=[str(tmp_path)],
        output_dir=out_dir,
        manifest_dir=None,
        codec="h264",
        container="mp4",
        crf=None,
        preset="medium",
        audio_bitrate="128k",
        resolution=None,
        jobs=1,
        on_exist=EXIST_SKIP,
        no_recursive=False,
        ext=None,
    )


def test_cli_batch_success_exit_zero(tmp_path, monkeypatch, capsys):
    _touch(str(tmp_path / "a.mp4"))
    out_dir = str(tmp_path / "out")
    monkeypatch.setattr("bvp.batch.core_module.transcode", _success_core)

    code = cli._run_batch(_batch_namespace(tmp_path, out_dir))
    assert code == 0
    printed = capsys.readouterr().out
    assert "成功 1" in printed
    assert "对账结果: 通过" in printed


def test_cli_batch_failure_exit_eight(tmp_path, monkeypatch, capsys):
    _touch(str(tmp_path / "bad.mp4"))
    monkeypatch.setattr("bvp.batch.core_module.transcode", _failing_core)

    capsys.readouterr()  # 清空既有缓冲，隔离本测试输出
    code = cli._run_batch(_batch_namespace(tmp_path, str(tmp_path / "out")))
    captured = capsys.readouterr()
    # 存在失败条目即返回 8（即使失败条目自身对账干净、无残留文件）
    assert code == 8
    assert "失败 1" in captured.out


def test_cli_reports_reconciliation_mismatch_to_stderr(tmp_path, monkeypatch,
                                                        capsys):
    """构造对账不符：成功条目落盘后，删除其产物再让 CLI 复核。"""
    _touch(str(tmp_path / "a.mp4"))
    out_dir = str(tmp_path / "out")
    monkeypatch.setattr("bvp.batch.core_module.transcode", _success_core)
    cli._run_batch(_batch_namespace(tmp_path, out_dir))

    # 成功清单已写、产物已存在；现在删掉产物，制造清单与磁盘不一致
    produced = os.path.join(out_dir, "a_h264.mp4")
    assert os.path.isfile(produced)
    os.remove(produced)

    capsys.readouterr()
    code = cli.main(["verify", out_dir])
    captured = capsys.readouterr()
    assert code == 8
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert any(
        m["kind"] == "success_output_missing" for m in payload["mismatches"]
    )


def test_cli_usage_error_raises_systemexit(tmp_path):
    # argparse 缺必填位置参数时抛 SystemExit(2)，由 main 直接传播
    try:
        cli.main(["batch"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("缺少 inputs 必须以 SystemExit(2) 退出")


def test_cli_verify_subcommand_reads_manifest(tmp_path, monkeypatch, capsys):
    _touch(str(tmp_path / "a.mp4"))
    out_dir = str(tmp_path / "out")
    monkeypatch.setattr("bvp.batch.core_module.transcode", _success_core)
    cli._run_batch(_batch_namespace(tmp_path, out_dir))

    capsys.readouterr()  # 清空批量阶段的输出，只保留 verify 的 JSON
    code = cli.main(["verify", out_dir])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["summary"]["success"] == 1


def test_parser_rejects_unknown_codec(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["batch", str(tmp_path), "--codec", "av1"])
