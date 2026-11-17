"""规则引擎相关 CLI 测试：rules-check、坏配置拒绝启动、watch 热加载（无 ffmpeg）。"""

from __future__ import annotations

import os

from bvp import cli


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


def _good_config(tmp_path) -> str:
    path = tmp_path / "rules.json"
    path.write_text("""{
  "version": 1,
  "rules": [
    {"name": "media", "dir": ".", "params": {"codec": "h265"}}
  ]
}""", encoding="utf-8")
    return str(path)


def test_rules_check_passes_valid_config(tmp_path, capsys):
    path = _good_config(tmp_path)
    code = cli.main(["rules-check", path, str(tmp_path / "a.mp4")])
    assert code == 0
    out = capsys.readouterr().out
    assert "校验通过" in out
    assert "media" in out


def test_rules_check_bad_config_exit_two(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("""{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "av1"}}
  ]
}""", encoding="utf-8")
    code = cli.main(["rules-check", str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "av1" in err


def test_batch_with_bad_rules_refuses_startup(tmp_path, capsys, monkeypatch):
    _touch(str(tmp_path / "a.mp4"))
    path = tmp_path / "bad.json"
    path.write_text('{"version": 1, "rules": []}', encoding="utf-8")
    monkeypatch.setattr("bvp.batch.core_module.transcode", _success_core)
    code = cli.main(["batch", str(tmp_path), "--rules", str(path),
                     "-o", str(tmp_path / "out")])
    assert code == 2
    err = capsys.readouterr().err
    assert "rules" in err


def test_batch_with_rules_uses_rule_params(tmp_path, monkeypatch, capsys):
    _touch(str(tmp_path / "a.mp4"))
    path = _good_config(tmp_path)
    captured: list = []

    def _core(input_path, output_path=None, **kwargs):
        captured.append(kwargs)
        return _success_core(input_path, output_path, **kwargs)

    monkeypatch.setattr("bvp.batch.core_module.transcode", _core)
    code = cli.main(["batch", str(tmp_path), "--rules", path,
                     "-o", str(tmp_path / "out")])
    assert code == 0
    assert captured[0]["codec"] == "h265"  # 规则参数生效，非内置 h264


def test_explicit_cli_codec_overrides_rule(tmp_path, monkeypatch):
    _touch(str(tmp_path / "a.mp4"))
    path = _good_config(tmp_path)
    captured: list = []

    def _core(input_path, output_path=None, **kwargs):
        captured.append(kwargs)
        return _success_core(input_path, output_path, **kwargs)

    monkeypatch.setattr("bvp.batch.core_module.transcode", _core)
    code = cli.main(["batch", str(tmp_path), "--rules", path,
                     "--codec", "h264", "-o", str(tmp_path / "out")])
    assert code == 0
    assert captured[0]["codec"] == "h264"


def test_watch_mode_runs_multiple_cycles_and_reloads(tmp_path, monkeypatch):
    _touch(str(tmp_path / "a.mp4"))
    path = _good_config(tmp_path)
    monkeypatch.setattr("bvp.batch.core_module.transcode", _success_core)
    # 两轮批量后退出；轮询间隔压到最小，不真实睡眠太久
    code = cli.main([
        "batch", str(tmp_path), "--rules", path, "--watch",
        "--max-cycles", "2", "--poll-interval", "0.1",
        "-o", str(tmp_path / "out"),
    ])
    assert code == 0


def test_watch_bad_reload_keeps_old_config(tmp_path, monkeypatch, capsys):
    _touch(str(tmp_path / "a.mp4"))
    path = _good_config(tmp_path)
    monkeypatch.setattr("bvp.batch.core_module.transcode", _success_core)

    real_maybe_reload = None
    from bvp.rules.store import ConfigStore, ReloadResult
    from bvp.rules.schema import RuleConfigError, RuleConfigIssue

    def _flaky_reload(self, *, force=False):
        # 第一轮返回坏配置错误，验证现行快照不被污染、批量仍成功
        return ReloadResult(
            changed=True, loaded=False, rule_set=self.snapshot(),
            error=RuleConfigError(
                [RuleConfigIssue(3, "模拟的坏配置")], str(path)
            ),
            path=str(path),
        )

    monkeypatch.setattr(ConfigStore, "maybe_reload", _flaky_reload)
    code = cli.main([
        "batch", str(tmp_path), "--rules", path, "--watch",
        "--max-cycles", "1", "--poll-interval", "0.1",
        "-o", str(tmp_path / "out"),
    ])
    # 坏配置只告警，不中断服务，批量本身成功
    assert code == 0
    assert "热加载失败" in capsys.readouterr().err
