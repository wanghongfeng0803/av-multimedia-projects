"""规则引擎与批量编排的集成测试（注入假核心，不依赖 ffmpeg）。

锁定端到端行为：规则排除不产生清单条目、逐任务参数确实透传到核心、
CLI 显式参数覆盖规则、规则名写入清单、以及规则扩展名白名单生效。
"""

from __future__ import annotations

import json
import os

from bvp.batch import batch_transcode
from bvp.rules.engine import RuleSet
from bvp.rules.schema import build_rule_set

_CODECS = frozenset({"h264", "h265"})


def _touch(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\x00")
    return path


class _FakeResult:
    def __init__(self, output_path):
        self.output_path = output_path


def _fake_core(calls: list | None = None):
    def _transcode(input_path, output_path=None, **kwargs):
        if calls is not None:
            calls.append((input_path, output_path, kwargs))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(b"fake")
        return _FakeResult(output_path)
    return _transcode


def _rule_set(tmp_path, text: str) -> RuleSet:
    path = tmp_path / "rules.json"
    path.write_text(text, encoding="utf-8")
    config = build_rule_set(str(path), available_codecs=_CODECS)
    return RuleSet(config=config, config_path=str(path))


def test_excluded_files_get_no_manifest_entry(tmp_path):
    for name in ("keep.mp4", "drop.tmp.mp4"):
        _touch(str(tmp_path / name))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "all", "dir": ".", "ext": [".mp4"],
     "params": {"codec": "h264"}},
    {"name": "skip-temp", "dir": ".", "exclude": ["*.tmp.mp4"]}
  ]
}""")
    result = batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        transcode_fn=_fake_core(),
    )
    inputs = {os.path.basename(e.input_path) for e in result.manifest.entries}
    assert inputs == {"keep.mp4"}
    assert result.summary["total"] == 1


def test_per_file_params_reach_core(tmp_path):
    _touch(str(tmp_path / "media" / "a.mp4"))
    _touch(str(tmp_path / "media" / "sub" / "b.mp4"))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "base", "abstract": true,
     "params": {"codec": "h265", "container": "mkv"}},
    {"name": "media", "dir": "./media", "inherits": ["base"]},
    {"name": "sub-override", "priority": 5, "dir": "./media/sub",
     "params": {"codec": "h264"}}
  ]
}""")
    calls: list = []
    result = batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        transcode_fn=_fake_core(calls),
    )
    assert result.summary["success"] == 2
    by_name = {
        os.path.basename(inp): kwargs for inp, _out, kwargs in calls
    }
    assert by_name["a.mp4"]["codec"] == "h265"
    assert by_name["a.mp4"]["container"] == "mkv"
    assert by_name["b.mp4"]["codec"] == "h264"   # 高优先级覆盖
    assert by_name["b.mp4"]["container"] == "mkv"  # 继承字段仍生效


def test_rule_name_recorded_in_manifest(tmp_path):
    _touch(str(tmp_path / "media" / "a.mp4"))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "media", "dir": "./media", "params": {"codec": "h265"}}
  ]
}""")
    result = batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        transcode_fn=_fake_core(),
    )
    entry = result.manifest.entries[0]
    assert entry.rule_name == "media"
    # 落盘回读后仍保留
    with open(result.manifest_paths["full"], encoding="utf-8") as handle:
        doc = json.load(handle)
    assert doc["entries"][0]["rule_name"] == "media"


def test_explicit_cli_kwargs_override_rule_params(tmp_path):
    _touch(str(tmp_path / "media" / "a.mp4"))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "media", "dir": "./media",
     "params": {"codec": "h265", "crf": 30}}
  ]
}""")
    calls: list = []
    batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        codec="h264",  # 显式覆盖，压过规则的 h265
        transcode_fn=_fake_core(calls),
    )
    assert calls[0][2]["codec"] == "h264"
    # 未被显式覆盖的规则字段保持
    assert calls[0][2]["crf"] == 30


def test_rule_extension_whitelist_filters_scan(tmp_path):
    _touch(str(tmp_path / "a.mp4"))
    _touch(str(tmp_path / "b.mkv"))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "only-mp4", "dir": ".", "ext": [".mp4"],
     "params": {"codec": "h264"}}
  ]
}""")
    calls: list = []
    result = batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        transcode_fn=_fake_core(calls),
    )
    scanned = {os.path.basename(inp) for inp, _o, _k in calls}
    assert scanned == {"a.mp4"}
    assert result.summary["total"] == 1


def test_legacy_path_without_rules_unchanged(tmp_path):
    # 不传 rule_set 时，旧的全批统一参数路径行为不变
    _touch(str(tmp_path / "a.mp4"))
    calls: list = []
    result = batch_transcode(
        [str(tmp_path)], output_dir=str(tmp_path / "out"),
        codec="h265", transcode_fn=_fake_core(calls),
    )
    assert result.summary["success"] == 1
    assert calls[0][2]["codec"] == "h265"
    assert result.manifest.entries[0].rule_name is None


def test_jobs_and_recursive_taken_from_defaults(tmp_path):
    _touch(str(tmp_path / "a.mp4"))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "defaults": {"jobs": 2, "recursive": false},
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "h264"}}
  ]
}""")
    result = batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        transcode_fn=_fake_core(),
    )
    assert result.manifest.run_info["jobs"] == 2
    assert result.manifest.run_info["recursive"] is False
    assert result.manifest.run_info["rules_config"].endswith("rules.json")


def test_per_file_output_container_changes_extension(tmp_path):
    _touch(str(tmp_path / "media" / "a.mp4"))
    rules = _rule_set(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "mkv-rule", "dir": "./media",
     "params": {"container": "mkv", "codec": "h265"}}
  ]
}""")
    result = batch_transcode(
        [str(tmp_path)], rule_set=rules,
        output_dir=str(tmp_path / "out"),
        transcode_fn=_fake_core(),
    )
    output = result.manifest.success_entries[0].output_path
    assert output.endswith(".mkv")
