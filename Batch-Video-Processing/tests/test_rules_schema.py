"""规则配置 schema 与**全局可满足性校验**测试（不依赖 ffmpeg）。

校验分三层，本文件逐一锁定：

1. 词法/结构：JSON 语法、重复键、闭合键集合、版本号；
2. 静态值域：参数类型、取值区间、分辨率语法、动态表达式拒绝；
3. 全局组合：同优先级矛盾、不支持编码器、悬空/循环继承。

所有失败都必须带具体行号并一次性收齐，而非遇到第一个就停。
"""

from __future__ import annotations

import os

import pytest

from bvp.rules.schema import RuleConfigError, build_rule_set

_CODECS = frozenset({"h264", "h265"})


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _issues(path: str, codecs=_CODECS):
    with pytest.raises(RuleConfigError) as exc_info:
        build_rule_set(path, available_codecs=codecs)
    return exc_info.value.issues


def test_valid_minimal_config_builds(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "h264"}}
  ]
}""")
    config = build_rule_set(path, available_codecs=_CODECS)
    assert config.rules[0].name == "r1"
    # 相对目录以配置文件所在目录为基准
    assert config.rules[0].dirs == (str(tmp_path),)


def test_json_syntax_error_reports_line(tmp_path):
    path = _write(tmp_path, "r.json", '{\n  "version": 1,\n  broken\n}')
    issues = _issues(path)
    assert any(i.line == 3 for i in issues)


def test_duplicate_key_rejected_with_line(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [{"name": "r1", "dir": ".", "params": {"codec": "h264"}}],
  "rules": []
}""")
    issues = _issues(path)
    assert any("重复键" in i.message and i.line == 4 for i in issues)


def test_unsupported_version_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 99,
  "rules": [{"name": "r1", "dir": "."}]
}""")
    issues = _issues(path)
    assert any("版本" in i.message for i in issues)


def test_unknown_top_level_and_rule_keys_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "bogus": 1,
  "rules": [{"name": "r1", "dir": ".", "weird": 2}]
}""")
    issues = _issues(path)
    messages = " ".join(i.message for i in issues)
    assert "bogus" in messages
    assert "weird" in messages


# ---------------------------------------------------------------------------
# 静态值域
# ---------------------------------------------------------------------------

def test_crf_out_of_range_rejected_with_line(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"crf": 99}}
  ]
}""")
    issues = _issues(path)
    issue = next(i for i in issues if "crf" in i.message)
    assert issue.line == 4


def test_bad_resolution_syntax_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"resolution": "1080q"}}
  ]
}""")
    issues = _issues(path)
    assert any("分辨率" in i.message for i in issues)


def test_valid_resolution_forms_accepted(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "a", "dir": "a", "params": {"resolution": "720p"}},
    {"name": "b", "dir": "b", "params": {"resolution": "50%"}},
    {"name": "c", "dir": "c", "params": {"resolution": "1280x720"}}
  ]
}""")
    config = build_rule_set(path, available_codecs=_CODECS)
    assert len(config.rules) == 3


def test_dynamic_attribute_dependency_rejected_even_if_well_typed(tmp_path):
    # 关键契约：依赖文件运行期属性（分辨率/时长）的取值在启动期拒绝，
    # 而不是推迟到任务执行时
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"resolution": "${width/2}"}}
  ]
}""")
    issues = _issues(path)
    assert any("运行期属性" in i.message for i in issues)


def test_equals_prefix_and_object_params_rejected_as_dynamic(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"crf": "= 20 + 1"}}
  ]
}""")
    issues = _issues(path)
    assert any("运行期属性" in i.message for i in issues)


def test_unknown_param_key_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"bitrate": "2M"}}
  ]
}""")
    issues = _issues(path)
    assert any("未知参数" in i.message and "bitrate" in i.message
               for i in issues)


def test_bad_bitrate_and_preset_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".",
     "params": {"audio_bitrate": "fast", "preset": "turbo"}}
  ]
}""")
    issues = _issues(path)
    messages = [i.message for i in issues]
    assert any("audio_bitrate" in m for m in messages)
    assert any("preset" in m for m in messages)


def test_on_exist_domain_checked(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"on_exist": "nuke"}}
  ]
}""")
    issues = _issues(path)
    assert any("on_exist" in i.message for i in issues)


# ---------------------------------------------------------------------------
# 编码器两级校验
# ---------------------------------------------------------------------------

def test_unsupported_build_codec_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "av1"}}
  ]
}""")
    issues = _issues(path)
    assert any("编码器" in i.message and "av1" in i.message for i in issues)


def test_build_supported_but_machine_unavailable_codec_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "h265"}}
  ]
}""")
    # 构建支持 h265，但本机探测只返回 h264
    issues = _issues(path, codecs=frozenset({"h264"}))
    assert any("本机不支持" in i.message and "h265" in i.message
               for i in issues)


def test_defaults_codec_also_validated(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "defaults": {"codec": "av1"},
  "rules": [{"name": "r1", "dir": "."}]
}""")
    issues = _issues(path)
    assert any("defaults 引用的编码器" in i.message for i in issues)


# ---------------------------------------------------------------------------
# 继承
# ---------------------------------------------------------------------------

def test_dangling_inheritance_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "inherits": ["ghost"]}
  ]
}""")
    issues = _issues(path)
    assert any("不存在的规则" in i.message for i in issues)


def test_inheritance_cycle_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "a", "abstract": true, "inherits": ["b"]},
    {"name": "b", "abstract": true, "inherits": ["a"]},
    {"name": "c", "dir": ".", "inherits": ["a"]}
  ]
}""")
    issues = _issues(path)
    assert any("循环" in i.message for i in issues)


def test_abstract_rule_with_match_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "base", "abstract": true, "dir": ".",
     "params": {"codec": "h264"}}
  ]
}""")
    issues = _issues(path)
    assert any("抽象规则" in i.message for i in issues)


def test_concrete_rule_without_match_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "params": {"codec": "h264"}}
  ]
}""")
    issues = _issues(path)
    assert any("匹配条件" in i.message for i in issues)


def test_duplicate_rule_name_rejected(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "dup", "dir": "a"},
    {"name": "dup", "dir": "b"}
  ]
}""")
    issues = _issues(path)
    assert any("重复" in i.message for i in issues)


# ---------------------------------------------------------------------------
# 全局矛盾（规则对组合，非逐字段）
# ---------------------------------------------------------------------------

def test_same_priority_conflicting_params_rejected_with_param_line(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "h264"}},
    {"name": "r2", "dir": ".", "params": {"codec": "h265"}}
  ]
}""")
    issues = _issues(path)
    issue = next(i for i in issues if "矛盾" in i.message)
    # 行号必须定位到 r2 的 params.codec（第 5 行），而非文件首行
    assert issue.line == 5
    assert "h265" in issue.message and "h264" in issue.message


def test_different_priority_is_override_not_conflict(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "low", "priority": 1, "dir": ".",
     "params": {"codec": "h264"}},
    {"name": "high", "priority": 5, "dir": ".",
     "params": {"codec": "h265"}}
  ]
}""")
    # 不同优先级语义明确（高覆盖低），不得报错
    config = build_rule_set(path, available_codecs=_CODECS)
    assert len(config.rules) == 2


def test_disjoint_extensions_do_not_conflict(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "mp4", "dir": ".", "ext": [".mp4"],
     "params": {"codec": "h264"}},
    {"name": "mkv", "dir": ".", "ext": [".mkv"],
     "params": {"codec": "h265"}}
  ]
}""")
    config = build_rule_set(path, available_codecs=_CODECS)
    assert len(config.rules) == 2


def test_same_value_is_not_conflict(tmp_path):
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": "a", "params": {"crf": 24}},
    {"name": "r2", "dir": "b", "params": {"crf": 24}}
  ]
}""")
    config = build_rule_set(path, available_codecs=_CODECS)
    assert len(config.rules) == 2


def test_multiple_issues_collected_at_once(tmp_path):
    # 不允许遇到第一个错误就停：一次启动校验应收齐全部问题
    path = _write(tmp_path, "r.json", """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"crf": 99, "codec": "av1"}},
    {"name": "r2", "dir": ".", "params": {"crf": 80}}
  ]
}""")
    issues = _issues(path)
    assert len(issues) >= 3  # 两个 crf 越界 + 一个坏编码器
