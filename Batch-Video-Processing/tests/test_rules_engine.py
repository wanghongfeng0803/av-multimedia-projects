"""规则引擎求值测试：匹配、继承、覆盖、组合收敛与可解释轨迹。

这些用例锁定「同一文件命中多条规则时收敛到唯一有效参数集」的确定性，
以及求值顺序（排除 → 命中收集 → 继承展开 → 优先级覆盖 → 缺省兜底）。
"""

from __future__ import annotations

import os

from bvp.rules.engine import DEFAULT_RULE_NAME, RuleSet
from bvp.rules.schema import build_rule_set

_CODECS = frozenset({"h264", "h265"})


def _build(tmp_path, text: str) -> RuleSet:
    path = tmp_path / "rules.json"
    path.write_text(text, encoding="utf-8")
    config = build_rule_set(str(path), available_codecs=_CODECS)
    return RuleSet(config=config, config_path=str(path))


def _p(tmp_path, *parts: str) -> str:
    return os.path.abspath(str(tmp_path.joinpath(*parts)))


def test_dir_and_ext_matching_uses_and_semantics(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "mp4-in-media", "dir": "./media", "ext": [".mp4"],
     "params": {"codec": "h265"}}
  ]
}""")
    hit = rules.resolve(_p(tmp_path, "media", "a.mp4"))
    miss_ext = rules.resolve(_p(tmp_path, "media", "a.mkv"))
    miss_dir = rules.resolve(_p(tmp_path, "other", "b.mp4"))

    assert hit.selected and hit.winning_rule == "mp4-in-media"
    assert hit.params.codec == "h265"
    # 未命中具体规则时落回隐式默认，文件仍入选（参数完整确定）
    assert miss_ext.selected and miss_ext.winning_rule == DEFAULT_RULE_NAME
    assert miss_ext.params.codec == "h264"
    assert miss_dir.selected and miss_dir.winning_rule == DEFAULT_RULE_NAME


def test_exclude_rule_filters_files_before_param_evaluation(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "all", "dir": "./media", "ext": [".mp4"],
     "params": {"codec": "h265"}},
    {"name": "skip-temp", "dir": "./media", "exclude": ["*.tmp.mp4"]}
  ]
}""")
    excluded = rules.resolve(_p(tmp_path, "media", "x.tmp.mp4"))
    included = rules.resolve(_p(tmp_path, "media", "x.mp4"))
    assert excluded.selected is False
    assert excluded.excluded_by == "skip-temp"
    assert included.winning_rule == "all"
    # 纯排除规则不出现在参数轨迹里
    assert all(step.rule != "skip-temp" for step in included.trace)


def test_inheritance_expands_base_params_with_child_override(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "base", "abstract": true,
     "params": {"codec": "h265", "container": "mkv", "preset": "slow"}},
    {"name": "child", "dir": "./media", "inherits": ["base"],
     "params": {"preset": "fast"}}
  ]
}""")
    result = rules.resolve(_p(tmp_path, "media", "a.mp4"))
    assert result.params.codec == "h265"          # 继承自基类
    assert result.params.container == "mkv"       # 继承自基类
    assert result.params.preset == "fast"         # 子类覆盖
    trace = result.trace[0]
    assert trace.rule == "child"
    assert trace.inherited_from == ("base",)


def test_higher_priority_overrides_lower_field_by_field(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "low", "priority": 1, "dir": "./media",
     "params": {"codec": "h264", "crf": 23, "preset": "medium"}},
    {"name": "high", "priority": 9, "dir": "./media",
     "params": {"crf": 30}}
  ]
}""")
    result = rules.resolve(_p(tmp_path, "media", "a.mp4"))
    assert result.params.crf == 30                # 高优先级覆盖
    assert result.params.codec == "h264"          # 低优先级独有字段保留
    assert result.params.preset == "medium"
    assert result.winning_rule == "high"          # 最高优先级规则裁决
    # 轨迹按优先级从高到低排列
    assert [step.rule for step in result.trace] == ["high", "low"]


def test_same_priority_merges_in_definition_order(tmp_path):
    # 同优先级但参数不重叠（矛盾情况已在 schema 层拒绝），按定义顺序组合
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "first", "dir": "./media", "params": {"codec": "h265"}},
    {"name": "second", "dir": "./media", "ext": [".mp4"],
     "params": {"crf": 28}}
  ]
}""")
    result = rules.resolve(_p(tmp_path, "media", "a.mp4"))
    assert result.params.codec == "h265"
    assert result.params.crf == 28


def test_defaults_layer_below_rules_above_builtin(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "defaults": {"crf": 26, "audio_bitrate": "192k", "jobs": 3},
  "rules": [
    {"name": "r1", "dir": "./media", "params": {"crf": 30}}
  ]
}""")
    result = rules.resolve(_p(tmp_path, "media", "a.mp4"))
    assert result.params.crf == 30                    # 规则压过 defaults
    assert result.params.audio_bitrate == "192k"      # defaults 生效
    assert result.params.preset == "medium"           # 内置默认兜底
    assert result.params.jobs == 3


def test_every_resolution_has_complete_param_set(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": "./media", "params": {"codec": "h265"}}
  ]
}""")
    for path in (_p(tmp_path, "media", "a.mp4"),
                 _p(tmp_path, "nowhere", "b.mkv")):
        result = rules.resolve(path)
        assert result.selected
        kwargs = result.params.core_kwargs()
        # 核心所需六个字段全部有确定值，无 KeyNames 缺失
        assert set(kwargs) == {
            "codec", "container", "crf", "preset",
            "audio_bitrate", "resolution",
        }


def test_relative_output_dir_resolved_against_config_dir(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": "./media",
     "params": {"output_dir": "./out"}}
  ]
}""")
    result = rules.resolve(_p(tmp_path, "media", "a.mp4"))
    assert result.params.output_dir == _p(tmp_path, "out")


def test_explain_contains_trace_and_effective_params(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": "./media", "params": {"codec": "h265"}}
  ]
}""")
    text = rules.explain(_p(tmp_path, "media", "a.mp4"))
    assert "r1" in text
    assert "h265" in text
    assert "裁决规则" in text


def test_extension_union_drives_scanner_whitelist(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "a", "dir": "./x", "ext": [".mp4"]},
    {"name": "b", "dir": "./y", "ext": [".mov", ".mkv"]}
  ]
}""")
    assert rules.all_extensions() == frozenset({".mp4", ".mov", ".mkv"})


def test_exclude_glob_with_subdir_path_pattern(tmp_path):
    # 回归：含分隔符的 glob（drafts/*）必须按相对路径匹配，且排除规则的
    # dir/ext 只作为生效子树前置条件，不能把 exclude 自身误当不命中
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "all", "dir": "./archive", "params": {"codec": "h265"}},
    {"name": "skip-drafts", "dir": "./archive", "exclude": ["drafts/*"]}
  ]
}""")
    excluded = rules.resolve(_p(tmp_path, "archive", "drafts", "x.mp4"))
    included = rules.resolve(_p(tmp_path, "archive", "keep.mp4"))
    assert excluded.selected is False
    assert excluded.excluded_by == "skip-drafts"
    assert included.selected is True


def test_nested_dir_prefix_boundary_no_false_match(tmp_path):
    rules = _build(tmp_path, """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": "./media", "params": {"codec": "h265"}}
  ]
}""")
    # /media-backup 不能被前缀误判为 /media 的子树
    result = rules.resolve(_p(tmp_path, "media-backup", "a.mp4"))
    assert result.winning_rule == DEFAULT_RULE_NAME
    inside = rules.resolve(_p(tmp_path, "media", "sub", "a.mp4"))
    assert inside.winning_rule == "r1"
