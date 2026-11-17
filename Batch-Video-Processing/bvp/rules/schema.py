"""v1 规则配置的模式定义、行号标注与**启动期全局可满足性校验**。

配置文件是一份静态声明，规则语言在 v1 刻意保持**纯字面量、可静态判定**：

- 匹配条件只允许路径维度（``dir`` / ``ext`` / ``exclude``），不允许引用
  被匹配文件的运行期属性（分辨率/时长/编码）；
- 参数值只允许标量字面量（字符串/整数/布尔），任何表达式、插值或属性引用
  （``"${...}"``、``"=..."`` 等）一律在启动时按行拒绝；
- 因而「同一份配置对任意可能输入会产出什么参数」在启动前就能完整推理，
  全局可满足性校验（见 :func:`build_rule_set`）不是逐字段类型检查，而是
  对**规则对之间的组合关系**做裁决。

可满足性校验覆盖三类全局问题，任何一类命中都拒绝启动：

1. **矛盾参数**：两条同优先级、可能命中同一文件的规则对同一参数给出不同
   字面值（不同优先级不是矛盾——高优先级覆盖低优先级，语义明确）；
2. **不支持的编码器**：参数引用的 codec 不在构建支持集合，或本机 ffmpeg
   实际不提供该编码器（编码器探测可注入，纯逻辑测试不依赖 ffmpeg）；
3. **属性依赖**：参数取值需要被匹配文件的分辨率/时长等运行期属性才能确定，
   本版本只接受静态字面量，命中即报错并定位到具体行。

所有问题都带 JSON 路径对应的源文件行号，:class:`RuleConfigError` 一次性
收集全部问题输出，而不是遇到第一个就停。
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass

from ..core import parse_resolution_spec, supported_codecs, supported_containers
from ..errors import InvalidInputError
from ..planning import EXIST_OVERWRITE, EXIST_SKIP, EXIST_SUFFIX
from .linemap import LineMap, build_line_map

__all__ = [
    "RULE_CONFIG_VERSION",
    "RULE_PARAM_KEYS",
    "MATCH_KEYS",
    "RuleConfigIssue",
    "RuleConfigError",
    "RuleConfig",
    "load_config_text",
    "build_rule_set",
]

#: 配置格式版本，结构不兼容变化时递增
RULE_CONFIG_VERSION = 1

#: 每条规则可声明的转码参数键（闭合集合，出现其他键即报错）
RULE_PARAM_KEYS: frozenset[str] = frozenset({
    "codec", "container", "crf", "preset", "audio_bitrate",
    "resolution", "on_exist", "output_dir",
})

#: defaults 段允许的额外键（批量级、非单文件转码参数）
_DEFAULTS_ONLY_KEYS: frozenset[str] = frozenset({"jobs", "recursive"})

#: 匹配条件键（闭合集合）
MATCH_KEYS: frozenset[str] = frozenset({"dir", "ext", "exclude"})

_PRESETS: frozenset[str] = frozenset({
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo",
})
_EXIST_POLICIES: frozenset[str] = frozenset(
    {EXIST_SKIP, EXIST_OVERWRITE, EXIST_SUFFIX}
)
_BITRATE_RE = re.compile(r"^\d+(?:\.\d+)?[kKmM]?$")
#: 属性依赖/表达式写法的可疑前缀与插值（v1 不支持动态取值）
_DYNAMIC_MARKERS = ("${", "{{", "= ")


@dataclass(frozen=True)
class RuleConfigIssue:
    """一条配置问题，带源文件行号（无法定位时 line 为 None）。"""

    line: int | None
    message: str
    rule: str | None = None

    def format(self, path: str) -> str:
        location = f"{path}:{self.line}" if self.line else path
        rule = f"（规则 {self.rule!r}）" if self.rule else ""
        return f"{location}: {self.message}{rule}"


class RuleConfigError(ValueError):
    """配置无法通过全局可满足性校验；携带全部问题与首个问题的行号。"""

    def __init__(self, issues: list[RuleConfigIssue], config_path: str):
        self.issues = issues
        self.config_path = config_path
        self.line = next((i.line for i in issues if i.line), None)
        body = "\n".join(issue.format(config_path) for issue in issues)
        super().__init__(
            f"规则配置校验失败，共 {len(issues)} 个问题：\n{body}"
        )


@dataclass(frozen=True)
class RuleDefinition:
    """一条通过校验的规则定义（不可变，可在线程间安全共享）。"""

    name: str
    priority: int
    dirs: tuple[str, ...]               # 已规范化为绝对路径（配置目录为基准）
    exts: frozenset[str]                # 小写、含点
    exclude: tuple[str, ...]            # glob 原样保留，求值期匹配
    params: dict                        # 已做静态值域校验的标量参数
    abstract: bool
    inherits: tuple[str, ...]
    line: int
    config_dir: str


@dataclass
class RuleConfig:
    """解析并校验通过的完整配置（defaults + 规则序列）。"""

    defaults: dict
    rules: tuple[RuleDefinition, ...]
    line_map: LineMap
    config_dir: str
    source_text: str

    def rule_param_line(self, rule_name: str, key: str) -> int | None:
        """定位某规则 params.<key> 的源文件行号，缺失时回退到规则起点。"""
        for index, rule in enumerate(self.rules):
            if rule.name != rule_name:
                continue
            line = self.line_map.line_of_path(
                ("rules", index, "params", key)
            )
            return line if line is not None else rule.line
        return None


# ---------------------------------------------------------------------------
# 加载与解析
# ---------------------------------------------------------------------------

def load_config_text(path: str) -> tuple[dict, LineMap, str]:
    """读取配置文本并解析为 JSON，同时建立行号映射。

    :raises RuleConfigError: 文件不可读、JSON 语法错误或存在重复键
    """
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise RuleConfigError(
            [RuleConfigIssue(None, f"配置文件无法读取: {exc}")], abs_path
        ) from exc

    line_map = build_line_map(text)
    if line_map.duplicates:
        issues = [
            RuleConfigIssue(
                dup.line,
                f"对象内存在重复键 {dup.key!r}，标准 JSON 会静默取后者，"
                "规则引擎要求显式消除歧义。",
            )
            for dup in line_map.duplicates
        ]
        raise RuleConfigError(issues, abs_path)

    import json
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuleConfigError(
            [RuleConfigIssue(exc.lineno, f"JSON 语法错误: {exc.msg}")],
            abs_path,
        ) from exc
    if not isinstance(data, dict):
        raise RuleConfigError(
            [RuleConfigIssue(1, "配置顶层必须是 JSON 对象。")], abs_path
        )
    return data, line_map, text


# ---------------------------------------------------------------------------
# 校验与构造
# ---------------------------------------------------------------------------

def build_rule_set(
    path: str,
    *,
    available_codecs: frozenset[str] | None = None,
) -> RuleConfig:
    """加载并执行完整的启动期校验，返回不可变 :class:`RuleConfig`。

    :param available_codecs: 本机 ffmpeg 实际支持的 codec 规范名集合；
        None 时只校验「构建支持」（:func:`bvp.core.supported_codecs`），
        不探测本机环境，便于无 ffmpeg 的纯逻辑测试
    :raises RuleConfigError: 任何全局可满足性问题（含行号清单）
    """
    data, line_map, text = load_config_text(path)
    config_dir = os.path.dirname(os.path.abspath(path))
    issues: list[RuleConfigIssue] = []

    _check_top_level(data, line_map, issues)
    defaults = _validate_defaults(data, line_map, config_dir, issues)
    rules = _validate_rules(data, line_map, config_dir, issues)

    # 名字引用（继承）与唯一性
    names = {rule.name for rule in rules}
    for rule in rules:
        for parent in rule.inherits:
            if parent not in names:
                issues.append(RuleConfigIssue(
                    rule.line,
                    f"规则 {rule.name!r} 继承了不存在的规则 {parent!r}。",
                    rule.name,
                ))
        if _inherits_cycle(rule.name, rules):
            issues.append(RuleConfigIssue(
                rule.line,
                f"规则继承链存在循环（涉及 {rule.name!r}）。", rule.name,
            ))

    # 编码器两级校验：构建支持 + 本机可用
    build_codecs = frozenset(supported_codecs())
    usable_codecs = (
        build_codecs if available_codecs is None
        else frozenset(available_codecs) & build_codecs
    )
    _check_codecs(defaults, rules, line_map, usable_codecs, issues)

    # 全局矛盾：同优先级规则对的参数冲突
    _check_pairwise_conflicts(rules, line_map, issues)

    if issues:
        raise RuleConfigError(issues, os.path.abspath(path))
    return RuleConfig(
        defaults=defaults,
        rules=rules,
        line_map=line_map,
        config_dir=config_dir,
        source_text=text,
    )


def _check_top_level(data: dict, line_map: LineMap,
                     issues: list[RuleConfigIssue]) -> None:
    allowed = {"version", "defaults", "rules"}
    for key in data:
        if key not in allowed:
            issues.append(RuleConfigIssue(
                line_map.line_of(key), f"顶层出现未知键 {key!r}。",
            ))
    version = data.get("version")
    if version is None:
        issues.append(RuleConfigIssue(
            line_map.line_of("version"),
            "缺少 version 字段（当前支持版本 1）。",
        ))
    elif version != RULE_CONFIG_VERSION:
        issues.append(RuleConfigIssue(
            line_map.line_of("version"),
            f"不支持的配置版本 {version!r}，当前仅支持 {RULE_CONFIG_VERSION}。",
        ))
    rules_node = data.get("rules")
    if not isinstance(rules_node, list) or not rules_node:
        issues.append(RuleConfigIssue(
            line_map.line_of("rules"),
            "rules 必须是非空数组，至少定义一条规则。",
        ))


def _validate_defaults(
    data: dict, line_map: LineMap, config_dir: str,
    issues: list[RuleConfigIssue],
) -> dict:
    raw = data.get("defaults")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        issues.append(RuleConfigIssue(
            line_map.line_of("defaults"), "defaults 必须是 JSON 对象。",
        ))
        return {}
    return _validate_param_object(
        raw, line_map, ("defaults",), config_dir, issues,
        allow_batch_keys=True, rule_name=None,
    )


def _validate_rules(
    data: dict, line_map: LineMap, config_dir: str,
    issues: list[RuleConfigIssue],
) -> list[RuleDefinition]:
    rules: list[RuleDefinition] = []
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        return rules

    seen_names: set[str] = set()
    for index, raw in enumerate(raw_rules):
        base = ("rules", index)
        if not isinstance(raw, dict):
            issues.append(RuleConfigIssue(
                line_map.line_of_path(base),
                f"rules[{index}] 必须是 JSON 对象。",
            ))
            continue

        name = raw.get("name")
        if not isinstance(name, str) or not name:
            issues.append(RuleConfigIssue(
                line_map.line_of_path(base + ("name",)),
                f"rules[{index}] 缺少非空字符串 name。",
            ))
            continue
        if name in seen_names:
            issues.append(RuleConfigIssue(
                line_map.line_of_path(base + ("name",)),
                f"规则名 {name!r} 重复，规则名必须唯一。", name,
            ))
        seen_names.add(name)
        # 规则起点行：优先 name 键所在行（必然存在且稳定），再退回 rules 数组
        rule_line = line_map.line_of_path(base + ("name",)) \
            or line_map.line_of("rules") or 1

        for key in raw:
            if key not in {"name", "priority", "inherits", "abstract",
                           "params"} | MATCH_KEYS:
                issues.append(RuleConfigIssue(
                    line_map.line_of_path(base + (key,)),
                    f"规则 {name!r} 出现未知键 {key!r}。", name,
                ))

        priority = _validate_priority(raw, base, line_map, name, issues)
        inherits = _validate_inherits(raw, base, line_map, name, issues)
        abstract = bool(raw.get("abstract", False))
        dirs, exts, exclude = _validate_match(
            raw, base, line_map, config_dir, name, issues,
        )
        params = _validate_params(
            raw, base, line_map, config_dir, name, issues,
        )

        has_match = bool(raw.get("dir") or raw.get("ext") or raw.get("exclude"))
        if not abstract and not has_match:
            issues.append(RuleConfigIssue(
                rule_line,
                f"规则 {name!r} 没有任何匹配条件（dir/ext/exclude）；"
                "仅可被继承的基类规则须显式声明 abstract: true。", name,
            ))
        if abstract and has_match:
            issues.append(RuleConfigIssue(
                rule_line,
                f"抽象规则 {name!r} 不应携带匹配条件，它只用于被继承。", name,
            ))

        rules.append(RuleDefinition(
            name=name,
            priority=priority,
            dirs=dirs,
            exts=exts,
            exclude=exclude,
            params=params,
            abstract=abstract,
            inherits=tuple(inherits),
            line=rule_line,
            config_dir=config_dir,
        ))
    return rules


def _validate_priority(raw, base, line_map, name, issues) -> int:
    if "priority" not in raw:
        return 0
    value = raw["priority"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        issues.append(RuleConfigIssue(
            line_map.line_of_path(base + ("priority",)),
            f"规则 {name!r} 的 priority 必须是 >= 0 的整数。", name,
        ))
        return 0
    return value


def _validate_inherits(raw, base, line_map, name, issues) -> list[str]:
    value = raw.get("inherits", [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        issues.append(RuleConfigIssue(
            line_map.line_of_path(base + ("inherits",)),
            f"规则 {name!r} 的 inherits 必须是规则名字符串数组。", name,
        ))
        return []
    return value


def _validate_match(
    raw, base, line_map, config_dir, name, issues,
) -> tuple[tuple[str, ...], frozenset[str], tuple[str, ...]]:
    dirs: tuple[str, ...] = ()
    raw_dir = raw.get("dir")
    if raw_dir is not None:
        if isinstance(raw_dir, str):
            raw_dir = [raw_dir]
        if not isinstance(raw_dir, list) or not all(
            isinstance(item, str) and item for item in raw_dir
        ):
            issues.append(RuleConfigIssue(
                line_map.line_of_path(base + ("dir",)),
                f"规则 {name!r} 的 dir 必须是字符串或字符串数组。", name,
            ))
        else:
            dirs = tuple(
                _resolve_dir(item, config_dir) for item in raw_dir
            )

    exts = frozenset()
    raw_ext = raw.get("ext")
    if raw_ext is not None:
        if isinstance(raw_ext, str):
            raw_ext = [raw_ext]
        if not isinstance(raw_ext, list) or not all(
            isinstance(item, str) and item for item in raw_ext
        ):
            issues.append(RuleConfigIssue(
                line_map.line_of_path(base + ("ext",)),
                f"规则 {name!r} 的 ext 必须是字符串或字符串数组。", name,
            ))
        else:
            normalized = set()
            for item in raw_ext:
                item = item.strip().lower()
                normalized.add(item if item.startswith(".") else "." + item)
            exts = frozenset(normalized)

    exclude: tuple[str, ...] = ()
    raw_exclude = raw.get("exclude")
    if raw_exclude is not None:
        if isinstance(raw_exclude, str):
            raw_exclude = [raw_exclude]
        if not isinstance(raw_exclude, list) or not all(
            isinstance(item, str) and item for item in raw_exclude
        ):
            issues.append(RuleConfigIssue(
                line_map.line_of_path(base + ("exclude",)),
                f"规则 {name!r} 的 exclude 必须是字符串或字符串数组。", name,
            ))
        else:
            exclude = tuple(raw_exclude)
    return dirs, exts, exclude


def _validate_params(
    raw, base, line_map, config_dir, name, issues,
) -> dict:
    params_node = raw.get("params")
    if params_node is None:
        return {}
    if not isinstance(params_node, dict):
        issues.append(RuleConfigIssue(
            line_map.line_of_path(base + ("params",)),
            f"规则 {name!r} 的 params 必须是 JSON 对象。", name,
        ))
        return {}
    return _validate_param_object(
        params_node, line_map, base + ("params",), config_dir, issues,
        allow_batch_keys=False, rule_name=name,
    )


def _validate_param_object(
    node: dict, line_map: LineMap, base: tuple, config_dir: str,
    issues: list[RuleConfigIssue], *, allow_batch_keys: bool,
    rule_name: str | None,
) -> dict:
    validated: dict = {}
    allowed = RULE_PARAM_KEYS | (
        _DEFAULTS_ONLY_KEYS if allow_batch_keys else frozenset()
    )
    for key, value in node.items():
        path = base + (key,)
        if key not in allowed:
            issues.append(RuleConfigIssue(
                line_map.line_of_path(path),
                f"出现未知参数 {key!r}；可识别参数: "
                f"{', '.join(sorted(allowed))}。", rule_name,
            ))
            continue
        if _is_dynamic(value):
            issues.append(RuleConfigIssue(
                line_map.line_of_path(path),
                f"参数 {key!r} 的取值依赖表达式或文件运行期属性；v1 规则语言"
                "只接受静态字面量，不支持按分辨率/时长等动态取值。",
                rule_name,
            ))
            continue
        ok = _validate_domain(key, value, line_map, path, config_dir,
                              issues, rule_name)
        if ok:
            validated[key] = value
    return validated


def _is_dynamic(value: object) -> bool:
    """属性依赖判定：非字面量结构或带表达式标记的字符串都拒绝。"""
    if isinstance(value, (dict, list)):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return (
            stripped.startswith(("=", "$", "@"))
            or any(marker in stripped for marker in _DYNAMIC_MARKERS)
        )
    return False


def _validate_domain(
    key: str, value: object, line_map: LineMap, path: tuple,
    config_dir: str, issues: list[RuleConfigIssue], rule_name: str | None,
) -> bool:
    """逐参数静态值域校验；返回是否通过。"""
    line = line_map.line_of_path(path)
    if key == "codec":
        if not isinstance(value, str) or not value:
            issues.append(RuleConfigIssue(
                line, "codec 必须是非空字符串。", rule_name,
            ))
            return False
        return True  # 是否被构建/本机支持在 _check_codecs 统一裁决
    if key == "container":
        if value not in supported_containers():
            issues.append(RuleConfigIssue(
                line,
                f"不支持的容器 {value!r}，可选: "
                f"{', '.join(supported_containers())}。", rule_name,
            ))
            return False
        return True
    if key == "crf":
        if not isinstance(value, int) or isinstance(value, bool) \
                or not (0 <= value <= 51):
            issues.append(RuleConfigIssue(
                line, f"crf 必须是 0-51 的整数，收到 {value!r}。", rule_name,
            ))
            return False
        return True
    if key == "preset":
        if value not in _PRESETS:
            issues.append(RuleConfigIssue(
                line,
                f"preset {value!r} 不在 x264/x265 档位集合内: "
                f"{', '.join(sorted(_PRESETS))}。", rule_name,
            ))
            return False
        return True
    if key == "audio_bitrate":
        if not isinstance(value, str) or not _BITRATE_RE.match(value):
            issues.append(RuleConfigIssue(
                line,
                f"audio_bitrate {value!r} 写法非法，支持如 128k / 192k / 320k。",
                rule_name,
            ))
            return False
        return True
    if key == "resolution":
        if not isinstance(value, str) or not value:
            issues.append(RuleConfigIssue(
                line, "resolution 必须是非空字符串。", rule_name,
            ))
            return False
        try:
            parse_resolution_spec(value)  # 纯语法校验，不读任何文件属性
        except InvalidInputError as exc:
            issues.append(RuleConfigIssue(line, str(exc), rule_name))
            return False
        return True
    if key == "on_exist":
        if value not in _EXIST_POLICIES:
            issues.append(RuleConfigIssue(
                line,
                f"on_exist 必须是 {', '.join(sorted(_EXIST_POLICIES))} 之一，"
                f"收到 {value!r}。", rule_name,
            ))
            return False
        return True
    if key == "output_dir":
        if not isinstance(value, str) or not value:
            issues.append(RuleConfigIssue(
                line, "output_dir 必须是非空字符串。", rule_name,
            ))
            return False
        return True
    if key == "jobs":
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            issues.append(RuleConfigIssue(
                line, f"jobs 必须是 >= 1 的整数，收到 {value!r}。", rule_name,
            ))
            return False
        return True
    if key == "recursive":
        if not isinstance(value, bool):
            issues.append(RuleConfigIssue(
                line, f"recursive 必须是布尔值，收到 {value!r}。", rule_name,
            ))
            return False
        return True
    issues.append(RuleConfigIssue(line, f"参数 {key!r} 暂无校验规则。",
                                  rule_name))
    return False


def _check_codecs(
    defaults: dict, rules: list[RuleDefinition], line_map: LineMap,
    usable_codecs: frozenset[str], issues: list[RuleConfigIssue],
) -> None:
    """编码器两级校验：所有引用点都必须落在「构建支持且本机可用」集合。"""
    if "codec" in defaults and defaults["codec"] not in usable_codecs:
        issues.append(RuleConfigIssue(
            line_map.line_of("defaults", "codec"),
            f"defaults 引用的编码器 {defaults['codec']!r} 当前构建/本机不支持；"
            f"可用: {', '.join(sorted(usable_codecs))}。",
        ))
    for rule in rules:
        codec = rule.params.get("codec")
        if codec is not None and codec not in usable_codecs:
            issues.append(RuleConfigIssue(
                rule.line,
                f"规则 {rule.name!r} 引用的编码器 {codec!r} 当前构建/本机不支持；"
                f"可用: {', '.join(sorted(usable_codecs))}。", rule.name,
            ))


def _inherits_cycle(name: str, rules: list[RuleDefinition]) -> bool:
    by_name = {rule.name: rule for rule in rules}
    seen: set[str] = set()
    current = name
    while current in by_name:
        if current in seen:
            return True
        seen.add(current)
        parents = by_name[current].inherits
        if not parents:
            return False
        current = parents[0]  # 多继承的每条链在各自规则上都会被遍历到
    return False


# ---------------------------------------------------------------------------
# 全局矛盾检测（规则对组合，而非逐字段）
# ---------------------------------------------------------------------------

def _check_pairwise_conflicts(
    rules: list[RuleDefinition], line_map: LineMap,
    issues: list[RuleConfigIssue],
) -> None:
    # 纯排除规则（无自身参数且不继承）不可能与任何规则产生参数矛盾；
    # 只继承的规则经展开后仍携带基类参数，需纳入检测
    concrete = [
        rule for rule in rules
        if not rule.abstract and (rule.params or rule.inherits)
    ]
    index_by_name = {rule.name: i for i, rule in enumerate(rules)}
    for i, left in enumerate(concrete):
        for right in concrete[i + 1:]:
            if left.priority != right.priority:
                continue  # 不同优先级：高优先级覆盖，语义确定，非矛盾
            if left.name in right.inherits or right.name in left.inherits:
                continue  # 继承链上的同名参数是覆盖，不是矛盾
            if not _can_co_match(left, right):
                continue
            right_index = index_by_name[right.name]
            for key in sorted(set(left.params) & set(right.params)):
                if left.params[key] != right.params[key]:
                    line = line_map.line_of_path(
                        ("rules", right_index, "params", key)
                    ) or right.line
                    issues.append(RuleConfigIssue(
                        line,
                        f"规则 {right.name!r} 与同优先级规则 {left.name!r} "
                        f"可能命中同一文件，却对参数 {key!r} 给出矛盾取值 "
                        f"{right.params[key]!r} != {left.params[key]!r}；"
                        "请设置不同 priority 显式裁决，或收紧匹配条件。",
                        right.name,
                    ))


def _can_co_match(left: RuleDefinition, right: RuleDefinition) -> bool:
    """保守判断两条规则是否**可能**命中同一文件（宁可多报不漏报）。"""
    if left.dirs and right.dirs and not _dir_sets_overlap(left.dirs, right.dirs):
        return False
    if left.exts and right.exts and not (left.exts & right.exts):
        return False
    # glob 模式不做可满足性求解：无法证明互斥就按可能重叠处理（sound）
    return True


def _dir_sets_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    for a in left:
        for b in right:
            if a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep):
                return True
    return False


def _resolve_dir(raw_dir: str, config_dir: str) -> str:
    """相对目录以配置文件所在目录为基准规范化为绝对路径。"""
    if os.path.isabs(raw_dir):
        return os.path.abspath(raw_dir)
    return os.path.abspath(os.path.join(config_dir, raw_dir))


# 供引擎复用的 glob 判断（保留在 schema 便于冲突分析共享语义）
def path_matches_pattern(path: str, pattern: str) -> bool:
    """glob 模式匹配，统一使用正斜杠的跨平台路径语义。

    规则语言对 ``exclude`` 模式定义两种形态：

    - **不含分隔符**（``*.tmp.mp4``）：匹配任意目录下的同名文件，即对
      完整路径的文件名段做 :func:`fnmatch.fnmatch`；
    - **含分隔符**（``drafts/*``、``build/tmp/*.mp4``）：按相对子树匹配。
      :func:`fnmatch.fnmatch` 的 ``*`` 不跨目录分隔符，因此除直接 glob 外，
      若模式的某段是路径的祖先目录，也视为命中（``drafts/*`` 命中
      ``archive/drafts/scratch.mp4``），符合「排除整个子树」的直觉。
    """
    norm_path = path.replace(os.sep, "/")
    norm_pattern = pattern.replace(os.sep, "/")
    if "/" not in norm_pattern:
        return fnmatch.fnmatch(os.path.basename(norm_path), norm_pattern)
    if fnmatch.fnmatch(norm_path, norm_pattern):
        return True
    # 子树匹配：模式的字面量（无通配符）段构成祖先目录，只要该目录作为
    # 完整路径段出现在路径中，其下所有文件都命中（drafts/* → 任意深度的
    # drafts 子树）。用 / 包裹判断，避免 draftsy 这类前缀误匹配。
    segments = [seg for seg in norm_pattern.split("/") if seg]
    literal_parts = [
        seg for seg in segments if not any(ch in seg for ch in "*?[")
    ]
    if literal_parts:
        ancestor = "/".join(literal_parts)
        return ("/" + ancestor + "/") in ("/" + norm_path)
    return False
