"""规则引擎子包：把配置文件解析为每个文件确定、可解释的一组处理参数。

模块划分（依赖方向自上而下，全部零三方依赖）：

- :mod:`bvp.rules.linemap`：配置文本的 JSON 路径 → 行号定位，及重复键检测；
- :mod:`bvp.rules.schema`：v1 配置的数据模型、行号标注与启动期全局校验；
- :mod:`bvp.rules.engine`：规则求值（匹配/继承/覆盖/组合/冲突解决）；
- :mod:`bvp.rules.store`：原子热加载的 :class:`ConfigStore`。

设计硬约束见 :mod:`bvp.rules.schema` 与 :mod:`bvp.rules.engine` 的模块说明。
"""

from .engine import (
    MatchSpec,
    Resolution,
    ResolvedParams,
    RuleDefinition,
    RuleSet,
    RuleTraceEntry,
)
from .schema import RuleConfigError, RuleConfigIssue
from .store import ConfigStore, ReloadResult, load_rule_config

__all__ = [
    "MatchSpec",
    "RuleDefinition",
    "ResolvedParams",
    "Resolution",
    "RuleTraceEntry",
    "RuleSet",
    "RuleConfigError",
    "RuleConfigIssue",
    "ConfigStore",
    "ReloadResult",
    "load_rule_config",
]
