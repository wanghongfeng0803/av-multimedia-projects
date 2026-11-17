"""规则求值引擎：把单个文件路径收敛为**唯一、确定、可解释**的参数集。

求值顺序（对任意一个输入文件，过程完全确定，与文件系统枚举次序无关）：

1. **排除优先**：任何规则的 ``exclude`` glob 命中该文件，文件立即不入选，
   不再参与后续任何规则（排除项不携带参数，语义上是硬过滤器）；
2. **收集命中**：其余具体规则按 ``(priority 降序, 配置中的定义顺序)`` 求值，
   匹配条件取 AND（``dir``/``ext``/多值内部取 OR，三个维度之间取 AND）；
3. **继承展开**：每条命中规则的有效参数 = 其继承链上基类参数（按继承顺序
   先到先合并）+ 自身参数覆盖；抽象规则不直接匹配，只通过继承生效；
4. **覆盖收敛**：全部命中规则按 priority 从低到高、同优先级按定义顺序
   依次合并，后写者逐字段覆盖先写者——同优先级矛盾已在启动期
   (:func:`bvp.rules.schema.build_rule_set`) 拒绝，运行期不存在歧义；
5. **缺省兜底**：没有任何规则命中时，落回隐式 ``__default__`` 规则
   （配置 defaults + 内置默认值），保证每个入选文件都有完整参数集。

合并优先级的完整链条（高 → 低）：
CLI 显式参数 → 命中规则参数（含继承）→ 配置 ``defaults`` → 内置默认值。
本模块负责后三层；CLI 覆盖由批量层在调用前叠加。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .schema import (
    RuleConfig,
    RuleDefinition,
    path_matches_pattern,
)

__all__ = [
    "MatchSpec",
    "ResolvedParams",
    "Resolution",
    "RuleTraceEntry",
    "RuleSet",
    "DEFAULT_RULE_NAME",
]

#: 无规则命中时的隐式兜底规则名（写入清单便于解释）
DEFAULT_RULE_NAME = "__default__"

#: 内置默认值（最低优先级，与历史 CLI 默认严格对齐，保证旧行为不变）
BUILTIN_DEFAULTS: dict = {
    "codec": "h264",
    "container": "mp4",
    "crf": None,            # None 表示交给核心按 codec 取默认 CRF
    "preset": "medium",
    "audio_bitrate": "128k",
    "resolution": None,
    "on_exist": "skip",
    "output_dir": None,
}
_BATCH_BUILTINS: dict = {"jobs": 1, "recursive": True}


@dataclass(frozen=True)
class MatchSpec:
    """规则匹配条件的运行期视图（从 :class:`RuleDefinition` 投影）。"""

    dirs: tuple[str, ...]
    exts: frozenset[str]
    exclude: tuple[str, ...]


@dataclass(frozen=True)
class RuleTraceEntry:
    """解释轨迹中的一步：哪条规则在何种匹配下贡献了哪些字段。"""

    rule: str
    priority: int
    matched_via: tuple[str, ...]      # 实际命中的匹配维度（dir/ext）
    contributed: tuple[str, ...]      # 本次合并真正写入/覆盖的参数键
    inherited_from: tuple[str, ...]   # 展开继承时经过的基类名


@dataclass(frozen=True)
class ResolvedParams:
    """收敛后的完整有效参数（八个转码字段全部有确定值，可直接透传核心）。"""

    codec: str
    container: str
    crf: int | None
    preset: str
    audio_bitrate: str
    resolution: str | None
    on_exist: str
    output_dir: str | None
    jobs: int = 1
    recursive: bool = True

    def core_kwargs(self) -> dict:
        """投影为 :func:`bvp.core.transcode` 接受的关键字参数。"""
        return {
            "codec": self.codec,
            "container": self.container,
            "crf": self.crf,
            "preset": self.preset,
            "audio_bitrate": self.audio_bitrate,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class Resolution:
    """单个文件的求值结果：入选参数 + 完整解释轨迹。"""

    selected: bool
    params: ResolvedParams | None
    winning_rule: str | None
    trace: tuple[RuleTraceEntry, ...]
    excluded_by: str | None = None


@dataclass
class RuleSet:
    """不可变规则集快照：构造后只读，可被任意线程安全共享。

    热加载的实现方式是整体替换 :class:`RuleSet` 对象（见
    :mod:`bvp.rules.store`），对象内部不做任何就地修改，因此进行中的任务
    持有旧快照时永远看到一致的旧配置，不存在中间态。
    """

    config: RuleConfig
    config_path: str

    @property
    def rules(self) -> tuple[RuleDefinition, ...]:
        return self.config.rules

    @property
    def defaults(self) -> dict:
        return self.config.defaults

    def all_extensions(self) -> frozenset[str]:
        """扫描器扩展名白名单：所有规则显式 ``ext`` 的并集。

        若没有任何规则声明 ``ext``（规则只按目录匹配），返回空集合，由
        批量层回退到扫描器内置默认视频扩展名集合——**空并集不等于空白名单**，
        否则只按目录匹配的规则会把所有文件都过滤掉。
        """
        union: set[str] = set()
        for rule in self.config.rules:
            if rule.abstract:
                continue
            union.update(rule.exts)
        return frozenset(union)

    def should_recurse(self) -> bool:
        return bool(self.defaults.get("recursive", _BATCH_BUILTINS["recursive"]))

    def batch_jobs(self) -> int:
        return int(self.defaults.get("jobs", _BATCH_BUILTINS["jobs"]))

    # ------------------------------------------------------------------
    # 核心求值
    # ------------------------------------------------------------------

    def resolve(self, file_path: str) -> Resolution:
        """把一个绝对文件路径求值为唯一参数集（或判定排除）。"""
        absolute = os.path.abspath(file_path)

        # 阶段 1：排除优先，任何 exclude 命中即硬过滤
        excluded_by = self._match_exclude(absolute)
        if excluded_by is not None:
            return Resolution(
                selected=False, params=None, winning_rule=None,
                trace=(), excluded_by=excluded_by,
            )

        # 阶段 2：收集命中的具体规则，按 (priority 降序, 定义顺序)。
        # 纯排除规则（只有 exclude、自身无参数且不继承）不参与参数求值，
        # 它的过滤职责已在阶段 1 完成；只继承不自带参数的规则仍需参与，
        # 因为其基类参数会在继承展开时生效。
        matched = [
            (index, rule)
            for index, rule in enumerate(self.config.rules)
            if not rule.abstract
            and (rule.params or rule.inherits)
            and self._matches(rule, absolute)
        ]
        matched.sort(key=lambda item: (-item[1].priority, item[0]))

        if not matched:
            params = self._fallback_params()
            return Resolution(
                selected=True, params=params,
                winning_rule=DEFAULT_RULE_NAME, trace=(),
            )

        # 阶段 3+4：继承展开后按优先级逐层合并。
        # matched 已按 (priority 降序, 定义顺序) 排列；合并必须从**低优先级
        # 到高优先级**进行，使高优先级最后写入完成覆盖，因此反向遍历。
        merged: dict = {}
        trace: list[RuleTraceEntry] = []
        for _, rule in reversed(matched):
            effective, inherited = self._expand_inheritance(rule)
            contributed = tuple(
                key for key in effective if merged.get(key) != effective[key]
            )
            merged.update(effective)
            trace.append(RuleTraceEntry(
                rule=rule.name,
                priority=rule.priority,
                matched_via=self._matched_dimensions(rule, absolute),
                contributed=contributed,
                inherited_from=tuple(inherited),
            ))
        # 合并是低→高，trace 自然按低优先级在前；展示时反转为高→低，
        # 与 matched 的裁决顺序一致，便于人读
        trace.reverse()

        params = self._materialize(merged)
        return Resolution(
            selected=True, params=params,
            winning_rule=matched[0][1].name,
            trace=tuple(trace),
        )

    def explain(self, file_path: str) -> str:
        """生成人读解释文本（``rules-check`` 子命令与排查使用）。"""
        result = self.resolve(file_path)
        if not result.selected:
            return f"{file_path}\n  → 排除（命中规则 {result.excluded_by!r} 的 exclude）"
        lines = [file_path, f"  → 入选，裁决规则: {result.winning_rule}"]
        for step in result.trace:
            inherited = (
                f"，继承自 {list(step.inherited_from)}"
                if step.inherited_from else ""
            )
            lines.append(
                f"    - 规则 {step.rule!r}（priority={step.priority}）"
                f"匹配维度 {list(step.matched_via)}{inherited}"
                f"，贡献字段 {list(step.contributed)}"
            )
        lines.append(f"  有效参数: {_compact(result.params)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 匹配与继承的内部实现
    # ------------------------------------------------------------------

    def _match_exclude(self, absolute: str) -> str | None:
        for rule in self.config.rules:
            if rule.abstract or not rule.exclude:
                continue
            # 排除规则的 dir/ext 是它的生效子树前置条件；glob 命中才排除。
            # 不能复用 _matches：该函数对普通规则会把 exclude 当作不命中条件。
            if rule.dirs and not any(
                _within(absolute, dir_) for dir_ in rule.dirs
            ):
                continue
            if rule.exts and os.path.splitext(absolute)[1].lower() \
                    not in rule.exts:
                continue
            if any(path_matches_pattern(absolute, pattern)
                   for pattern in rule.exclude):
                return rule.name
        return None

    def _matches(self, rule: RuleDefinition, absolute: str) -> bool:
        """普通参数规则的匹配：dir/ext 前置条件 AND，且未被自身 exclude 排除。"""
        if rule.dirs and not any(_within(absolute, dir_) for dir_ in rule.dirs):
            return False
        if rule.exts and os.path.splitext(absolute)[1].lower() not in rule.exts:
            return False
        # 普通参数规则：若声明了 exclude，被排除的文件不算命中
        if rule.exclude and any(
            path_matches_pattern(absolute, pattern)
            for pattern in rule.exclude
        ):
            return False
        return True

    def _matched_dimensions(
        self, rule: RuleDefinition, absolute: str
    ) -> tuple[str, ...]:
        via: list[str] = []
        if rule.dirs and any(_within(absolute, d) for d in rule.dirs):
            via.append("dir")
        if rule.exts and os.path.splitext(absolute)[1].lower() in rule.exts:
            via.append("ext")
        return tuple(via)

    def _expand_inheritance(
        self, rule: RuleDefinition
    ) -> tuple[dict, list[str]]:
        """展开继承链：基类参数按 inherits 顺序先合并，自身参数最后覆盖。"""
        by_name = {item.name: item for item in self.config.rules}
        merged: dict = {}
        chain: list[str] = []
        for parent_name in rule.inherits:
            parent = by_name.get(parent_name)
            if parent is None:
                continue  # 启动校验已拦截悬空继承，这里防御性跳过
            parent_params, parent_chain = self._expand_inheritance(parent)
            merged.update(parent_params)
            chain.extend(parent_chain)
            chain.append(parent_name)
        merged.update(rule.params)
        return merged, chain

    def _fallback_params(self) -> ResolvedParams:
        return self._materialize({})

    def _materialize(self, merged: dict) -> ResolvedParams:
        """把任意层级的参数叠加到 defaults 与内置默认之上，产出完整对象。"""
        layered: dict = {}
        layered.update(BUILTIN_DEFAULTS)
        layered.update({
            key: value for key, value in self.config.defaults.items()
            if key in BUILTIN_DEFAULTS
        })
        layered.update(merged)

        output_dir = layered["output_dir"]
        if output_dir is not None and not os.path.isabs(output_dir):
            output_dir = os.path.abspath(
                os.path.join(self.config.config_dir, output_dir)
            )
        return ResolvedParams(
            codec=layered["codec"],
            container=layered["container"],
            crf=layered["crf"],
            preset=layered["preset"],
            audio_bitrate=layered["audio_bitrate"],
            resolution=layered["resolution"],
            on_exist=layered["on_exist"],
            output_dir=output_dir,
            jobs=int(self.config.defaults.get(
                "jobs", _BATCH_BUILTINS["jobs"])),
            recursive=bool(self.config.defaults.get(
                "recursive", _BATCH_BUILTINS["recursive"])),
        )


def _within(absolute: str, directory: str) -> bool:
    """判断文件是否位于目录内（含目录本身边界，避免前缀误匹配）。"""
    return absolute == directory or absolute.startswith(directory + os.sep)


def _compact(params: ResolvedParams | None) -> dict:
    if params is None:
        return {}
    return {
        key: value
        for key, value in (
            ("codec", params.codec),
            ("container", params.container),
            ("crf", params.crf),
            ("preset", params.preset),
            ("audio_bitrate", params.audio_bitrate),
            ("resolution", params.resolution),
            ("on_exist", params.on_exist),
            ("output_dir", params.output_dir),
        )
        if value is not None
    }
