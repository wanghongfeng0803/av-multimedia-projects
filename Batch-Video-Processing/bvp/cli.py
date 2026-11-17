"""命令行入口：``python -m bvp batch <输入...> [选项]``。

退出码约定（与错误分类契约对齐，便于脚本编排）：

- 0：全部成功或跳过，且清单对账通过；
- 8：批量跑完但存在失败条目，或对账发现清单与磁盘不符；
- 2：启动前参数错误（未扫描、未转码即快速失败），含规则配置校验失败。

CLI 只做参数解析与结果打印，所有编排逻辑都在 :mod:`bvp.batch`。

参数合并优先级（高 → 低）：**CLI 显式选项 → 规则命中参数 → 配置
defaults → 内置默认值**。实现上用 :data:`argparse.SUPPRESS` 让未给出的
选项在 Namespace 中缺席（而非落入 argparse 默认值），从而把「显式覆盖」
与「未指定」严格区分开。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .batch import batch_transcode
from .core import supported_codecs, supported_containers
from .planning import EXIST_OVERWRITE, EXIST_SKIP, EXIST_SUFFIX
from .rules.schema import RuleConfigError, build_rule_set
from .rules.store import ConfigStore, probe_available_codecs
from .verify import reconcile_manifest_files

__all__ = ["main", "build_parser"]

_EXIT_OK = 0
_EXIT_BATCH_FAILURE = 8
_EXIT_USAGE = 2

#: 未显式给出的可覆盖选项统一缺席，交给规则/defaults 裁决
_UNSET = object()

#: 旧式 Namespace（旧测试/嵌入调用方直接构造）携带的内置默认值。
#: 这些值等于「未指定」而非「显式覆盖」，否则会错误压过规则参数。
_LEGACY_DEFAULTS: dict = {
    "codec": "h264",
    "container": "mp4",
    "preset": "medium",
    "audio_bitrate": "128k",
    "on_exist": EXIST_SKIP,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bvp",
        description="可复核的目录批量视频转码（成功/失败/跳过清单落盘）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser(
        "batch", help="对一个或多个文件/目录执行批量转码"
    )
    p_batch.add_argument(
        "inputs", nargs="+", help="输入视频文件或目录（目录可嵌套递归）"
    )
    p_batch.add_argument(
        "-o", "--output-dir", default=argparse.SUPPRESS,
        help="统一输出目录，镜像输入子目录；缺省由规则或核心默认决定",
    )
    p_batch.add_argument(
        "--manifest-dir", default=None,
        help="清单目录；缺省取输出目录，再退化为当前工作目录",
    )
    p_batch.add_argument(
        "--rules", default=None,
        help="规则配置文件路径（JSON v1）；启用后逐文件按规则裁决参数",
    )
    p_batch.add_argument(
        "--watch", action="store_true",
        help="监视规则配置文件并原子热加载：每轮批量后检查变化，"
             "进行中任务沿用旧快照，新任务使用新快照",
    )
    p_batch.add_argument(
        "--poll-interval", type=float, default=2.0,
        help="--watch 模式下检查配置变化的间隔秒数（默认 2.0）",
    )
    p_batch.add_argument(
        "--max-cycles", type=int, default=0,
        help="watch 模式最多运行的批量轮数；0 表示持续运行（测试用）",
    )
    p_batch.add_argument(
        "--codec", default=argparse.SUPPRESS,
        choices=tuple(supported_codecs()),
        help="目标视频编码（显式覆盖规则；缺省由规则/defaults 决定）",
    )
    p_batch.add_argument(
        "--container", default=argparse.SUPPRESS,
        choices=tuple(supported_containers()),
        help="目标容器（显式覆盖规则）",
    )
    p_batch.add_argument(
        "--crf", type=int, default=argparse.SUPPRESS, help="质量参数 CRF"
    )
    p_batch.add_argument(
        "--preset", default=argparse.SUPPRESS, help="x264/x265 速度档位"
    )
    p_batch.add_argument(
        "--audio-bitrate", default=argparse.SUPPRESS,
        help="音频重编码码率（显式覆盖规则）",
    )
    p_batch.add_argument(
        "--resolution", default=argparse.SUPPRESS,
        help="目标分辨率，如 1280x720 / 720p / 50%%（显式覆盖规则）",
    )
    p_batch.add_argument(
        "-j", "--jobs", type=int, default=argparse.SUPPRESS,
        help="并发工作线程数（显式覆盖规则 defaults）",
    )
    p_batch.add_argument(
        "--on-exist", default=argparse.SUPPRESS,
        choices=(EXIST_SKIP, EXIST_OVERWRITE, EXIST_SUFFIX),
        help="输出已存在策略（显式覆盖规则）",
    )
    p_batch.add_argument(
        "--no-recursive", action="store_true",
        help="目录只扫描一层，不递归子目录",
    )
    p_batch.add_argument(
        "--ext", action="append", default=None,
        help="自定义扫描扩展名，可重复给出，如 --ext .mp4 --ext .mov",
    )

    p_check = sub.add_parser(
        "rules-check",
        help="只校验规则配置并打印求值说明，不执行任何转码",
    )
    p_check.add_argument("rules", help="规则配置文件路径（JSON v1）")
    p_check.add_argument(
        "sample", nargs="*",
        help="可选的示例文件路径；给出时打印每个文件的规则求值解释",
    )

    p_verify = sub.add_parser(
        "verify", help="对清单目录执行独立对账（不运行转码）"
    )
    p_verify.add_argument("manifest_dir", help="存放三份清单的目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        return _run_verify(args.manifest_dir)
    if args.command == "rules-check":
        return _run_rules_check(args.rules, args.sample)
    return _run_batch(args)


def _safe_codec_probe():
    """返回本机编码器探测回调；ffmpeg 不可用时返回 None 跳过本机级校验。"""
    def _probe():
        try:
            return probe_available_codecs()
        except Exception:  # noqa: BLE001 - 探测失败降级，不阻断规则加载
            return None
    return _probe


def _explicit(args: argparse.Namespace, name: str):
    """读取显式选项；未给出或（旧式 Namespace 下）等于内置默认时返回 None。

    两条解析路径要区分开：

    - 新 argparse 路径用 :data:`argparse.SUPPRESS`，未指定的选项在 Namespace
      中直接缺席，``hasattr`` 为假即「未指定」；显式给出 ``h264`` 等与内置
      默认同值的选项时属性存在，必须当作真正的覆盖；
    - 旧式 Namespace（旧测试/嵌入调用方直接构造）没有 ``rules`` 等新字段，
      且携带内置默认值（``codec="h264"`` 等），这些值语义上是「未指定」，
      不能当成显式覆盖压过规则参数。
    """
    if not hasattr(args, name):
        return None
    is_legacy_namespace = not hasattr(args, "rules")
    value = getattr(args, name)
    if is_legacy_namespace and name in _LEGACY_DEFAULTS \
            and value == _LEGACY_DEFAULTS[name]:
        return None
    return value


def _run_batch(args: argparse.Namespace) -> int:
    extensions = (
        frozenset(args.ext) if getattr(args, "ext", None) else None
    )
    recursive = False if getattr(args, "no_recursive", False) else None
    common_kwargs = dict(
        output_dir=_explicit(args, "output_dir"),
        recursive=recursive,
        jobs=_explicit(args, "jobs"),
        on_exist=_explicit(args, "on_exist"),
        codec=_explicit(args, "codec"),
        container=_explicit(args, "container"),
        crf=_explicit(args, "crf"),
        preset=_explicit(args, "preset"),
        audio_bitrate=_explicit(args, "audio_bitrate"),
        resolution=_explicit(args, "resolution"),
        extensions=extensions,
        manifest_dir=getattr(args, "manifest_dir", None),
    )

    rules_path = getattr(args, "rules", None)
    if not rules_path:
        return _run_single_batch(args.inputs, None, common_kwargs)

    # --- 规则配置路径：先做一次启动校验，坏配置拒绝启动（退出码 2） -----
    # 本机编码器探测尽力而为：ffmpeg 不可用时降级为仅构建级校验，
    # 不阻塞无 ffmpeg 环境下的规则校验与纯逻辑使用。
    try:
        store = ConfigStore(rules_path, codec_probe=_safe_codec_probe())
    except RuleConfigError as exc:
        _print_config_error(exc)
        return _EXIT_USAGE

    if not getattr(args, "watch", False):
        return _run_single_batch(args.inputs, store.snapshot(), common_kwargs)
    return _run_watch(args, store, common_kwargs)


def _run_single_batch(inputs: list[str], rule_set, common_kwargs: dict) -> int:
    try:
        result = batch_transcode(
            list(inputs), rule_set=rule_set, **common_kwargs,
        )
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    _print_result(result)
    if result.success:
        return _EXIT_OK
    return _EXIT_BATCH_FAILURE


def _run_watch(
    args: argparse.Namespace, store: ConfigStore, common_kwargs: dict,
) -> int:
    """热加载循环：每轮批量前原子取得当前快照，轮询配置变化并整体切换。"""
    cycles = 0
    last_failure = _EXIT_OK
    while True:
        cycles += 1
        reload_result = store.maybe_reload()
        if reload_result.error is not None:
            # 坏配置不替换现行快照：打印告警，继续用旧配置服务
            _print_config_error(reload_result.error, prefix="配置热加载失败，"
                                                           "继续沿用旧配置")
        elif reload_result.loaded:
            print(f"[watch] 检测到新配置，已原子切换: {store.path}")

        code = _run_single_batch(args.inputs, store.snapshot(), common_kwargs)
        last_failure = code

        if args.max_cycles and cycles >= args.max_cycles:
            return last_failure
        time.sleep(max(0.1, args.poll_interval))


def _print_result(result) -> None:
    summary = result.summary
    print("=" * 60)
    print("批量转码结束")
    print(
        f"  总计 {summary['total']} | 成功 {summary['success']} | "
        f"失败 {summary['failed']} | 跳过 {summary['skipped']}"
    )
    for label, key in (("全量清单", "full"), ("失败清单", "failed"),
                       ("跳过清单", "skipped")):
        print(f"  {label}: {result.manifest_paths[key]}")
    print(f"  对账结果: {'通过' if result.reconciliation.ok else '不符'}")
    if not result.reconciliation.ok:
        for mismatch in result.reconciliation.mismatches:
            print(f"    - [{mismatch.kind}] {mismatch.message}", file=sys.stderr)


def _print_config_error(exc: RuleConfigError, *, prefix: str | None = None) -> None:
    header = prefix or "规则配置校验失败，拒绝启动"
    print(f"{header}，共 {len(exc.issues)} 个问题：", file=sys.stderr)
    for issue in exc.issues:
        print(f"  {issue.format(exc.config_path)}", file=sys.stderr)


def _run_rules_check(rules_path: str, samples: list[str]) -> int:
    try:
        config = build_rule_set(rules_path)
    except RuleConfigError as exc:
        _print_config_error(exc)
        return _EXIT_USAGE

    from .rules.engine import RuleSet
    rule_set = RuleSet(config=config, config_path=rules_path)
    print(f"规则配置校验通过: {rules_path}")
    print(f"  defaults: {config.defaults or '（无）'}")
    print(f"  规则数: {len(config.rules)}")
    for rule in config.rules:
        kind = "抽象基类" if rule.abstract else "具体规则"
        print(
            f"  - {rule.name}（{kind}，priority={rule.priority}，"
            f"行 {rule.line}）"
        )
    for sample in samples:
        print()
        print(rule_set.explain(sample))
    return _EXIT_OK


def _run_verify(manifest_dir: str) -> int:
    result = reconcile_manifest_files(manifest_dir)
    payload = {
        "ok": result.ok,
        "summary": result.summary,
        "produced_outputs": result.produced_outputs,
        "existing_outputs": result.existing_outputs,
        "mismatches": [
            {
                "index": m.index,
                "input_path": m.input_path,
                "kind": m.kind,
                "message": m.message,
            }
            for m in result.mismatches
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return _EXIT_OK if result.ok else _EXIT_BATCH_FAILURE
