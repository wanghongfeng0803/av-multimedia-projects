"""规则配置的原子热加载存储。

热加载要满足三条硬约束：

1. **替换无中间态**：配置切换是一次不可变 :class:`~bvp.rules.engine.RuleSet`
   快照的整体指针替换（锁内只做一次引用赋值），任何线程读到的要么是旧
   快照、要么是新快照，绝不会读到半解析的配置；
2. **进行中任务不受影响**：批量任务在启动时通过 :meth:`ConfigStore.snapshot`
   取走快照引用，之后整个任务生命周期都持有该不可变对象；配置替换只改变
   存储里的指针，旧对象不被修改、不受影响；
3. **坏配置不污染现行配置**：重新加载若校验失败，保留旧快照继续服务，
   错误通过 :class:`ReloadResult` 上报，由调用方决定告警还是退出。

轮询基于 ``(mtime_ns, size)`` 判定文件是否变化（零依赖、无需文件系统事件
支持）；真正的原子性由配置维护方以「同目录临时文件 + ``os.replace``」替换
配置文件保证，本模块只在观察到变化后读取一个完整的新文件。
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

from ..binaries import find_ffmpeg
from ..core import supported_codecs
from .engine import RuleSet
from .schema import RuleConfigError, build_rule_set

__all__ = [
    "ConfigStore",
    "ReloadResult",
    "load_rule_config",
    "probe_available_codecs",
]

#: 编码器探测函数签名：返回本机可用 codec 规范名集合（None 表示跳过本机探测）
CodecProbe = Callable[[], frozenset[str] | None]


def probe_available_codecs() -> frozenset[str]:
    """探测本机 ffmpeg 实际提供的编码器，映射回构建支持的 codec 规范名。

    复用核心的编码器注册表口径（规范名 → ffmpeg 编码器名），只返回构建
    认识且本机确实可用的交集；ffmpeg 不可用时抛出，由调用方决定是否降级。
    """
    from ..core import _CODECS  # 复用核心注册表，避免第二处编码映射

    ffmpeg_path = find_ffmpeg()
    proc = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    listing = proc.stdout or ""
    available: set[str] = set()
    for name in supported_codecs():
        spec = _CODECS[name]
        if all(encoder in listing for encoder in spec["encoders"]):
            available.add(name)
    return frozenset(available)


@dataclass(frozen=True)
class ReloadResult:
    """一次（尝试）重新加载的结果，供调用方记录与告警。"""

    changed: bool
    loaded: bool
    rule_set: RuleSet | None
    error: RuleConfigError | None
    path: str

    @property
    def ok(self) -> bool:
        return self.error is None


def load_rule_config(
    path: str,
    *,
    codec_probe: CodecProbe | None = None,
) -> RuleSet:
    """一次性加载并封装为不可变规则快照（库调用便捷入口）。"""
    available = codec_probe() if codec_probe is not None else None
    config = build_rule_set(path, available_codecs=available)
    return RuleSet(config=config, config_path=os.path.abspath(path))


class ConfigStore:
    """线程安全的规则快照持有者，支持轮询式原子热加载。"""

    def __init__(
        self,
        path: str,
        *,
        codec_probe: CodecProbe | None = None,
    ):
        self._path = os.path.abspath(path)
        self._codec_probe = codec_probe
        self._lock = threading.Lock()
        self._current: RuleSet | None = None
        self._signature: tuple[int, int] | None = None
        # 首次加载失败直接抛出：没有有效配置就不允许启动
        first = self._load()
        if first.error is not None:
            raise first.error
        self._current = first.rule_set
        self._signature = self._file_signature()

    # ------------------------------------------------------------------
    # 读取：取快照是锁内一次引用拷贝，无中间态
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    def snapshot(self) -> RuleSet:
        """返回当前有效规则快照（不可变，可被任务长期持有）。"""
        with self._lock:
            assert self._current is not None  # 构造函数已保证首个快照有效
            return self._current

    def current_signature(self) -> tuple[int, int] | None:
        with self._lock:
            return self._signature

    # ------------------------------------------------------------------
    # 热加载
    # ------------------------------------------------------------------

    def maybe_reload(self, *, force: bool = False) -> ReloadResult:
        """检测配置文件变化并原子切换；无变化或校验失败时保留现行快照。

        :param force: 跳过 mtime/size 比较，强制重新解析（测试与手动刷新用）
        """
        signature = self._file_signature()
        if not force and signature == self._signature:
            with self._lock:
                return ReloadResult(
                    changed=False, loaded=False, rule_set=self._current,
                    error=None, path=self._path,
                )

        result = self._load()
        if result.error is not None:
            # 关键策略：坏配置绝不替换现行快照，服务不中断
            return result
        with self._lock:
            self._current = result.rule_set
            self._signature = signature
        return result

    def _load(self) -> ReloadResult:
        try:
            rule_set = load_rule_config(
                self._path, codec_probe=self._codec_probe,
            )
        except RuleConfigError as exc:
            with self._lock:
                return ReloadResult(
                    changed=True, loaded=False, rule_set=self._current,
                    error=exc, path=self._path,
                )
        return ReloadResult(
            changed=True, loaded=True, rule_set=rule_set,
            error=None, path=self._path,
        )

    def _file_signature(self) -> tuple[int, int]:
        try:
            stat = os.stat(self._path)
        except OSError:
            # 文件暂时不可见时返回不可能与正常签名相等的哨兵，
            # 强制下一轮重新尝试加载并由校验环节报告具体错误
            return (-1, -1)
        return (stat.st_mtime_ns, stat.st_size)
