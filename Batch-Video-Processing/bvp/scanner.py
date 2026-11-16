"""确定性目录扫描：手写 ``os.scandir`` 递归，顺序不依赖文件系统返回次序。

批量转码要求清单可复核，因此遍历顺序必须**跨运行、跨平台稳定**：

- 每一个目录层内的条目按 ``(小写名, 原名)`` 排序后再访问，大小写差异
  靠第二键保持确定而不是靠平台默认行为；
- 目录用显式栈做深度优先遍历（子目录逆序压栈、顺序弹出），不使用
  ``os.walk``——后者的返回次序由文件系统决定，不可作为清单顺序；
- 扫描只保存路径字符串，不读取文件内容，万级文件的内存开销仅为路径名。

无法访问的文件/目录不会让整轮扫描崩溃，而是作为 :class:`ScanProblem`
逐条上报，由批量层按稳定契约归入失败（``INPUT_NOT_FOUND``），保证
「遇错继续 + 清单对得平」。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "DEFAULT_VIDEO_EXTENSIONS",
    "ScanProblem",
    "ScanResult",
    "scan_inputs",
    "normalize_extensions",
]

#: 默认纳入扫描的视频扩展名（小写、含点）。
#: 扩展名只做第一轮筛选；文件是否真为有效视频由核心 ffprobe 按内容裁决，
#: 伪装文件最终以 NOT_A_VIDEO 进入跳过清单，而不是污染整批。
DEFAULT_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
    ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts",
    ".3gp", ".3g2", ".ogv", ".rmvb", ".rm", ".vob", ".dv",
})


@dataclass(frozen=True)
class ScanProblem:
    """扫描阶段发现的不可访问路径（批量层据此记一条失败）。"""

    path: str           # 出问题的绝对路径
    kind: str           # missing / not_regular / access_denied / io_error
    message: str        # 人读说明，批量层不解析、只透传到清单


@dataclass(frozen=True)
class ScanResult:
    """扫描产物：有序候选文件 + 不可访问问题清单。"""

    files: tuple[str, ...]
    problems: tuple[ScanProblem, ...]


def normalize_extensions(
    extensions: frozenset[str] | None,
) -> frozenset[str]:
    """把扩展名集合规范为小写、含点的形式；None 取内置默认集合。"""
    if extensions is None:
        return DEFAULT_VIDEO_EXTENSIONS
    normalized: set[str] = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else "." + ext)
    if not normalized:
        raise ValueError("扩展名过滤集合为空，至少需要一个扩展名。")
    return frozenset(normalized)


def scan_inputs(
    inputs: list[str],
    *,
    recursive: bool = True,
    extensions: frozenset[str] | None = None,
    exclude_dirs: frozenset[str] | None = None,
) -> ScanResult:
    """从若干文件/目录路径中确定性地收集候选视频。

    :param inputs: 文件或目录路径列表。显式点名的文件**绕过扩展名筛选**
        （用户直接给出就应入选，是否为有效视频交给核心按内容裁决）；
        目录按扩展名筛选其中的普通文件
    :param recursive: 目录是否递归进入子目录（默认递归，正确处理嵌套目录）
    :param extensions: 自定义扩展名集合；None 用 :data:`DEFAULT_VIDEO_EXTENSIONS`
    :param exclude_dirs: 递归时整体跳过的目录绝对路径集合（典型用途：
        输出目录位于输入目录内，避免把上一轮产物当成新输入）
    :returns: :class:`ScanResult`，``files`` 已去重并按确定顺序排列；
        不存在/无法访问的路径进入 ``problems`` 而非被静默丢弃
    """
    exts = normalize_extensions(extensions)
    excluded = {os.path.abspath(p) for p in (exclude_dirs or frozenset())}

    seen: set[str] = set()
    collected: list[str] = []
    problems: list[ScanProblem] = []

    # 输入根本身也按确定顺序处理，保证多根场景下结果可复现
    for raw in sorted((p for p in inputs if p), key=lambda p: os.path.abspath(p).lower()):
        path = os.path.abspath(raw)
        if os.path.abspath(path) in excluded and os.path.isdir(path):
            continue
        _visit_entry(path, recursive, exts, excluded, seen,
                     collected, problems, explicit_file=True)

    # collected 在遍历中已按确定次序追加；这里再做一次整体排序，
    # 让多根合并后的最终顺序仍然完全确定、与根的给法无关
    collected.sort(key=lambda p: p.lower())
    return ScanResult(
        files=tuple(_dedupe_sorted(collected)),
        problems=tuple(problems),
    )


def _dedupe_sorted(sorted_paths: list[str]) -> list[str]:
    """已排序列表上去重，保持首次出现顺序（大小写路径在 Windows 可能重复）。"""
    result: list[str] = []
    seen: set[str] = set()
    for path in sorted_paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _visit_entry(
    path: str,
    recursive: bool,
    exts: frozenset[str],
    excluded: set[str],
    seen: set[str],
    collected: list[str],
    problems: list[ScanProblem],
    *,
    explicit_file: bool,
) -> None:
    """处置一个路径：文件直接入选，目录展开扫描，异常上报为 ScanProblem。"""
    try:
        is_dir = os.path.isdir(path)
        is_file = os.path.isfile(path)
    except OSError as exc:
        problems.append(ScanProblem(
            path=path, kind="io_error",
            message=f"无法访问该路径: {exc}",
        ))
        return

    if is_dir:
        if recursive:
            _scan_directory(path, recursive, exts, excluded,
                            seen, collected, problems)
        else:
            _scan_flat_directory(path, exts, seen, collected, problems)
        return

    if is_file:
        if explicit_file:
            _collect(path, seen, collected)
        elif os.path.splitext(path)[1].lower() in exts:
            _collect(path, seen, collected)
        return

    # 既不是目录也不是普通文件：不存在、悬空链接或被权限拦截
    kind, message = _classify_missing(path)
    problems.append(ScanProblem(path=path, kind=kind, message=message))


def _classify_missing(path: str) -> tuple[str, str]:
    """区分不存在与权限/IO 问题，给出稳定 kind 码。"""
    if not os.path.exists(path):
        # 断链 symlink 在 Windows 上 lexists 为真但 exists 为假
        if os.path.islink(path):
            return "access_denied", f"符号链接目标不可访问: {path}"
        return "missing", f"路径不存在: {path}"
    if not os.path.isfile(path):
        return "not_regular", f"既不是普通文件也不是目录: {path}"
    return "io_error", f"路径无法访问: {path}"


def _scan_flat_directory(
    directory: str,
    exts: frozenset[str],
    seen: set[str],
    collected: list[str],
    problems: list[ScanProblem],
) -> None:
    """只扫描目录一层（不递归）。"""
    try:
        names = sorted(os.listdir(directory), key=lambda n: (n.lower(), n))
    except OSError as exc:
        problems.append(ScanProblem(
            path=directory, kind="access_denied",
            message=f"无法读取目录内容: {exc}",
        ))
        return
    for name in names:
        full = os.path.join(directory, name)
        try:
            if os.path.isfile(full) \
                    and os.path.splitext(full)[1].lower() in exts:
                _collect(os.path.abspath(full), seen, collected)
        except OSError as exc:
            problems.append(ScanProblem(
                path=os.path.abspath(full), kind="io_error",
                message=f"无法访问该文件: {exc}",
            ))


def _scan_directory(
    directory: str,
    recursive: bool,
    exts: frozenset[str],
    excluded: set[str],
    seen: set[str],
    collected: list[str],
    problems: list[ScanProblem],
) -> None:
    """深度优先递归扫描；显式栈 + 每层排序，保证顺序确定。"""
    stack: list[str] = [directory]
    while stack:
        current = stack.pop()
        if os.path.abspath(current) in excluded:
            continue
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError as exc:
            problems.append(ScanProblem(
                path=current, kind="access_denied",
                message=f"无法读取目录内容: {exc}",
            ))
            continue

        # 关键确定性：排序在访问之前完成，不接受文件系统的自然返回次序
        entries.sort(key=lambda entry: (entry.name.lower(), entry.name))
        sub_dirs: list[str] = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if os.path.abspath(entry.path) not in excluded:
                        sub_dirs.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError as exc:
                problems.append(ScanProblem(
                    path=os.path.abspath(entry.path), kind="io_error",
                    message=f"无法访问该条目: {exc}",
                ))
                continue
            if os.path.splitext(entry.name)[1].lower() in exts:
                _collect(os.path.abspath(entry.path), seen, collected)

        # 逆序压栈使排序第一的子目录最先弹出，维持 DFS 字典序
        stack.extend(reversed(sub_dirs))


def _collect(abs_path: str, seen: set[str], collected: list[str]) -> None:
    if abs_path in seen:
        return
    seen.add(abs_path)
    collected.append(abs_path)
