"""参数转义与引号包裹（注入防护）。

两道防线：
1. 子进程始终以 argv 列表启动（不经 shell），路径/文本作为独立元素
   经 CreateProcess/execve 原样传递，shell 元字符无法变成命令；
2. 需要嵌入 filtergraph 的路径按 ffmpeg 两级转义处理
   （escape_filter_path）；主管线用 -i + 流标签避开这种嵌套。

render_shell_command 仅用于日志展示，不参与实际执行。
"""

from __future__ import annotations

import sys

_WINDOWS_SPECIAL = set(' \t\n\v"')
_POSIX_SPECIAL = set(" \t\n\"'\\$`!&|;()<>*?[]{}#~^=")


def escape_filter_path(path: str) -> str:
    """转义为可嵌入 ffmpeg filtergraph 的单引号字面量（两级反斜杠规则）。"""
    level1: list[str] = []
    for char in path:
        if char in ("\\", "'"):
            level1.append("\\")
        level1.append(char)
    level2: list[str] = []
    for char in level1:
        if char in ("\\", ":"):
            level2.append("\\")
        level2.append(char)
    return "'" + "".join(level2) + "'"


def posix_shell_quote(token: str) -> str:
    """POSIX 单引号法，``'`` 替换为 ``'\\''``。"""
    if not token:
        return "''"
    if not any(char in _POSIX_SPECIAL for char in token):
        return token
    return "'" + token.replace("'", "'\\''") + "'"


def windows_argv_quote(token: str) -> str:
    """Windows CreateProcess 双引号规则（CommandLineToArgvW 口径）。"""
    if not token:
        return '""'
    if not any(char in _WINDOWS_SPECIAL for char in token):
        return token
    out: list[str] = []
    backslashes = 0
    for char in token:
        if char == "\\":
            backslashes += 1
            continue
        if char == '"':
            out.append("\\" * (backslashes * 2 + 1))
            out.append('"')
        else:
            out.append("\\" * backslashes)
            backslashes = 0
            out.append(char)
    out.append("\\" * (backslashes * 2))
    return '"' + "".join(out) + '"'


def shell_quote(token: str) -> str:
    """按当前平台选择引号规则。"""
    if sys.platform.startswith("win"):
        return windows_argv_quote(token)
    return posix_shell_quote(token)


def render_shell_command(argv: list[str]) -> str:
    """把 argv 渲染为可粘贴的 shell 命令行（仅日志用）。"""
    return " ".join(shell_quote(str(part)) for part in argv)
