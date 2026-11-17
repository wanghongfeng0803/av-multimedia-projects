"""配置文本的行号定位与重复键检测（零依赖字符级 JSON 扫描）。

规则引擎要求任何启动期错误都能**定位到具体行**，而标准库 ``json`` 解析后
只丢回纯数据结构、不保留来源位置。本模块用一个只识别 JSON 词法的状态机
扫描原始文本，建立「JSON 路径 → 起始行号」映射：

- 对象键的路径形如 ``("rules", 2, "params", "codec")``（数组段用整数下标）；
- 每个值（含标量）也记录其路径起始行，便于值域错误按行报告；
- 同一对象内重复键直接记录为问题（标准 ``json`` 默认静默取后者，会掩盖
  配置错误，规则引擎不允许这种隐式行为）。

扫描器只做词法定位，不做语法裁决：真正的 JSON 语法错误仍由
``json.JSONDecodeError`` 报告（其自带 ``lineno``），本模块在遇到非法词法时
尽力继续，定位结果只用于后续能解析成功的配置。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LineMap", "DuplicateKey", "build_line_map"]


@dataclass(frozen=True)
class DuplicateKey:
    """同一对象内重复出现的键。"""

    path: tuple
    line: int          # 第二次出现的行号
    key: str


@dataclass
class LineMap:
    """JSON 路径（元组）→ 起始行号（1 起）的映射，附带重复键清单。"""

    lines: dict[tuple, int]
    duplicates: tuple[DuplicateKey, ...]

    def line_of(self, *path) -> int | None:
        """查询某 JSON 路径的起始行；不存在返回 None。"""
        return self.lines.get(tuple(path))

    def line_of_path(self, path: tuple) -> int | None:
        return self.lines.get(tuple(path))


def build_line_map(text: str) -> LineMap:
    """扫描 JSON 文本，建立路径行号映射并收集重复键。"""
    lines: dict[tuple, int] = {}
    duplicates: list[DuplicateKey] = []
    # 容器栈：
    # - 对象元素 ("object", 键名|None)：键名是「下一个值」的待写入键，
    #   值落地后仍保留它，使该值的 current_path() 完整指向键路径；
    # - 数组元素 ("array", 当前下标)
    stack: list[tuple[str, object]] = []
    index = 0
    length = len(text)
    line = 1

    def current_path() -> tuple:
        return tuple(segment for _, segment in stack if segment is not None)

    while index < length:
        char = text[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        if char in " \t\r":
            index += 1
            continue

        if char == "{":
            stack.append(("object", None))
            index += 1
            continue
        if char == "[":
            stack.append(("array", 0))
            index += 1
            continue
        if char in "}]":
            # 容器闭合：仅弹栈。父容器的下标推进交给它自己的逗号/下一值，
            # 不能在这里推进，否则数组嵌套会让下标多计一次。
            if stack:
                stack.pop()
            index += 1
            continue
        if char == ",":
            # 逗号才是「数组下一个元素 / 对象等待下一个键」的分界点
            if stack and stack[-1][0] == "array":
                kind, segment = stack[-1]
                stack[-1] = (kind, (segment if isinstance(segment, int) else 0) + 1)
            elif stack and stack[-1][0] == "object":
                stack[-1] = ("object", None)
            index += 1
            continue
        if char == ":":
            index += 1
            continue

        start_line = line
        if char == '"':
            value, index, line = _scan_string(text, index, line)
            is_key = _next_nonspace_is_colon(text, index)
            if is_key:
                key_path = current_path() + (value,)
                if key_path in lines:
                    duplicates.append(DuplicateKey(
                        path=key_path, line=start_line, key=value,
                    ))
                lines[key_path] = start_line
                # 登记对象栈顶的待写入键：随后的值路径自动带上该键名
                if stack and stack[-1][0] == "object":
                    stack[-1] = ("object", value)
            else:
                lines[current_path()] = start_line
            continue

        # 裸标量：true / false / null / 数字
        index, line = _scan_scalar(text, index, line)
        lines[current_path()] = start_line

    return LineMap(lines=lines, duplicates=tuple(duplicates))


def _next_nonspace_is_colon(text: str, index: int) -> bool:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index < len(text) and text[index] == ":"


def _scan_string(text: str, index: int, line: int) -> tuple[str, int, int]:
    """从起始引号扫描到结束引号，返回 (字符串值, 新下标, 新行号)。"""
    index += 1  # 跳过起始引号
    chars: list[str] = []
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\n":
            line += 1
        if char == "\\":
            # 转义对：跳过两个字符，按 JSON 规则还原常见转义
            if index + 1 < length:
                escaped = text[index + 1]
                mapping = {
                    '"': '"', "\\": "\\", "/": "/",
                    "b": "\b", "f": "\f", "n": "\n",
                    "r": "\r", "t": "\t",
                }
                chars.append(mapping.get(escaped, escaped))
                index += 2
                continue
            index += 1
            continue
        if char == '"':
            return "".join(chars), index + 1, line
        chars.append(char)
        index += 1
    # 未闭合字符串（语法错误场景）：交回 json 模块报错，这里尽力返回
    return "".join(chars), index, line


def _scan_scalar(text: str, index: int, line: int) -> tuple[int, int]:
    """跳过一个裸标量 token，返回 (新下标, 新行号)。"""
    length = len(text)
    while index < length and text[index] not in " \t\r\n,]}":
        if text[index] == "\n":
            line += 1
        index += 1
    return index, line
