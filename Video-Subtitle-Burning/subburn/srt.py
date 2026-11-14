"""SRT 字幕的状态机解析与时间码换算（从零实现，不使用任何第三方库）。

SRT 文件由若干「块」构成，块之间以一个或多个空白行分隔，块内部依次为::

    序号（可选，实践中经常缺失或错号，解析时不依赖其正确性）
    时间码行：  00:00:01,500 --> 00:00:04,000
    文本行（可多行，可为空）

本模块的关键设计：

* :func:`decode_srt` 负责编码探测与换行归一化，对 BOM、UTF-16、GB18030
  以及 CRLF / CR 混用做防御处理；
* :func:`parse_srt` 是一个显式的四状态行级状态机（见 :class:`_State`），
  不依赖「空行一定出现」「序号一定正确」等假设，因此能容忍粘连块、
  缺失序号、多余空行等真实世界里的脏数据；
* :func:`parse_timecode` 把 ``HH:MM:SS,mmm`` 换算为整数毫秒，同时兼容
  小数点（``.``）、缺小时（``MM:SS,mmm``）以及秒小数位不足 3 位等变体。

解析结果为 :class:`Cue` 列表，时间一律以**毫秒整数**表示，避免浮点误差
在长视频上累积。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import BinaryIO

__all__ = ["Cue", "SRTParseError", "normalize_newlines",
           "parse_srt", "parse_srt_file", "parse_timecode", "decode_srt"]


class SRTParseError(ValueError):
    """strict 模式下遇到无法恢复的格式问题时抛出。"""


@dataclass(frozen=True)
class Cue:
    """一条字幕：半开时间区间 ``[start_ms, end_ms)`` 内显示 ``text``。"""

    index: int          # 解析时分配的连续序号（从 1 开始），不取自文件
    start_ms: int
    end_ms: int
    text: str           # 多行文本以 ``\\n`` 连接，已去除行尾空白
    raw_index: str = ""  # 文件里实际写的序号，调试/诊断用


# ---------------------------------------------------------------------------
# 时间码：00:00:01,500  ->  1500
# ---------------------------------------------------------------------------

# 时间码主体：[HH:]MM:SS 后接逗号或点与小数秒。数字位数先宽松捕获，再做范围校验。
_TIMECODE_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*$"
)
_SHORT_TIMECODE_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*$"
)
# 时间码行整体：左侧 --> 右侧，右侧允许尾随定位信息（如 X1:.. X2:.. Y1:.. Y2:..）
_TIME_LINE_RE = re.compile(r"^\s*(.+?)\s*-->\s*(.+?)\s*$")


def parse_timecode(token: str) -> int:
    """把单个 SRT 时间码换算为毫秒整数。

    支持的变体：
      * ``HH:MM:SS,mmm`` / ``HH:MM:SS.mmm``（逗号与点等价）；
      * ``MM:SS,mmm``（缺小时）；
      * 小数秒为 1~3 位，不足 3 位右侧补零（``.5`` 视为 500ms）。

    >>> parse_timecode("00:00:01,500")
    1500
    >>> parse_timecode("1:02:03.4")
    3723400
    >>> parse_timecode("00:90,000")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    SRTParseError: 时间码越界或格式错误: '00:90,000'
    """
    m = _TIMECODE_RE.match(token)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        frac_str = m.group(4)
    else:
        m = _SHORT_TIMECODE_RE.match(token)
        if not m:
            raise SRTParseError(f"无法识别的时间码: {token!r}")
        hh = 0
        mm, ss = int(m.group(1)), int(m.group(2))
        frac_str = m.group(3)

    # 范围校验：分钟/秒必须落在 60 进制内，小时不做人为上限
    if mm >= 60 or ss >= 60:
        raise SRTParseError(f"时间码越界或格式错误: {token!r}")
    # 小数按原始位数右侧补零到毫秒：".5"->500, ".05"->50, ".005"->5。
    # 必须按字符串位数补齐，不能按整数字数（",004" 的整数 4 不代表 400ms）。
    ms = int(frac_str.ljust(3, "0")[:3])
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def _parse_time_line(line: str) -> tuple[int, int] | None:
    """解析整行时间码；不是时间码行则返回 None。定位信息在切分后被剥离。"""
    m = _TIME_LINE_RE.match(line)
    if not m:
        return None
    left, right = m.group(1), m.group(2)
    # 右侧时间码后可能带 " X1:.. X2:.. Y1:.. Y2:.."（SRT 定位扩展），取首个空白前部分
    right = right.split()[0] if right.split() else right
    try:
        return parse_timecode(left), parse_timecode(right)
    except SRTParseError:
        return None


# ---------------------------------------------------------------------------
# 编码探测与换行归一化
# ---------------------------------------------------------------------------

def normalize_newlines(text: str) -> str:
    """把 CRLF / 孤立 CR 统一为 LF，状态机只需面对一种换行。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_srt(data: bytes) -> str:
    """把 SRT 原始字节解码为文本，并做换行归一化（CRLF/CR -> LF）。

    判定依据：
      * UTF-16 BOM（``FF FE`` / ``FE FF``）优先识别；
      * 无 BOM 时按候选编码逐个尝试严格解码，首个成功者胜出；
      * 全部失败时退回 ``latin-1``（单字节必然成功，保证不抛异常）。
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        text = None
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = data.decode("latin-1")  # 兜底：按字节映射，永不失败
    return normalize_newlines(text)


def read_srt_text(source: str | BinaryIO) -> str:
    """从路径或二进制流读取并解码 SRT 文本。"""
    if hasattr(source, "read"):
        return decode_srt(source.read())
    with open(source, "rb") as f:
        return decode_srt(f.read())


# ---------------------------------------------------------------------------
# 行级状态机
# ---------------------------------------------------------------------------

class _State(Enum):
    IDLE = "idle"          # 块外：等待序号行或时间码行
    GOT_INDEX = "got_index"  # 已收序号，等待时间码行
    GOT_TIME = "got_time"    # 已收时间码，等待/收集文本行
    GOT_TEXT = "got_text"    # 已收到至少一行文本，等待块结束或下一块


def parse_srt(
    text: str,
    *,
    strict: bool = False,
) -> tuple[list[Cue], list[str]]:
    """解析 SRT 文本，返回 ``(字幕列表, 诊断信息列表)``。

    :param strict: 为 True 时，任何被容忍的异常都会抛出
        :class:`SRTParseError`；默认宽松模式，仅记录到诊断列表并尽量恢复。
    """
    text = normalize_newlines(text)
    warnings: list[str] = []

    def complain(msg: str) -> None:
        if strict:
            raise SRTParseError(msg)
        warnings.append(msg)

    cues: list[Cue] = []

    # 当前块的暂存区
    state = _State.IDLE
    raw_index = ""
    start_ms = end_ms = -1
    text_lines: list[str] = []

    def flush() -> None:
        """把暂存区收敛成一条 Cue；非法块只告警不产出。"""
        nonlocal state, raw_index, start_ms, end_ms, text_lines
        if state in (_State.GOT_TIME, _State.GOT_TEXT):
            if start_ms < 0 or end_ms < 0:
                complain(f"块 #{raw_index or '?'} 缺少时间码，已丢弃")
            elif end_ms < start_ms:
                complain(f"块 #{raw_index or '?'} 结束时间早于开始时间，已丢弃")
            else:
                cues.append(
                    Cue(
                        index=len(cues) + 1,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text="\n".join(text_lines).strip(),
                        raw_index=raw_index,
                    )
                )
        # 其余状态（IDLE / GOT_INDEX 遇空行）直接丢弃暂存
        state = _State.IDLE
        raw_index, start_ms, end_ms = "", -1, -1
        text_lines = []

    for lineno, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.rstrip()          # 去行尾空白/制表符；保留行首空白以兼容缩进
        is_blank = (line.strip() == "")
        parsed_time = None if is_blank else _parse_time_line(line)
        is_numeric = (not is_blank) and line.strip().isdigit()

        if state == _State.IDLE:
            if is_blank:
                continue  # 块间多余空行：幂等吞掉
            if parsed_time is not None:
                # 没有序号也能开块（序号在 SRT 里本就可选）
                start_ms, end_ms = parsed_time
                state = _State.GOT_TIME
            elif is_numeric:
                raw_index, state = line.strip(), _State.GOT_INDEX
            else:
                complain(f"第 {lineno} 行无法识别（既非序号也非时间码），已跳过: {line!r}")

        elif state == _State.GOT_INDEX:
            if is_blank:
                continue  # 序号与时间码之间的空行，容忍
            if parsed_time is not None:
                start_ms, end_ms = parsed_time
                state = _State.GOT_TIME
            elif is_numeric:
                raw_index = line.strip()  # 连续两个数字行：以后者为准
                complain(f"第 {lineno} 行出现重复序号，已采用后者")
            else:
                complain(f"第 {lineno} 行应为时间码，当前块被放弃: {line!r}")
                state = _State.IDLE

        elif state in (_State.GOT_TIME, _State.GOT_TEXT):
            if is_blank:
                flush()  # 规范的块结束
            elif parsed_time is not None:
                # 上一块缺空行直接粘连下一块：先收尾再开新块
                if state == _State.GOT_TIME:
                    complain(f"第 {lineno} 行前缺少空行，上一条字幕文本为空")
                flush()
                start_ms, end_ms = parsed_time
                state = _State.GOT_TIME
            elif is_numeric and state == _State.GOT_TEXT:
                # 文本后紧跟数字行且无空行：按新块序号处理（数字作为文本的情形极罕见）
                complain(f"第 {lineno} 行前缺少空行，按新字幕块处理")
                flush()
                raw_index, state = line.strip(), _State.GOT_INDEX
            else:
                text_lines.append(line)
                state = _State.GOT_TEXT

    flush()  # 文件末尾没有收尾空行时兜底

    # 零宽字幕（start == end）永远不会在半开区间内显示，单独提示
    for c in cues:
        if c.start_ms == c.end_ms:
            complain(f"字幕 #{c.index} 起止时间相同，将不会显示")
    return cues, warnings


def parse_srt_file(
    source: str | BinaryIO,
    *,
    strict: bool = False,
) -> tuple[list[Cue], list[str]]:
    """从路径/二进制流一步完成解码与解析。"""
    return parse_srt(read_srt_text(source), strict=strict)
