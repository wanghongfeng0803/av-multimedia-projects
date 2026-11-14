"""ASS/SSA 字幕解析：脚本段、样式表、事件与行内覆盖标签。

本模块与 :mod:`subburn.srt` 同级，负责把 ASS（Advanced SubStation Alpha，
含旧版 SSA v4）文本解析成与渲染器对接的结构化数据，**不做任何像素工作**。

ASS 与 SRT 的根本区别在于「样式」与「特效」：

* 样式在文件头的 ``[V4+ Styles]`` 段集中声明（字体、字号、颜色、描边、
  对齐、边距……），``[Events]`` 段的每条对话只引用样式名，可再用行内
  覆盖标签（override tags，``{\\...}``）做局部修改；
* 颜色是 ASS 私有的 ``&HAABBGGRR``（注意 BGR 顺序、alpha 0=不透明）；
* 时间码为 ``H:MM:SS.cc``（百分秒，两位小数），与 SRT 的毫秒逗号格式不同；
* 覆盖标签能在一行文本中途切换粗斜体、颜色、字号，还能给出定位
  （``\\pos``/``\\move``）、淡入淡出（``\\fad``）与卡拉OK（``\\k``）等特效。

设计要点：

* 编码探测/换行归一化复用 :func:`subburn.srt.decode_srt`，与 SRT 同一套
  BOM/UTF-16/GB18030 防御逻辑，不重复造轮子；
* 行内标签用「块级扫描 + 标签正则」的显式状态机处理，块分两类——
  ``{...}`` 内是标签块（``{\\...}`` 生效、``{...}`` 无反斜杠视为注释），
  块外是文本，扫描过程中把不断变化的样式状态快照成不可变 :class:`ASSRun`；
* 解析结果全部为冻结 dataclass，可哈希、可做渲染缓存键；
* 与 SRT 一致提供 ``strict`` 开关：宽松模式把异常收进诊断列表并尽量恢复，
  严格模式直接抛 :class:`ASSParseError`。

支持的标签子集见 :func:`_parse_tag_block`；未识别标签一律安静跳过，
保证真实世界里带复杂特效的文件也能降级渲染而不是报错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import BinaryIO

from .srt import decode_srt

__all__ = [
    "ASSParseError", "ASSColor", "ASSStyle", "ASSRun", "ASSLine",
    "ASSCue", "ASSDocument",
    "parse_ass_time", "parse_ass_text",
    "parse_ass", "parse_ass_file",
]


class ASSParseError(ValueError):
    """strict 模式下遇到无法恢复的 ASS 格式问题时抛出。"""


# ---------------------------------------------------------------------------
# 颜色：&HAABBGGRR（BGR，alpha 0=不透明、255=全透明）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ASSColor:
    """ASS 颜色的规范表示：8-bit RGBA（与 :class:`~subburn.style.Color` 同构）。

    解析时即完成 ``&HAABBGGRR`` -> RGBA 的通道交换与 alpha 反转，渲染层
    不再接触 ASS 的私有字节序。
    """

    r: int
    g: int
    b: int
    a: int = 255

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b

    @property
    def rgba(self) -> tuple[int, int, int, int]:
        return self.r, self.g, self.b, self.a

    @classmethod
    def parse(cls, token: str) -> "ASSColor":
        """解析 ``&HAABBGGRR`` / ``&HBBGGRR`` / ``&HGGGRR`` 等变体。

        ASS 颜色以 ``&H`` 开头、可带尾随 ``&``；按 BGR 从右向左落位，
        缺位通道补 0；头部剩余的两位按 alpha 解释并反转（ASS alpha 是
        「透明度」：0 不透明、255 全透明）。
        """
        s = token.strip().upper()
        if not s.startswith("&H"):
            raise ASSParseError(f"ASS 颜色需以 &H 开头: {token!r}")
        s = s[2:].rstrip("&").lstrip("-")
        if not s:
            raise ASSParseError(f"ASS 颜色为空: {token!r}")
        if not re.fullmatch(r"[0-9A-F]+", s):
            raise ASSParseError(f"ASS 颜色含非法十六进制字符: {token!r}")
        if len(s) > 8:
            raise ASSParseError(f"ASS 颜色超过 8 位十六进制: {token!r}")
        if len(s) <= 6:
            # 无 alpha：BGR 按右侧对齐补零，&HFF -> 0000FF（纯蓝），
            # &HFF00 -> 00FF00（纯绿），与 libass 的短色值解释一致
            s = s.rjust(6, "0")
            aa = "00"
        else:
            s = s.zfill(8)
            aa = s[:-6]
        body = s[-6:]
        b, g, r = int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16)
        return cls(r, g, b, 255 - int(aa, 16))


# ---------------------------------------------------------------------------
# 时间码：H:MM:SS.cc（百分秒；SSA v4 用单字符百分秒，兼容之）
# ---------------------------------------------------------------------------

def parse_ass_time(token: str) -> int:
    """把 ``H:MM:SS.cc`` 换算为毫秒整数。

    小数部分按百分秒解释：1~2 位右侧补零到 2 位再乘 10（``.5`` -> 500ms，
    ``.05`` -> 50ms）；三位小数按毫秒直接接受（容错部分非标准文件）。
    分钟/秒做 60 进制越界校验，与 SRT 时间码的严格度一致。
    """
    m = re.fullmatch(
        r"\s*(\d+):(\d{1,2}):(\d{1,2})[.](\d{1,3})\s*", token
    )
    if not m:
        raise ASSParseError(f"无法识别的 ASS 时间码: {token!r}")
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac_str = m.group(4)  # 保留原始字符串：前导零有位值意义（.05 != .5）
    if mm >= 60 or ss >= 60:
        raise ASSParseError(f"ASS 时间码越界: {token!r}")
    if len(frac_str) == 3:
        ms = int(frac_str)
    else:
        ms = int(frac_str.ljust(2, "0")[:2]) * 10
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


# ---------------------------------------------------------------------------
# 样式表：[V4+ Styles] / [V4 Styles]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ASSStyle:
    """一条 ``Style:`` 行的规范化表示。

    颜色统一为 :class:`ASSColor`（RGBA）；``bold``/``italic`` 归一为布尔
    （ASS 里 -1/1/True 都表示开启）；数值字段保留 ASS 的 PlayRes 坐标语义，
    缩放到真实画面分辨率由渲染器负责。
    """

    name: str
    font_name: str = "Arial"
    font_size: float = 20.0
    primary_color: ASSColor = ASSColor(255, 255, 255)
    secondary_color: ASSColor = ASSColor(255, 255, 0)
    outline_color: ASSColor = ASSColor(0, 0, 0)
    back_color: ASSColor = ASSColor(0, 0, 0)
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    border_style: int = 1
    outline_w: float = 2.0
    shadow_w: float = 2.0
    alignment: int = 2
    margin_l: int = 10
    margin_r: int = 10
    margin_v: int = 10
    scale_x: int = 100
    scale_y: int = 100
    spacing: int = 0
    encoding: int = 1

    @classmethod
    def from_fields(cls, name: str, fields: list[str],
                    fmt: list[str]) -> "ASSStyle":
        """按 Format 声明的列名把字段值对位填入，缺列/坏值回退默认。

        列名驱动而非位置驱动，使 V4（旧 SSA，列集合不同）与 V4+ 共用一套
        构造逻辑；数值列解析失败时保留默认值，不让一条坏样式拖垮整文件。
        """
        data: dict[str, str] = {}
        for i, col in enumerate(fmt):
            data[col.strip().lower()] = fields[i].strip() if i < len(fields) else ""

        def num(key: str, default: float) -> float:
            try:
                return float(data.get(key, ""))
            except ValueError:
                return default

        def color(key: str, default: ASSColor) -> ASSColor:
            raw = data.get(key, "")
            if not raw:
                return default
            try:
                return ASSColor.parse(raw)
            except ASSParseError:
                return default

        def flag(key: str) -> bool:
            v = data.get(key, "0").strip().lower()
            return v in ("-1", "1", "true")

        # SSA v4 的 Alignment 是 1..11（旧九/十一档方位），换算成 ASS 的
        # numpad 1..9；V4+ 本身就是 numpad，原样保留。V4 样式列里有
        # TertiaryColour/AlphaLevel，V4+ 没有，据此区分两种 Format。
        align = int(num("alignment", 2))
        columns = {c.strip().lower() for c in fmt}
        if "tertiarycolour" in columns or "alphalevel" in columns:
            align = _SSA4_TO_NUMPAD.get(align, 2)
        return cls(
            name=name,
            font_name=data.get("fontname", "Arial") or "Arial",
            font_size=num("fontsize", 20.0),
            primary_color=color("primarycolour", cls.primary_color),
            secondary_color=color("secondarycolour", cls.secondary_color),
            outline_color=color("outlinecolour", cls.outline_color),
            back_color=color("backcolour", cls.back_color),
            bold=flag("bold"),
            italic=flag("italic"),
            underline=flag("underline"),
            strikeout=flag("strikeout"),
            border_style=int(num("borderstyle", 1)),
            outline_w=num("outline", 2.0),
            shadow_w=num("shadow", 2.0),
            alignment=align,
            margin_l=int(num("marginl", 10)),
            margin_r=int(num("marginr", 10)),
            margin_v=int(num("marginv", 10)),
            scale_x=int(num("scalex", 100)),
            scale_y=int(num("scaley", 100)),
            spacing=int(num("spacing", 0)),
            encoding=int(num("encoding", 1)),
        )


#: SSA v4 的对齐编号 -> ASS numpad 对齐（libass 逆映射）
_SSA4_TO_NUMPAD = {1: 1, 2: 2, 3: 3, 4: 8, 5: 4, 6: 5, 7: 6,
                   8: 7, 9: 7, 10: 8, 11: 9}


# ---------------------------------------------------------------------------
# 事件文本：行内覆盖标签 -> 样式 run 序列
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ASSRun:
    """共享同一组行内样式的一段文本。

    :param k_dur_ms: 该 run 对应的卡拉OK 音节时长（``\\k`` 系列），0 表示
        非卡拉OK 文本；
    :param k_start_ms: 该音节相对对话起点的开始时间，由前面的 ``\\k`` 累积；
    :param k_fill: 卡拉OK 高亮方式：``"k"``（到点整体变色）/``"kf"``（扫过
        变色，本实现按到点变色近似）/``"ko"``（同 k，跳过边界清场语义）。
    """

    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    font_name: str = "Arial"
    font_size: float = 20.0
    primary: ASSColor = ASSColor(255, 255, 255)
    spacing: float = 0.0
    scale_x: int = 100
    scale_y: int = 100
    k_dur_ms: int = 0
    k_start_ms: int = 0
    k_fill: str = ""


@dataclass(frozen=True)
class ASSLine:
    """一条对话的一个显示行（``\\N`` 硬换行切分；``\\n`` 同样按换行处理）。"""

    runs: tuple[ASSRun, ...]


@dataclass(frozen=True)
class ASSCue:
    """一条 Dialogue 事件：半开 ``[start_ms, end_ms)`` 内显示若干行。"""

    index: int
    start_ms: int
    end_ms: int
    style_name: str
    lines: tuple[ASSLine, ...]
    layer: int = 0
    # -- 行内 cue 级特效（标签块里出现时覆盖样式表默认）------------------
    alignment: int | None = None
    pos: tuple[float, float] | None = None          # \pos(x,y)
    move: tuple[float, float, float, float] | None = None  # \move(x1,y1,x2,y2)
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    raw_text: str = ""


@dataclass(frozen=True)
class ASSDocument:
    """整个 ASS 文件的解析产物。"""

    styles: dict[str, ASSStyle]
    cues: tuple[ASSCue, ...]
    play_res_x: int
    play_res_y: int
    wrap_style: int = 0
    warnings: tuple[str, ...] = ()

    def style_of(self, cue: ASSCue) -> ASSStyle:
        """取对话引用的样式；缺失时回退首个样式或内置默认（永不抛异常）。"""
        st = self.styles.get(cue.style_name)
        if st is not None:
            return st
        if self.styles:
            return next(iter(self.styles.values()))
        return ASSStyle(name="Default")


# -- 标签分词：带括号参数的标签（\pos(...)）与裸标签（\b1、\k40）分开匹配，
#    每次只吃一个标签，游标推进路径简单且可单测。裸标签的参数允许为空，
#    否则单个无参标签（如裸 \r）会让整个分词循环提前退出 -----------------
_TAG_PAREN_RE = re.compile(r"([a-zA-Z]+)\(([^()]*)\)")
_TAG_BARE_RE = re.compile(r"([a-zA-Z]+)([^\\]*)")
_TAG_NAME_RE = re.compile(r"[a-zA-Z]+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class _RunState:
    """标签扫描过程中的可变样式状态；快照后凝固为 :class:`ASSRun`。"""

    def __init__(self, st: ASSStyle):
        self.bold = st.bold
        self.italic = st.italic
        self.underline = st.underline
        self.strikeout = st.strikeout
        self.font_name = st.font_name
        self.font_size = st.font_size
        self.primary = st.primary_color
        self.spacing = float(st.spacing)
        self.scale_x = st.scale_x
        self.scale_y = st.scale_y
        self.k_cursor_ms = 0
        self.k_pending_ms = 0
        self.k_pending_fill = ""
        self.drawing = False

    def snapshot(self, text: str) -> ASSRun:
        return ASSRun(
            text=text, bold=self.bold, italic=self.italic,
            underline=self.underline, strikeout=self.strikeout,
            font_name=self.font_name, font_size=self.font_size,
            primary=self.primary, spacing=self.spacing,
            scale_x=self.scale_x, scale_y=self.scale_y,
            k_dur_ms=self.k_pending_ms, k_start_ms=self.k_cursor_ms,
            k_fill=self.k_pending_fill,
        )


def _truthy_tag(value: str, default: bool) -> bool:
    """``\\b1``/``\\i0`` 一类开关：空参数按开启，0 关闭，非零开启。

    ``\\b`` 也可能携带字重数（如 ``\\b700``），>=600 视为粗体，与
    libass 的阈值一致。
    """
    if value == "":
        return True
    try:
        n = int(float(value))
    except ValueError:
        return default
    return n != 0 and (n == 1 or n < 100 or n >= 600)


def _first_number(args: str) -> float | None:
    m = _NUM_RE.search(args or "")
    return float(m.group(0)) if m else None


def _parse_tag_block(block: str, state: _RunState, styles: dict[str, ASSStyle],
                     default_style: ASSStyle,
                     cue_overrides: dict) -> None:
    """解析一个 ``{...}`` 块内的零到多个标签，就地修改状态/cue 覆盖项。

    块首不是反斜杠时按 ASS 注释处理（整个块忽略）；分词循环每次吃掉一个
    标签——先试带括号形态（``\\pos(x,y)``），再退到裸标签（``\\b1``、
    ``\\k40``），参数统一收集为字符串交给分派逻辑。
    """
    if not block.startswith("\\"):
        return
    i, n = 1, len(block)
    while i < n:
        if block[i] == "\\":  # 连续反斜杠：跳过分隔符
            i += 1
            continue
        if block[i:i + 1] == "r":
            # \r[样式名]：名字取到下一个反斜杠或块尾（裸 \r 时名为空串）
            nxt = block.find("\\", i)
            token = block[i:] if nxt < 0 else block[i:nxt]
            _apply_reset(state, styles, default_style, token[1:])
            i = n if nxt < 0 else nxt
            continue
        m = _TAG_PAREN_RE.match(block, i)
        if m:
            raw_name, args = m.group(1), m.group(2)
            i = m.end()
        else:
            m = _TAG_BARE_RE.match(block, i)
            if not m:
                return
            raw_name, args = m.group(1), m.group(2)
            i = m.end()
        # 大写 \K 是 \kf 的历史别名；其余标签名统一小写分派
        name = "kf" if raw_name == "K" else raw_name.lower()

        if name == "b":
            state.bold = _truthy_tag(args.strip(), state.bold)
        elif name == "i":
            state.italic = _truthy_tag(args.strip(), state.italic)
        elif name == "u":
            state.underline = _truthy_tag(args.strip(), state.underline)
        elif name == "s":
            state.strikeout = _truthy_tag(args.strip(), state.strikeout)
        elif name == "fn":
            if args.strip():
                state.font_name = args.strip()
        elif name == "fs":
            v = _first_number(args)
            if v is not None and v > 0:
                state.font_size = v
        elif name in ("c", "1c"):
            try:
                state.primary = ASSColor.parse(args.strip())
            except ASSParseError:
                pass
        elif name in ("alpha", "1alpha"):
            mm = re.search(r"[0-9A-Fa-f]{1,2}", args)
            if mm:
                state.primary = ASSColor(
                    state.primary.r, state.primary.g, state.primary.b,
                    255 - int(mm.group(0), 16),
                )
        elif name == "fsp":
            v = _first_number(args)
            if v is not None and v >= 0:
                state.spacing = v
        elif name == "fscx":
            v = _first_number(args)
            if v is not None and v > 0:
                state.scale_x = int(round(v))
        elif name == "fscy":
            v = _first_number(args)
            if v is not None and v > 0:
                state.scale_y = int(round(v))
        elif name == "an":
            v = _first_number(args)
            if v is not None and 1 <= int(v) <= 9:
                cue_overrides["alignment"] = int(v)
        elif name == "a":
            # 旧式 SSA 对齐（1..4/5..11），映射到 numpad
            v = _first_number(args)
            if v is not None and int(v) in _SSA4_TO_NUMPAD:
                cue_overrides["alignment"] = _SSA4_TO_NUMPAD[int(v)]
        elif name == "pos":
            nums = _NUM_RE.findall(args)
            if len(nums) >= 2:
                cue_overrides["pos"] = (float(nums[0]), float(nums[1]))
        elif name == "move":
            nums = _NUM_RE.findall(args)
            if len(nums) >= 4:
                cue_overrides["move"] = tuple(float(x) for x in nums[:4])
        elif name == "fad":
            nums = _NUM_RE.findall(args)
            if len(nums) >= 2:
                cue_overrides["fade_in_ms"] = max(0, int(float(nums[0])))
                cue_overrides["fade_out_ms"] = max(0, int(float(nums[1])))
        elif name == "fade":
            # \fade(a1,a2,a3,t1,t2,t3,t4)：本实现只取淡入(t2-t1)与淡出(t4-t3)
            nums = _NUM_RE.findall(args)
            if len(nums) >= 7:
                cue_overrides["fade_in_ms"] = max(
                    0, int(float(nums[4])) - int(float(nums[3])))
                cue_overrides["fade_out_ms"] = max(
                    0, int(float(nums[6])) - int(float(nums[5])))
        elif name in ("k", "kf", "ko"):
            v = _first_number(args)
            if v is not None:
                # 先结清上一个音节的游标，再挂起新音节；文本在块外切段时落账
                state.k_cursor_ms += state.k_pending_ms
                state.k_pending_ms = max(0, int(round(v * 10)))
                # 裸标签正则只吃小写，大写 \K 在入口小写化后落到这里，语义同 kf
                state.k_pending_fill = name
        elif name == "p":
            # 矢量绘图开关：\p1 进入绘图模式，\p0 退出。绘图内容本版本不
            # 实现，进入后跳过产生的文本 run。
            v = _first_number(args)
            state.drawing = v is not None and int(v) != 0
        elif name == "r":
            pass  # 已在分词循环入口特殊处理（需要携带可选样式名）
        # 其余标签（q/n/fr/fax/blur/bord/shad/2c/3c/4c/t……）安静忽略


def _apply_reset(state: _RunState, styles: dict[str, ASSStyle],
                 default_style: ASSStyle, args: str) -> None:
    """``\\r[样式名]``：把行内状态重置回指定样式（缺省为对话默认样式）。"""
    name = args.strip()
    st = styles.get(name, default_style)
    state.__init__(st)  # 重新走一遍 __init__ 即完成全字段复位


def _unescape_ass_text(text: str) -> str:
    """处理标签块外的 ASS 文本转义。"""
    return (text.replace("\\h", " ").replace("\\n", "\n")
                .replace("\\N", "\n").replace("\\ ", " "))


def parse_event_text(raw: str, st: ASSStyle,
                     styles: dict[str, ASSStyle]) -> tuple[tuple[ASSLine, ...], dict]:
    """把一条 Dialogue 的 Text 字段解析为 ``(行序列, cue 级覆盖字典)``。

    扫描按字符推进，维护「当前文本缓冲」与 :class:`_RunState`：

    * 遇到 ``{`` 找到配对 ``}``：先把缓冲落成一个 run，再解析标签块；
      ``\\r`` 类块在解析前后各切一刀，保证重置只影响其后的文本；
    * 标签块外的 ``\\N``/``\\n`` 转成换行，最终按行切分 :class:`ASSLine`；
    * 绘图模式（``\\p1``）期间的文本直接丢弃。
    """
    overrides: dict = {}
    state = _RunState(st)
    runs: list[ASSRun] = []
    buf: list[str] = []

    def flush() -> None:
        if buf and not state.drawing:
            piece = _unescape_ass_text("".join(buf))
            if piece:
                runs.append(state.snapshot(piece))
        buf.clear()

    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == "{":
            end = raw.find("}", i + 1)
            if end < 0:  # 未闭合的标签块：剩余文本按普通文本处理
                break
            flush()
            block = raw[i + 1:end]
            _parse_tag_block(block, state, styles, st, overrides)
            i = end + 1
            continue
        buf.append(ch)
        i += 1
    flush()

    # 按换行切行；空文本行保留为空 run 序列的行（与 SRT 多行语义一致）
    lines: list[ASSLine] = []
    current: list[ASSRun] = []
    for run in runs:
        parts = run.text.split("\n")
        for j, part in enumerate(parts):
            if j > 0:
                lines.append(ASSLine(tuple(current)))
                current = []
            if part:
                # 冻结 dataclass 用 replace 派生换文本的同族 run
                current.append(replace(run, text=part))
    lines.append(ASSLine(tuple(current)))
    return tuple(lines), overrides


# ---------------------------------------------------------------------------
# 文件级解析：段 -> 样式表 -> 事件
# ---------------------------------------------------------------------------

def _split_fields(text: str) -> list[str]:
    """按逗号切分字段；Text 列允许含逗号，调用方对最后一列另行整体取回。"""
    return text.split(",")


def parse_ass_text(text: str, *, strict: bool = False
                   ) -> tuple[ASSDocument, list[str]]:
    """解析 ASS/SSA 文本，返回 ``(文档对象, 诊断列表)``。

    :param strict: 为 True 时任何被容忍的异常抛 :class:`ASSParseError`，
        否则收进诊断列表并尽量恢复。
    """
    from .srt import normalize_newlines
    text = normalize_newlines(text)
    warnings: list[str] = []

    def complain(msg: str) -> None:
        if strict:
            raise ASSParseError(msg)
        warnings.append(msg)

    section = ""
    style_format: list[str] = []
    event_format: list[str] = []
    styles: dict[str, ASSStyle] = {}
    raw_cues: list[tuple[int, str]] = []
    play_res_x = play_res_y = 0
    wrap_style = 0

    for lineno, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip().lower(), value.strip()

        if section == "script info":
            if key == "playresx":
                play_res_x = _safe_int(value, 0)
            elif key == "playresy":
                play_res_y = _safe_int(value, 0)
            elif key == "wrapstyle":
                wrap_style = _safe_int(value, 0)
        elif section in ("v4+ styles", "v4 styles"):
            if key == "format":
                style_format = [c.strip().lower() for c in value.split(",")]
            elif key == "style":
                if not style_format:
                    complain(f"第 {lineno} 行 Style 出现在 Format 之前，已忽略")
                    continue
                fields = _split_fields(value)
                name = fields[0].strip() if fields else f"Style{len(styles) + 1}"
                try:
                    styles[name] = ASSStyle.from_fields(
                        name, fields, style_format)
                except Exception as exc:  # 防御：单条坏样式不拖垮全局
                    complain(f"第 {lineno} 行样式 {name!r} 解析失败已跳过: {exc}")
        elif section == "events":
            if key == "format":
                event_format = [c.strip().lower() for c in value.split(",")]
            elif key in ("dialogue", "comment"):
                raw_cues.append((lineno, line))

    if not event_format:
        # 缺 Format 声明时按 V4+ 标准列兜底
        event_format = [
            "layer", "start", "end", "style", "name",
            "marginl", "marginr", "marginv", "effect", "text",
        ]

    cues: list[ASSCue] = []
    for seq, (lineno, line) in enumerate(raw_cues, start=1):
        kind, rest = line.split(":", 1)
        cols = rest.split(",", len(event_format) - 1)
        if len(cols) < len(event_format):
            complain(f"第 {lineno} 行事件字段不足，已忽略")
            continue
        row = {col: (cols[i].strip() if i < len(cols) else "")
               for i, col in enumerate(event_format)}
        if kind.strip().lower() == "comment":
            continue
        try:
            start_ms = parse_ass_time(row.get("start", ""))
            end_ms = parse_ass_time(row.get("end", ""))
        except ASSParseError as exc:
            complain(f"第 {lineno} 行时间码解析失败，事件已忽略: {exc}")
            continue
        if end_ms < start_ms:
            complain(f"第 {lineno} 行结束早于开始，事件已忽略")
            continue

        style_name = row.get("style", "Default")
        base_style = styles.get(style_name)
        if base_style is None:
            complain(f"第 {lineno} 行引用了未定义样式 {style_name!r}，回退默认")
            base_style = (next(iter(styles.values()))
                          if styles else ASSStyle(name="Default"))
        raw_text = row.get("text", "")
        lines, overrides = parse_event_text(raw_text, base_style, styles)
        if not any(lr.runs for lr in lines):
            complain(f"第 {lineno} 行事件无可见文本，已忽略")
            continue
        cues.append(ASSCue(
            index=len(cues) + 1,
            start_ms=start_ms, end_ms=end_ms,
            style_name=style_name, lines=lines,
            layer=_safe_int(row.get("layer", "0"), 0),
            alignment=overrides.get("alignment"),
            pos=overrides.get("pos"),
            move=overrides.get("move"),
            fade_in_ms=overrides.get("fade_in_ms", 0),
            fade_out_ms=overrides.get("fade_out_ms", 0),
            raw_text=raw_text,
        ))

    doc = ASSDocument(
        styles=styles,
        cues=tuple(cues),
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        wrap_style=wrap_style,
        warnings=tuple(warnings),
    )
    return doc, warnings


def _safe_int(token: str, default: int) -> int:
    try:
        return int(float(token.strip()))
    except (ValueError, AttributeError):
        return default


def parse_ass(source: str | BinaryIO, *, strict: bool = False
              ) -> tuple[ASSDocument, list[str]]:
    """从路径/二进制流一步完成解码与解析（编码探测复用 SRT 通道）。"""
    if hasattr(source, "read"):
        text = decode_srt(source.read())
    else:
        with open(source, "rb") as f:
            text = decode_srt(f.read())
    return parse_ass_text(text, strict=strict)


#: 向后兼容别名，命名与 :func:`subburn.srt.parse_srt_file` 对齐
def parse_ass_file(source: str | BinaryIO, *, strict: bool = False
                   ) -> tuple[ASSDocument, list[str]]:
    return parse_ass(source, strict=strict)
