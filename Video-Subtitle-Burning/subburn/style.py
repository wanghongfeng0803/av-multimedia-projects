"""字幕样式描述：字体 / 颜色 / 位置三类可配置参数的数据结构与归一化。

本模块只负责「样式参数」的表示、解析与校验，不接触任何像素；渲染层
（:mod:`subburn.render`）把这些描述映射为具体的光栅化与排版动作。数据结构
全部为冻结 dataclass / Enum，可哈希、可做缓存键，也便于序列化成 JSON。

三个子系统：

* **颜色**：:class:`Color` 以 8-bit RGBA 为规范表示，支持 ``#rgb(a)`` /
  ``rgb()`` / ``rgba()`` / ``hsl()`` / 常见命名色 / 元组 / 整数等多种输入，
  并提供 RGB↔HSL、RGB↔HSV 的整数精确颜色空间转换；
* **字体**：:class:`FontSpec` 描述字体族（按优先级排列的回退链）、字号与
  :class:`FontWeight` 字重，字体文件是否存在由 :mod:`subburn.fonts` 解析；
* **位置**：:class:`Alignment`（水平/垂直对齐）+ :class:`Margins`（四边边距）
  组成 :class:`Position`，并在 :meth:`Position.anchor_origin` 中统一处理
  边距越界与画面夹紧。

顶层 :class:`SubtitleStyle` 聚合全部参数，并提供 :meth:`from_legacy` 与旧版
:class:`subburn.render.Style` 互操作，保证既有调用方零改动。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Iterable

__all__ = [
    "Color", "FontWeight", "FontSpec",
    "HAlign", "VAlign", "Margins", "Position", "SubtitleStyle",
]


# ---------------------------------------------------------------------------
# 颜色：规范表示为 8-bit RGBA
# ---------------------------------------------------------------------------

#: 常见 CSS 命名色（小写名 -> 8-bit RGB）。覆盖基础 16 色与高频扩展色。
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0), "silver": (192, 192, 192), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "white": (255, 255, 255),
    "maroon": (128, 0, 0), "red": (255, 0, 0), "darkred": (139, 0, 0),
    "purple": (128, 0, 128), "fuchsia": (255, 0, 255),
    "magenta": (255, 0, 255), "pink": (255, 192, 203),
    "green": (0, 128, 0), "darkgreen": (0, 100, 0), "lime": (0, 255, 0),
    "olive": (128, 128, 0), "yellow": (255, 255, 0), "gold": (255, 215, 0),
    "navy": (0, 0, 128), "blue": (0, 0, 255), "darkblue": (0, 0, 139),
    "teal": (0, 128, 128), "aqua": (0, 255, 255), "cyan": (0, 255, 255),
    "orange": (255, 165, 0),
    "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169),
    "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
    "transparent": (0, 0, 0),
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3,8})$")
_FUNC_RE = re.compile(
    r"^(rgba?|hsla?)\(\s*([^)]*?)\s*\)$", re.IGNORECASE
)


def _clamp_byte(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


def _parse_percent(token: str) -> float:
    """把 ``'70%'`` 解析为 0.7；非百分号数字按 0~1 浮点处理。"""
    token = token.strip()
    if token.endswith("%"):
        return float(token[:-1].strip()) / 100.0
    return float(token)


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """HSL(色相 0~360 度，S/L 为 0~1) -> 8-bit RGB，标准六分区算法。"""
    h = h % 360.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return (
        round((r1 + m) * 255),
        round((g1 + m) * 255),
        round((b1 + m) * 255),
    )


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """HSV(色相 0~360 度，S/V 为 0~1) -> 8-bit RGB。"""
    h = h % 360.0
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return (
        round((r1 + m) * 255),
        round((g1 + m) * 255),
        round((b1 + m) * 255),
    )


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rn, gn, bn), min(rn, gn, bn)
    delta = cmax - cmin
    h = _hue_from_sector(rn, gn, bn, cmax, delta)
    l = (cmax + cmin) / 2.0
    s = 0.0 if delta == 0.0 else delta / (1.0 - abs(2.0 * l - 1.0))
    return h, s, l


def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rn, gn, bn), min(rn, gn, bn)
    delta = cmax - cmin
    h = _hue_from_sector(rn, gn, bn, cmax, delta)
    s = 0.0 if cmax == 0.0 else delta / cmax
    return h, s, cmax


def _hue_from_sector(rn: float, gn: float, bn: float,
                     cmax: float, delta: float) -> float:
    if delta == 0.0:
        return 0.0
    if cmax == rn:
        h = 60.0 * (((gn - bn) / delta) % 6.0)
    elif cmax == gn:
        h = 60.0 * ((bn - rn) / delta + 2.0)
    else:
        h = 60.0 * ((rn - gn) / delta + 4.0)
    return h % 360.0


@dataclass(frozen=True)
class Color:
    """规范化的 8-bit RGBA 颜色。

    构造时逐通道校验范围；需要宽容输入（用户配置可能越界）时用
    :meth:`parse` 的 ``clamp=True``。透明度 ``a=0`` 表示全透明，``255``
    不透明，与瓦片的 alpha 混合公式直接对接。
    """

    r: int
    g: int
    b: int
    a: int = 255

    def __post_init__(self) -> None:
        for name, v in (("r", self.r), ("g", self.g),
                        ("b", self.b), ("a", self.a)):
            if not isinstance(v, int) or not 0 <= v <= 255:
                raise ValueError(f"颜色通道 {name} 必须是 0~255 的整数，收到 {v!r}")

    # -- 便捷构造 --------------------------------------------------------
    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> "Color":
        return cls(r, g, b, 255)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float,
                 a: int = 255) -> "Color":
        """HSL 构造：h 为 0~360 度（自动取模），s/l 为 0~1。"""
        s, l = _unit_interval("s", s), _unit_interval("l", l)
        r, g, b = _hsl_to_rgb(h, s, l)
        return cls(r, g, b, a)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float,
                 a: int = 255) -> "Color":
        """HSV 构造：h 为 0~360 度（自动取模），s/v 为 0~1。"""
        s, v = _unit_interval("s", s), _unit_interval("v", v)
        r, g, b = _hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    def with_alpha(self, a: int) -> "Color":
        """返回替换透明度后的新颜色（不可变对象的派生手段）。"""
        return Color(self.r, self.g, self.b, a)

    # -- 多格式解析 ------------------------------------------------------
    @classmethod
    def parse(cls, spec: "Color | str | int | Iterable[Any]",
              *, clamp: bool = False) -> "Color":
        """把多种颜色描述解析为 :class:`Color`。

        支持：

        * 已是 :class:`Color`：原样返回；
        * 字符串：``#rgb`` / ``#rgba`` / ``#rrggbb`` / ``#rrggbbaa``、
          ``rgb(r,g,b)`` / ``rgba(r,g,b,a)``、``hsl(h,s%,l%)`` /
          ``hsla(...)``、CSS 命名色（``red``、``transparent`` 等）；
        * 整数：``0xRRGGBB`` 或 ``0xRRGGBBAA``；
        * 序列：3~4 个元素，整数按 0~255、浮点按 0~1 解释。

        :param clamp: 为 True 时越界通道夹紧而不是抛异常，用于消化
            不受信任的外部配置。
        """
        if isinstance(spec, Color):
            return spec
        if isinstance(spec, int):
            return cls._from_int(spec, clamp)
        if isinstance(spec, str):
            return cls._from_str(spec.strip(), clamp)
        if isinstance(spec, (tuple, list)):
            return cls._from_seq(spec, clamp)
        raise ValueError(f"无法解析颜色: {spec!r}")

    @classmethod
    def _from_int(cls, value: int, clamp: bool) -> "Color":
        if value < 0 or value > 0xFFFFFFFF:
            if not clamp:
                raise ValueError(f"整数颜色超出 0xRRGGBB(AA) 范围: {value!r}")
            value = max(0, min(value, 0xFFFFFFFF))
        if value <= 0xFFFFFF:
            return cls((value >> 16) & 255, (value >> 8) & 255,
                       value & 255, 255)
        return cls((value >> 24) & 255, (value >> 16) & 255,
                   (value >> 8) & 255, value & 255)

    @classmethod
    def _from_str(cls, text: str, clamp: bool) -> "Color":
        low = text.lower()
        if low in _NAMED_COLORS:
            r, g, b = _NAMED_COLORS[low]
            return cls(r, g, b, 0 if low == "transparent" else 255)

        m = _HEX_RE.match(text)
        if m:
            return cls._from_hex(m.group(1), clamp)

        m = _FUNC_RE.match(text)
        if m:
            kind = m.group(1).lower()
            parts = [p.strip() for p in m.group(2).split(",") if p.strip() != ""]
            if kind.startswith("hsl"):
                return cls._from_hsl_func(parts, clamp)
            return cls._from_rgb_func(parts, clamp)

        raise ValueError(f"无法识别的颜色格式: {text!r}")

    @classmethod
    def _from_hex(cls, digits: str, clamp: bool) -> "Color":
        if len(digits) not in (3, 4, 6, 8):
            raise ValueError(f"十六进制颜色位数非法: #{digits}")
        if len(digits) <= 4:  # #rgb / #rgba：每位展开成两位
            pairs = [ch * 2 for ch in digits]
        else:
            pairs = [digits[i:i + 2] for i in range(0, len(digits), 2)]
        vals = [int(p, 16) for p in pairs]
        if len(vals) == 3:
            vals.append(255)
        return cls._build(vals, clamp)

    @classmethod
    def _from_rgb_func(cls, parts: list[str], clamp: bool) -> "Color":
        if len(parts) not in (3, 4):
            raise ValueError(f"rgb()/rgba() 需要 3~4 个参数，收到 {len(parts)}")
        vals: list[int] = []
        for i, tok in enumerate(parts):
            if i < 3:
                vals.append(_byte_from_token(tok, clamp))
            else:
                vals.append(_alpha_from_token(tok, clamp))
        if len(vals) == 3:
            vals.append(255)
        return cls._build(vals, clamp)

    @classmethod
    def _from_hsl_func(cls, parts: list[str], clamp: bool) -> "Color":
        if len(parts) not in (3, 4):
            raise ValueError(f"hsl()/hsla() 需要 3~4 个参数，收到 {len(parts)}")
        try:
            h = float(parts[0].rstrip("deg").strip()) % 360.0
            s = _parse_percent(parts[1])
            l = _parse_percent(parts[2])
        except ValueError as exc:
            raise ValueError(f"hsl() 参数无法解析: {parts!r}") from exc
        if clamp:
            s, l = max(0.0, min(1.0, s)), max(0.0, min(1.0, l))
        elif not 0.0 <= s <= 1.0 or not 0.0 <= l <= 1.0:
            raise ValueError(f"hsl() 饱和度/亮度必须在 0~1 之间: {parts!r}")
        a = _alpha_from_token(parts[3], clamp) if len(parts) == 4 else 255
        r, g, b = _hsl_to_rgb(h, s, l)
        return cls(r, g, b, a)

    @classmethod
    def _from_seq(cls, seq: Iterable[Any], clamp: bool) -> "Color":
        vals = list(seq)
        if len(vals) not in (3, 4):
            raise ValueError(f"颜色序列长度必须为 3 或 4，收到 {len(vals)}")
        out: list[int] = []
        for v in vals:
            if isinstance(v, float):  # 浮点通道按 0~1 解释
                if not clamp and not 0.0 <= v <= 1.0:
                    raise ValueError(f"浮点颜色通道必须在 0~1 之间: {v!r}")
                out.append(_clamp_byte(round(v * 255)))
            elif isinstance(v, int):
                out.append(_clamp_byte(v) if clamp else v)
            else:
                raise ValueError(f"颜色通道必须是 int/float: {v!r}")
        if len(out) == 3:
            out.append(255)
        return cls._build(out, clamp)

    @classmethod
    def _build(cls, vals: list[int], clamp: bool) -> "Color":
        if clamp:
            vals = [_clamp_byte(v) for v in vals]
        return cls(vals[0], vals[1], vals[2], vals[3])

    # -- 输出与颜色空间 --------------------------------------------------
    @property
    def rgb(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b

    @property
    def rgba(self) -> tuple[int, int, int, int]:
        return self.r, self.g, self.b, self.a

    def to_hsl(self) -> tuple[float, float, float]:
        """返回 (h[0,360), s[0,1], l[0,1])。"""
        return _rgb_to_hsl(self.r, self.g, self.b)

    def to_hsv(self) -> tuple[float, float, float]:
        """返回 (h[0,360), s[0,1], v[0,1])。"""
        return _rgb_to_hsv(self.r, self.g, self.b)

    def to_hex(self) -> str:
        """十六进制表示；不透明时输出 ``#rrggbb``，带透明度时追加两位 alpha。"""
        base = f"#{self.r:02x}{self.g:02x}{self.b:02x}"
        return base if self.a == 255 else base + f"{self.a:02x}"

    def __str__(self) -> str:
        return self.to_hex()


def _byte_from_token(tok: str, clamp: bool) -> int:
    if tok.endswith("%"):  # rgb 通道百分比：按比例映射到 0~255
        ratio = _parse_percent(tok)
        if not clamp and not 0.0 <= ratio <= 1.0:
            raise ValueError(f"rgb 百分比越界: {tok!r}")
        return _clamp_byte(round(ratio * 255))
    v = int(float(tok))
    return _clamp_byte(v) if clamp else v


def _alpha_from_token(tok: str, clamp: bool) -> int:
    """alpha 标记：百分比 / 0~1 浮点 / 0~255 整数（整数按字节解释）。"""
    if tok.endswith("%"):
        ratio = _parse_percent(tok)
        if not clamp and not 0.0 <= ratio <= 1.0:
            raise ValueError(f"alpha 百分比越界: {tok!r}")
        return _clamp_byte(round(ratio * 255))
    if "." in tok:  # 浮点 alpha 固定按 0~1
        ratio = float(tok)
        if not clamp and not 0.0 <= ratio <= 1.0:
            raise ValueError(f"alpha 浮点必须在 0~1 之间: {tok!r}")
        return _clamp_byte(round(ratio * 255))
    v = int(tok)
    return _clamp_byte(v) if clamp else v


def _unit_interval(name: str, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} 必须落在 0~1，收到 {v!r}")
    return v


# ---------------------------------------------------------------------------
# 字体：字重 + 字体规格
# ---------------------------------------------------------------------------

class FontWeight(Enum):
    """CSS 风格的九档字重，值为 OpenType 权重数。"""

    THIN = 100
    EXTRALIGHT = 200
    LIGHT = 300
    REGULAR = 400
    MEDIUM = 500
    SEMIBOLD = 600
    BOLD = 700
    EXTRABOLD = 800
    BLACK = 900

    @classmethod
    def nearest(cls, value: int) -> "FontWeight":
        """把任意整数权重吸附到最近的标准档（夹紧到 100~900）。"""
        step = max(100, min(900, int(value)))
        step = int(round(step / 100.0) * 100)
        return cls(step)

    @classmethod
    def parse(cls, token: "FontWeight | int | str") -> "FontWeight":
        """接受枚举、整数（就近吸附）与名称（含 ``normal``/``bold`` 别名）。"""
        if isinstance(token, FontWeight):
            return token
        if isinstance(token, int):
            return cls.nearest(token)
        if isinstance(token, str):
            low = token.strip().lower().replace("-", "").replace("_", "")
            # 别名与枚举名统一按小写比较（枚举名 BOLD -> "bold"）
            aliases = {"normal": "regular",
                       "demibold": "semibold", "heavy": "black",
                       "ultrabold": "extrabold", "ultralight": "extralight"}
            low = aliases.get(low, low)
            for w in cls:
                if w.name.lower() == low:
                    return w
            if token.strip().isdigit():
                return cls.nearest(int(token.strip()))
        raise ValueError(f"无法解析字重: {token!r}")


@dataclass(frozen=True)
class FontSpec:
    """字体规格：字体族回退链 + 字号 + 字重。

    :param families: 按优先级排列的字体族名/字体文件路径；空元组表示交给
        :mod:`subburn.fonts` 走系统默认探测链。
    :param size: 像素字号；``0`` 表示按视频高度自动选择。
    """

    families: tuple[str, ...] = ()
    size: int = 0
    weight: FontWeight = FontWeight.REGULAR

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError(f"字号必须是非负整数（0 为自动），收到 {self.size!r}")
        if not isinstance(self.families, tuple):
            object.__setattr__(self, "families", tuple(self.families))

    @classmethod
    def parse(cls, spec: "FontSpec | dict | None") -> "FontSpec":
        if spec is None:
            return cls()
        if isinstance(spec, FontSpec):
            return spec
        if isinstance(spec, dict):
            families = spec.get("families") or spec.get("family") or ()
            if isinstance(families, str):
                families = [families]
            return cls(
                families=tuple(families),
                size=int(spec.get("size", 0)),
                weight=FontWeight.parse(spec.get("weight", FontWeight.REGULAR)),
            )
        raise ValueError(f"无法解析字体规格: {spec!r}")


# ---------------------------------------------------------------------------
# 位置：对齐 + 四边边距
# ---------------------------------------------------------------------------

class HAlign(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VAlign(Enum):
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


def _parse_align(enum_cls: type[Enum], token: "Enum | str",
                 aliases: dict[str, str] | None = None) -> Enum:
    if isinstance(token, enum_cls):
        return token
    aliases = aliases or {}
    key = str(token).strip().lower()
    key = aliases.get(key, key)
    for m in enum_cls:
        if m.value == key:
            return m
    raise ValueError(f"无法解析 {enum_cls.__name__}: {token!r}")


@dataclass(frozen=True)
class Margins:
    """四边边距（像素），顺序为 左/上/右/下，均为非负整数。"""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"边距 {name} 必须是非负整数，收到 {v!r}")

    @classmethod
    def uniform(cls, value: int) -> "Margins":
        return cls(value, value, value, value)

    @classmethod
    def parse(cls, spec: "Margins | int | dict | Iterable[int] | None") -> "Margins":
        """接受整数（四边相同）、四元序列、含 left/top/right/bottom 的字典。"""
        if spec is None:
            return cls()
        if isinstance(spec, Margins):
            return spec
        if isinstance(spec, int):
            return cls.uniform(spec)
        if isinstance(spec, dict):
            return cls(**{k: int(spec.get(k, 0))
                          for k in ("left", "top", "right", "bottom")})
        vals = list(spec)
        if len(vals) != 4:
            raise ValueError(f"边距序列必须含 4 个值（左上右下），收到 {len(vals)}")
        return cls(*(int(v) for v in vals))

    def clamp_to_frame(self, frame_w: int, frame_h: int) -> "Margins":
        """处理边距越界：单边夹紧到画面内；对边之和占满画面时双双归零。

        若不做这层归一化，``left + right >= 画面宽`` 之类的配置会算出负的
        可用宽度，把字幕推到画面外。归零时保留「至少能显示」的确定性结果。
        """
        left, right = min(self.left, frame_w), min(self.right, frame_w)
        top, bottom = min(self.top, frame_h), min(self.bottom, frame_h)
        if left + right >= frame_w and frame_w > 0:
            left = right = 0
        if top + bottom >= frame_h and frame_h > 0:
            top = bottom = 0
        return Margins(left, top, right, bottom)


@dataclass(frozen=True)
class Position:
    """字幕在画面中的锚定方式：水平/垂直对齐 + 四边边距。"""

    halign: HAlign = HAlign.CENTER
    valign: VAlign = VAlign.BOTTOM
    margins: Margins = field(default_factory=lambda: Margins.uniform(36))

    @classmethod
    def parse(cls, spec: "Position | dict | None") -> "Position":
        if spec is None:
            return cls()
        if isinstance(spec, Position):
            return spec
        if not isinstance(spec, dict):
            raise ValueError(f"无法解析位置配置: {spec!r}")
        return cls(
            halign=_parse_align(
                HAlign, spec.get("halign", HAlign.CENTER),
                {"centre": "center", "middle": "center"},
            ),
            valign=_parse_align(
                VAlign, spec.get("valign", VAlign.BOTTOM),
                {"centre": "middle", "center": "middle"},
            ),
            margins=Margins.parse(spec.get("margins", spec.get("margin", 36))),
        )

    def anchor_origin(self, frame_w: int, frame_h: int,
                      box_w: int, box_h: int) -> tuple[int, int]:
        """计算字幕块左上角坐标，夹紧到画面内（越界配置也不会推出画面）。

        块比画面小时严格落在边距围成的安全区内；块比画面还大时退化为贴
        边（坐标 0），超出部分由混合函数裁剪，保证至少左上角内容可见。
        """
        m = self.margins.clamp_to_frame(frame_w, frame_h)
        if self.halign is HAlign.LEFT:
            x = m.left
        elif self.halign is HAlign.RIGHT:
            x = frame_w - m.right - box_w
        else:
            x = (frame_w - box_w) // 2
        if self.valign is VAlign.TOP:
            y = m.top
        elif self.valign is VAlign.MIDDLE:
            y = (frame_h - box_h) // 2
        else:
            y = frame_h - m.bottom - box_h
        x = max(0, min(x, max(0, frame_w - box_w)))
        y = max(0, min(y, max(0, frame_h - box_h)))
        return x, y


# ---------------------------------------------------------------------------
# 顶层样式聚合
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubtitleStyle:
    """完整字幕样式描述：字体 + 颜色 + 描边/阴影/底盒 + 位置与排版间距。

    默认值刻意与旧版 :class:`subburn.render.Style` 的像素行为一致，使默认
    渲染结果不发生任何变化。
    """

    font: FontSpec = field(default_factory=FontSpec)
    color: Color = Color(255, 255, 255)              # 填充色（含文字透明度）
    outline_color: Color = Color(0, 0, 0)
    outline_width: int = 2
    shadow_offset: int = 2
    shadow_color: Color = Color(0, 0, 0, 140)        # 与旧硬编码阴影透明度一致
    position: Position = field(default_factory=Position)
    line_gap: int = 6
    tile_gap: int = 12
    box: bool = False
    box_color: Color = Color(0, 0, 0, 128)           # 与旧 box_alpha 默认值一致
    box_padding: int = 12

    def __post_init__(self) -> None:
        for name in ("outline_width", "shadow_offset", "line_gap",
                     "tile_gap", "box_padding"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} 必须是非负整数，收到 {v!r}")

    # -- 与旧版 Style 的桥接 --------------------------------------------
    @classmethod
    def from_legacy(cls, legacy: "object") -> "SubtitleStyle":
        """把旧版 :class:`subburn.render.Style` 逐项翻译为新样式。"""
        from .render import Style  # 局部导入避免模块环依赖

        if isinstance(legacy, SubtitleStyle):
            return legacy
        if not isinstance(legacy, Style):
            raise TypeError(f"from_legacy 仅接受旧版 Style，收到 {type(legacy)!r}")
        return cls(
            color=Color(*legacy.color, 255),
            outline_color=Color(*legacy.outline_color, 255),
            outline_width=legacy.outline_width,
            shadow_offset=legacy.shadow_offset,
            position=Position(margins=Margins.uniform(legacy.margin)),
            line_gap=legacy.line_gap,
            tile_gap=legacy.tile_gap,
            box=legacy.box,
            box_color=Color(*legacy.box_color, legacy.box_alpha),
            box_padding=legacy.box_padding,
        )

    # -- 字典 / JSON 形态 ------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, *, clamp: bool = False) -> "SubtitleStyle":
        """从部分字段字典构造（嵌套 font/position/color 均支持）。

        未提供的字段沿用默认值；颜色与枚举经各自的 ``parse`` 消化，
        ``clamp=True`` 时数值越界走夹紧策略而非抛异常。
        """
        if not isinstance(data, dict):
            raise ValueError("样式配置必须是字典")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"样式配置含未知字段: {sorted(unknown)}")

        def color_of(key: str, default: Color) -> Color:
            return Color.parse(data[key], clamp=clamp) if key in data else default

        return cls(
            font=FontSpec.parse(data.get("font")),
            color=color_of("color", cls.color),
            outline_color=color_of("outline_color", cls.outline_color),
            outline_width=int(data.get("outline_width", cls.outline_width)),
            shadow_offset=int(data.get("shadow_offset", cls.shadow_offset)),
            shadow_color=color_of("shadow_color", cls.shadow_color),
            position=Position.parse(data.get("position")),
            line_gap=int(data.get("line_gap", cls.line_gap)),
            tile_gap=int(data.get("tile_gap", cls.tile_gap)),
            box=bool(data.get("box", cls.box)),
            box_color=color_of("box_color", cls.box_color),
            box_padding=int(data.get("box_padding", cls.box_padding)),
        )

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的纯字典（与 :meth:`from_dict` 往返一致）。"""
        return {
            "font": {
                "families": list(self.font.families),
                "size": self.font.size,
                "weight": self.font.weight.name.lower(),
            },
            "color": self.color.to_hex(),
            "outline_color": self.outline_color.to_hex(),
            "outline_width": self.outline_width,
            "shadow_offset": self.shadow_offset,
            "shadow_color": self.shadow_color.to_hex(),
            "position": {
                "halign": self.position.halign.value,
                "valign": self.position.valign.value,
                "margins": {
                    "left": self.position.margins.left,
                    "top": self.position.margins.top,
                    "right": self.position.margins.right,
                    "bottom": self.position.margins.bottom,
                },
            },
            "line_gap": self.line_gap,
            "tile_gap": self.tile_gap,
            "box": self.box,
            "box_color": self.box_color.to_hex(),
            "box_padding": self.box_padding,
        }
