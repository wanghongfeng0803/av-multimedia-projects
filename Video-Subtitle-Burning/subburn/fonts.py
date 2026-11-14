"""字体光栅化：内置位图字体（零依赖兜底）+ 可选 Pillow TrueType 渲染。

设计为两级：

1. :class:`BitmapFont` —— 纯标准库实现的 5x7 点阵字体，覆盖 ASCII 字母、
   数字与常见标点。任何环境都能工作，是渲染正确性的保底路径，也是测试
   不依赖系统字体的关键。字形用 7 行 ``#``/空格的字符串声明，运行期转成
   位掩码，缩放整数倍输出；
2. :class:`PillowFont` —— 安装了 Pillow 且能找到 TTF 时启用，支持任意
   字号与中文等全字符集；Pillow 仅负责「字形 -> 带 alpha 的位图」，排版、
   描边、混合仍由 :mod:`subburn.render` 自行完成。

两级实现同一个 :class:`Font` 协议，渲染层不感知字体来源。

样式系统（:mod:`subburn.style` 的 :class:`FontSpec`）在其上增加了三层能力：

* **字体族回退链**：:class:`FontRegistry` 把常见系统字体登记为族名（支持
  ``Arial`` / ``微软雅黑`` 等别名），按用户给出的优先级逐个匹配，全部未
  命中时退回系统默认字体，最后还有位图字体保底；
* **字重选择**：字体文件名携带 Bold/Light 等线索时按 :class:`FontWeight`
  就近选档；选不到目标字重时保留常规字形并标记合成粗体；
* **合成粗体**：位图字体通过二次叠印 1 像素偏移模拟 bold，Pillow 字体在
  无 bold 文件时同样以微小横向偏移描厚，保证「请求 bold 一定更粗」。
"""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

from .style import FontSpec, FontWeight

__all__ = [
    "Font", "BitmapFont", "PillowFont", "FontRegistry",
    "load_font", "resolve_font", "get_registry",
]

# 5(宽) x 7(高) 点阵：# 为墨水像素。小写字母在查表时归一化为大写。
_GLYPHS: dict[str, tuple[str, ...]] = {
    " ": ("     ",) * 7,
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".####", "#....", "#....", "#..##", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "##..#", "#.#.#", "#..##", "#..##", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": (".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."),
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    ",": (".....", ".....", ".....", ".....", ".##..", ".##..", "#...."),
    "!": ("..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    ":": (".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."),
    ";": (".....", ".##..", ".##..", ".....", ".##..", ".##..", "#...."),
    "'": ("..#..", "..#..", ".#...", ".....", ".....", ".....", "....."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "(": ("...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."),
    ")": (".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."),
    "/": ("....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "%": ("##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"),
    "&": (".##..", "#..#.", "#..#.", ".##..", "#.#.#", "#..#.", ".##.#"),
}
# 未收录字符统一渲染为空心方框
_BOX_GLYPH = (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.")

_GLYPH_W, _GLYPH_H = 5, 7


@runtime_checkable
class Font(Protocol):
    """渲染层依赖的最小字体接口。"""

    #: 行高（像素），多行排版时每行的步进
    line_height: int
    #: 是否为合成粗体（无原生 bold 字形时由光栅化层描厚）
    bold: bool

    def text_size(self, text: str) -> tuple[int, int]:
        """单行文本渲染后的像素宽高。"""

    def rasterize_line(self, text: str, color: tuple[int, int, int]) -> bytearray:
        """把单行文本光栅化为 RGBA 缓冲（长度 = 宽*高*4）。

        墨迹像素 alpha=255，背景像素 alpha=0；以 alpha 而非颜色区分墨迹，
        因此任意颜色（含纯黑）的文字都能正确渲染。
        """


class BitmapFont:
    """5x7 点阵字体，按整数倍缩放，零第三方依赖。

    ``bold=True`` 时每个墨水像素向右多印一列，用二次叠印模拟合成粗体；
    宽度相应增加 1 个缩放单位，避免新墨迹被裁掉。
    """

    def __init__(self, scale: int = 3, bold: bool = False):
        if scale < 1:
            raise ValueError("scale 必须 >= 1")
        self.scale = scale
        self.bold = bool(bold)
        # 每个字形后留 1 个像素的字间距，行高留 1 像素下行余量
        self.advance = (_GLYPH_W + 1) * scale
        self.line_height = (_GLYPH_H + 1) * scale
        # 合成粗体额外占用的宽度（像素）
        self._bold_extra = scale if bold else 0

    def _glyph(self, ch: str) -> tuple[str, ...]:
        return _GLYPHS.get(ch.upper(), _BOX_GLYPH)

    def text_size(self, text: str) -> tuple[int, int]:
        if not text:
            return 0, self.line_height
        # 末尾不留字间距；粗体外扩计入整体宽度
        w = len(text) * self.advance - self.scale + self._bold_extra
        return w, _GLYPH_H * self.scale

    def rasterize_line(self, text: str, color: tuple[int, int, int]) -> bytearray:
        width, height = self.text_size(text)
        buf = bytearray(width * height * 4)  # RGBA，背景保持全透明
        s = self.scale
        for ci, ch in enumerate(text):
            x0 = ci * self.advance
            for ry, row in enumerate(self._glyph(ch)):
                for gx, px in enumerate(row):
                    if px != "#":
                        continue
                    # 墨水像素放大为 s*s 的实心方块；粗体再右移一格叠印
                    offsets = (0, s) if self.bold else (0,)
                    for b_off in offsets:
                        for dy in range(s):
                            base = (
                                (ry * s + dy) * width + x0 + gx * s + b_off
                            ) * 4
                            for dx in range(s):
                                off = base + dx * 4
                                buf[off:off + 3] = bytes(color)
                                buf[off + 3] = 255
        return buf


class PillowFont:
    """基于 Pillow 的 TTF 光栅化；仅在 Pillow 与字体文件都可用时构造。

    ``synthetic_bold=True`` 表示文件本身不含 bold 字重，绘制时以 1 像素
    横向偏移重复盖印模拟加粗（:class:`FontRegistry` 选档失败时使用）。
    """

    def __init__(self, font_path: str, size: int, synthetic_bold: bool = False):
        from PIL import ImageFont  # 懒加载：无 Pillow 环境不受影响

        self.font_path = font_path
        self.font = ImageFont.truetype(font_path, size)
        self.bold = bool(synthetic_bold)
        # 用 ascender/descender 算稳定行高，避免不同字形度量抖动
        asc, desc = self.font.getmetrics()
        self.line_height = asc + desc
        self._ascent = asc

    def text_size(self, text: str) -> tuple[int, int]:
        if not text:
            return 0, self.line_height
        box = self.font.getbbox(text)
        width = box[2] - box[0] + (1 if self.bold else 0)
        return width, box[3] - box[1]

    def rasterize_line(self, text: str, color: tuple[int, int, int]) -> bytearray:
        from PIL import Image, ImageDraw

        width, height = self.text_size(text)
        if width == 0:
            return bytearray()
        # RGBA 透明底：墨迹 alpha=255，背景 alpha=0，渲染层凭 alpha 取形
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # getbbox 的左上可能非 0，绘制时反向偏移，使墨迹对齐 (0,0)
        box = self.font.getbbox(text)
        draw = ImageDraw.Draw(img)
        draw.text((-box[0], -box[1]), text, font=self.font,
                  fill=color + (255,))
        if self.bold:  # 合成粗体：右移 1px 再盖一次，笔画横向增厚
            draw.text((-box[0] + 1, -box[1]), text, font=self.font,
                      fill=color + (255,))
        return bytearray(img.tobytes())


# ---------------------------------------------------------------------------
# 字体族注册表与回退链
# ---------------------------------------------------------------------------

# 各平台常见的可直接尝试的字体路径；命中即用，避免依赖 fontconfig 之类的系统服务
_CANDIDATE_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",      # Windows 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑 Bold
    "C:/Windows/Fonts/simhei.ttf",    # Windows 黑体
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
    "/System/Library/Fonts/PingFang.ttc",        # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
)

#: 族名别名：用户可能写中文/显示名/短名，统一映射到注册表内部族键
_FAMILY_ALIASES = {
    "微软雅黑": "microsoft yahei", "雅黑": "microsoft yahei",
    "msyh": "microsoft yahei", "microsoftyahei": "microsoft yahei",
    "黑体": "simhei", "heiti": "simhei",
    "平苹方": "pingfang", "苹方": "pingfang", "pingfangsc": "pingfang",
    "helveticaneue": "helvetica",
    "noto": "noto sans cjk", "notosans": "noto sans cjk",
    "notosanscjk": "noto sans cjk", "思源黑体": "noto sans cjk",
    "dejavu": "dejavu sans",
    "sans": "sans-serif", "sansserif": "sans-serif",
}

#: 文件名粗体线索：token/后缀 -> 权重档（连体缩写如 bd/bi 一并覆盖）
_WEIGHT_TOKEN_MAP = {
    "thin": FontWeight.THIN,
    "extralight": FontWeight.EXTRALIGHT, "ultralight": FontWeight.EXTRALIGHT,
    "light": FontWeight.LIGHT, "regular": FontWeight.REGULAR,
    "normal": FontWeight.REGULAR, "medium": FontWeight.MEDIUM,
    "semibold": FontWeight.SEMIBOLD, "demibold": FontWeight.SEMIBOLD,
    "extrabold": FontWeight.EXTRABOLD, "ultrabold": FontWeight.EXTRABOLD,
    "bold": FontWeight.BOLD, "bd": FontWeight.BOLD, "bi": FontWeight.BOLD,
    "black": FontWeight.BLACK, "heavy": FontWeight.BLACK,
}
#: 斜体线索（斜体不影响字重，识别后直接剥离，避免污染族名）
_ITALIC_TOKENS = {"italic", "ital", "oblique", "it"}

#: 已知连体缩写词表（Windows 命名习惯，整体匹配以免误伤族名末字母，
#: 例如 simkai 不能因为末尾的 i 被当成斜体剥掉）
_STEM_ALIASES = {
    "arial": ("arial", FontWeight.REGULAR),
    "arialbd": ("arial", FontWeight.BOLD),
    "ariali": ("arial", FontWeight.REGULAR),
    "arialbi": ("arial", FontWeight.BOLD),
    "msyh": ("microsoft yahei", FontWeight.REGULAR),
    "msyhbd": ("microsoft yahei", FontWeight.BOLD),
    "msyhl": ("microsoft yahei", FontWeight.LIGHT),
    "simhei": ("simhei", FontWeight.REGULAR),
}

#: 已知连体文件名（无分隔符）的后缀，按长度降序匹配
_CONCAT_SUFFIXES = sorted(
    set(_WEIGHT_TOKEN_MAP) | _ITALIC_TOKENS, key=len, reverse=True
)


def parse_font_filename(path: str) -> tuple[str, FontWeight]:
    """从字体文件名解析 ``(内部族键, 字重)``。

    同时支持分隔符写法（``DejaVuSans-Bold``）与连体缩写（``arialbd``、
    ``msyhbd``、``DejaVuSans-BoldOblique``）；斜体标记只剥离不参与字重。
    连体后缀要求长度 >=2，避免误伤 ``simkai`` 这类以单个字母结尾的族名。
    """
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem in _STEM_ALIASES:
        return _STEM_ALIASES[stem]

    weight = FontWeight.REGULAR
    family_parts: list[str] = []
    for token in re.split(r"[-_ .]", stem):
        if not token:
            continue
        if token in _WEIGHT_TOKEN_MAP:
            weight = _WEIGHT_TOKEN_MAP[token]
            continue
        if token in _ITALIC_TOKENS:
            continue
        # 连体情况：循环剥离后缀，处理 bold+oblique 这类组合
        stripped = token
        while stripped:
            if stripped in _WEIGHT_TOKEN_MAP:
                weight = _WEIGHT_TOKEN_MAP[stripped]
                stripped = ""
                break
            if stripped in _ITALIC_TOKENS:
                stripped = ""
                break
            matched = False
            for suf in _CONCAT_SUFFIXES:
                if len(suf) >= 2 and len(stripped) > len(suf) \
                        and stripped.endswith(suf):
                    stripped = stripped[: -len(suf)]
                    if suf in _WEIGHT_TOKEN_MAP:
                        weight = _WEIGHT_TOKEN_MAP[suf]
                    matched = True
                    break
            if not matched:
                break
        family_parts.append(stripped)
    family = "".join(family_parts)
    fixed = {
        "msyh": "microsoft yahei", "simhei": "simhei",
        "arial": "arial", "pingfang": "pingfang",
        "helvetica": "helvetica", "dejavusans": "dejavu sans",
        "notosanscjk": "noto sans cjk",
    }
    return fixed.get(family, family), weight


# 向后兼容的两个薄封装（旧内部调用点）
def _family_from_filename(path: str) -> str:
    return parse_font_filename(path)[0]


def _weight_from_filename(path: str) -> FontWeight:
    return parse_font_filename(path)[1]


def _normalize_family(name: str) -> str:
    """把用户给出的族名归一为内部族键（小写、去空格/连字符、查别名表）。"""
    key = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _FAMILY_ALIASES.get(key, key)


class FontRegistry:
    """扫描已知字体路径建立 ``族 -> 字重 -> 文件`` 索引，带实例缓存。

    扫描是惰性且一次性的（首次查询触发），之后只做字典查找；
    :class:`PillowFont` 实例按 ``(路径, 字号, 合成粗体)`` 缓存，避免同一
    字体在多条字幕上重复触发 FreeType 加载。
    """

    def __init__(self, candidate_paths: tuple[str, ...] = _CANDIDATE_FONTS):
        self._candidates = candidate_paths
        self._by_family: dict[str, dict[FontWeight, str]] = {}
        self._scanned = False
        self._instance_cache: dict[tuple[str, int, bool], Font] = {}

    # -- 扫描 ------------------------------------------------------------
    def _scan(self) -> None:
        if self._scanned:
            return
        for path in self._candidates:
            if not os.path.isfile(path):
                continue
            family = _family_from_filename(path)
            weight = _weight_from_filename(path)
            # 同档重复时保留先登记的（候选表本身按平台优先级排列）
            self._by_family.setdefault(family, {}).setdefault(weight, path)
        self._scanned = True

    def available_families(self) -> list[str]:
        self._scan()
        return sorted(self._by_family)

    # -- 族解析 ----------------------------------------------------------
    def _resolve_family(self, family: str, weight: FontWeight,
                        size: int) -> Font | None:
        self._scan()
        if os.path.isfile(family):
            # 直接给了字体文件路径：绕过族表，按文件名猜字重，缺失的字重
            # 交给合成粗体兜底
            native = _weight_from_filename(family)
            return self._make_font(
                family, size,
                synthetic=weight is not native and weight.value >= FontWeight.BOLD.value,
            )
        key = _normalize_family(family)
        table = self._by_family.get(key)
        if not table:
            return None
        path = table.get(weight)
        synthetic = False
        if path is None:
            # 就近选档：优先同方向差距最小的可用权重（注意 min 的结果是
            # 权重键，真正要打开的是 table 里登记的路径）
            chosen = min(table, key=lambda w: abs(w.value - weight.value))
            path = table[chosen]
            # 想要 bold 及以上但只找到更细的字体：合成加粗
            synthetic = (
                weight.value >= FontWeight.BOLD.value
                and chosen.value < FontWeight.BOLD.value
            )
        return self._make_font(path, size, synthetic)

    def _make_font(self, path: str, size: int, synthetic: bool) -> Font | None:
        cache_key = (path, size, synthetic)
        cached = self._instance_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            font: Font = PillowFont(path, size, synthetic_bold=synthetic)
        except Exception:
            # 文件损坏/格式不支持：从本次候选中剔除，交由上层回退
            return None
        self._instance_cache[cache_key] = font
        return font

    def resolve(self, spec: FontSpec, *, fallback_size: int) -> Font:
        """按 :class:`FontSpec` 解析字体，执行完整回退链。

        顺序：族链逐个匹配 -> 系统默认候选 -> 内置位图字体。位图字体仅
        支持常规与合成粗体两档，且字号折算为最接近的整数缩放档。
        """
        size = spec.size if spec.size > 0 else fallback_size
        for family in spec.families:
            font = self._resolve_family(family, spec.weight, size)
            if font is not None:
                return font
        # 族链全部未命中：退回候选表中第一个真正可加载的字体
        self._scan()
        for table in self._by_family.values():
            for weight, path in sorted(table.items(), key=lambda kv: kv[0].value):
                synthetic = (
                    spec.weight.value >= FontWeight.BOLD.value
                    and weight.value < FontWeight.BOLD.value
                )
                font = self._make_font(path, size, synthetic)
                if font is not None:
                    return font
        # 最终保底：零依赖位图字体（同样进实例缓存，重复解析零成本）
        scale = max(1, round(size / _GLYPH_H))
        bold = spec.weight.value >= FontWeight.BOLD.value
        cache_key = ("__bitmap__", scale, bold)
        cached = self._instance_cache.get(cache_key)
        if cached is not None:
            return cached
        font = BitmapFont(scale=scale, bold=bold)
        self._instance_cache[cache_key] = font
        return font


#: 进程级单例（惰性扫描），测试可构造独立注册表替换
_registry: FontRegistry | None = None


def get_registry() -> FontRegistry:
    global _registry
    if _registry is None:
        _registry = FontRegistry()
    return _registry


def resolve_font(spec: FontSpec | None = None, *,
                 frame_height: int = 0, registry: FontRegistry | None = None) -> Font:
    """样式系统的字体入口：:class:`FontSpec` + 视频高度 -> 可渲染 :class:`Font`。

    ``frame_height > 0`` 且规格里字号为自动(0)时，沿用旧规则
    ``max(20, 高度//24)`` 决定像素字号。
    """
    spec = spec or FontSpec()
    fallback_size = max(20, frame_height // 24) if frame_height else 28
    reg = registry or get_registry()
    return reg.resolve(spec, fallback_size=fallback_size)


def load_font(
    font_path: str | None = None,
    size: int = 28,
    *,
    allow_pillow: bool = True,
    weight: FontWeight | int | str | None = None,
) -> Font:
    """按优先级构造字体：显式路径 -> 系统 TTF（需 Pillow）-> 内置位图。

    保留旧签名（``font_path/size/allow_pillow``），新增可选 ``weight``：
    位图模式下 ``size`` 会被折算为最接近的缩放档（5x7 字形的整数倍）。
    """
    fw = FontWeight.REGULAR if weight is None else FontWeight.parse(weight)
    if allow_pillow:
        spec = FontSpec(
            families=(font_path,) if font_path else (),
            size=size, weight=fw,
        )
        # 显式路径用一次性注册表（不污染进程单例），否则走全局系统字体表
        registry = FontRegistry() if font_path else get_registry()
        return registry.resolve(spec, fallback_size=size)
    scale = max(1, round(size / _GLYPH_H))
    return BitmapFont(scale=scale, bold=fw.value >= FontWeight.BOLD.value)
