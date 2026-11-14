"""字体发现与按码点回退链。

回退优先级（每码点独立决策，混排文本可由多个字体拼接）：
1. 显式字体文件（按给定顺序）；
2. 系统字体目录的 TrueType（CJK 字体优先排序）；
3. 内置 5x7 点阵（仅可打印 ASCII，零依赖最终兜底）；
4. 仍无法覆盖的码点画豆腐块占位，不静默丢字。

字体懒加载，解析失败记入缓存不反复尝试。
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass

from ..errors import WatermarkError
from . import bitmap_font
from .ttf_font import TrueTypeFont, is_truetype_file, load_font

_IS_WINDOWS = sys.platform.startswith("win")
_IS_MACOS = sys.platform == "darwin"
_FONT_SUFFIXES = (".ttf", ".ttc", ".otf")

_PREFERRED_FONT_NAMES = (
    "msyh", "simhei", "simsun", "simkai", "simfang",
    "pingfang", "stheiti", "hiragino",
    "notosanscjk", "notoserifcjk", "sourcehansans", "sourcehanserif",
    "wqy", "wenquanyi", "droidsansfallback",
    "segoeui", "arialuni",
    "dejavusans", "freesans", "liberationsans",
    "arial", "calibri",
)

_WHITESPACE = {
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20,
    0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
}


def _font_search_dirs():
    dirs = []
    if _IS_WINDOWS:
        root = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
        dirs.append(os.path.join(root, "Fonts"))
        appdata = os.environ.get("LOCALAPPDATA")
        if appdata:
            dirs.append(os.path.join(appdata, "Microsoft", "Windows", "Fonts"))
    elif _IS_MACOS:
        dirs.extend(["/System/Library/Fonts", "/Library/Fonts",
                     os.path.expanduser("~/Library/Fonts")])
    else:
        dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts",
                     os.path.expanduser("~/.fonts"),
                     os.path.expanduser("~/.local/share/fonts")])
    extra = os.environ.get("HLS_PACKAGER_FONT_DIR")
    if extra:
        dirs.extend(p for p in extra.split(os.pathsep) if p)

    seen, result = set(), []
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen and os.path.isdir(d):
            seen.add(key)
            result.append(d)
    return result


def _preference_rank(path):
    name = os.path.basename(path).lower()
    for i, keyword in enumerate(_PREFERRED_FONT_NAMES):
        if keyword in name:
            return i
    return len(_PREFERRED_FONT_NAMES)


def discover_system_fonts() -> list[str]:
    """扫描字体目录，返回按偏好排序的字体路径。"""
    found, seen = [], set()
    for directory in _font_search_dirs():
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if not entry.lower().endswith(_FONT_SUFFIXES):
                continue
            path = os.path.join(directory, entry)
            if not os.path.isfile(path):
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    found.sort(key=lambda p: (_preference_rank(p), os.path.basename(p).lower()))
    return found


@dataclass(frozen=True)
class ResolvedFace:
    """单个码点的渲染决策：ttf / bitmap / tofu。"""

    codepoint: int
    kind: str
    font: TrueTypeFont | None = None
    glyph_id: int = 0


class FontStack:
    """按码点在显式字体 -> 系统字体 -> 内置点阵之间回退。"""

    def __init__(self, explicit_fonts=None, *, system_fonts=None):
        self._paths = list(explicit_fonts or [])
        for path in (system_fonts if system_fonts is not None else discover_system_fonts()):
            if path not in self._paths:
                self._paths.append(path)
        self._loaded: dict[str, TrueTypeFont | None] = {}

    def _load(self, path):
        if path in self._loaded:
            return self._loaded[path]
        try:
            font = load_font(path) if is_truetype_file(path) else None
        except (WatermarkError, OSError, ValueError, struct.error):
            font = None
        self._loaded[path] = font
        return font

    def _iter_fonts(self):
        for path in self._paths:
            font = self._load(path)
            if font is not None:
                yield font

    def is_whitespace(self, codepoint):
        return codepoint in _WHITESPACE

    def resolve(self, codepoint: int) -> ResolvedFace:
        if self.is_whitespace(codepoint):
            for font in self._iter_fonts():
                return ResolvedFace(codepoint, "ttf", font=font,
                                    glyph_id=font.glyph_id(codepoint))
            return ResolvedFace(codepoint, "bitmap")

        for font in self._iter_fonts():
            glyph_id = font.glyph_id(codepoint)
            if glyph_id != 0:
                return ResolvedFace(codepoint, "ttf", font=font, glyph_id=glyph_id)

        if bitmap_font.supports(codepoint):
            return ResolvedFace(codepoint, "bitmap")
        return ResolvedFace(codepoint, "tofu")

    def fallback_used(self, codepoint):
        face = self.resolve(codepoint)
        if face.kind == "ttf":
            return f"ttf:{os.path.basename(face.font.source_path)}"
        return face.kind
