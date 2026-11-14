"""字体光栅化：内置位图字体（零依赖兜底）+ 可选 Pillow TrueType 渲染。

设计为两级：

1. :class:`BitmapFont` —— 纯标准库实现的 5x7 点阵字体，覆盖 ASCII 字母、
   数字与常见标点。任何环境都能工作，是渲染正确性的保底路径，也是测试
   不依赖系统字体的关键。字形用 7 行 ``#``/空格的字符串声明，运行期转成
   位掩码，缩放整数倍输出。
2. :class:`PillowFont` —— 安装了 Pillow 且能找到 TTF 时启用，支持任意
   字号与中文等全字符集；Pillow 仅负责「字形 -> 带 alpha 的位图」，排版、
   描边、混合仍由 :mod:`subburn.render` 自行完成。

两级实现同一个 :class:`Font` 协议，渲染层不感知字体来源。
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

__all__ = ["Font", "BitmapFont", "PillowFont", "load_font"]

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

    def text_size(self, text: str) -> tuple[int, int]:
        """单行文本渲染后的像素宽高。"""

    def rasterize_line(self, text: str, color: tuple[int, int, int]) -> bytearray:
        """把单行文本光栅化为 RGBA 缓冲（长度 = 宽*高*4）。

        墨迹像素 alpha=255，背景像素 alpha=0；以 alpha 而非颜色区分墨迹，
        因此任意颜色（含纯黑）的文字都能正确渲染。
        """


class BitmapFont:
    """5x7 点阵字体，按整数倍缩放，零第三方依赖。"""

    def __init__(self, scale: int = 3):
        if scale < 1:
            raise ValueError("scale 必须 >= 1")
        self.scale = scale
        # 每个字形后留 1 个像素的字间距，行高留 1 像素下行余量
        self.advance = (_GLYPH_W + 1) * scale
        self.line_height = (_GLYPH_H + 1) * scale

    def _glyph(self, ch: str) -> tuple[str, ...]:
        return _GLYPHS.get(ch.upper(), _BOX_GLYPH)

    def text_size(self, text: str) -> tuple[int, int]:
        if not text:
            return 0, self.line_height
        # 末尾不留字间距
        w = len(text) * self.advance - self.scale
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
                    # 墨水像素放大为 s*s 的实心方块
                    for dy in range(s):
                        base = ((ry * s + dy) * width + x0 + gx * s) * 4
                        for dx in range(s):
                            off = base + dx * 4
                            buf[off:off + 3] = bytes(color)
                            buf[off + 3] = 255
        return buf


class PillowFont:
    """基于 Pillow 的 TTF 光栅化；仅在 Pillow 与字体文件都可用时构造。"""

    def __init__(self, font_path: str, size: int):
        from PIL import ImageFont  # 懒加载：无 Pillow 环境不受影响

        self.font_path = font_path
        self.font = ImageFont.truetype(font_path, size)
        # 用 ascender/descender 算稳定行高，避免不同字形度量抖动
        asc, desc = self.font.getmetrics()
        self.line_height = asc + desc
        self._ascent = asc

    def text_size(self, text: str) -> tuple[int, int]:
        if not text:
            return 0, self.line_height
        box = self.font.getbbox(text)
        return box[2] - box[0], box[3] - box[1]

    def rasterize_line(self, text: str, color: tuple[int, int, int]) -> bytearray:
        from PIL import Image, ImageDraw

        width, height = self.text_size(text)
        if width == 0:
            return bytearray()
        # RGBA 透明底：墨迹 alpha=255，背景 alpha=0，渲染层凭 alpha 取形
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # getbbox 的左上可能非 0，绘制时反向偏移，使墨迹对齐 (0,0)
        box = self.font.getbbox(text)
        ImageDraw.Draw(img).text(
            (-box[0], -box[1]), text, font=self.font, fill=color + (255,)
        )
        return bytearray(img.tobytes())


# 各平台常见的可直接尝试的字体路径；命中即用，避免依赖 fontconfig 之类的系统服务
_CANDIDATE_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",      # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",    # Windows 黑体
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux 常见
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def load_font(
    font_path: str | None = None,
    size: int = 28,
    *,
    allow_pillow: bool = True,
) -> Font:
    """按优先级构造字体：显式路径 -> 系统 TTF（需 Pillow）-> 内置位图。

    位图模式下 ``size`` 会被折算为最接近的缩放档（5x7 字形的整数倍）。
    """
    candidates = [font_path] if font_path else list(_CANDIDATE_FONTS)
    if allow_pillow:
        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    return PillowFont(path, size)
                except Exception:
                    # 字体文件损坏/格式不支持等：继续尝试下一个，最后还有位图兜底
                    continue
    # 位图缩放：让字形高度尽量接近请求字号
    scale = max(1, round(size / _GLYPH_H))
    return BitmapFont(scale=scale)
