"""字幕文本的排版、光栅化与逐像素叠加（alpha 混合自行实现）。

渲染分两层概念：

* **瓦片(tile)**：一条字幕文本渲染成的小 RGBA 位图，带描边、阴影与可选
  半透明底盒。瓦片与视频帧尺寸无关，可以按文本内容缓存，同一字幕在其
  生命周期的几十上百帧里只光栅化一次；
* **混合(blend)**：把瓦片按计算好的左上角坐标贴到 RGB24 帧缓冲上，
  :func:`blend_rgba_tile` 用标准源覆盖公式逐像素混合::

      dst = src_rgb * src_a + dst_rgb * (1 - src_a)

  不依赖 OpenCV/PIL 的贴图能力，混合数学完全可见、可测。

排版规则：整体水平居中；多条字幕自下而上堆叠；瓦片之间保留固定垂直间距。
"""

from __future__ import annotations

from dataclasses import dataclass

from .fonts import Font

__all__ = ["Style", "OverlayRenderer", "blend_rgba_tile"]


@dataclass(frozen=True)
class Style:
    """字幕视觉样式。颜色为 8-bit RGB 三元组。"""

    color: tuple[int, int, int] = (255, 255, 255)       # 填充色
    outline_color: tuple[int, int, int] = (0, 0, 0)     # 描边色
    outline_width: int = 2                              # 描边厚度（像素）
    shadow_offset: int = 2                              # 阴影偏移（0 关闭）
    margin: int = 36                                    # 距画面底边/侧边的留白
    line_gap: int = 6                                   # 多行文本的行间距
    tile_gap: int = 12                                  # 多条字幕瓦片的垂直间距
    box: bool = False                                   # 是否加半透明底盒
    box_color: tuple[int, int, int] = (0, 0, 0)
    box_alpha: int = 128                                # 底盒不透明度 0-255
    box_padding: int = 12


def _set_px(buf: bytearray, w: int, x: int, y: int,
            rgb: tuple[int, int, int], alpha: int = 255) -> None:
    """写一个 RGBA 像素（边界由调用方保证）。"""
    off = (y * w + x) * 4
    buf[off:off + 3] = bytes(rgb)
    buf[off + 3] = alpha


def blend_rgba_tile(frame: bytearray, frame_w: int, frame_h: int,
                    tile: bytearray, tile_w: int, tile_h: int,
                    x0: int, y0: int) -> None:
    """把 RGBA 瓦片以源覆盖混合贴到 RGB24 帧缓冲上，自动裁剪越界部分。

    瓦片每像素 4 字节，帧每像素 3 字节。alpha=0 直接跳过（最常见路径），
    alpha=255 走不透明快路径，中间值按整数运算混合（>>8 代替除法）。
    """
    # 与画面矩形求交，落在画面外的行/列直接裁掉
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(tile_w, frame_w - x0), min(tile_h, frame_h - y0)
    for ty in range(sy0, sy1):
        fy = y0 + ty
        for tx in range(sx0, sx1):
            t_off = (ty * tile_w + tx) * 4
            a = tile[t_off + 3]
            if a == 0:
                continue
            f_off = (fy * frame_w + x0 + tx) * 3
            if a == 255:
                frame[f_off:f_off + 3] = tile[t_off:t_off + 3]
            else:
                inv = 255 - a
                for c in range(3):
                    frame[f_off + c] = (
                        tile[t_off + c] * a + frame[f_off + c] * inv
                    ) >> 8


class OverlayRenderer:
    """把字幕 Cue 渲染为瓦片并按帧叠加，带 LRU 风格的瓦片缓存。"""

    def __init__(self, font: Font, style: Style | None = None,
                 cache_size: int = 256):
        self.font = font
        self.style = style or Style()
        self._cache: dict[str, tuple[bytearray, int, int]] = {}
        self._cache_order: list[str] = []  # 简单 FIFO 淘汰，控制内存上限
        self._cache_size = cache_size

    # ------------------------------------------------------------------
    # 瓦片构建
    # ------------------------------------------------------------------
    def _build_tile(self, text: str) -> tuple[bytearray, int, int]:
        """把多行文本光栅化为一张 RGBA 瓦片（含描边/阴影/底盒）。"""
        st = self.style
        lines = text.split("\n")
        # 每行墨迹尺寸（描边与阴影会在四周再外扩）
        line_sizes = [self.font.text_size(line) for line in lines]
        text_w = max((w for w, _ in line_sizes), default=0)
        text_h = (
            sum(h for _, h in line_sizes)
            + st.line_gap * max(0, len(lines) - 1)
        )
        pad = st.box_padding if st.box else 0
        # 瓦片 = 文本区 + 描边四向外扩 + 右下阴影空间 + 底盒内边距
        inset = st.outline_width + pad
        tile_w = text_w + inset * 2 + st.shadow_offset
        tile_h = text_h + inset * 2 + st.shadow_offset
        tile = bytearray(tile_w * tile_h * 4)  # 全透明初始化

        # 1) 半透明底盒：覆盖整张瓦片（含 padding），文本稍后叠在上面
        if st.box:
            for y in range(tile_h - st.shadow_offset):
                for x in range(tile_w - st.shadow_offset):
                    _set_px(tile, tile_w, x, y, st.box_color, st.box_alpha)

        # 逐行绘制，y 为当前行墨迹顶部在瓦片中的坐标
        y = inset
        for line, (lw, lh) in zip(lines, line_sizes):
            x = inset + (text_w - lw) // 2  # 行内水平居中
            self._paint_line(tile, tile_w, line, x, y, lw, lh)
            y += lh + st.line_gap
        return tile, tile_w, tile_h

    def _paint_line(self, tile: bytearray, tile_w: int, line: str,
                    x: int, y: int, lw: int, lh: int) -> None:
        """在瓦片上绘制单行：阴影 -> 描边 -> 填充，三层依次覆盖。"""
        st = self.style
        if not line:
            return
        fill = self.font.rasterize_line(line, st.color)
        # 描边层用描边色重新光栅化同一段文本，按 8 邻域偏移铺开
        outline = (
            self.font.rasterize_line(line, st.outline_color)
            if st.outline_width > 0 else None
        )

        def stamp(rgb_buf: bytearray, ox: int, oy: int, alpha: int) -> None:
            """把一张 RGBA 墨迹缓冲按偏移盖到瓦片上（仅写墨迹像素）。

            源缓冲以 alpha=255 标记墨迹；``alpha`` 参数是本次叠印的不透明度
            （阴影用低值，描边/填充用 255）。
            """
            max_y = len(tile) // (tile_w * 4)
            for ry in range(lh):
                for rx in range(lw):
                    src = (ry * lw + rx) * 4
                    if rgb_buf[src + 3] == 0:  # 透明背景，跳过
                        continue
                    px, py = ox + rx, oy + ry
                    if 0 <= px < tile_w and 0 <= py < max_y:
                        _set_px(
                            tile, tile_w, px, py,
                            (rgb_buf[src], rgb_buf[src + 1], rgb_buf[src + 2]),
                            alpha,
                        )

        # 阴影：整体平移一次，半透明黑
        if st.shadow_offset > 0:
            stamp(fill, x + st.shadow_offset, y + st.shadow_offset, 140)
        # 描边：厚度通过逐层扩大偏移量实现
        if outline is not None:
            for d in range(1, st.outline_width + 1):
                for dx in range(-d, d + 1):
                    for dy in range(-d, d + 1):
                        if dx == 0 and dy == 0:
                            continue
                        stamp(outline, x + dx, y + dy, 255)
        # 填充最后盖在最上层
        stamp(fill, x, y, 255)

    def _get_tile(self, text: str) -> tuple[bytearray, int, int]:
        """带淘汰的瓦片缓存：key 为文本（样式绑定在 renderer 实例上）。"""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        tile = self._build_tile(text)
        self._cache[text] = tile
        self._cache_order.append(text)
        if len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return tile

    # ------------------------------------------------------------------
    # 帧叠加
    # ------------------------------------------------------------------
    def render_frame(self, frame: bytearray, width: int, height: int,
                     texts: list[str]) -> None:
        """把当前帧应显示的字幕文本列表叠加到 RGB24 帧缓冲上。

        列表首项画在最下方，后续项依次向上堆叠（与调度器的返回顺序配合）。
        """
        if not texts:
            return
        st = self.style
        tiles = [self._get_t(t) for t in texts if t.strip()]
        total_h = sum(h for _, _, h in tiles) + st.tile_gap * (len(tiles) - 1)
        # 垂直放置：优先锚定在底边留白之上；字幕组比可用空间还高时（小画面
        # 或字幕过多）回退到贴顶，避免负坐标把整块字幕推到画面外被裁光。
        avail_h = height - 2 * st.margin
        if total_h <= avail_h:
            cursor_y = height - st.margin - total_h
        else:
            cursor_y = max(0, (height - total_h) // 2)
        for tile, tw, th in tiles:
            # 水平居中；瓦片比画面宽时夹紧到 0，由 blend 负责横向裁剪
            x0 = max(0, (width - tw) // 2)
            blend_rgba_tile(frame, width, height, tile, tw, th, x0, cursor_y)
            cursor_y += th + st.tile_gap

    def _get_t(self, text: str) -> tuple[bytearray, int, int]:
        return self._get_tile(text)
