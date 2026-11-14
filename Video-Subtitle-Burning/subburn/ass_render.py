"""ASS 对话的排版与逐像素渲染：多样式 run、定位特效与瓦片缓存。

本模块与 :mod:`subburn.render` 平级，复用其 :func:`blend_rgba_tile`
源覆盖混合，但排版模型完全按 ASS 语义重建，原因是 SRT 路径的
:class:`~subburn.render.OverlayRenderer` 假设「整条瓦片共享一种样式」，
而 ASS 的一行文本可在中途切换粗斜体、颜色、字号（:class:`~subburn.ass.ASSRun`），
还带 numpad 对齐、``\\pos``/``\\move`` 定位、卡拉OK 与淡入淡出。

设计与既有渲染层保持一致的两条主线：

* **瓦片化 + 缓存**：一条对话在某一帧的可见内容光栅化成一张 RGBA 瓦片，
  瓦片与视频帧尺寸无关；ASS 瓦片的键是 ``(cue 身份, 帧序号)``——同一条
  对话在其几十上百个生命周期帧里，无逐帧特效时完全复用同一瓦片，有
  卡拉OK/淡入淡出时也只在样式实际变化的帧重新光栅化；
* **混合数学不重写**：瓦片到帧的叠加仍走 :func:`blend_rgba_tile`，
  裁剪、alpha=0/255 快路径、整数混合公式全部继承，避免两套混合实现漂移。

ASS 坐标基于脚本的 PlayResX/PlayResY，与真实画面可能不一致，因此所有
几何量先经统一缩放系数换算；样式字号则映射成 :class:`~subburn.style.FontSpec`
交给 :mod:`subburn.fonts` 的回退链解析（位图字体仍是零依赖保底）。

明确的能力边界（README 同步声明）：不做智能自动换行（``\\q`` 忽略，
``\\N`` 硬换行生效）、不实现矢量绘图（``\\p`` 内文本丢弃）、旋转/模糊/
``\\t`` 动画等未覆盖标签安静跳过，保证文件可降级渲染而非报错。
"""

from __future__ import annotations

from .ass import ASSCue, ASSColor, ASSDocument, ASSRun, ASSStyle
from .fonts import Font, FontRegistry, get_registry
from .render import blend_rgba_tile
from .style import FontSpec, FontWeight

__all__ = ["ASSRenderer", "numpad_align"]


def _set_px(buf: bytearray, w: int, x: int, y: int,
            rgb: tuple[int, int, int], alpha: int = 255) -> None:
    """写一个 RGBA 像素（边界由调用方保证），与 render 层内存布局一致。"""
    off = (y * w + x) * 4
    buf[off:off + 3] = bytes(rgb)
    buf[off + 3] = alpha


# ---------------------------------------------------------------------------
# numpad 对齐（1..9 的小键盘布局）
# ---------------------------------------------------------------------------

def numpad_align(alignment: int) -> tuple[str, str]:
    """把 ASS numpad 对齐号翻译成 ``(水平, 垂直)`` 三档方位。

    小键盘::

        7 8 9      左上  上中  右上
        4 5 6  ->  左中  居中  右中
        1 2 3      左下  下中  右下
    """
    a = alignment if 1 <= alignment <= 9 else 2
    halign = "left" if a in (1, 4, 7) else "right" if a in (3, 6, 9) else "center"
    valign = "top" if a in (7, 8, 9) else "middle" if a in (4, 5, 6) else "bottom"
    return halign, valign


# ---------------------------------------------------------------------------
# RGBA 缓冲最近邻缩放（fscx/fscy）
# ---------------------------------------------------------------------------

def _scale_rgba(buf: bytearray, src_w: int, src_h: int,
                dst_w: int, dst_h: int) -> bytearray:
    """把 RGBA 缓冲按最近邻采样缩放到目标尺寸（零依赖、整数索引）。

    输出像素直接取源图对应位置，不做插值：ASS 的 ``\\fscx``/``\\fscy``
    通常是整百分比的轻度缩放，最近邻速度快且不会产生半透明灰边污染
    alpha 通道。
    """
    if dst_w <= 0 or dst_h <= 0:
        return bytearray()
    out = bytearray(dst_w * dst_h * 4)
    for dy in range(dst_h):
        sy = min(src_h - 1, dy * src_h // dst_h)
        src_row = sy * src_w * 4
        dst_row = dy * dst_w * 4
        for dx in range(dst_w):
            sx = min(src_w - 1, dx * src_w // dst_w)
            si = src_row + sx * 4
            di = dst_row + dx * 4
            out[di:di + 4] = buf[si:si + 4]
    return out


class ASSRenderer:
    """把 :class:`~subburn.ass.ASSCue` 逐帧渲染到 RGB24 帧缓冲。

    :param document: 解析后的 ASS 文档（提供样式表与 PlayRes）；
    :param frame_width/frame_height: 真实视频帧尺寸，PlayRes 到帧的缩放、
        字号换算都在构造时确定，逐帧渲染只做整数像素运算；
    :param registry: 字体注册表，默认进程单例，测试可注入空表强制位图路径。
    """

    def __init__(self, document: ASSDocument, frame_width: int,
                 frame_height: int, registry: FontRegistry | None = None,
                 cache_size: int = 128):
        self.doc = document
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.registry = registry or get_registry()
        # PlayRes -> 帧的统一缩放；PlayRes 缺失（0）时按 1:1 处理
        self.scale = (
            frame_height / document.play_res_y
            if document.play_res_y > 0 else 1.0
        )
        self._font_cache: dict[tuple, Font] = {}
        self._tile_cache: dict[tuple, tuple[bytearray, int, int]] = {}
        self._cache_order: list[tuple] = []
        self._cache_size = cache_size

    # ------------------------------------------------------------------
    # 字体解析（按 run 的字体名/字号/粗斜体，带实例缓存）
    # ------------------------------------------------------------------
    def _font_for(self, run: ASSRun, style: ASSStyle) -> Font:
        size_px = max(1, round(run.font_size * self.scale))
        key = (run.font_name, size_px, run.bold, run.italic)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        spec = FontSpec(
            families=(run.font_name,),
            size=size_px,
            weight=FontWeight.BOLD if run.bold else FontWeight.REGULAR,
            italic=run.italic,
        )
        font = self.registry.resolve(spec, fallback_size=size_px)
        self._font_cache[key] = font
        return font

    # ------------------------------------------------------------------
    # 卡拉OK：当前帧该 run 用主色还是次色
    # ------------------------------------------------------------------
    def _run_color(self, run: ASSRun, style: ASSStyle,
                   local_ms: int) -> ASSColor:
        """返回 run 在帧时刻的填充色。

        ``\\k`` 语义：音节到点前用次色（SecondaryColour），到点后切主色；
        ``\\kf``/``\\ko`` 的扫光/清场动画本实现按同一到点切换近似（README
        已声明）。非卡拉OK run 恒为行内主色（可能已被 ``\\c`` 覆盖）。
        """
        if not run.k_dur_ms:
            return run.primary
        filled = local_ms >= run.k_start_ms
        if filled:
            return run.primary
        # 未到点的卡拉OK 音节：仅当 run 未被 \c 单独改色时才回退样式次色，
        # 判定依据是 run 主色与样式主色相同
        return style.secondary_color if run.primary == style.primary_color \
            else run.primary

    # ------------------------------------------------------------------
    # 单行排版：返回每个 run 的墨迹缓冲与放置坐标
    # ------------------------------------------------------------------
    def _layout_line(self, line, style: ASSStyle, local_ms: int
                     ) -> tuple[list[dict], int, int]:
        """把一行的 runs 排版为 ``(图元列表, 行宽, 行高)``。

        每个图元是含 RGBA 缓冲与左上角坐标的字典；``\\fsp`` 字间距非零时
        退化为逐字光栅化（间距只在这类 run 付出额外成本，常规 run 仍整段
        一次成型）；``\\fscx``/``\\fscy`` 在墨迹缓冲上做最近邻缩放。
        """
        pieces: list[dict] = []
        cursor_x = 0
        line_w = line_h = 0
        for run in line.runs:
            font = self._font_for(run, style)
            color = self._run_color(run, style, local_ms)
            sx = max(1, run.scale_x) / 100.0
            sy = max(1, run.scale_y) / 100.0
            gap = round(run.spacing * self.scale)

            def place(text: str, extra_gap: int) -> None:
                nonlocal cursor_x, line_w, line_h
                raw_w, raw_h = font.text_size(text)
                if raw_w == 0:
                    cursor_x += extra_gap
                    return
                buf = font.rasterize_line(text, color.rgb)
                dst_w, dst_h = max(1, round(raw_w * sx)), max(1, round(raw_h * sy))
                if (dst_w, dst_h) != (raw_w, raw_h):
                    buf = _scale_rgba(buf, raw_w, raw_h, dst_w, dst_h)
                pieces.append({
                    "buf": buf, "w": dst_w, "h": dst_h,
                    "x": cursor_x,
                    "underline": run.underline, "strikeout": run.strikeout,
                })
                cursor_x += dst_w + extra_gap
                line_w = max(line_w, cursor_x - extra_gap)
                line_h = max(line_h, dst_h)

            if gap > 0:
                for ch in run.text:  # 逐字铺，字间插间距
                    place(ch, gap)
            else:
                place(run.text, 0)
        return pieces, line_w, line_h

    # ------------------------------------------------------------------
    # 瓦片构建
    # ------------------------------------------------------------------
    def _build_tile(self, cue: ASSCue, frame_ms: int) -> tuple[bytearray, int, int]:
        """把一条对话在指定帧时刻光栅化为 RGBA 瓦片。"""
        st = self.doc.style_of(cue)
        local_ms = max(0, frame_ms - cue.start_ms)
        outline_w = max(0, round(st.outline_w * self.scale))
        shadow_w = max(0, round(st.shadow_w * self.scale))
        line_gap = round(st.font_size * self.scale * 0.3)

        # 1) 逐行排版，汇总几何
        laid: list[tuple[list[dict], int, int]] = [
            self._layout_line(line, st, local_ms) for line in cue.lines
        ]
        text_w = max((w for _, w, _ in laid), default=0)
        text_h = sum(h for _, _, h in laid) + line_gap * max(0, len(laid) - 1)

        # BorderStyle=3（不透明底盒）需要四边外扩 outline_w；普通描边同样
        # 外扩；阴影只占右下。
        pad = outline_w if st.border_style == 3 else 0
        inset = outline_w + pad
        tile_w = max(1, text_w + inset * 2 + shadow_w)
        tile_h = max(1, text_h + inset * 2 + shadow_w)
        tile = bytearray(tile_w * tile_h * 4)

        # 2) BorderStyle=3：不透明底盒先铺整张瓦片（ASS 规范为不透明矩形）
        if st.border_style == 3:
            bg = st.back_color
            for y in range(tile_h - shadow_w):
                for x in range(tile_w - shadow_w):
                    _set_px(tile, tile_w, x, y, bg.rgb, bg.a)

        # 3) 逐行绘制（阴影 -> 描边 -> 墨迹 -> 下划线/删除线）。
        #    每行按 numpad 水平方位在文本区内各自对齐：居中时短行居中、
        #    右对齐时贴右，而不是所有行统一左缘对齐。
        halign, _ = numpad_align(cue.alignment or st.alignment)
        y = inset
        for pieces, lw, lh in laid:
            if halign == "center":
                line_offset = (text_w - lw) // 2
            elif halign == "right":
                line_offset = text_w - lw
            else:
                line_offset = 0
            for pc in pieces:
                self._stamp_piece(tile, tile_w, tile_h, pc,
                                  inset + line_offset + pc["x"], y,
                                  st, outline_w, shadow_w)
            y += lh + line_gap

        # 4) 淡入淡出：全局 alpha 缩放（在合成好的瓦片上一次性处理，避免
        #    每层各自淡入导致的叠加偏差）
        fade_alpha = self._fade_factor(cue, local_ms)
        if fade_alpha < 255:
            for i in range(3, len(tile), 4):
                tile[i] = (tile[i] * fade_alpha) >> 8
        return tile, tile_w, tile_h

    def _fade_factor(self, cue: ASSCue, local_ms: int) -> int:
        """根据 ``\\fad``/``\\fade`` 计算当前帧的全局不透明度（0~255）。"""
        dur = cue.end_ms - cue.start_ms
        fin, fout = cue.fade_in_ms, cue.fade_out_ms
        if fin <= 0 and fout <= 0:
            return 255
        if fin and local_ms < fin:
            return max(0, min(255, round(255 * local_ms / fin)))
        if fout and local_ms > dur - fout:
            remain = max(0, dur - local_ms)
            return max(0, min(255, round(255 * remain / fout)))
        return 255

    def _stamp_piece(self, tile: bytearray, tile_w: int, tile_h: int,
                     pc: dict, x: int, y: int, st: ASSStyle,
                     outline_w: int, shadow_w: int) -> None:
        """把单个 run 图元盖到瓦片上，含阴影、描边、装饰线。"""
        buf, w, h = pc["buf"], pc["w"], pc["h"]
        if not buf:
            return

        def stamp(ox: int, oy: int, rgb: tuple[int, int, int],
                  alpha: int) -> None:
            for ry in range(h):
                for rx in range(w):
                    si = (ry * w + rx) * 4
                    if buf[si + 3] == 0:
                        continue
                    px, py = ox + rx, oy + ry
                    if 0 <= px < tile_w and 0 <= py < tile_h:
                        _set_px(tile, tile_w, px, py, rgb, alpha)

        # 阴影：用图元自身墨迹取形，颜色取 BackColour，整体右下平移
        if shadow_w > 0 and st.back_color.a > 0:
            stamp(x + shadow_w, y + shadow_w, st.back_color.rgb,
                  st.back_color.a)
        # 描边：8 邻域按厚度逐层铺开，颜色取 OutlineColour
        if outline_w > 0 and st.border_style != 3:
            for d in range(1, outline_w + 1):
                for dx in range(-d, d + 1):
                    for dy in range(-d, d + 1):
                        if dx == 0 and dy == 0:
                            continue
                        stamp(x + dx, y + dy, st.outline_color.rgb,
                              st.outline_color.a)
        # 主墨迹最后盖在最上层（保留图元自带 alpha，含半透明文字）
        for ry in range(h):
            for rx in range(w):
                si = (ry * w + rx) * 4
                if buf[si + 3] == 0:
                    continue
                px, py = x + rx, y + ry
                if 0 <= px < tile_w and 0 <= py < tile_h:
                    _set_px(tile, tile_w, px, py,
                            (buf[si], buf[si + 1], buf[si + 2]), buf[si + 3])
        # 下划线/删除线：厚度按缩放后字号取 1~2px，宽度与图元墨迹等宽
        thickness = max(1, round(st.font_size * self.scale / 14))
        line_rgb = self._ink_rgb(buf, w, h)
        if pc["underline"]:
            ly = y + h - 1
            for yy in range(ly, min(tile_h, ly + thickness)):
                for xx in range(x, min(tile_w, x + w)):
                    _set_px(tile, tile_w, xx, yy, line_rgb, 255)
        if pc["strikeout"]:
            ly = y + h // 2
            for yy in range(ly, min(tile_h, ly + thickness)):
                for xx in range(x, min(tile_w, x + w)):
                    _set_px(tile, tile_w, xx, yy, line_rgb, 255)

    @staticmethod
    def _ink_rgb(buf: bytearray, w: int, h: int) -> tuple[int, int, int]:
        """取图元首个墨迹像素的颜色作为装饰线颜色（同段 run 单色）。"""
        for i in range(0, len(buf), 4):
            if buf[i + 3] != 0:
                return buf[i], buf[i + 1], buf[i + 2]
        return (255, 255, 255)

    # ------------------------------------------------------------------
    # 缓存与帧叠加
    # ------------------------------------------------------------------
    def _state_key(self, cue: ASSCue, frame_ms: int) -> tuple:
        """计算瓦片的「视觉状态键」。

        瓦片只在影响其像素的状态变化时才需要重建：

        * 淡入淡出的全局 alpha（``_fade_factor``，稳态恒为 255）；
        * 各卡拉OK 音节是否已到点（决定主/次色）。

        ``\\move`` 不进键——它只改变瓦片贴到画面的位置（:meth:`_anchor`），
        同一张瓦片在移动过程中全程复用。于是无特效对话整条生命周期只有
        一张瓦片，带淡入的对话在淡入结束后也回到共享状态，避免逐帧重建。
        """
        local_ms = max(0, frame_ms - cue.start_ms)
        fade = self._fade_factor(cue, local_ms)
        k_state = tuple(
            (r.k_dur_ms, local_ms >= r.k_start_ms)
            for line in cue.lines for r in line.runs if r.k_dur_ms
        )
        return (cue.index, fade, k_state)

    def _tile_for(self, cue: ASSCue, frame_ms: int
                  ) -> tuple[bytearray, int, int]:
        key = self._state_key(cue, frame_ms)
        cached = self._tile_cache.get(key)
        if cached is not None:
            return cached
        tile = self._build_tile(cue, frame_ms)
        self._tile_cache[key] = tile
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._tile_cache.pop(old, None)
        return tile

    def _anchor(self, cue: ASSCue, st: ASSStyle, tw: int, th: int,
                frame_ms: int) -> tuple[int, int]:
        """计算瓦片左上角：\\pos/\\move 优先，否则按 numpad 对齐 + 边距。"""
        # \move：在 [start,end] 时间窗内线性插值（PlayRes 坐标 -> 帧坐标）
        anchor = cue.pos
        if cue.move is not None:
            x1, y1, x2, y2 = cue.move
            dur = max(1, cue.end_ms - cue.start_ms)
            t = min(1.0, max(0.0, (frame_ms - cue.start_ms) / dur))
            anchor = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
        if anchor is not None:
            # \pos 的坐标是对齐锚点（numpad 决定锚在瓦片的哪个角/边）
            ax, ay = anchor[0] * self.scale, anchor[1] * self.scale
            halign, valign = numpad_align(cue.alignment or st.alignment)
            ox = ax if halign == "left" else ax - tw if halign == "right" \
                else ax - tw // 2
            oy = ay if valign == "top" else ay - th if valign == "bottom" \
                else ay - th // 2
            return self._clamp(round(ox), round(oy), tw, th)

        halign, valign = numpad_align(cue.alignment or st.alignment)
        ml = round(st.margin_l * self.scale)
        mr = round(st.margin_r * self.scale)
        mv = round(st.margin_v * self.scale)
        if halign == "left":
            x = ml
        elif halign == "right":
            x = self.frame_w - mr - tw
        else:
            x = (self.frame_w - tw) // 2
        if valign == "top":
            y = mv
        elif valign == "middle":
            y = (self.frame_h - th) // 2
        else:
            y = self.frame_h - mv - th
        return self._clamp(x, y, tw, th)

    def _clamp(self, x: int, y: int, tw: int, th: int) -> tuple[int, int]:
        """夹紧到画面内，超出部分交由混合函数裁剪（与 SRT 路径同策略）。"""
        return (max(0, min(x, max(0, self.frame_w - tw))),
                max(0, min(y, max(0, self.frame_h - th))))

    def render_frame(self, frame: bytearray, width: int, height: int,
                     cues: list[ASSCue], frame_ms: int) -> None:
        """把当前帧在场的 ASS 对话列表叠加到 RGB24 帧缓冲（就地修改）。

        列表按 layer 再按开始时间排序后依次叠加，后画的覆盖先画的，与
        libass 的层叠语义一致；同一 layer 的多条对话沿垂直方向自然排布
        由各自的对齐/定位决定，本实现不做碰撞推开。
        """
        if not cues:
            return
        for cue in sorted(cues, key=lambda c: (c.layer, c.start_ms, c.index)):
            st = self.doc.style_of(cue)
            tile, tw, th = self._tile_for(cue, frame_ms)
            x, y = self._anchor(cue, st, tw, th, frame_ms)
            blend_rgba_tile(frame, width, height, tile, tw, th, x, y)
