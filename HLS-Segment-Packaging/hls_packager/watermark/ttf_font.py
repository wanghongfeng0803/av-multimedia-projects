"""自研 TrueType（.ttf/.ttc）解析与字形轮廓提取。

手工解析 sfnt 目录、cmap（format 0/4/12）、head/hhea/maxp/hmtx/loca/glyf
六张必要表，把 TrueType 二次贝塞尔轮廓扁平化为多边形，交给
rasterizer 做扫描线栅格化。CFF（OTTO）/WOFF/损坏文件以受控异常上抛，
由 font_stack 走回退链。
"""

from __future__ import annotations

import struct

from ..errors import WatermarkError

_SFNT_TRUE = 0x00010000
_SFNT_TRUE_2 = 0x74727565
_SFNT_TYPO = 0x74797031
_TTC_TAG = 0x74746366

_FLAG_ON_CURVE = 0x01
_FLAG_X_SHORT = 0x02
_FLAG_Y_SHORT = 0x04
_FLAG_REPEAT = 0x08
_FLAG_X_SAME = 0x10
_FLAG_Y_SAME = 0x20

_ARG_1_AND_2_ARE_WORDS = 0x0001
_WE_HAVE_A_SCALE = 0x0008
_MORE_COMPONENTS = 0x0020
_WE_HAVE_AN_X_AND_Y_SCALE = 0x0040
_WE_HAVE_A_TWO_BY_TWO = 0x0080

_CURVE_MAX_DEPTH = 6
_COMPOSITE_MAX_DEPTH = 12


class _Reader:
    """大端只读视图。"""

    __slots__ = ("data",)

    def __init__(self, data: bytes):
        self.data = data

    def u8(self, offset): return self.data[offset]

    def s8(self, offset):
        v = self.data[offset]
        return v - 256 if v >= 128 else v

    def u16(self, offset): return struct.unpack_from(">H", self.data, offset)[0]

    def s16(self, offset): return struct.unpack_from(">h", self.data, offset)[0]

    def u32(self, offset): return struct.unpack_from(">I", self.data, offset)[0]

    def tag(self, offset): return bytes(self.data[offset:offset + 4])

    def check(self, offset, size, what):
        if offset < 0 or offset + size > len(self.data):
            raise WatermarkError(f"字体文件损坏：读取 {what} 时越界")


def _parse_cmap_subtable(reader, table_offset, sub_offset):
    base = table_offset + sub_offset
    fmt = reader.u16(base)

    if fmt == 0:
        reader.check(base, 262, "cmap format 0")
        return {cp: reader.u8(base + 6 + cp) for cp in range(256)}

    if fmt == 4:
        length = reader.u16(base + 2)
        reader.check(base, length, "cmap format 4")
        seg_count = reader.u16(base + 6) // 2
        end_codes = [reader.u16(base + 14 + 2 * i) for i in range(seg_count)]
        start_base = base + 14 + seg_count * 2 + 2
        start_codes = [reader.u16(start_base + 2 * i) for i in range(seg_count)]
        delta_base = start_base + seg_count * 2
        range_base = delta_base + seg_count * 2
        mapping = {}
        for i, (start, end) in enumerate(zip(start_codes, end_codes)):
            if start == 0xFFFF and end == 0xFFFF:
                continue
            delta = reader.s16(delta_base + 2 * i)
            ro_pos = range_base + 2 * i
            range_offset = reader.u16(ro_pos)
            for cp in range(start, end + 1):
                if range_offset == 0:
                    glyph_id = (cp + delta) & 0xFFFF
                else:
                    glyph_id = reader.u16(ro_pos + range_offset + 2 * (cp - start))
                    if glyph_id != 0:
                        glyph_id = (glyph_id + delta) & 0xFFFF
                if glyph_id:
                    mapping[cp] = glyph_id
        return mapping

    if fmt == 12:
        num_groups = reader.u32(base + 12)
        reader.check(base, 16 + 12 * num_groups, "cmap format 12")
        mapping = {}
        for i in range(num_groups):
            g = base + 16 + 12 * i
            start_cp, end_cp, start_glyph = reader.u32(g), reader.u32(g + 4), reader.u32(g + 8)
            if start_glyph + (end_cp - start_cp) > 0xFFFF:
                continue
            for cp in range(start_cp, end_cp + 1):
                mapping[cp] = start_glyph + (cp - start_cp)
        return mapping

    return None


def _parse_cmap(reader, table_offset):
    reader.check(table_offset, 4, "cmap")
    num_records = reader.u16(table_offset + 2)
    records = []
    for i in range(num_records):
        pos = table_offset + 4 + 8 * i
        records.append((reader.u16(pos), reader.u16(pos + 2), reader.u32(pos + 4)))

    def rank(rec):
        p, e, _ = rec
        if p == 3 and e in (10, 1):
            return 3 if e == 10 else 2
        return 1 if p == 0 else 0

    best = {}
    for rec in sorted(records, key=rank, reverse=True):
        parsed = _parse_cmap_subtable(reader, table_offset, rec[2])
        if parsed and len(parsed) > len(best):
            best = parsed
    return best


def _read_simple_glyph(reader, offset, contour_count):
    end_pts = [reader.u16(offset + 10 + 2 * i) for i in range(contour_count)]
    point_count = end_pts[-1] + 1
    instruction_len = reader.u16(offset + 10 + 2 * contour_count)
    pos = offset + 10 + 2 * contour_count + 2 + instruction_len

    flags = []
    while len(flags) < point_count:
        flag = reader.u8(pos)
        pos += 1
        repeat = reader.u8(pos) if flag & _FLAG_REPEAT else 0
        if flag & _FLAG_REPEAT:
            pos += 1
        flags.extend([flag] * (repeat + 1))
    flags = flags[:point_count]

    xs = [0] * point_count
    x = 0
    for i, flag in enumerate(flags):
        if flag & _FLAG_X_SHORT:
            delta = reader.u8(pos); pos += 1
            if not (flag & _FLAG_X_SAME):
                delta = -delta
        elif flag & _FLAG_X_SAME:
            delta = 0
        else:
            delta = reader.s16(pos); pos += 2
        x += delta
        xs[i] = x

    ys = [0] * point_count
    y = 0
    for i, flag in enumerate(flags):
        if flag & _FLAG_Y_SHORT:
            delta = reader.u8(pos); pos += 1
            if not (flag & _FLAG_Y_SAME):
                delta = -delta
        elif flag & _FLAG_Y_SAME:
            delta = 0
        else:
            delta = reader.s16(pos); pos += 2
        y += delta
        ys[i] = y

    points = [((float(xs[i]), float(ys[i])), bool(flags[i] & _FLAG_ON_CURVE))
              for i in range(point_count)]
    contours = []
    start = 0
    for end in end_pts:
        contours.append(points[start:end + 1])
        start = end + 1
    return contours


def _expand_implicit_oncurve(points):
    """相邻 off-curve 间插入隐式 on-curve 中点。"""
    if not points:
        return [], []
    exp_pts, exp_on = [], []
    n = len(points)
    for i in range(n):
        pt, on = points[i]
        _, prev_on = points[(i - 1) % n]
        if not on and not prev_on:
            prev_pt, _ = points[(i - 1) % n]
            exp_pts.append(((prev_pt[0] + pt[0]) / 2.0, (prev_pt[1] + pt[1]) / 2.0))
            exp_on.append(True)
        exp_pts.append(pt)
        exp_on.append(on)
    if exp_pts and not exp_on[0] and not exp_on[-1]:
        mid = ((exp_pts[-1][0] + exp_pts[0][0]) / 2.0,
               (exp_pts[-1][1] + exp_pts[0][1]) / 2.0)
        exp_pts.insert(0, mid)
        exp_on.insert(0, True)
    return exp_pts, exp_on


def _flatten_quadratic(p0, p1, p2, depth, out):
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        d2 = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    else:
        cross = dx * (p1[1] - p0[1]) - dy * (p1[0] - p0[0])
        d2 = (cross * cross) / l2
    if depth >= _CURVE_MAX_DEPTH or d2 < 1.0:
        out.append(p2)
        return
    m01 = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    m12 = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    m012 = ((m01[0] + m12[0]) / 2.0, (m01[1] + m12[1]) / 2.0)
    _flatten_quadratic(p0, m01, m012, depth + 1, out)
    _flatten_quadratic(m012, m12, p2, depth + 1, out)


def _contour_to_polygon(points, on_curve):
    polygon = [points[0]]
    n = len(points)
    i = 0
    while i < n:
        if on_curve[i]:
            j = (i + 1) % n
            if on_curve[j]:
                polygon.append(points[j])
                i = j
            else:
                k = (i + 2) % n
                _flatten_quadratic(points[i], points[j], points[k], 0, polygon)
                i = k
        else:
            i += 1
        if i == 0:
            break
    return polygon


def _apply_transform(points, a, b, c, d, e, f):
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in points]


class TrueTypeFont:
    """已解析必要表的 TrueType 字体（含 TTC 单个 face）。

    TTC 表目录偏移按规范相对整个文件起点，故不切片数据；
    base_offset 仅用于定位 sfnt 头与表目录。
    """

    def __init__(self, data, source_path, face_index=0, base_offset=0):
        self._reader = _Reader(data)
        self.data = data
        self.source_path = source_path
        self.face_index = face_index
        self._base = base_offset
        self._tables = {}
        self._cmap = {}
        self._units_per_em = 1000
        self._index_to_loc_format = 0
        self._num_glyphs = 0
        self._hmetrics = []
        self._number_of_h_metrics = 0
        self._glyph_cache = {}
        self._parse()

    def _parse(self):
        reader = self._reader
        base = self._base
        reader.check(base, 12, "sfnt 头")
        if reader.u32(base) not in (_SFNT_TRUE, _SFNT_TRUE_2, _SFNT_TYPO):
            raise WatermarkError(
                f"不支持的字体轮廓格式：{self.source_path}"
                "（仅支持 TrueType 轮廓，CFF/OpenType 需回退其它字体）"
            )
        num_tables = reader.u16(base + 4)
        reader.check(base, 12 + 16 * num_tables, "sfnt 表目录")
        for i in range(num_tables):
            entry = base + 12 + 16 * i
            self._tables[reader.tag(entry)] = reader.u32(entry + 8)

        for required in (b"cmap", b"head", b"hhea", b"maxp", b"hmtx", b"loca", b"glyf"):
            if required not in self._tables:
                raise WatermarkError(
                    f"字体缺少必要表 {required.decode('ascii', 'replace')}：{self.source_path}"
                )

        head = self._tables[b"head"]
        self._units_per_em = reader.u16(head + 18)
        if not (16 <= self._units_per_em <= 16384):
            raise WatermarkError(f"字体 unitsPerEm 非法：{self._units_per_em}")
        self._index_to_loc_format = reader.s16(head + 50)
        self._num_glyphs = reader.u16(self._tables[b"maxp"] + 4)
        self._number_of_h_metrics = reader.u16(self._tables[b"hhea"] + 34)
        self._parse_hmtx()
        self._cmap = _parse_cmap(reader, self._tables[b"cmap"])

    def _parse_hmtx(self):
        reader = self._reader
        hmtx = self._tables[b"hmtx"]
        metrics = []
        for i in range(self._number_of_h_metrics):
            pos = hmtx + 4 * i
            metrics.append((reader.u16(pos), reader.s16(pos + 2)))
        if metrics:
            last_advance = metrics[-1][0]
            lsb_base = hmtx + 4 * self._number_of_h_metrics
            for i in range(max(0, self._num_glyphs - self._number_of_h_metrics)):
                metrics.append((last_advance, reader.s16(lsb_base + 2 * i)))
        self._hmetrics = metrics

    @property
    def units_per_em(self):
        return self._units_per_em

    def glyph_id(self, codepoint):
        return self._cmap.get(codepoint, 0)

    def advance_width(self, glyph_id):
        if 0 <= glyph_id < len(self._hmetrics):
            return self._hmetrics[glyph_id][0]
        return self._units_per_em // 2

    def _glyph_offsets(self, glyph_id):
        reader = self._reader
        loca = self._tables[b"loca"]
        if self._index_to_loc_format == 1:
            start = reader.u32(loca + 4 * glyph_id)
            end = reader.u32(loca + 4 * (glyph_id + 1))
        else:
            start = 2 * reader.u16(loca + 2 * glyph_id)
            end = 2 * reader.u16(loca + 2 * (glyph_id + 1))
        glyf = self._tables[b"glyf"]
        return glyf + start, glyf + end

    def glyph_outlines(self, glyph_id, _depth=0):
        """返回字形全部轮廓；空字形返回 None；复合字形递归展开。"""
        if glyph_id in self._glyph_cache:
            cached = self._glyph_cache[glyph_id]
            return [list(c) for c in cached] if cached is not None else None
        if _depth > _COMPOSITE_MAX_DEPTH:
            raise WatermarkError("复合字形嵌套过深，疑似损坏字体")

        start, end = self._glyph_offsets(glyph_id)
        if end - start < 2:
            self._glyph_cache[glyph_id] = None
            return None

        reader = self._reader
        n_contours = reader.s16(start)
        if n_contours > 0:
            raw = _read_simple_glyph(reader, start, n_contours)
            polygons = []
            for r in raw:
                pts, ons = _expand_implicit_oncurve(r)
                polygons.append(_contour_to_polygon(pts, ons))
            self._glyph_cache[glyph_id] = tuple(tuple(p) for p in polygons)
            return polygons
        if n_contours == -1:
            polygons = self._read_composite_glyph(reader, start, _depth)
            self._glyph_cache[glyph_id] = tuple(tuple(p) for p in polygons)
            return polygons
        self._glyph_cache[glyph_id] = None
        return None

    def _read_composite_glyph(self, reader, start, depth):
        pos = start + 10
        result = []
        while True:
            flags = reader.u16(pos)
            glyph_id = reader.u16(pos + 2)
            pos += 4
            if flags & _ARG_1_AND_2_ARE_WORDS:
                arg1, arg2 = reader.s16(pos), reader.s16(pos + 2)
                pos += 4
            else:
                arg1, arg2 = reader.s8(pos), reader.s8(pos + 1)
                pos += 2
            e, f = float(arg1), float(arg2)

            if flags & _WE_HAVE_A_TWO_BY_TWO:
                a = reader.s16(pos) / 16384.0
                b = reader.s16(pos + 2) / 16384.0
                c = reader.s16(pos + 4) / 16384.0
                d = reader.s16(pos + 6) / 16384.0
                pos += 8
            elif flags & _WE_HAVE_AN_X_AND_Y_SCALE:
                a = d = reader.s16(pos) / 16384.0
                b = c = 0.0
                pos += 2
            elif flags & _WE_HAVE_A_SCALE:
                a = d = reader.s16(pos) / 16384.0
                b = c = 0.0
                pos += 2
            else:
                a = d = 1.0
                b = c = 0.0

            child = self.glyph_outlines(glyph_id, depth + 1)
            if child is not None:
                for contour in child:
                    result.append(_apply_transform(contour, a, b, c, d, e, f))
            if not (flags & _MORE_COMPONENTS):
                break
        return result


def load_font(path: str, face_index: int = 0) -> TrueTypeFont:
    """加载字体；TTC 取第 face_index 个 face，失败抛 WatermarkError。"""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise WatermarkError(f"无法读取字体文件 {path!r}：{exc}") from exc
    if len(data) < 12:
        raise WatermarkError(f"字体文件过小或已损坏：{path}")

    reader = _Reader(data)
    if reader.u32(0) == _TTC_TAG:
        num_fonts = reader.u32(8)
        if not (0 <= face_index < num_fonts):
            raise WatermarkError(f"TTC 字体集只有 {num_fonts} 个 face，请求第 {face_index} 个")
        face_offset = reader.u32(12 + 4 * face_index)
        return TrueTypeFont(data, path, face_index, base_offset=face_offset)
    return TrueTypeFont(data, path, face_index)


def is_truetype_file(path: str) -> bool:
    """快速魔数判断文件是否像 TrueType/TTC。"""
    try:
        with open(path, "rb") as handle:
            magic = handle.read(4)
    except OSError:
        return False
    if len(magic) < 4:
        return False
    value = struct.unpack(">I", magic)[0]
    return value == _TTC_TAG or value in (_SFNT_TRUE, _SFNT_TRUE_2, _SFNT_TYPO)
