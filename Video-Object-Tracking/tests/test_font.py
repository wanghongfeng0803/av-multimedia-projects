"""内置点阵字库测试：尺寸、栅格化内容与兜底字形。"""

from vtrack.font import (
    GLYPH_ADVANCE,
    GLYPH_H,
    GLYPH_W,
    render_text_mask,
    text_width,
)


def test_text_width_formula():
    assert text_width("") == 0
    assert text_width("A") == GLYPH_W
    assert text_width("AB") == GLYPH_ADVANCE + GLYPH_W
    assert text_width("HELLO") == 5 * GLYPH_ADVANCE - 1


def test_render_mask_dims():
    w, h, mask = render_text_mask("AB")
    assert (w, h) == (text_width("AB"), GLYPH_H)
    assert len(mask) == w * h
    assert set(mask) <= {0, 1}


def test_known_glyph_has_expected_pixels():
    # 字库按列编码。用稳定的几何不变量校验，不绑定单个 bit：
    # 数字 0 的两侧列是竖边（点亮行数 >=5），顶/底横边横跨多列
    w, h, mask = render_text_mask("0")
    col0 = [mask[r * w + 0] for r in range(h)]
    col4 = [mask[r * w + 4] for r in range(h)]
    assert sum(col0) >= 5 and sum(col4) >= 5
    top_row = [mask[c] for c in range(w)]
    bot_row = [mask[(h - 1) * w + c] for c in range(w)]
    assert sum(top_row) >= 3 and sum(bot_row) >= 3
    # 与数字 1 的字形必须不同（防字库退化成同形）
    _, _, mask1 = render_text_mask("1")
    assert mask != mask1
    # 对照：空格字形全暗
    _, _, blank = render_text_mask(" ")
    assert sum(blank) == 0


def test_lowercase_uses_uppercase_glyph():
    wu, _, upper = render_text_mask("A")
    wl, _, lower = render_text_mask("a")
    assert upper == lower


def test_unknown_char_uses_fallback_block():
    w, h, mask = render_text_mask("~")  # ~ 不在字库（ASCII 126）
    # 兜底块 5 列全部有点亮
    col0 = sum(mask[c] for c in range(0, w * h, w))
    assert col0 > 0


def test_empty_text_empty_mask():
    w, h, mask = render_text_mask("")
    assert (w, h) == (0, GLYPH_H)
    assert mask == b""


def test_spaces_advance_but_stay_blank():
    w, h, mask = render_text_mask(" A")
    # 首字符空格：第一列区域应全 0
    first_glyph = [mask[r * w + c]
                   for r in range(h) for c in range(GLYPH_W)]
    assert not any(first_glyph)
