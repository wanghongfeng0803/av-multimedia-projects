"""内置点阵字体、TrueType 解析与字体回退链测试。"""

from __future__ import annotations

import os

import pytest

from hls_packager.watermark import bitmap_font
from hls_packager.watermark.font_stack import FontStack, discover_system_fonts
from hls_packager.watermark.rasterizer import rasterize_glyph
from hls_packager.watermark.ttf_font import is_truetype_file, load_font

_SYSTEM_FONTS = discover_system_fonts()
_HAS_TTF = bool(_SYSTEM_FONTS)


def test_bitmap_font_covers_all_printable_ascii():
    for codepoint in range(0x20, 0x7F):
        assert bitmap_font.supports(codepoint), f"缺少 ASCII 字形 {chr(codepoint)!r}"
    assert len(bitmap_font.GLYPHS) == 0x7F - 0x20


def test_bitmap_font_does_not_cover_non_ascii():
    assert not bitmap_font.supports(ord("水"))
    assert not bitmap_font.supports(0x1F600)
    assert bitmap_font.glyph_rows(0x1F600) is None


def test_bitmap_glyph_rows_have_fixed_shape():
    rows = bitmap_font.glyph_rows(ord("A"))
    assert rows is not None and len(rows) == bitmap_font.GLYPH_HEIGHT
    assert all(0 <= row < (1 << bitmap_font.GLYPH_WIDTH) for row in rows)


def test_letter_a_has_ink_but_space_does_not():
    assert any(bitmap_font.glyph_rows(ord("A")))
    assert not any(bitmap_font.glyph_rows(ord(" ")))


def test_empty_font_stack_falls_back_to_bitmap_for_ascii():
    stack = FontStack([], system_fonts=[])
    assert stack.resolve(ord("A")).kind == "bitmap"
    assert stack.resolve(ord(" ")).kind == "bitmap"


def test_empty_font_stack_uses_tofu_for_uncovered_codepoint():
    stack = FontStack([], system_fonts=[])
    assert stack.resolve(ord("水")).kind == "tofu"
    assert stack.resolve(0x1F600).kind == "tofu"


def test_explicit_missing_font_path_does_not_crash(tmp_path):
    missing = str(tmp_path / "no-such-font.ttf")
    stack = FontStack([missing], system_fonts=[])
    assert stack.resolve(ord("A")).kind == "bitmap"


def test_corrupted_font_is_skipped(tmp_path):
    bad = tmp_path / "broken.ttf"
    bad.write_bytes(b"\x00\x01\x00\x00not-a-real-font" + b"\x00" * 64)
    stack = FontStack([str(bad)], system_fonts=[])
    assert stack.resolve(ord("A")).kind == "bitmap"
    assert stack._loaded[str(bad)] is None


def test_is_truetype_file_rejects_garbage(tmp_path):
    garbage = tmp_path / "garbage.ttf"
    garbage.write_bytes(b"plain text, not a font")
    assert is_truetype_file(str(garbage)) is False
    assert is_truetype_file(str(tmp_path / "missing.ttf")) is False


@pytest.mark.skipif(not _HAS_TTF, reason="系统中未发现任何 TrueType 字体")
def test_load_system_font_and_read_metrics():
    font = load_font(_SYSTEM_FONTS[0])
    assert 16 <= font.units_per_em <= 16384
    assert font.glyph_id(ord("A")) != 0
    assert font.glyph_id(0x1FAFAFA) == 0


@pytest.mark.skipif(not _HAS_TTF, reason="系统中未发现任何 TrueType 字体")
def test_rasterized_latin_letter_has_ink():
    stack = FontStack()
    face = stack.resolve(ord("A"))
    assert face.kind == "ttf"
    bitmap = rasterize_glyph(face.font, face.glyph_id, 32)
    assert bitmap.width > 0 and bitmap.height > 0
    assert any(c > 0 for c in bitmap.coverage)
    assert bitmap.advance > 0


@pytest.mark.skipif(not _HAS_TTF, reason="系统中未发现任何 TrueType 字体")
def test_space_glyph_is_blank_but_has_advance():
    stack = FontStack()
    face = stack.resolve(ord(" "))
    bitmap = rasterize_glyph(face.font, face.glyph_id, 32)
    assert not any(bitmap.coverage)
    assert bitmap.advance > 0


@pytest.mark.skipif(not _HAS_TTF, reason="系统中未发现任何 TrueType 字体")
def test_ttc_collection_loads_and_renders_cjk():
    ttc_paths = [p for p in _SYSTEM_FONTS if p.lower().endswith(".ttc")]
    if not ttc_paths:
        pytest.skip("系统中没有 TTC 字体集")
    font = load_font(ttc_paths[0])
    glyph_id = font.glyph_id(ord("水"))
    if glyph_id == 0:
        pytest.skip("该 TTC 不含 CJK 字形")
    bitmap = rasterize_glyph(font, glyph_id, 32)
    assert any(c > 0 for c in bitmap.coverage)


@pytest.mark.skipif(not _HAS_TTF, reason="系统中未发现任何 TrueType 字体")
def test_font_stack_prefers_real_font_over_bitmap():
    stack = FontStack()
    assert stack.resolve(ord("A")).kind == "ttf"
    for codepoint in (ord("A"), ord("0"), ord("水")):
        assert stack.resolve(codepoint).kind in ("ttf", "bitmap", "tofu")


def test_discovery_only_returns_existing_files():
    for path in _SYSTEM_FONTS:
        assert os.path.isfile(path)
        assert path.lower().endswith((".ttf", ".ttc", ".otf"))
