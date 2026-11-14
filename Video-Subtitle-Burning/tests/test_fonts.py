"""字体层测试：位图粗体、文件名解析、注册表回退链与字重选择。

涉及真实 TTF 的用例在无可用字体文件时自动 skip；纯位图/解析逻辑不依赖
系统环境，任何机器都能跑。
"""

from __future__ import annotations

import os

import pytest

from subburn.fonts import (
    BitmapFont,
    FontRegistry,
    load_font,
    parse_font_filename,
    resolve_font,
)
from subburn.style import FontSpec, FontWeight


class TestBitmapFontBold:
    def test_bold_is_wider_than_regular(self):
        regular = BitmapFont(scale=2, bold=False)
        bold = BitmapFont(scale=2, bold=True)
        assert bold.bold is True and regular.bold is False
        assert bold.text_size("AB")[0] > regular.text_size("AB")[0]

    def test_bold_renders_more_ink_pixels(self):
        regular = BitmapFont(1, bold=False)
        bold = BitmapFont(1, bold=True)
        a = regular.rasterize_line("A", (255, 255, 255))
        b = bold.rasterize_line("A", (255, 255, 255))
        ink = lambda buf: sum(1 for i in range(3, len(buf), 4) if buf[i] == 255)
        assert ink(b) > ink(a)

    def test_bold_keeps_glyph_metrics_stable(self):
        bold = BitmapFont(2, bold=True)
        w, h = bold.text_size("AB")
        assert h == 14  # 高度不随粗体变化
        assert bold.line_height == 16

    def test_legacy_constructor_defaults_regular(self):
        assert BitmapFont(1).bold is False


class TestFilenameParsing:
    def test_regular_filenames(self):
        assert parse_font_filename("arial.ttf") == ("arial", FontWeight.REGULAR)
        assert parse_font_filename("msyh.ttc") == (
            "microsoft yahei", FontWeight.REGULAR)
        assert parse_font_filename("NotoSansCJK-Regular.ttc") == (
            "noto sans cjk", FontWeight.REGULAR)

    def test_separated_weight_suffix(self):
        assert parse_font_filename("DejaVuSans-Bold.ttf") == (
            "dejavu sans", FontWeight.BOLD)
        assert parse_font_filename("Foo-Light.otf") == ("foo", FontWeight.LIGHT)
        assert parse_font_filename("Foo_Black.ttf") == ("foo", FontWeight.BLACK)

    def test_concatenated_abbreviated_suffix(self):
        # Windows 常见连体缩写
        assert parse_font_filename("arialbd.ttf") == ("arial", FontWeight.BOLD)
        assert parse_font_filename("msyhbd.ttc") == (
            "microsoft yahei", FontWeight.BOLD)
        assert parse_font_filename("arialbi.ttf") == ("arial", FontWeight.BOLD)

    def test_italic_does_not_change_weight(self):
        assert parse_font_filename("ariali.ttf") == ("arial", FontWeight.REGULAR)
        family, weight = parse_font_filename("DejaVuSans-BoldOblique.ttf")
        assert family == "dejavu sans" and weight is FontWeight.BOLD


class TestRegistryFallback:
    def test_empty_candidates_falls_back_to_bitmap(self):
        reg = FontRegistry(candidate_paths=())
        font = reg.resolve(FontSpec(size=28), fallback_size=28)
        assert isinstance(font, BitmapFont)

    def test_bitmap_bold_when_native_unavailable(self):
        reg = FontRegistry(candidate_paths=())
        font = reg.resolve(
            FontSpec(size=28, weight=FontWeight.BOLD), fallback_size=28)
        assert isinstance(font, BitmapFont) and font.bold is True

    def test_missing_family_chain_degrades_to_default(self):
        reg = FontRegistry(candidate_paths=())
        # 全链不存在：最终位图保底，不抛异常
        font = reg.resolve(
            FontSpec(families=("NoSuchFont", "AlsoMissing"), size=20),
            fallback_size=20)
        assert isinstance(font, BitmapFont)

    def test_broken_font_path_is_skipped(self, tmp_path):
        fake = tmp_path / "broken.ttf"
        fake.write_bytes(b"not a real font")
        reg = FontRegistry(candidate_paths=(str(fake),))
        font = reg.resolve(FontSpec(families=(str(fake),), size=20),
                           fallback_size=20)
        # 损坏文件无法加载 -> 位图保底
        assert isinstance(font, BitmapFont)

    def test_instance_cache_returns_same_object(self, tmp_path):
        # 用两个不存在的路径构造空表场景；缓存行为在位图回退上同样可验证
        reg = FontRegistry(candidate_paths=())
        spec = FontSpec(size=21)
        assert reg.resolve(spec, fallback_size=21) is reg.resolve(
            spec, fallback_size=21)


def _has_any_ttf() -> bool:
    reg = FontRegistry()
    reg._scan()
    return bool(reg._by_family)


@pytest.mark.skipif(not _has_any_ttf(), reason="系统无可用 TTF 字体")
class TestRegistryWithRealFonts:
    def setup_method(self):
        self.registry = FontRegistry()

    def test_family_resolves_to_real_file(self):
        family = self.registry.available_families()[0]
        font = self.registry.resolve(
            FontSpec(families=(family,), size=24), fallback_size=24)
        assert not isinstance(font, BitmapFont)
        assert os.path.isfile(font.font_path)

    def test_bold_weight_prefers_native_bold(self):
        reg = self.registry
        # 找到同时登记了 bold 的族才有意义
        for family, table in reg._by_family.items():
            if FontWeight.BOLD in table and FontWeight.REGULAR in table:
                font = reg.resolve(
                    FontSpec(families=(family,), size=24,
                             weight=FontWeight.BOLD),
                    fallback_size=24)
                assert font.font_path == table[FontWeight.BOLD]
                assert font.bold is False
                return
        pytest.skip("本机没有同时含 regular/bold 的字体族")

    def test_synthetic_bold_when_only_regular_exists(self):
        reg = self.registry
        for family, table in reg._by_family.items():
            weights = set(table)
            if FontWeight.REGULAR in weights and not any(
                w.value >= FontWeight.BOLD.value for w in weights
            ):
                font = reg.resolve(
                    FontSpec(families=(family,), size=24,
                             weight=FontWeight.BOLD),
                    fallback_size=24)
                assert font.bold is True  # 无原生 bold -> 合成
                return
        pytest.skip("本机没有仅含 regular 的字体族")

    def test_chain_skips_unknown_family(self):
        known = self.registry.available_families()[0]
        font = self.registry.resolve(
            FontSpec(families=("DefinitelyMissing", known), size=24),
            fallback_size=24)
        assert not isinstance(font, BitmapFont)

    def test_explicit_path_bypasses_family_table(self):
        self.registry._scan()
        any_path = next(
            iter(next(iter(self.registry._by_family.values())).values()))
        font = self.registry.resolve(
            FontSpec(families=(any_path,), size=20,
                     weight=FontWeight.BOLD),
            fallback_size=24)
        assert font.font_path == any_path


class TestResolveAndLoadHelpers:
    def test_resolve_font_auto_size_from_frame_height(self):
        font = resolve_font(
            FontSpec(), frame_height=480,
            registry=FontRegistry(candidate_paths=()))
        # 位图缩放档：max(20, 480//24)=20 -> scale=round(20/7)=3
        assert isinstance(font, BitmapFont) and font.scale == 3

    def test_resolve_font_explicit_size_wins(self):
        font = resolve_font(
            FontSpec(size=14), frame_height=480,
            registry=FontRegistry(candidate_paths=()))
        assert font.scale == 2  # round(14/7)=2

    def test_load_font_without_pillow_is_bitmap(self):
        font = load_font(allow_pillow=False, size=28, weight="bold")
        assert isinstance(font, BitmapFont) and font.bold is True

    def test_load_font_regular_weight(self):
        font = load_font(allow_pillow=False, size=28)
        assert isinstance(font, BitmapFont) and font.bold is False
