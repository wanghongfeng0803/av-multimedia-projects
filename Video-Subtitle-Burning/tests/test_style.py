"""样式描述层测试：颜色解析/颜色空间转换、字重、位置越界、序列化。"""

from __future__ import annotations

import pytest

from subburn.style import (
    Color,
    FontSpec,
    FontWeight,
    HAlign,
    Margins,
    Position,
    SubtitleStyle,
    VAlign,
)


class TestColorParsing:
    def test_hex_shorthand_and_long(self):
        assert Color.parse("#f00") == Color(255, 0, 0)
        assert Color.parse("#F00") == Color(255, 0, 0)
        assert Color.parse("#00ff00") == Color(0, 255, 0)
        assert Color.parse("0000ff") == Color(0, 0, 255)  # 允许省略 #

    def test_hex_with_alpha(self):
        assert Color.parse("#ff000080").rgba == (255, 0, 0, 128)
        assert Color.parse("#f008").rgba == (255, 0, 0, 0x88)
        assert Color.parse("#ff0000").a == 255

    def test_named_colors(self):
        assert Color.parse("red") == Color(255, 0, 0)
        assert Color.parse("White") == Color(255, 255, 255)
        assert Color.parse("transparent").a == 0

    def test_rgb_and_rgba_functions(self):
        assert Color.parse("rgb(255, 128, 0)").rgb == (255, 128, 0)
        c = Color.parse("rgba(0, 255, 0, 0.25)")
        assert c.rgb == (0, 255, 0) and c.a == 64
        # 百分比通道与百分比 alpha
        assert Color.parse("rgb(100%, 50%, 0%)").rgb == (255, 128, 0)
        assert Color.parse("rgba(0,0,0,50%)").a == 128

    def test_hsl_function_and_conversion_primaries(self):
        assert Color.parse("hsl(0, 100%, 50%)").rgb == (255, 0, 0)
        assert Color.parse("hsl(120, 100%, 50%)").rgb == (0, 255, 0)
        assert Color.parse("hsl(240, 100%, 50%)").rgb == (0, 0, 255)
        # 灰度（饱和度 0）：色相不影响结果
        assert Color.parse("hsl(12, 0%, 40%)").rgb == (102, 102, 102)

    def test_integer_and_sequence_inputs(self):
        assert Color.parse(0x00FF00) == Color(0, 255, 0)
        assert Color.parse(0xFF000080).a == 128
        assert Color.parse((255, 0, 0)) == Color(255, 0, 0)
        # 浮点序列按 0~1 解释
        assert Color.parse((1.0, 0.5, 0.0, 0.25)).rgba == (255, 128, 0, 64)

    def test_passthrough_and_unknown_format(self):
        c = Color(1, 2, 3, 4)
        assert Color.parse(c) is c
        with pytest.raises(ValueError):
            Color.parse("not-a-color")
        with pytest.raises(ValueError):
            Color.parse("#12")  # 非法位数

    def test_strict_range_rejects_out_of_bounds(self):
        with pytest.raises(ValueError):
            Color(256, 0, 0)
        with pytest.raises(ValueError):
            Color.parse((300, -1, 0))
        with pytest.raises(ValueError):
            Color.parse("rgba(999,0,0,1)")

    def test_clamp_swallows_out_of_bounds(self):
        assert Color.parse((300, -5, 0), clamp=True).rgb == (255, 0, 0)
        # 越界通道夹紧；整数 alpha 按 0~255 字节解释，2 仍是 2
        assert Color.parse("rgba(999,0,0,2)", clamp=True).rgba == (255, 0, 0, 2)
        # 整数超出 32 位：夹紧到不透明白
        assert Color.parse(0x1FFFFFFFF, clamp=True) == Color(255, 255, 255, 255)


class TestColorSpaceRoundTrip:
    @pytest.mark.parametrize("rgb", [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (20, 180, 99), (200, 200, 200), (17, 42, 137), (0, 0, 0),
    ])
    def test_hsl_round_trip(self, rgb):
        h, s, l = Color(*rgb).to_hsl()
        assert Color.from_hsl(h, s, l) == Color(*rgb)

    @pytest.mark.parametrize("rgb", [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (20, 180, 99), (17, 42, 137),
    ])
    def test_hsv_round_trip(self, rgb):
        h, s, v = Color(*rgb).to_hsv()
        assert Color.from_hsv(h, s, v) == Color(*rgb)

    def test_known_hsl_values(self):
        # 纯红：色相 0、满饱和、亮度 0.5
        h, s, l = Color(255, 0, 0).to_hsl()
        assert (h, s, l) == pytest.approx((0.0, 1.0, 0.5))
        # 色相环绕：360 度与 0 度等价
        assert Color.from_hsl(360, 1, 0.5).rgb == (255, 0, 0)

    def test_hex_serialization_omits_opaque_alpha(self):
        assert Color(255, 0, 0).to_hex() == "#ff0000"
        assert Color(255, 0, 0, 128).to_hex() == "#ff000080"
        assert str(Color(0, 255, 0)) == "#00ff00"

    def test_with_alpha_preserves_rgb(self):
        c = Color(10, 20, 30, 255).with_alpha(64)
        assert c.rgba == (10, 20, 30, 64)


class TestFontWeight:
    def test_enum_values_match_opentype(self):
        assert FontWeight.REGULAR.value == 400
        assert FontWeight.BOLD.value == 700
        assert FontWeight.BLACK.value == 900

    def test_parse_name_and_aliases(self):
        assert FontWeight.parse("bold") is FontWeight.BOLD
        assert FontWeight.parse("BOLD") is FontWeight.BOLD
        assert FontWeight.parse("normal") is FontWeight.REGULAR
        assert FontWeight.parse("heavy") is FontWeight.BLACK
        assert FontWeight.parse(FontWeight.LIGHT) is FontWeight.LIGHT

    def test_parse_integer_snaps_to_nearest_step(self):
        assert FontWeight.parse(700) is FontWeight.BOLD
        assert FontWeight.parse(550) is FontWeight.SEMIBOLD
        assert FontWeight.parse(120) is FontWeight.THIN
        assert FontWeight.parse(999) is FontWeight.BLACK  # 夹紧
        assert FontWeight.nearest(440) is FontWeight.REGULAR

    def test_invalid_token(self):
        with pytest.raises(ValueError):
            FontWeight.parse("chunky")


class TestFontSpec:
    def test_defaults_and_normalization(self):
        spec = FontSpec.parse({"family": "Arial", "size": 30, "weight": "bold"})
        assert spec.families == ("Arial",)
        assert spec.size == 30 and spec.weight is FontWeight.BOLD

    def test_family_chain_from_list(self):
        spec = FontSpec.parse({"families": ["Arial", "微软雅黑", "DejaVu Sans"]})
        assert spec.families == ("Arial", "微软雅黑", "DejaVu Sans")

    def test_none_and_passthrough(self):
        assert FontSpec.parse(None) == FontSpec()
        spec = FontSpec(("A",), 12, FontWeight.MEDIUM)
        assert FontSpec.parse(spec) is spec

    def test_rejects_negative_size(self):
        with pytest.raises(ValueError):
            FontSpec(size=-1)


class TestMargins:
    def test_parse_forms(self):
        assert Margins.parse(8) == Margins(8, 8, 8, 8)
        assert Margins.parse([1, 2, 3, 4]) == Margins(1, 2, 3, 4)
        assert Margins.parse({"left": 5}) == Margins(5, 0, 0, 0)
        assert Margins.parse(None) == Margins()

    def test_reject_negative_and_bad_length(self):
        with pytest.raises(ValueError):
            Margins(left=-1)
        with pytest.raises(ValueError):
            Margins.parse([1, 2, 3])

    def test_clamp_single_side_overflow(self):
        # 单边先夹紧到画面宽；夹紧后与对边之和占满画面 -> 触发对边归零
        # （否则可用宽度为负，字幕会被推出画面），这是 clamp_to_frame 的
        # 明确设计：优先保证「能显示」而非「保留边距」
        m = Margins(400, 0, 10, 0).clamp_to_frame(320, 240)
        assert (m.left, m.right) == (0, 0)
        # 对边之和未占满画面：普通大边距原样保留
        m2 = Margins(200, 0, 10, 0).clamp_to_frame(320, 240)
        assert (m2.left, m2.right) == (200, 10)

    def test_clamp_opposing_sides_collapse_to_zero(self):
        # 左右之和占满宽度：双双归零，保证可用宽度为正
        m = Margins(300, 0, 300, 0).clamp_to_frame(320, 240)
        assert (m.left, m.right) == (0, 0)
        m = Margins(0, 200, 0, 200).clamp_to_frame(320, 240)
        assert (m.top, m.bottom) == (0, 0)


class TestPosition:
    def test_anchor_by_alignment(self):
        box = (100, 40)
        bottom_center = Position()
        assert bottom_center.anchor_origin(320, 240, *box) == (110, 240 - 36 - 40)

        right_top = Position(HAlign.RIGHT, VAlign.TOP, Margins.uniform(10))
        assert right_top.anchor_origin(320, 240, *box) == (210, 10)

        left_middle = Position(HAlign.LEFT, VAlign.MIDDLE, Margins.uniform(10))
        x, y = left_middle.anchor_origin(320, 240, *box)
        assert x == 10 and y == (240 - 40) // 2

    def test_oversized_box_clamps_to_origin(self):
        p = Position(HAlign.RIGHT, VAlign.BOTTOM, Margins.uniform(10))
        # 块比画面还大：任何对齐都退化为 (0,0)，由混合层负责裁剪
        assert p.anchor_origin(320, 240, 400, 300) == (0, 0)

    def test_parse_dict_with_aliases(self):
        p = Position.parse({"halign": "centre", "valign": "center",
                            "margin": 12})
        assert p.halign is HAlign.CENTER and p.valign is VAlign.MIDDLE
        assert p.margins == Margins.uniform(12)

    def test_parse_invalid_alignment(self):
        with pytest.raises(ValueError):
            Position.parse({"halign": "sideways"})


class TestSubtitleStyle:
    def test_legacy_bridge_maps_every_field(self):
        from subburn.render import Style

        legacy = Style(
            color=(10, 20, 30), outline_color=(40, 50, 60),
            outline_width=3, shadow_offset=4, margin=22,
            line_gap=7, tile_gap=13, box=True,
            box_color=(1, 2, 3), box_alpha=99, box_padding=11,
        )
        new = SubtitleStyle.from_legacy(legacy)
        assert new.color.rgba == (10, 20, 30, 255)
        assert new.outline_color.rgb == (40, 50, 60)
        assert new.outline_width == 3 and new.shadow_offset == 4
        assert new.position.margins == Margins.uniform(22)
        assert (new.line_gap, new.tile_gap) == (7, 13)
        assert new.box and new.box_color.rgba == (1, 2, 3, 99)
        assert new.box_padding == 11

    def test_from_dict_partial_keeps_defaults(self):
        st = SubtitleStyle.from_dict({"color": "#ff0000", "box": True})
        assert st.color == Color(255, 0, 0)
        assert st.box is True
        assert st.outline_width == SubtitleStyle().outline_width  # 默认值保留

    def test_from_dict_nested_font_and_position(self):
        st = SubtitleStyle.from_dict({
            "font": {"families": ["Arial"], "size": 22, "weight": 700},
            "position": {"halign": "left", "valign": "top",
                         "margins": {"left": 5, "top": 6, "right": 7, "bottom": 8}},
        })
        assert st.font.families == ("Arial",)
        assert st.font.weight is FontWeight.BOLD
        assert st.position.margins == Margins(5, 6, 7, 8)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError):
            SubtitleStyle.from_dict({"nope": 1})

    def test_dict_round_trip(self):
        st = SubtitleStyle(
            font=FontSpec(("Arial", "微软雅黑"), 24, FontWeight.BOLD),
            color=Color(1, 2, 3, 200),
            position=Position(HAlign.RIGHT, VAlign.TOP, Margins(1, 2, 3, 4)),
            box=True,
        )
        assert SubtitleStyle.from_dict(st.to_dict()) == st

    def test_rejects_negative_numeric_fields(self):
        with pytest.raises(ValueError):
            SubtitleStyle(outline_width=-1)
        with pytest.raises(ValueError):
            SubtitleStyle(line_gap=-2)

    def test_frozen_and_hashable(self):
        with pytest.raises(Exception):
            SubtitleStyle().box = True  # type: ignore[misc]
        hash(SubtitleStyle())
        hash(Color(1, 2, 3, 4))
