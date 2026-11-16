"""3D LUT（.cube）解析、三线性插值与强度混合的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import math

import pytest

from color_grading.errors import LUTDomainError, LUTParseError
from color_grading.lut3d import (
    MAX_GRID_SIZE,
    CubeLUT,
    apply_chunk,
    clamp_strength,
    load_cube,
    parse_cube,
)


# --------------------------------------------------------------- 构造工具


def make_cube_text(size: int, func=None, *, header_extra: str = "") -> str:
    """生成一张 N³ 的 .cube 文本；func(r,g,b)->(R,G,B) 归一化，默认恒等。

    数据严格按 R 最快、G 次之、B 最慢的 Adobe 展平顺序写出。
    """
    if func is None:
        func = lambda r, g, b: (r, g, b)
    lines = [f"LUT_3D_SIZE {size}", 'TITLE "generated"']
    if header_extra:
        lines.append(header_extra)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rr, gg, bb = func(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append(f"{rr:.6f} {gg:.6f} {bb:.6f}")
    return "\n".join(lines) + "\n"


def identity_cube(size: int = 4) -> CubeLUT:
    return parse_cube(make_cube_text(size))


# size=2 的恒等网格恰好 8 个点（R 最快、G 次之、B 最慢）。
IDENTITY_2X2X2 = (
    "0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
    "0 0 1\n1 0 1\n0 1 1\n1 1 1\n"
)


# --------------------------------------------------------------- 解析容错


def test_parse_basic_identity_size_and_length():
    lut = identity_cube(4)
    assert lut.size == 4
    assert len(lut.data) == 4 ** 3 * 3
    assert lut.title == "generated"


def test_parse_tolerates_comments_blank_lines_crlf_and_spacing():
    text = (
        "#  leading comment\r\n"
        "\r\n"
        "  LUT_3D_SIZE   2  # trailing comment\r\n"
        'TITLE   "t"\r\n'
        "\n"
        + IDENTITY_2X2X2.replace("\n", "\r\n", 1)
    )
    lut = parse_cube(text)
    assert lut.size == 2
    assert len(lut.data) == 24
    # 第一个数据点 (R=0,G=0,B=0) 与最后一个 (1,1,1) 落位正确。
    assert lut.data[:3] == (0.0, 0.0, 0.0)
    assert lut.data[-3:] == (1.0, 1.0, 1.0)


def test_parse_keyword_case_insensitive():
    text = "lut_3d_size 2\n" + IDENTITY_2X2X2
    assert parse_cube(text).size == 2


def test_parse_title_without_quotes():
    text = 'LUT_3D_SIZE 2\nTITLE bare title\n' + IDENTITY_2X2X2
    assert parse_cube(text).title == "bare title"


def test_parse_missing_size_raises_with_location():
    with pytest.raises(LUTParseError) as ei:
        parse_cube("# only comment\n\n")
    assert ei.value.line_number is None
    assert "LUT_3D_SIZE" in str(ei.value)


def test_parse_duplicate_size_raises_with_line():
    text = "LUT_3D_SIZE 2\nLUT_3D_SIZE 3\n"
    with pytest.raises(LUTParseError) as ei:
        parse_cube(text)
    assert ei.value.line_number == 2
    assert "重复" in str(ei.value)


def test_parse_data_before_size_raises():
    text = "0 0 0\nLUT_3D_SIZE 2\n" + IDENTITY_2X2X2
    with pytest.raises(LUTParseError) as ei:
        parse_cube(text)
    assert ei.value.line_number == 1


def test_parse_wrong_data_count_raises():
    # 声明 size=3（需 27 点），只给 2 点。
    text = "LUT_3D_SIZE 3\n0 0 0\n1 0 0\n"
    with pytest.raises(LUTParseError) as ei:
        parse_cube(text)
    assert "27" in str(ei.value) and "2 个点" in str(ei.value)


def test_parse_non_numeric_token_raises_with_line():
    text = "LUT_3D_SIZE 2\n0 0 0\nfoo 0 0\n0 1 0\n0 0 1\n" + IDENTITY_2X2X2
    with pytest.raises(LUTParseError) as ei:
        parse_cube(text)
    assert ei.value.line_number == 3
    assert "foo" in str(ei.value)


def test_parse_nan_inf_rejected():
    bad = ("nan", "inf", "-inf", "1e999")
    for token in bad:
        text = "LUT_3D_SIZE 2\n0 0 0\n" + f"{token} 0 0\n" + IDENTITY_2X2X2
        with pytest.raises(LUTParseError):
            parse_cube(text)


def test_parse_unknown_keyword_line_raises_instead_of_silent_skip():
    text = "LUT_3D_SIZE 2\nWEIRD_KEYWORD 1 2 3\n" + IDENTITY_2X2X2
    with pytest.raises(LUTParseError):
        parse_cube(text)


def test_parse_1d_lut_rejected_not_forced():
    text = "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n"
    with pytest.raises(LUTParseError, match="1D"):
        parse_cube(text)


@pytest.mark.parametrize("bad_size", [1, 0, -3, 257, 1000])
def test_parse_size_out_of_range_rejected(bad_size):
    with pytest.raises(LUTParseError):
        parse_cube(f"LUT_3D_SIZE {bad_size}\n")


def test_parse_wrong_column_count_raises():
    text = "LUT_3D_SIZE 2\n0 0\n" + IDENTITY_2X2X2
    with pytest.raises(LUTParseError) as ei:
        parse_cube(text)
    assert ei.value.line_number == 2


# --------------------------------------------------------------- 域匹配


def test_default_domain_accepted():
    lut = parse_cube("LUT_3D_SIZE 2\n" + IDENTITY_2X2X2)
    assert lut.size == 2


def test_explicit_unit_domain_accepted():
    text = (
        "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n"
        + IDENTITY_2X2X2
    )
    assert parse_cube(text).size == 2


def test_non_unit_domain_min_rejected():
    text = (
        "LUT_3D_SIZE 2\nDOMAIN_MIN -0.05 -0.05 -0.05\n"
        + IDENTITY_2X2X2
    )
    with pytest.raises(LUTDomainError):
        parse_cube(text)


def test_non_unit_domain_max_rejected():
    text = (
        "LUT_3D_SIZE 2\nDOMAIN_MAX 1.2 1.2 1.2\n"
        + IDENTITY_2X2X2
    )
    with pytest.raises(LUTDomainError):
        parse_cube(text)


def test_domain_wrong_component_count_rejected():
    text = "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0\n" + IDENTITY_2X2X2
    with pytest.raises(LUTParseError):
        parse_cube(text)


# --------------------------------------------------------------- load_cube


def test_load_cube_reads_file(tmp_path):
    p = tmp_path / "id.cube"
    p.write_text(make_cube_text(3), encoding="utf-8")
    lut = load_cube(str(p))
    assert lut.size == 3
    assert lut.file_path == str(p)


def test_load_cube_missing_file_raises():
    with pytest.raises(LUTParseError):
        load_cube("definitely-not-here.cube")


def test_load_cube_utf8_bom(tmp_path):
    p = tmp_path / "bom.cube"
    p.write_bytes("﻿".encode("utf-8") + make_cube_text(2).encode("utf-8"))
    assert load_cube(str(p)).size == 2

def test_error_carries_file_path(tmp_path):
    p = tmp_path / "bad.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n", encoding="utf-8")
    with pytest.raises(LUTParseError) as ei:
        load_cube(str(p))
    assert ei.value.file_path == str(p)
    assert str(p) in str(ei.value)


# --------------------------------------------------------------- 三线性插值


def test_identity_lut_is_byte_exact_at_corners_and_interior():
    lut = identity_cube(8)
    for px in [(0, 0, 0), (255, 255, 255), (17, 99, 240), (128, 64, 32)]:
        assert lut.apply_pixel(*px) == px


def test_grid_corners_hit_stored_values_exactly():
    # 自定义 LUT：输出 = 输入的简单缩放，网格点处必须零插值误差。
    size = 5
    lut = parse_cube(make_cube_text(size, lambda r, g, b: (r * 0.5, g * 0.5, b * 0.5)))
    for gi in (0, size - 1):
        v = round(gi / (size - 1) * 255)
        got = lut.apply_pixel(v, v, v)
        expect = round((gi / (size - 1) * 0.5) * 255)
        assert got == (expect, expect, expect)


def test_trilinear_matches_manual_reference_on_random_pixels():
    """apply_pixel 与一个独立、直白的三线性参考实现逐像素一致。"""
    size = 6
    lut = parse_cube(make_cube_text(size, lambda r, g, b: (r * r, 1 - g, b * 0.7 + 0.1)))

    def reference(lut: CubeLUT, r: int, g: int, b: int):
        n = lut.size
        coords = []
        fracs = []
        for v in (r, g, b):
            f = v / 255 * (n - 1)
            lo = min(int(f), n - 2)
            coords.append((lo, lo + 1))
            fracs.append(f - lo)
        acc = []
        for ch in range(3):
            total = 0.0
            for bi in (0, 1):
                for gi in (0, 1):
                    for ri in (0, 1):
                        idx = lut.index_of(
                            coords[0][ri], coords[1][gi], coords[2][bi]
                        )
                        w = (
                            (fracs[0] if ri else 1 - fracs[0])
                            * (fracs[1] if gi else 1 - fracs[1])
                            * (fracs[2] if bi else 1 - fracs[2])
                        )
                        total += lut.data[idx + ch] * w
            q = math.floor(total * 255 + 0.5)
            acc.append(min(255, max(0, q)))
        return tuple(acc)

    for rv in (0, 1, 33, 127, 200, 255):
        for gv in (0, 45, 130, 255):
            for bv in (7, 88, 177, 254):
                assert lut.apply_pixel(rv, gv, bv) == reference(lut, rv, gv, bv)


def test_apply_pixel_and_apply_chunk_equivalent():
    """热路径 apply_chunk 与参考 apply_pixel 在所有像素上逐字节一致。"""
    lut = parse_cube(make_cube_text(5, lambda r, g, b: (b, r, g)))  # 通道轮换
    data = bytes((10, 200, 77, 0, 0, 0, 255, 255, 255, 123, 45, 67))
    out = apply_chunk(data, lut, 1.0)
    ref = bytearray()
    for i in range(0, len(data), 3):
        ref += bytes(lut.apply_pixel(data[i], data[i + 1], data[i + 2]))
    assert out == bytes(ref)


def test_apply_chunk_rejects_bad_length():
    lut = identity_cube(2)
    with pytest.raises(ValueError):
        apply_chunk(b"00", lut)


# --------------------------------------------------------------- 强度边界


def test_strength_zero_is_byte_exact_original():
    lut = parse_cube(make_cube_text(4, lambda r, g, b: (1 - r, 1 - g, 1 - b)))  # 强反色
    data = bytes((0, 100, 255, 12, 240, 33))
    assert apply_chunk(data, lut, 0.0) == data


def test_strength_one_equals_full_lut():
    lut = parse_cube(make_cube_text(4, lambda r, g, b: (1 - r, 1 - g, 1 - b)))
    data = bytes((0, 100, 255, 12, 240, 33, 255, 1, 128))
    assert apply_chunk(data, lut, 1.0) == apply_chunk(data, lut)


def test_strength_midpoint_is_byte_space_blend():
    lut = identity_cube(8)
    # 用一个确定的非恒等 LUT：每个输出固定偏亮 +0.25（钳到 1）。
    lut = parse_cube(make_cube_text(8, lambda r, g, b: (min(1, r + 0.25),) * 3))
    data = bytes((100, 100, 100))
    full = apply_chunk(data, lut, 1.0)
    half = apply_chunk(data, lut, 0.5)
    orig = 100
    expect = math.floor(orig + (full[0] - orig) * 0.5 + 0.5)
    assert half[0] == half[1] == half[2] == expect
    # 单调性：0 < 0.5 < 1 对该加亮 LUT 应严格不下降
    assert data[0] <= half[0] <= full[0]


def test_strength_deterministic_across_calls():
    lut = parse_cube(make_cube_text(4, lambda r, g, b: (b, g * 0.5, r)))
    data = bytes((i % 256 for i in range(3 * 50)))
    first = apply_chunk(data, lut, 0.37)
    for _ in range(5):
        assert apply_chunk(data, lut, 0.37) == first


def test_clamp_strength_bounds_and_messages():
    assert clamp_strength(0.5) == (0.5, [])
    with pytest.warns(UserWarning, match="strength"):
        v, msgs = clamp_strength(2.0)
    assert v == 1.0 and msgs and "钳制" in msgs[0]
    with pytest.warns(UserWarning):
        v, _ = clamp_strength(-1.0)
    assert v == 0.0


def test_clamp_strength_endpoints_silent():
    assert clamp_strength(0.0) == (0.0, [])
    assert clamp_strength(1.0) == (1.0, [])


def test_max_grid_size_constant():
    assert MAX_GRID_SIZE == 256
