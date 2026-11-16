"""像素格式自适应测试（纯逻辑）。"""

import random

import pytest

from vdenoise import (
    SUPPORTED_FORMATS,
    encoder_for,
    filter_dims,
    frame_size,
    join_planes,
    pix_fmt_matches,
    plane_layout,
    split_planes,
    working_format,
)


# ---------- working_format ----------

@pytest.mark.parametrize("src,expect", [
    ("yuv420p", "yuv420p"),
    ("yuvj420p", "yuv420p"),   # 全范围 YUV 归一
    ("yuv422p", "yuv422p"),
    ("yuv444p", "yuv444p"),
    ("gray", "gray"),
    ("gray8", "gray"),
    ("rgb24", "rgb24"),
    ("bgr24", "rgb24"),
    ("gbrp", "gbrp"),
    ("yuv420p10le", "yuv420p"),  # 不支持的位深回退
    ("nv12", "yuv420p"),         # 半平面格式回退
    ("xyz12le", "yuv420p"),
])
def test_working_format_mapping(src, expect):
    assert working_format(src) == expect


# ---------- encoder_for ----------

@pytest.mark.parametrize("fmt", ["rgb24", "gbrp"])
def test_rgb_formats_use_libx264rgb(fmt):
    codec, out_pix = encoder_for(fmt)
    assert codec == "libx264rgb"
    assert out_pix is None


@pytest.mark.parametrize("fmt", ["yuv420p", "yuv422p", "yuv444p"])
def test_yuv_formats_use_libx264(fmt):
    codec, out_pix = encoder_for(fmt)
    assert codec == "libx264"
    assert out_pix is None


def test_gray_converts_to_yuv420p_on_output():
    codec, out_pix = encoder_for("gray")
    assert codec == "libx264"
    assert out_pix == "yuv420p"


# ---------- pix_fmt_matches ----------

def test_rgb24_gbrp_equivalence():
    # H.264 RGB 编码产物被 ffprobe 探为 gbrp 而非 rgb24
    assert pix_fmt_matches("gbrp", "rgb24")
    assert pix_fmt_matches("rgb24", "gbrp")
    assert pix_fmt_matches("gbrp", "gbrp")


def test_pix_fmt_matches_exact_and_alias():
    assert pix_fmt_matches("yuv420p", "yuv420p")
    assert pix_fmt_matches("yuvj420p", "yuv420p")
    assert not pix_fmt_matches("yuv422p", "yuv420p")
    assert not pix_fmt_matches("gray", "yuv420p")


# ---------- 平面布局 ----------

def test_plane_layout_yuv420p_odd_dims():
    layout = plane_layout("yuv420p", 5, 3)
    assert layout == [(5, 3, 1), (3, 2, 1), (3, 2, 1)]


def test_frame_size_matches_layout():
    for fmt in SUPPORTED_FORMATS:
        layout = plane_layout(fmt, 7, 5)
        assert frame_size(fmt, 7, 5) == sum(w * h * b for w, h, b in layout)


def test_filter_dims_rgb24_three_full_planes():
    assert filter_dims("rgb24", 6, 4) == [(6, 4)] * 3


def test_filter_dims_yuv420p():
    assert filter_dims("yuv420p", 8, 4) == [(8, 4), (4, 2), (4, 2)]


# ---------- split/join 往返 ----------

@pytest.mark.parametrize("fmt,w,h", [
    ("gray", 7, 5),
    ("yuv420p", 8, 6),
    ("yuv420p", 7, 5),   # 奇数尺寸，色度向上取整
    ("yuv422p", 8, 6),
    ("yuv444p", 5, 5),
    ("gbrp", 6, 4),
    ("rgb24", 6, 4),
])
def test_split_join_roundtrip(fmt, w, h):
    rng = random.Random(hash((fmt, w, h)) & 0xFFFF)
    data = bytes(rng.randrange(256) for _ in range(frame_size(fmt, w, h)))
    planes, dims = split_planes(data, fmt, w, h)
    assert join_planes(planes, fmt, w, h) == data
    # 平面尺寸与 filter_dims 一致
    assert dims == filter_dims(fmt, w, h)


def test_split_rgb24_deinterleaves_channels():
    # 像素 (R,G,B) = (1,2,3), (4,5,6)
    data = bytes([1, 2, 3, 4, 5, 6])
    planes, dims = split_planes(data, "rgb24", 2, 1)
    assert planes == [bytes([1, 4]), bytes([2, 5]), bytes([3, 6])]
    assert dims == [(2, 1)] * 3
