"""自研 PNG 编解码测试（5 种滤波器、RGB/RGBA、损坏输入）。"""

import pytest

from watermark_overlay.errors import ImageDecodeError
from watermark_overlay.image import Image, read_png, write_png

from conftest import make_png


def test_write_then_read_roundtrip_rgba(tmp_path):
    img = Image(3, 2)
    for y in range(2):
        for x in range(3):
            img.set_pixel(x, y, x * 80, y * 120, x + y * 30, 128 + x)
    path = tmp_path / "rgba.png"
    write_png(str(path), img)
    decoded = read_png(str(path))
    assert (decoded.width, decoded.height) == (3, 2)
    assert bytes(decoded.pixels) == bytes(img.pixels)


@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
def test_decode_all_five_filters(tmp_path, filter_type):
    path = tmp_path / f"f{filter_type}.png"
    make_png(str(path), 5, 4, channels=4, filter_type=filter_type)
    img = read_png(str(path))
    assert (img.width, img.height) == (5, 4)
    for i in range(5 * 4):
        assert tuple(img.pixels[i * 4:i * 4 + 4]) == (10, 20, 30, 255)


def test_decode_rgb_png_adds_opaque_alpha(tmp_path):
    path = tmp_path / "rgb.png"
    make_png(str(path), 3, 3, channels=3, fill=(7, 8, 9))
    img = read_png(str(path))
    for i in range(9):
        assert tuple(img.pixels[i * 4:i * 4 + 4]) == (7, 8, 9, 255)


def test_decode_rejects_bad_signature(tmp_path):
    path = tmp_path / "notpng.png"
    path.write_bytes(b"this is definitely not a png file" * 3)
    with pytest.raises(ImageDecodeError, match="签名"):
        read_png(str(path))


def test_decode_rejects_bad_crc(tmp_path):
    path = tmp_path / "badcrc.png"
    make_png(str(path), 2, 2, bad_crc=True)
    with pytest.raises(ImageDecodeError, match="CRC"):
        read_png(str(path))


def test_decode_rejects_unsupported_bit_depth(tmp_path):
    path = tmp_path / "bit16.png"
    make_png(str(path), 2, 2, bit_depth=16)
    with pytest.raises(ImageDecodeError, match="位深"):
        read_png(str(path))


def test_decode_rejects_interlaced(tmp_path):
    path = tmp_path / "interlace.png"
    make_png(str(path), 2, 2, interlace=1)
    with pytest.raises(ImageDecodeError, match="隔行"):
        read_png(str(path))


def test_decode_rejects_unsupported_color_type(tmp_path):
    path = tmp_path / "gray.png"
    make_png(str(path), 2, 2, channels=1, color_type=0)
    with pytest.raises(ImageDecodeError, match="色彩类型"):
        read_png(str(path))


def test_decode_rejects_truncated_file(tmp_path):
    path = tmp_path / "trunc.png"
    make_png(str(path), 4, 4)
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises(ImageDecodeError):
        read_png(str(path))


def test_decode_missing_file_raises_decode_error(tmp_path):
    with pytest.raises(ImageDecodeError, match="无法读取"):
        read_png(str(tmp_path / "nope.png"))


def test_image_buffer_validation():
    with pytest.raises(ValueError):
        Image(2, 2, bytearray(10))  # 长度不匹配
    with pytest.raises(ValueError):
        Image(0, 2)
    with pytest.raises(TypeError):
        Image(2.0, 2)
