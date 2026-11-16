"""像素格式自适应与平面布局。

工作格式 = 输入格式（若受支持），否则回退 ``yuv420p``。
packed RGB（rgb24）在滤波前拆成 3 个单字节平面，滤完再交错回去。

RGB 相关三个 ffmpeg 坑（实测）：
1. libx264 只收 YUV 系像素格式，RGB 工作格式必须换 libx264rgb；
2. H.264 RGB 编码产物被 ffprobe 探为 ``gbrp`` 而非 ``rgb24``，
   校验输出时两种都要认（:func:`pix_fmt_matches`）；
3. rawvideo rgb24 封装进 avi/mkv 原生不支持，只有 nut 容器可行
   （测试输入用 ``.nut``）。

另：gray 无法直接进 libx264，输出时转 ``yuv420p``（补 128 色度）。
"""

__all__ = [
    "SUPPORTED_FORMATS",
    "plane_layout",
    "filter_dims",
    "frame_size",
    "working_format",
    "encoder_for",
    "pix_fmt_matches",
    "split_planes",
    "join_planes",
]

# 格式名 -> 存储平面布局 [(宽除数, 高除数, 每像素字节)]
_LAYOUTS = {
    "gray": [(1, 1, 1)],
    "yuv420p": [(1, 1, 1), (2, 2, 1), (2, 2, 1)],
    "yuv422p": [(1, 1, 1), (2, 1, 1), (2, 1, 1)],
    "yuv444p": [(1, 1, 1), (1, 1, 1), (1, 1, 1)],
    "gbrp": [(1, 1, 1), (1, 1, 1), (1, 1, 1)],
    "rgb24": [(1, 1, 3)],
}

#: 支持作为工作格式的像素格式
SUPPORTED_FORMATS = tuple(_LAYOUTS)

# ffprobe 报告名 -> 工作格式名
_ALIASES = {
    "gray8": "gray",
    "yuvj420p": "yuv420p",
    "yuvj422p": "yuv422p",
    "yuvj444p": "yuv444p",
    "bgr24": "rgb24",
}

# H.264 RGB 产物会被探为 gbrp，与 rgb24 视为等价
_RGB_EQUIV = frozenset(("rgb24", "gbrp"))


def _normalize(pix_fmt):
    return _ALIASES.get(pix_fmt, pix_fmt)


def _ceil_div(n, d):
    return -(-n // d)


def plane_layout(pix_fmt, width, height):
    """返回存储布局 ``[(平面宽, 平面高, 每像素字节), ...]``。"""
    fmt = _normalize(pix_fmt)
    layout = _LAYOUTS[fmt]
    return [(_ceil_div(width, wd), _ceil_div(height, hd), bpp)
            for wd, hd, bpp in layout]


def filter_dims(pix_fmt, width, height):
    """滤波用的平面尺寸列表：packed RGB 拆成 3 个 (width, height) 平面。"""
    fmt = _normalize(pix_fmt)
    layout = _LAYOUTS[fmt]
    if len(layout) == 1 and layout[0][2] > 1:
        return [(width, height)] * layout[0][2]
    return [(_ceil_div(width, wd), _ceil_div(height, hd))
            for wd, hd, _ in layout]


def frame_size(pix_fmt, width, height):
    """一帧原始字节数。"""
    return sum(pw * ph * bpp for pw, ph, bpp in plane_layout(pix_fmt, width, height))


def working_format(src_pix_fmt):
    """由 ffprobe 报告的像素格式推导工作格式；不支持的回退 yuv420p。"""
    fmt = _normalize(src_pix_fmt)
    if fmt in _LAYOUTS:
        return fmt
    return "yuv420p"


def encoder_for(work_fmt):
    """返回 ``(视频编码器, 输出像素格式或 None)``。

    RGB 工作格式换 libx264rgb（libx264 只收 YUV 系）；gray 无法直接
    进 libx264，输出转 yuv420p；其余原格式进 libx264。
    """
    if work_fmt in _RGB_EQUIV:
        return "libx264rgb", None
    if work_fmt == "gray":
        return "libx264", "yuv420p"
    return "libx264", None


def pix_fmt_matches(probed, expected):
    """ffprobe 探测格式与期望格式是否一致（rgb24/gbrp 视为等价）。"""
    a = _normalize(probed)
    b = _normalize(expected)
    if a == b:
        return True
    return a in _RGB_EQUIV and b in _RGB_EQUIV


def split_planes(data, pix_fmt, width, height):
    """把一帧原始字节拆成 ``(平面列表, [(平面宽, 平面高), ...])``。

    packed RGB 按通道拆分（``data[c::3]``），每个平面独立滤波。
    """
    fmt = _normalize(pix_fmt)
    layout = _LAYOUTS[fmt]
    if len(layout) == 1 and layout[0][2] > 1:
        bpp = layout[0][2]
        planes = [data[c::bpp] for c in range(bpp)]
        return planes, [(width, height)] * bpp
    planes = []
    dims = []
    off = 0
    for wd, hd, bpp in layout:
        pw = _ceil_div(width, wd)
        ph = _ceil_div(height, hd)
        n = pw * ph * bpp
        planes.append(data[off:off + n])
        dims.append((pw, ph))
        off += n
    return planes, dims


def join_planes(planes, pix_fmt, width, height):
    """把滤波后的平面列表拼回一帧原始字节（:func:`split_planes` 的逆）。"""
    fmt = _normalize(pix_fmt)
    layout = _LAYOUTS[fmt]
    if len(layout) == 1 and layout[0][2] > 1:
        bpp = layout[0][2]
        out = bytearray(width * height * bpp)
        for c in range(bpp):
            out[c::bpp] = planes[c]
        return bytes(out)
    return b"".join(planes)
