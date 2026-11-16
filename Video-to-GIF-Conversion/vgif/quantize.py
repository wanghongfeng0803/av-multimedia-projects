"""颜色量化：纯 Python 中位切分（median-cut）全局调色板。

GIF 每帧最多 256 色调色板且条目数须为 2 的幂（3..256 之间取
2/4/8/…/256）。本模块只依赖标准库：

1. :func:`median_cut_palette` 对采样像素做中位切分——每轮沿 RGB 中
   跨度最大的通道在中位数处切箱，直到箱数达到目标色数（不足则保留
   实际箱数），箱内颜色均值即调色板条目；
2. :func:`map_frame_to_index` 用 RGB 三维立方缓存做最近邻映射，
   平滑素材命中率极高，未命中再全表欧氏距离比较；
3. :func:`quantize_frames` 串起多帧，共享一张全局调色板（GIF 全局
   颜色表），保证跨帧颜色一致、不闪色。

量化在缩放之后进行（RGB 路径默认），因此同一套抽帧产物可以用不同
尺寸/色深反复量化，无需重新解码源视频。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MAX_GIF_COLORS",
    "QuantizeError",
    "QuantizedFrame",
    "median_cut_palette",
    "map_frame_to_index",
    "quantize_frames",
    "next_pow2_colors",
]

#: GIF 全局颜色表上限。
MAX_GIF_COLORS = 256


class QuantizeError(ValueError):
    """量化参数或数据非法。"""


def next_pow2_colors(n: int) -> int:
    """把请求色数规整为 GIF 合法的 2 的幂（2..256）。

    中位切分实际产出可能少于请求数，最终写出时颜色表长度再向上取整
    到 2 的幂并以重复条目补齐（见 :mod:`vgif.gif_writer`）。
    """
    if not 1 <= n <= MAX_GIF_COLORS:
        raise QuantizeError(f"色数须在 1..{MAX_GIF_COLORS} 之间：{n}")
    p = 1
    while p < n:
        p <<= 1
    return max(2, p)


def _split_box(box: list) -> tuple:
    """沿最大跨度通道在中位数处把一个像素箱切成两箱。"""
    # 找跨度最大的通道。
    ranges = [
        max(p[c] for p in box) - min(p[c] for p in box) for c in range(3)
    ]
    channel = ranges.index(max(ranges))
    box.sort(key=lambda p: p[channel])
    mid = len(box) // 2
    return box[:mid], box[mid:]


def _box_mean(box: list) -> tuple:
    n = len(box)
    return tuple(
        int(sum(p[c] for p in box) / n + 0.5) for c in range(3)
    )


def median_cut_palette(pixels: list, max_colors: int) -> tuple:
    """中位切分生成全局调色板。

    :param pixels: ``[(r, g, b), ...]`` 采样像素列表（调用方负责抽样
        上限，避免超大素材的排序成本失控）。
    :param max_colors: 目标色数（1..256），实际条目可能更少（像素
        去重后颜色不足或箱无法再分）。
    :return: 调色板元组，每项为 ``(r, g, b)``，长度 ≥1。
    """
    if max_colors < 1 or max_colors > MAX_GIF_COLORS:
        raise QuantizeError(
            f"max_colors 须在 1..{MAX_GIF_COLORS}：{max_colors}"
        )
    if not pixels:
        raise QuantizeError("采样像素为空，无法生成调色板")

    # 去重既减少排序量，也让单颜色素材直接收敛。
    unique = list(set(pixels))
    if len(unique) <= max_colors:
        return tuple(sorted(unique))

    boxes = [unique]
    # 每轮切一个箱；空箱/单像素箱不可再分。
    while len(boxes) < max_colors:
        splittable = [i for i, b in enumerate(boxes) if len(b) >= 2]
        if not splittable:
            break
        # 优先切跨度最大的箱，减少颜色损失。
        def _spread(i):
            b = boxes[i]
            return max(
                max(p[c] for p in b) - min(p[c] for p in b)
                for c in range(3)
            )

        target = max(splittable, key=_spread)
        a, b = _split_box(boxes[target])
        boxes[target:target + 1] = [a, b]

    palette = tuple(_box_mean(b) for b in boxes if b)
    if not palette:  # 理论不可达，防御性处理。
        raise QuantizeError("中位切分未产出任何颜色")
    return palette


@dataclass(frozen=True)
class QuantizedFrame:
    """量化后的单帧：调色板下标 + 沿用的几何尺寸。

    :ivar indices: 长度 ``width*height`` 的索引字节。
    :ivar width: 帧宽。
    :ivar height: 帧高。
    """

    indices: bytes
    width: int
    height: int


def map_frame_to_index(frame: bytes, palette: tuple,
                       cache: dict | None = None) -> bytes:
    """把一帧 rgb24 映射为调色板索引（最近邻欧氏距离）。

    :param cache: 可选的 RGB→索引立方缓存（跨帧复用时由调用方持有
        同一个 dict）；平滑区域命中率高，可显著减少全表比较。
    """
    n_pix = len(frame) // 3
    out = bytearray(n_pix)
    if cache is None:
        cache = {}
    # 预乘平方，热点路径里少做乘法。
    pr = tuple(c[0] for c in palette)
    pg = tuple(c[1] for c in palette)
    pb = tuple(c[2] for c in palette)
    k = len(palette)
    for i in range(n_pix):
        r, g, b = frame[3 * i], frame[3 * i + 1], frame[3 * i + 2]
        key = (r, g, b)
        idx = cache.get(key)
        if idx is None:
            best = 0
            best_d = (
                (r - pr[0]) ** 2 + (g - pg[0]) ** 2 + (b - pb[0]) ** 2
            )
            for j in range(1, k):
                d = (
                    (r - pr[j]) ** 2
                    + (g - pg[j]) ** 2
                    + (b - pb[j]) ** 2
                )
                if d < best_d:
                    best_d, best = d, j
            idx = best
            cache[key] = idx
        out[i] = idx
    return bytes(out)


def quantize_frames(frames: list, width: int, height: int,
                    max_colors: int = MAX_GIF_COLORS,
                    sample_cap: int = 200_000) -> tuple:
    """对多帧 rgb24 字节生成共享调色板并逐帧量化。

    :param frames: rgb24 字节列表，每帧长度须为 ``width*height*3``。
    :param sample_cap: 生成调色板时的像素抽样上限（跨所有帧均匀
        跨步取样），防止超大素材中位切分排序过慢；映射阶段仍处理
        全部像素，不影响输出精度口径。
    :return: ``(palette, [QuantizedFrame, ...])``。
    """
    if not frames:
        raise QuantizeError("没有可量化的帧")
    frame_size = width * height * 3
    for i, f in enumerate(frames):
        if len(f) != frame_size:
            raise QuantizeError(
                f"第 {i} 帧长度 {len(f)} 与 {width}x{height} rgb24 不符"
            )

    total_pixels = len(frames) * width * height
    step = max(1, total_pixels // sample_cap)
    samples = []
    flat_stride = width * height
    for f in frames:
        for p in range(0, flat_stride, step):
            o = p * 3
            samples.append((f[o], f[o + 1], f[o + 2]))
    palette = median_cut_palette(samples, max_colors)
    # GIF 全局颜色表最多 256 条目，下标 0..255 恰可由单字节表示。
    cache: dict = {}
    qframes = [
        QuantizedFrame(
            indices=map_frame_to_index(f, palette, cache),
            width=width,
            height=height,
        )
        for f in frames
    ]
    return palette, qframes
