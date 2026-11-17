"""两种检测模式共享的底层机制：分块、局部 MAD 门控、形态学与滞后状态机。

:mod:`vmotion.detector`（旧帧差模式）与 :mod:`vmotion.background`
（背景建模模式）的差异只在「参考从哪来、怎么更新」：

- 帧差模式的参考是上一帧，每帧整体滚动；
- 背景模式的参考是逐块维护的运行背景，按前景/背景选择性更新。

而「块均值 → 局部 MAD 自适应门控 → 块掩码上采样 → 形态学精炼 →
连通域面积过滤 → 滞后状态机」这一整条后链路两者完全一致，统一收敛在
本模块，避免两套实现各自漂移。本模块只放无状态纯函数与不持有像素的
小工具，具体检测器负责自己的参考帧/背景模型生命周期。
"""

from dataclasses import dataclass

from .morphology import (
    close_mask,
    connected_components,
    open_mask,
)

__all__ = [
    "GateConfig",
    "integral_image",
    "window_sum",
    "block_means",
    "gate_block_mask",
    "upsample_block_mask",
    "refine_mask",
    "Hysteresis",
]


@dataclass
class GateConfig:
    """门控/精炼/滞后共用参数（两种检测模式配置的公共子集）。

    字段语义与 :class:`~vmotion.detector.MotionConfig` 同名参数一致，
    单独成类让背景配置与帧差配置可以各自演进，又能共用同一条后链路。
    """

    block: int = 8
    local_radius: int = 3
    pixel_threshold: float = 12.0
    gain: float = 3.0
    min_area: int = 64
    open_radius: int = 1
    close_radius: int = 1
    trigger_frames: int = 2
    release_frames: int = 3

    def validate_common(self):
        """公共参数合法性检查，非法时抛 :class:`ValueError`。"""
        if self.block <= 0:
            raise ValueError("block 必须为正整数")
        if self.local_radius < 0:
            raise ValueError("local_radius 不能为负")
        if not (0 <= self.pixel_threshold <= 255):
            raise ValueError("pixel_threshold 须落在 [0, 255]")
        if self.gain < 0:
            raise ValueError("gain 不能为负")
        if self.min_area <= 0:
            raise ValueError("min_area 必须为正")
        if self.open_radius < 0 or self.close_radius < 0:
            raise ValueError("形态学半径不能为负")
        if self.trigger_frames <= 0 or self.release_frames <= 0:
            raise ValueError("trigger_frames / release_frames 必须为正")
        return self


def integral_image(values, cols, rows):
    """一维紧凑值数组的积分图（Python int 无溢出风险）。

    返回长度 (cols+1)*(rows+1) 的列表，``I[r,c]`` 为左上 (0,0) 到
    (r-1,c-1) 的矩形和；首行首列恒 0，窗和可 O(1) 取得。
    """
    stride = cols + 1
    integ = [0] * (stride * (rows + 1))
    for r in range(rows):
        row_sum = 0
        src_base = r * cols
        dst_base = (r + 1) * stride
        prev_base = r * stride
        for c in range(cols):
            row_sum += values[src_base + c]
            integ[dst_base + c + 1] = integ[prev_base + c + 1] + row_sum
    return integ


def window_sum(integ, cols, rows, r, c, radius):
    """积分图上取以 (r,c) 为中心、半径 radius 的夹边窗和。"""
    r0 = max(0, r - radius)
    r1 = min(rows - 1, r + radius) + 1
    c0 = max(0, c - radius)
    c1 = min(cols - 1, c + radius) + 1
    stride = cols + 1
    return (integ[r1 * stride + c1] - integ[r0 * stride + c1]
            - integ[r1 * stride + c0] + integ[r0 * stride + c0])


def block_means(plane, width, height, block, bcols, brows):
    """灰度平面 → 块均值平面（float 列表，长度 bcols*brows）。

    两遍 O(WH) 实现，每个像素只访问一次：

    1. 逐行把像素累加到所在块列的行累加器（``row_sum[bx]``）；
    2. 每经过一个块的底行，把该块各行的累加器求和并除以块面积。

    边缘不满块按实际像素数归一。
    """
    means = [0.0] * (bcols * brows)
    row_sum = [0] * bcols
    for y in range(height):
        row_base = y * width
        bx = 0
        x = 0
        while x < width:
            x1 = min(width, x + block)
            row_sum[bx] += sum(plane[row_base + x:row_base + x1])
            bx += 1
            x = x1
        last_row_of_block = (y % block == block - 1) or (y == height - 1)
        if last_row_of_block:
            by = y // block
            y0 = by * block
            block_h = y - y0 + 1
            for c in range(bcols):
                x0 = c * block
                block_w = min(block, width - x0)
                means[by * bcols + c] = row_sum[c] / (block_h * block_w)
            row_sum = [0] * bcols
    return means


def gate_block_mask(cur_means, ref_means, bcols, brows, cfg):
    """块级绝对差 + 局部 MAD 门控，返回 ``(block_hit, stats)``。

    - ``block_hit``：长度 bcols*brows 的 bytearray，1 表示该块越过
      ``max(硬阈值, gain×局部MAD)`` 自适应门限；
    - ``stats``：``raw_pixels``（仅过硬阈值的块数）、``gated_pixels``
      （过自适应门限的块数）、``mad``（全块平均绝对差）、
      ``local_mad``（局部 MAD 的全帧均值）。

    局部 MAD 在差分块平面的积分图上取邻窗均值，O(块数) 完成。全局
    亮度变化在所有邻域同步抬高 MAD → 处处阈值抬高 → 整帧被门控；
    局部变化邻域外阈值不动，运动块仍可越限。
    """
    diffs = [abs(a - b) for a, b in zip(cur_means, ref_means)]
    integ = integral_image(diffs, bcols, brows)
    local_mad = [0.0] * len(diffs)
    for by in range(brows):
        for bx in range(bcols):
            r0 = max(0, by - cfg.local_radius)
            r1 = min(brows - 1, by + cfg.local_radius) + 1
            c0 = max(0, bx - cfg.local_radius)
            c1 = min(bcols - 1, bx + cfg.local_radius) + 1
            win = (r1 - r0) * (c1 - c0)
            local_mad[by * bcols + bx] = (
                window_sum(integ, bcols, brows, by, bx,
                           cfg.local_radius) / win
            )

    raw_pixels = 0
    gated_pixels = 0
    block_hit = bytearray(bcols * brows)
    for idx, d in enumerate(diffs):
        threshold = max(cfg.pixel_threshold, cfg.gain * local_mad[idx])
        if d > cfg.pixel_threshold:
            raw_pixels += 1
        if d > threshold:
            block_hit[idx] = 1
            gated_pixels += 1

    stats = {
        "raw_pixels": raw_pixels,
        "gated_pixels": gated_pixels,
        "mad": round(sum(diffs) / len(diffs), 4),
        "local_mad": round(sum(local_mad) / len(local_mad), 4),
    }
    return block_hit, stats


def upsample_block_mask(block_hit, width, height, block, bcols, brows):
    """块掩码上采样到全分辨率二值掩码（边缘块夹到画面尺寸）。"""
    mask = bytearray(width * height)
    for by in range(brows):
        y0 = by * block
        y1 = min(height, y0 + block)
        for bx in range(bcols):
            if not block_hit[by * bcols + bx]:
                continue
            x0 = bx * block
            x1 = min(width, x0 + block)
            strip = b"\x01" * (x1 - x0)
            for y in range(y0, y1):
                row_base = y * width
                mask[row_base + x0:row_base + x1] = strip
    return mask


def refine_mask(mask, width, height, cfg):
    """形态学清理 + 连通域 + 面积过滤，返回 ``(掩码 bytes, Region 列表)``。

    命中定义在精炼之后：门控命中但区域全部被 ``min_area`` 丢弃时，
    按存活标签重建掩码，保证对外掩码与区域严格一致、不产生空事件。
    """
    if cfg.open_radius > 0:
        mask = open_mask(mask, width, height, cfg.open_radius)
    if cfg.close_radius > 0:
        mask = close_mask(mask, width, height, cfg.close_radius)
    labels, regions = connected_components(mask, width, height)
    kept = [r for r in regions if r.area >= cfg.min_area]
    if len(kept) == len(regions):
        return bytes(mask), kept
    kept_labels = {r.label for r in kept}
    rebuilt = bytearray(width * height)
    for i, tag in enumerate(labels):
        if tag and tag in kept_labels:
            rebuilt[i] = 1
    return bytes(rebuilt), kept


class Hysteresis:
    """去抖/迟滞状态机：连续 trigger 帧命中置位，连续 release 帧未命中清位。

    与具体检测器解耦，只消费每帧「是否命中」布尔值；光照抑制等需要
    冻结状态的场景调用 :meth:`reset` 即可。
    """

    __slots__ = ("trigger_frames", "release_frames", "_hit_run",
                 "_miss_run", "_motion")

    def __init__(self, trigger_frames, release_frames):
        self.trigger_frames = trigger_frames
        self.release_frames = release_frames
        self._hit_run = 0
        self._miss_run = 0
        self._motion = False

    @property
    def motion(self):
        """当前对外运动态。"""
        return self._motion

    @property
    def hit_run(self):
        """当前连续命中计数（调试/测试口径）。"""
        return self._hit_run

    @property
    def miss_run(self):
        """当前连续未命中计数（调试/测试口径）。"""
        return self._miss_run

    def reset(self):
        """清空去抖计数并退回非运动态（光照抑制帧调用）。"""
        self._hit_run = 0
        self._miss_run = 0
        self._motion = False

    def update(self, hit):
        """喂入本帧命中布尔值，返回更新后的运动态。"""
        if hit:
            self._hit_run += 1
            self._miss_run = 0
        else:
            self._miss_run += 1
            self._hit_run = 0
        if not self._motion and self._hit_run >= self.trigger_frames:
            self._motion = True
        elif self._motion and self._miss_run >= self.release_frames:
            self._motion = False
        return self._motion
