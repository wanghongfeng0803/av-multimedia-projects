"""运动检测核心：分块帧差 + 局部自适应阈值 + 光照突变抑制。

设计针对的核心矛盾是「光照突变要抑制，但不能用简单全局归一化一棒子
打死」。直接对整帧做亮度归一化（如减去全局均值差）会把车灯扫过、
阴影移动这类**局部光照变化**也一并抹平，而它们恰恰是需要检出的目标。
本模块因此不做任何全局归一化，改用三道闸门：

1. **分块帧差**：先把亮度平面压成 ``block×block`` 的块均值平面再做
   绝对差。块均值等价于一个箱式低通，先压传感器噪声；块级局部统计
   也让「全局/局部」的区分有了空间尺度。
2. **局部自适应阈值（MAD 门控）**：每个像素的判定阈值不是全局常数，
   而是 ``max(绝对阈值, gain × 局部 MAD)``，局部 MAD 在以该块为中心的
   ``local_radius`` 块邻域内计算。全局亮度变化在所有邻域上同步抬高
   MAD → 处处阈值抬高 → 整帧被门控滤掉；车灯、阴影只影响局部邻域，
   邻域外阈值不动，运动物体仍可越过阈值检出。
3. **光照突变 holdoff**：帧间全局均值变化超过
   ``illum_change_threshold``（按 0..255 亮度量纲）判定为突变帧，
   该帧及随后 ``illum_hold_frames-1`` 帧不出掩码、冻结去抖状态；
   抑制期内再次突变则重新计时。参考帧在抑制期仍逐帧滚动（否则旧参考
   会让 holdoff 无限重触发），冻结的只是检测输出，避免突变后连锁误报。

输出侧再经形态学开闭、连通域与面积/滞后过滤，最终只产出二值掩码与
:class:`~vmotion.morphology.Region` 列表——检测模块不做任何可视化与
落盘，标记/事件模块只消费这里的数据结构。
"""

from dataclasses import asdict, dataclass, field

from .morphology import (
    close_mask,
    connected_components,
    open_mask,
)

__all__ = [
    "MotionConfig",
    "DEFAULT_CONFIG",
    "FrameDetection",
    "MotionDetector",
]


@dataclass
class MotionConfig:
    """检测参数（全部可配，阈值类参数均给保守默认值）。

    参数
    ----
    block:
        分块边长（像素）。块越大抗噪越强但定位越粗，须为正整数。
    local_radius:
        局部 MAD 邻域半径，单位是「块」；邻域边长 2r+1 块。
    pixel_threshold:
        块均值绝对差的硬阈值（0..255），任何自适应结果都不会低于它，
        防止静态画面局部 MAD≈0 时阈值塌成 0。
    gain:
        局部 MAD 增益系数，越大越保守（对全局波动越不敏感）。
    min_area:
        连通域最小前景像素数（全分辨率口径），低于此值的区域丢弃。
    open_radius, close_radius:
        形态学开/闭运算结构元半径（全分辨率像素）；0 表示跳过该步。
    trigger_frames:
        连续命中多少帧才确认运动（进入运动态的去抖帧数）。
    release_frames:
        连续多少帧无命中才退出运动态（迟滞释放，避免目标短暂停顿
        造成事件碎裂）。
    illum_change_threshold:
        帧间全局均值差（0..255）超过该值判定为光照突变。
    illum_hold_frames:
        突变后强制抑制的帧数；期间再次出现突变则重新计时。
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
    illum_change_threshold: float = 25.0
    illum_hold_frames: int = 2

    def validate(self):
        """参数合法性检查，非法时抛 :class:`ValueError`（不静默夹取）。"""
        if self.block <= 0:
            raise ValueError("block 必须为正整数")
        if self.local_radius < 0:
            raise ValueError("local_radius 不能为负")
        if self.pixel_threshold < 0 or self.pixel_threshold > 255:
            raise ValueError("pixel_threshold 须落在 [0, 255]")
        if self.gain < 0:
            raise ValueError("gain 不能为负")
        if self.min_area <= 0:
            raise ValueError("min_area 必须为正")
        if self.open_radius < 0 or self.close_radius < 0:
            raise ValueError("形态学半径不能为负")
        if self.trigger_frames <= 0 or self.release_frames <= 0:
            raise ValueError("trigger_frames / release_frames 必须为正")
        if not (0 <= self.illum_change_threshold <= 255):
            raise ValueError("illum_change_threshold 须落在 [0, 255]")
        if self.illum_hold_frames < 0:
            raise ValueError("illum_hold_frames 不能为负")
        return self


#: 默认参数（dataclass 字段的保守默认，单独导出供 CLI 帮助文本引用）
DEFAULT_CONFIG = MotionConfig()


@dataclass
class FrameDetection:
    """单帧检测结果：掩码 + 区域 + 调试统计，不含任何呈现逻辑。

    属性
    ----
    frame:
        帧序号，从 0 起、与解码顺序对齐。
    pts_time:
        帧显示时间（秒），由调用方按帧率填写。
    mask:
        全分辨率二值掩码（紧凑 bytes，非零即运动），与帧同尺寸。
    regions:
        本帧存活的 :class:`~vmotion.morphology.Region` 列表，
        标签在单帧内从 1 起编号（不做跨帧追踪，追踪留给上层）。
    motion:
        滞后状态机判定的本帧运动态（去抖/迟滞后的对外状态）。
    stats:
        调试统计：``raw_pixels`` 门控前命中数、``gated_pixels``
        门控后命中数、``mad`` 全帧均值差、``local_mad`` 局部 MAD
        均值、``mean_delta`` 全局均值变化（带符号）、``held_off``
        本帧是否处于光照抑制、``illum_level`` 突变强度 0..1。
    """

    frame: int
    pts_time: float
    mask: bytes
    regions: list = field(default_factory=list)
    motion: bool = False
    stats: dict = field(default_factory=dict)


def _integral_image(values, cols, rows):
    """一维紧凑值数组的积分图（uint64 口径，Python int 无溢出风险）。

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


def _window_sum(integ, cols, rows, r, c, radius):
    """积分图上取以 (r,c) 为中心、半径 radius 的夹边窗和。"""
    r0 = max(0, r - radius)
    r1 = min(rows - 1, r + radius) + 1
    c0 = max(0, c - radius)
    c1 = min(cols - 1, c + radius) + 1
    stride = cols + 1
    return (integ[r1 * stride + c1] - integ[r0 * stride + c1]
            - integ[r1 * stride + c0] + integ[r0 * stride + c0])


class MotionDetector:
    """逐帧运动检测器，持有参考帧与滞后/光照抑制状态。

    用法::

        det = MotionDetector(width, height, config)
        for i, plane in enumerate(gray_planes):
            result = det.update(plane, i, pts_time=i / fps)
            # result.mask / result.regions / result.motion

    检测器只认灰度紧凑平面（``gray`` rawvideo），不接触 ffmpeg；
    参考帧逐帧滚动更新（含光照抑制帧，抑制帧只冻结检测输出与去抖
    状态，见 :meth:`update` 与模块 docstring 的三闸门说明）。
    """

    def __init__(self, width, height, config=None):
        self.width = int(width)
        self.height = int(height)
        self.config = (config or DEFAULT_CONFIG)
        self.config.validate()
        self._reference = None          # 上一有效帧的灰度平面
        # 分块网格尺寸（ceil 取整，边缘块为不满块）
        self._bcols = (self.width + self.config.block - 1) // self.config.block
        self._brows = (self.height + self.config.block - 1) // self.config.block
        # 滞后状态机
        self._hit_run = 0
        self._miss_run = 0
        self._motion = False
        # 光照抑制：剩余抑制帧数；>0 表示当前处于 holdoff
        self._hold_remaining = 0

    # ------------------------------------------------------------ 内部
    def _block_means(self, plane):
        """灰度平面 → 块均值平面（float 列表，长度 bcols*brows）。

        两遍 O(WH) 实现，每个像素只访问一次：

        1. 逐行把像素累加到所在块列的行累加器（``row_sum[bx]``）；
        2. 每经过一个块的底行，把该块各行的累加器求和并除以块面积。

        边缘不满块按实际像素数归一，结果与嵌套窗口求和逐值一致。
        """
        b = self.config.block
        means = [0.0] * (self._bcols * self._brows)
        row_sum = [0] * self._bcols
        for y in range(self.height):
            row_base = y * self.width
            bx = 0
            x = 0
            while x < self.width:
                x1 = min(self.width, x + b)
                row_sum[bx] += sum(plane[row_base + x:row_base + x1])
                bx += 1
                x = x1
            last_row_of_block = (y % b == b - 1) or (y == self.height - 1)
            if last_row_of_block:
                by = y // b
                y0 = by * b
                block_h = y - y0 + 1
                for c in range(self._bcols):
                    x0 = c * b
                    block_w = min(b, self.width - x0)
                    means[by * self._bcols + c] = (
                        row_sum[c] / (block_h * block_w)
                    )
                row_sum = [0] * self._bcols
        return means

    def _global_mean(self, plane):
        """整帧灰度均值（O(WH) 一次求和）。"""
        return sum(plane) / (self.width * self.height)

    def _gate_mask(self, cur_means, ref_means):
        """块级差分 + 局部 MAD 门控，返回 (全分辨率掩码, 统计 dict)。"""
        cfg = self.config
        diffs = [abs(a - b) for a, b in zip(cur_means, ref_means)]
        # 局部 MAD：差分块平面的积分图 → 邻窗均值（O(块数)）
        integ = _integral_image(diffs, self._bcols, self._brows)
        local_mad = [0.0] * len(diffs)
        for by in range(self._brows):
            for bx in range(self._bcols):
                r0 = max(0, by - cfg.local_radius)
                r1 = min(self._brows - 1, by + cfg.local_radius) + 1
                c0 = max(0, bx - cfg.local_radius)
                c1 = min(self._bcols - 1, bx + cfg.local_radius) + 1
                win = (r1 - r0) * (c1 - c0)
                local_mad[by * self._bcols + bx] = (
                    _window_sum(integ, self._bcols, self._brows,
                                by, bx, cfg.local_radius) / win
                )

        raw_pixels = 0
        gated_pixels = 0
        block_hit = bytearray(self._bcols * self._brows)
        for idx, d in enumerate(diffs):
            threshold = max(cfg.pixel_threshold,
                            cfg.gain * local_mad[idx])
            if d > cfg.pixel_threshold:
                raw_pixels += 1
            if d > threshold:
                block_hit[idx] = 1
                gated_pixels += 1

        # 块掩码上采样到全分辨率（边缘块夹到画面尺寸），整段切片填充
        mask = bytearray(self.width * self.height)
        b = cfg.block
        for by in range(self._brows):
            y0 = by * b
            y1 = min(self.height, y0 + b)
            for bx in range(self._bcols):
                if not block_hit[by * self._bcols + bx]:
                    continue
                x0 = bx * b
                x1 = min(self.width, x0 + b)
                strip = b"\x01" * (x1 - x0)
                for y in range(y0, y1):
                    row_base = y * self.width
                    mask[row_base + x0:row_base + x1] = strip

        stats = {
            "raw_pixels": raw_pixels,
            "gated_pixels": gated_pixels,
            "mad": round(sum(diffs) / len(diffs), 4),
            "local_mad": round(sum(local_mad) / len(local_mad), 4),
        }
        return mask, stats

    def _refine(self, mask):
        """形态学清理 + 连通域 + 面积过滤，返回 (掩码, Region 列表)。"""
        cfg = self.config
        if cfg.open_radius > 0:
            mask = open_mask(mask, self.width, self.height, cfg.open_radius)
        if cfg.close_radius > 0:
            mask = close_mask(mask, self.width, self.height, cfg.close_radius)
        labels, regions = connected_components(mask, self.width, self.height)
        kept = [r for r in regions if r.area >= cfg.min_area]
        if len(kept) == len(regions):
            return bytes(mask), kept
        # 有区域被面积门槛丢弃：按存活标签重建掩码，保持两者严格一致
        kept_labels = {r.label for r in kept}
        rebuilt = bytearray(self.width * self.height)
        for i, tag in enumerate(labels):
            if tag and tag in kept_labels:
                rebuilt[i] = 1
        return bytes(rebuilt), kept

    # ------------------------------------------------------------ 对外
    def update(self, plane, frame_index, pts_time=None):
        """喂入一帧灰度平面，返回 :class:`FrameDetection`。

        首帧只建参考、不出运动；光照突变帧及 holdoff 内的帧返回空掩码。
        抑制期冻结的是「检测输出」与去抖状态，参考帧仍滚动到当前帧——
        否则下一帧相对突变前的旧参考会再次算出跳变，holdoff 被无限
        重新触发（详见 holdoff 分支内注释）。
        """
        if len(plane) != self.width * self.height:
            raise ValueError(
                f"平面尺寸 {len(plane)} 与 {self.width}x{self.height} 不符"
            )
        pts = 0.0 if pts_time is None else float(pts_time)
        empty = bytearray(self.width * self.height)

        if self._reference is None:
            self._reference = bytes(plane)
            return FrameDetection(
                frame=frame_index, pts_time=pts, mask=bytes(empty),
                regions=[], motion=False,
                stats={"raw_pixels": 0, "gated_pixels": 0, "mad": 0.0,
                       "local_mad": 0.0, "mean_delta": 0.0,
                       "held_off": False, "illum_level": 0.0},
            )

        cfg = self.config
        mean_delta = self._global_mean(plane) - self._global_mean(self._reference)
        illum_level = min(1.0, abs(mean_delta) / 255.0)

        # 闸门三：光照突变检测与 holdoff 计时。
        # 跳变帧本身即第 1 个抑制帧，装填 hold_frames-1 留给后续帧；
        # 抑制期内再次跳变则按新跳变重新装填（重新计时）。
        is_jump = abs(mean_delta) > cfg.illum_change_threshold
        if is_jump:
            self._hold_remaining = max(0, cfg.illum_hold_frames - 1)
            held = True
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1
            held = True
        else:
            held = False
        if held:
            # 抑制期不出掩码、清空去抖计数；参考帧仍需滚动到当前帧，
            # 否则下一帧相对突变前的旧参考会再次算出跳变，holdoff 被
            # 无限重新触发。冻结的是「检测输出」，不是参考帧本身。
            self._hit_run = 0
            self._miss_run = 0
            self._motion = False
            self._reference = bytes(plane)
            return FrameDetection(
                frame=frame_index, pts_time=pts, mask=bytes(empty),
                regions=[], motion=False,
                stats={"raw_pixels": 0, "gated_pixels": 0,
                       "mad": 0.0, "local_mad": 0.0,
                       "mean_delta": round(mean_delta, 4),
                       "held_off": True,
                       "illum_level": round(illum_level, 4)},
            )

        mask, stats = self._gate_mask(
            self._block_means(plane), self._block_means(self._reference)
        )
        stats["mean_delta"] = round(mean_delta, 4)
        stats["held_off"] = False
        stats["illum_level"] = round(illum_level, 4)

        # 先精炼，再定义「命中」：门控命中但区域全部被面积门槛丢弃时
        # 不算命中，保证运动态与对外掩码/区域严格一致，不产生空事件
        refined_mask, regions = self._refine(mask)
        hit = len(regions) > 0
        if hit:
            self._hit_run += 1
            self._miss_run = 0
        else:
            self._miss_run += 1
            self._hit_run = 0
        # 滞后状态机：连续 trigger 帧命中才置位，连续 release 帧无命中才清位
        if not self._motion and self._hit_run >= cfg.trigger_frames:
            self._motion = True
        elif self._motion and self._miss_run >= cfg.release_frames:
            self._motion = False

        if self._motion and hit:
            mask, regions = refined_mask, regions
        else:
            # 去抖未确认或迟滞释放中：不对外出掩码，避免事件碎裂
            mask, regions = bytes(empty), []

        # 参考帧滚动更新（抑制帧已在上方提前返回，不会污染参考）
        self._reference = bytes(plane)
        return FrameDetection(
            frame=frame_index, pts_time=pts, mask=mask,
            regions=regions, motion=self._motion, stats=stats,
        )

    def config_dict(self):
        """导出当前参数 dict，写入事件 JSON 的 params 段。"""
        return asdict(self.config)
