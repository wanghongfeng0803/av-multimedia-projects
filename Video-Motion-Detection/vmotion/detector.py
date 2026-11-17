"""运动检测核心（帧差模式）：分块帧差 + 局部自适应阈值 + 光照突变抑制。

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

本模式的参考是**上一帧整体滚动**；与背景建模模式
（:mod:`vmotion.background`，参考为逐块选择性更新的运行背景）共享
门控/形态学/滞后机制，公共实现见 :mod:`vmotion.detection_common`。

输出侧再经形态学开闭、连通域与面积/滞后过滤，最终只产出二值掩码与
:class:`~vmotion.morphology.Region` 列表——检测模块不做任何可视化与
落盘，标记/事件模块只消费这里的数据结构。
"""

from dataclasses import asdict, dataclass, field

from .detection_common import (
    GateConfig,
    Hysteresis,
    block_means,
    gate_block_mask,
    refine_mask,
    upsample_block_mask,
)

__all__ = [
    "MotionConfig",
    "DEFAULT_CONFIG",
    "FrameDetection",
    "MotionDetector",
]


@dataclass
class MotionConfig(GateConfig):
    """帧差模式检测参数（全部可配，阈值类参数均给保守默认值）。

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

    illum_change_threshold: float = 25.0
    illum_hold_frames: int = 2

    def validate(self):
        """参数合法性检查，非法时抛 :class:`ValueError`（不静默夹取）。"""
        self.validate_common()
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


class MotionDetector:
    """逐帧运动检测器（帧差模式），持有参考帧与滞后/光照抑制状态。

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
        self._hyst = Hysteresis(self.config.trigger_frames,
                                self.config.release_frames)
        # 光照抑制：剩余抑制帧数；>0 表示当前处于 holdoff
        self._hold_remaining = 0

    # 旧版本把去抖计数直接挂在检测器上，保留只读别名供既有测试/调试使用
    @property
    def _motion(self):
        return self._hyst.motion

    @property
    def _hit_run(self):
        return self._hyst.hit_run

    @property
    def _miss_run(self):
        return self._hyst.miss_run

    # ------------------------------------------------------------ 内部
    def _block_means(self, plane):
        """灰度平面 → 块均值平面（转发共享实现）。"""
        return block_means(plane, self.width, self.height,
                           self.config.block, self._bcols, self._brows)

    def _global_mean(self, plane):
        """整帧灰度均值（O(WH) 一次求和）。"""
        return sum(plane) / (self.width * self.height)

    def _gate_mask(self, cur_means, ref_means):
        """块级差分 + 局部 MAD 门控 + 上采样，返回 (全分辨率掩码, 统计)。"""
        cfg = self.config
        block_hit, stats = gate_block_mask(
            cur_means, ref_means, self._bcols, self._brows, cfg)
        mask = upsample_block_mask(
            block_hit, self.width, self.height,
            cfg.block, self._bcols, self._brows)
        return mask, stats

    def _refine(self, mask):
        """形态学清理 + 连通域 + 面积过滤（转发共享实现）。"""
        return refine_mask(mask, self.width, self.height, self.config)

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
            self._hyst.reset()
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
        self._hyst.update(hit)

        if self._hyst.motion and hit:
            mask, regions = refined_mask, regions
        else:
            # 去抖未确认或迟滞释放中：不对外出掩码，避免事件碎裂
            mask, regions = bytes(empty), []

        # 参考帧滚动更新（抑制帧已在上方提前返回，不会污染参考）
        self._reference = bytes(plane)
        return FrameDetection(
            frame=frame_index, pts_time=pts, mask=mask,
            regions=regions, motion=self._hyst.motion, stats=stats,
        )

    def config_dict(self):
        """导出当前参数 dict，写入事件 JSON 的 params 段。"""
        return asdict(self.config)
