"""背景建模检测模式：逐块运行背景 + 三时间尺度选择性更新。

帧差模式（:mod:`vmotion.detector`）的参考是上一帧，只能捕捉「帧间
变化沿」，物体停下后立刻与参考一致、既不会持续报警也无法回答「这块
区域现在算不算背景」。本模块把参考换成**逐块维护的运行背景模型**，
同时满足三个互相冲突的时间尺度目标：

1. **缓慢光照自适应**：被判为背景的块每帧以学习率 ``learning_rate``
   做指数滑动平均（EMA），背景随亮度漂移缓慢跟踪。慢到每帧只差几个
   灰度级的光照变化，块均值差始终越不过门限，块一直按背景更新，
   背景平滑跟过去且全程零误报；这正是固定参考帧做不到的。
2. **更新速度可配**：``learning_rate`` 即 EMA 系数 α，背景递推式
   ``B ← (1-α)B + αX``。α 越小跟踪越稳、抗瞬时扰动越强，α 越大
   适应越快；默认 0.05 给出约 20 帧量级的适应时间常数。
3. **前景停留足够久后吸收进背景**：每个块带一个「连续前景计数」。
   前景块默认**不更新**背景（移动物体不会污染模型），但连续被判
   前景达到 ``absorb_frames`` 帧后转入吸收态，开始以 α 向当前帧
   靠拢，背景追平后该块自动回到背景态——停下的车、放下的包裹最终
   成为新背景，不会无限期报警。

**初始化保护（预热期）**：背景模型未稳定前，任何差分都是「模型不准」
而非「真有运动」。前 ``warmup_frames`` 帧（不含首帧）只做精确累计
平均（``B = 前 n 帧的逐块均值``），不出掩码、不推进去抖状态机，
从源头杜绝开机阶段的大面积误报；预热结束后才切到 EMA 选择性更新。

与帧差模式共用块均值、局部 MAD 门控、形态学精炼和滞后状态机
（:mod:`vmotion.detection_common`），差异只在背景模型的维护方式；
光照突变 holdoff 语义保留且判定口径与帧差模式一致（帧间全局均值差，
相对上一帧而非背景模型），抑制期内对**全部块**做 EMA 更新，让模型
快速吸收跳变、避免解除后旧背景反复触发 holdoff。
"""

from dataclasses import asdict, dataclass

from .detection_common import (
    GateConfig,
    Hysteresis,
    block_means,
    gate_block_mask,
    refine_mask,
    upsample_block_mask,
)
from .detector import FrameDetection

__all__ = [
    "BackgroundConfig",
    "DEFAULT_BACKGROUND_CONFIG",
    "BackgroundDetector",
]


@dataclass
class BackgroundConfig(GateConfig):
    """背景建模模式参数。

    门控/形态学/滞后参数（block、local_radius、pixel_threshold、gain、
    min_area、open_radius、close_radius、trigger_frames、
    release_frames）语义与 :class:`~vmotion.detector.MotionConfig`
    完全一致；以下为背景建模专有参数。

    参数
    ----
    warmup_frames:
        预热帧数（首帧之后还需累计多少帧才完成建模）。预热期只做
        累计平均、不出掩码，防止背景未稳定时大量误报；须为非负整数，
        0 表示首帧即作为初始背景、第二帧起正常检测。
    learning_rate:
        背景 EMA 学习率 α，落在 (0, 1]。背景块每帧按
        ``B ← (1-α)B + αX`` 更新；吸收态块与光照抑制帧也用同一 α。
    absorb_frames:
        一个块连续被判前景多少帧后吸收进背景（转入 EMA 更新）。
        须为正整数；值越大越能容忍长时间停留的目标，越小越快承认
        「新常态」。
    illum_change_threshold:
        帧间全局均值差（0..255）超过该值判定为光照突变。
    illum_hold_frames:
        突变后强制抑制的帧数；期间再次出现突变则重新计时。
    """

    warmup_frames: int = 10
    learning_rate: float = 0.05
    absorb_frames: int = 60
    illum_change_threshold: float = 25.0
    illum_hold_frames: int = 2

    def validate(self):
        """参数合法性检查，非法时抛 :class:`ValueError`（不静默夹取）。"""
        self.validate_common()
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames 不能为负")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate 须落在 (0, 1]")
        if self.absorb_frames <= 0:
            raise ValueError("absorb_frames 必须为正")
        if not (0 <= self.illum_change_threshold <= 255):
            raise ValueError("illum_change_threshold 须落在 [0, 255]")
        if self.illum_hold_frames < 0:
            raise ValueError("illum_hold_frames 不能为负")
        return self


#: 背景建模默认参数
DEFAULT_BACKGROUND_CONFIG = BackgroundConfig()


class BackgroundDetector:
    """逐块背景建模检测器，持有运行背景、前景计数与滞后/抑制状态。

    用法与 :class:`~vmotion.detector.MotionDetector` 完全一致::

        det = BackgroundDetector(width, height, config)
        for i, plane in enumerate(gray_planes):
            result = det.update(plane, i, pts_time=i / fps)
            # result.mask / result.regions / result.motion

    内部状态
    --------
    _background:
        长度 bcols*brows 的 float 列表，逐块背景均值。
    _fg_runs:
        与块网格对齐的 int 列表，每个块连续被判前景的帧数；
        达到 ``absorb_frames`` 即进入吸收态（继续计数，块开始更新）。
    _seen:
        已喂入帧数（含首帧），用于预热期累计平均的归一化。
    """

    def __init__(self, width, height, config=None):
        self.width = int(width)
        self.height = int(height)
        self.config = config or DEFAULT_BACKGROUND_CONFIG
        self.config.validate()
        self._bcols = (self.width + self.config.block - 1) // self.config.block
        self._brows = (self.height + self.config.block - 1) // self.config.block
        nblocks = self._bcols * self._brows
        self._background = None          # 首帧后初始化为块均值列表
        self._fg_runs = [0] * nblocks
        self._seen = 0
        self._prev_global = None         # 上一帧全局灰度均值（holdoff 口径）
        self._hyst = Hysteresis(self.config.trigger_frames,
                                self.config.release_frames)
        self._hold_remaining = 0

    # ------------------------------------------------------------ 内部
    def _block_means(self, plane):
        return block_means(plane, self.width, self.height,
                           self.config.block, self._bcols, self._brows)

    def _global_mean(self, plane):
        return sum(plane) / (self.width * self.height)

    def _empty_detection(self, frame_index, pts, stats):
        """构造一帧空掩码检测结果（预热/抑制/未确认帧共用）。"""
        return FrameDetection(
            frame=frame_index, pts_time=pts,
            mask=bytes(self.width * self.height),
            regions=[], motion=False, stats=stats,
        )

    def _update_background_ema(self, cur_means, block_hit=None):
        """按选择性规则把当前块均值并入背景模型。

        ``block_hit`` 为 None 时（光照抑制帧需要全块快速吸收）对全部
        块做 EMA；否则前景块仅在吸收态更新，背景块每帧更新，同时
        维护连续前景计数。返回本帧新进入吸收态的块数。
        """
        alpha = self.config.learning_rate
        keep = 1.0 - alpha
        bg = self._background
        runs = self._fg_runs
        absorb = self.config.absorb_frames
        newly_absorbed = 0
        for idx, value in enumerate(cur_means):
            if block_hit is None:
                bg[idx] = keep * bg[idx] + alpha * value
                continue
            if block_hit[idx]:
                runs[idx] += 1
                if runs[idx] == absorb:
                    newly_absorbed += 1
                if runs[idx] >= absorb:
                    # 吸收态：承认「新常态」，背景开始向当前帧靠拢
                    bg[idx] = keep * bg[idx] + alpha * value
            else:
                # 瞬时前景已消失：计数清零，永不进入吸收
                runs[idx] = 0
                bg[idx] = keep * bg[idx] + alpha * value
        return newly_absorbed

    # ------------------------------------------------------------ 对外
    def update(self, plane, frame_index, pts_time=None):
        """喂入一帧灰度平面，返回 :class:`FrameDetection`。

        包装 :meth:`_update_impl` 只多做一件事：在每帧结束时记下当前
        全局均值供下一帧算 holdoff。光照突变的口径因此与帧差模式一致
        ——相对**上一帧**而非背景模型；否则学习率很小时背景追不上
        跳变，holdoff 会被旧背景无限重新触发。
        """
        result = self._update_impl(plane, frame_index, pts_time)
        self._prev_global = self._global_mean(plane)
        return result

    def _update_impl(self, plane, frame_index, pts_time=None):
        """帧阶段：首帧建模 → 预热累计平均 → holdoff → 门控+选择性更新。"""
        if len(plane) != self.width * self.height:
            raise ValueError(
                f"平面尺寸 {len(plane)} 与 {self.width}x{self.height} 不符"
            )
        pts = 0.0 if pts_time is None else float(pts_time)
        cfg = self.config
        cur = self._block_means(plane)

        # ---- 首帧：直接以块均值作为初始背景，无差分可做 ----
        if self._background is None:
            self._background = list(cur)
            self._seen = 1
            return self._empty_detection(frame_index, pts, {
                "raw_pixels": 0, "gated_pixels": 0, "mad": 0.0,
                "local_mad": 0.0, "mean_delta": 0.0,
                "held_off": False, "illum_level": 0.0,
                "warmup": True, "fg_blocks": 0, "absorbed_blocks": 0,
                "bg_mean": round(sum(self._background) / len(cur), 4),
            })

        mean_delta = self._global_mean(plane) - self._prev_global
        illum_level = min(1.0, abs(mean_delta) / 255.0)

        # ---- 预热期：精确累计平均 B_n = B_{n-1} + (X-B)/n ----
        if self._seen <= cfg.warmup_frames:
            n = self._seen + 1          # 含本帧后的样本数
            bg = self._background
            for idx, value in enumerate(cur):
                bg[idx] += (value - bg[idx]) / n
            self._seen = n
            # warmup 标志覆盖整个建模期（含首帧）：_seen 达到
            # warmup_frames+1 后下一帧才进入正常检测
            still_warmup = n <= cfg.warmup_frames + 1
            return self._empty_detection(frame_index, pts, {
                "raw_pixels": 0, "gated_pixels": 0, "mad": 0.0,
                "local_mad": 0.0,
                "mean_delta": round(mean_delta, 4),
                "held_off": False,
                "illum_level": round(illum_level, 4),
                "warmup": still_warmup, "fg_blocks": 0,
                "absorbed_blocks": 0,
                "bg_mean": round(sum(bg) / len(bg), 4),
            })

        # ---- 光照突变 holdoff（语义与帧差模式一致）----
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
            # 抑制期冻结检测输出与去抖；背景对全部块做 EMA，尽快吸收
            # 跳变后的新亮度，避免解除时旧背景再次触发 holdoff。
            self._hyst.reset()
            self._update_background_ema(cur, block_hit=None)
            for idx in range(len(self._fg_runs)):
                self._fg_runs[idx] = 0
            return self._empty_detection(frame_index, pts, {
                "raw_pixels": 0, "gated_pixels": 0, "mad": 0.0,
                "local_mad": 0.0,
                "mean_delta": round(mean_delta, 4),
                "held_off": True,
                "illum_level": round(illum_level, 4),
                "warmup": False, "fg_blocks": 0, "absorbed_blocks": 0,
                "bg_mean": round(
                    sum(self._background) / len(self._background), 4),
            })

        # ---- 正常检测：先对当前背景做门控，再选择性更新模型 ----
        block_hit, stats = gate_block_mask(
            cur, self._background, self._bcols, self._brows, cfg)
        newly_absorbed = self._update_background_ema(cur, block_hit)
        mask = upsample_block_mask(
            block_hit, self.width, self.height,
            cfg.block, self._bcols, self._brows)

        refined_mask, regions = refine_mask(
            mask, self.width, self.height, cfg)
        hit = len(regions) > 0
        self._hyst.update(hit)

        fg_blocks = sum(1 for r in self._fg_runs if r > 0)
        stats.update({
            "mean_delta": round(mean_delta, 4),
            "held_off": False,
            "illum_level": round(illum_level, 4),
            "warmup": False,
            "fg_blocks": fg_blocks,
            "absorbed_blocks": newly_absorbed,
            "bg_mean": round(
                sum(self._background) / len(self._background), 4),
        })

        if self._hyst.motion and hit:
            mask, regions = refined_mask, regions
        else:
            # 去抖未确认或迟滞释放中：不对外出掩码，避免事件碎裂
            mask, regions = bytes(self.width * self.height), []
        return FrameDetection(
            frame=frame_index, pts_time=pts, mask=mask,
            regions=regions, motion=self._hyst.motion, stats=stats,
        )

    def config_dict(self):
        """导出当前参数 dict，写入事件 JSON 的 params 段。"""
        return asdict(self.config)
