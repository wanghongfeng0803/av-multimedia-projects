"""单目标模板追踪状态机：追踪 → 丢失 → 重搜确认 → 恢复。

状态（内部三态，对外只暴露 tracked/lost 两态，见 :mod:`vtrack.model`）：

- ``TRACKED``：上一帧确认在跟。每帧只在最后已知位置周围的局部窗
  （边长 = 模板边 + 2·search_margin）内匹配，分数达标才移动框；
- ``LOST``：分数跌破阈值。立即上报丢失，框冻结在最后已知位置，
  之后每帧把搜索区按 ``lost_growth`` 复利扩大，直至铺满全帧——
  目标可能在遮挡期间移动到任意位置；
- ``REACQUIRE``：丢失后首次搜到达标候选。不立即采信（防误匹配
  「乱跟」），要求连续 ``reacquire_frames`` 帧在同一锚点附近且
  逐帧位移合理，全部达标才恢复 TRACKED；任何一帧不达标退回 LOST，
  重确认期间对外状态仍为 ``lost``。

本模块只产出逐帧 :class:`~vtrack.model.FrameRecord` 与状态转换时刻
（:attr:`TemplateTracker.loss_started` / :attr:`recovery_frame`），
丢失区间的 :class:`~vtrack.model.LossEvent` 由 pipeline 用真实 pts
组装，追踪器不持有时间信息，保持纯逻辑可单测。

防漂移：初始模板（anchor）全程保留，丢失重搜与重确认一律用 anchor；
可选的自适应更新（``update_template=True``，默认关闭）只在 TRACKED 且
分数优于 ``update_score`` 时对工作模板做凸组合，恢复时工作模板重置
回 anchor，避免错误模板被一路带到无法找回。
"""

from .model import LOST, TRACKED
from .template import (
    DEFAULT_THRESHOLDS,
    METHOD_SAD,
    TEMPLATE_METHODS,
    best_match,
    best_match_zncc_coarse_fine,
    extract_patch,
    higher_is_better,
    score_accepted,
)

__all__ = ["TemplateTracker", "STATE_TRACKED", "STATE_LOST", "STATE_REACQUIRE"]

#: 内部状态
STATE_TRACKED = "tracked"
STATE_LOST = "lost"
STATE_REACQUIRE = "reacquire"

#: 丢失后搜索区复利增长因子、重确认帧锚点最大位移
_DEFAULT_GROWTH = 1.5
_REACQUIRE_JUMP = 8
#: 自适应模板更新的默认混合系数
_DEFAULT_ALPHA = 0.2


class TemplateTracker:
    """逐帧驱动的单目标模板追踪状态机。

    参数
    ----
    width, height:
        帧平面尺寸。
    template:
        初始模板字节（tw×th 紧凑排列），同时作为永久 anchor。
    tw, th:
        模板宽高。
    init_x, init_y:
        初始模板在第 0 帧的左上角坐标。
    method:
        匹配方法，``sad`` / ``ssd`` / ``zncc``。
    threshold:
        接受阈值，方向由方法决定；缺省取
        :data:`vtrack.template.DEFAULT_THRESHOLDS`。
    search_margin:
        TRACKED 状态局部搜索区相对最后已知框外扩的像素边距。
    lost_growth:
        LOST 状态每帧搜索区边长的复利增长因子（>=1），铺满全帧后
        保持全帧搜索。
    reacquire_frames:
        重确认需要连续达标的帧数。
    reacquire_threshold:
        重确认专用阈值（缺省沿用主阈值）；可设得更严格以减少误恢复。
    update_template:
        是否启用工作模板自适应更新（默认关闭，纯 anchor 追踪）。
    update_alpha:
        启用更新时新模板的凸组合系数（0..1）。
    update_score:
        只有分数优于该值才更新模板，缺省取主阈值（即「达标才更新」）。
    """

    def __init__(self, width, height, template, tw, th, init_x, init_y,
                 method=METHOD_SAD, threshold=None, search_margin=12,
                 lost_growth=_DEFAULT_GROWTH,
                 reacquire_frames=3, reacquire_threshold=None,
                 update_template=False, update_alpha=_DEFAULT_ALPHA,
                 update_score=None):
        if method not in TEMPLATE_METHODS:
            raise ValueError(
                f"未知匹配方法 {method!r}，可选：{', '.join(TEMPLATE_METHODS)}"
            )
        if tw <= 0 or th <= 0 or tw > width or th > height:
            raise ValueError("模板尺寸非法或大于帧")
        if not (0 <= init_x <= width - tw and 0 <= init_y <= height - th):
            raise ValueError("初始框超出帧边界")
        if search_margin < 0:
            raise ValueError("search_margin 不能为负")
        if lost_growth < 1.0:
            raise ValueError("lost_growth 必须 >= 1")
        if reacquire_frames < 1:
            raise ValueError("reacquire_frames 必须 >= 1")
        if not 0.0 < update_alpha <= 1.0:
            raise ValueError("update_alpha 必须在 (0, 1]")

        self.width = width
        self.height = height
        self.tw = tw
        self.th = th
        self.method = method
        self.threshold = (
            DEFAULT_THRESHOLDS[method] if threshold is None else threshold
        )
        self.reacquire_threshold = (
            self.threshold if reacquire_threshold is None else reacquire_threshold
        )
        self.search_margin = search_margin
        self.lost_growth = lost_growth
        self.reacquire_frames = reacquire_frames
        self.update_template = update_template
        self.update_alpha = update_alpha
        self.update_score = (
            self.threshold if update_score is None else update_score
        )

        # anchor 永不改写；work 仅在启用自适应时变化
        self.anchor = bytes(template)
        self.work = bytes(template)

        # 最后已知确认位置与内部状态
        self.last_x = init_x
        self.last_y = init_y
        self.state = STATE_TRACKED
        self.lost_frames = 0          # 连续丢失计数（驱动搜索区扩大）
        self.loss_started = None      # 当前丢失区间的起始帧号，未丢失为 None
        self.recovery_frame = None    # 最近一次恢复确认的帧号，消费后清零
        self._reacquire_count = 0
        self._anchor_pos = None       # 重确认首帧的候选位置

    # ------------------------------------------------------------ 搜索区
    def local_search(self, x, y):
        """TRACKED 局部搜索区：最后已知框外扩 search_margin 后夹边界。"""
        m = self.search_margin
        return (x - m, y - m, x + m, y + m)

    def lost_search(self):
        """LOST 扩大搜索区：以最后已知框中心为中心，边长复利增长。

        第 k 次丢失（k 从 1 起）搜索边距取
        ``max(m0·k, m0·growth**(k-1))``：线性项保证小 margin、少帧数
        场景下搜索区每帧至少实打实外扩一个 m0，复利项在长丢失段加速
        扩大；搜索区一旦覆盖全帧即直接铺满，避免在大整数上空转。
        """
        m0 = self.search_margin
        k = self.lost_frames
        margin = max(m0 * k,
                     int(round(m0 * (self.lost_growth ** (k - 1)))))
        cx = self.last_x + self.tw // 2
        cy = self.last_y + self.th // 2
        half_w = self.tw // 2 + margin
        half_h = self.th // 2 + margin
        x0 = cx - half_w
        y0 = cy - half_h
        x1 = cx + half_w - self.tw
        y1 = cy + half_h - self.th
        if x0 <= 0 and y0 <= 0 and x1 >= self.width - self.tw \
                and y1 >= self.height - self.th:
            return (0, 0, self.width - self.tw, self.height - self.th)
        return (x0, y0, x1, y1)

    # ------------------------------------------------------------ 驱动
    def update(self, frame, frame_index, pts_time=0.0):
        """喂入下一帧亮度平面，返回本帧 :class:`~vtrack.model.FrameRecord`。

        首帧（``frame_index=0``）不做搜索，直接确认初始框，分数记为
        该方法的理想值（SAD/SSD=0，ZNCC=1），保证轨迹从第 0 帧对齐。
        """
        from .model import FrameRecord

        if frame_index == 0:
            return FrameRecord(
                frame=0, pts_time=pts_time, status=TRACKED,
                bbox=(self.last_x, self.last_y, self.tw, self.th),
                score=_ideal_score(self.method), confirmed=True,
                search=(self.last_x, self.last_y, self.last_x, self.last_y),
            )
        if self.state == STATE_TRACKED:
            return self._update_tracked(frame, frame_index, pts_time)
        if self.state == STATE_LOST:
            return self._update_lost(frame, frame_index, pts_time)
        return self._update_reacquire(frame, frame_index, pts_time)

    def take_recovery(self):
        """取走最近一次恢复帧号（无恢复返回 None），供 pipeline 闭合事件。"""
        frame = self.recovery_frame
        self.recovery_frame = None
        return frame

    # ------------------------------------------------------------ 三态
    def _update_tracked(self, frame, idx, pts):
        from .model import FrameRecord

        search = self.local_search(self.last_x, self.last_y)
        x, y, score, used = self._match(frame, self.work, search)
        if score_accepted(self.method, score, self.threshold):
            self.last_x, self.last_y = x, y
            self._maybe_update_template(frame, x, y, score)
            return FrameRecord(
                frame=idx, pts_time=pts, status=TRACKED,
                bbox=(x, y, self.tw, self.th), score=score,
                confirmed=True, search=used,
            )
        # 分数不达标：冻结框、进入丢失，不移动到任何候选位置（不乱跟）。
        # 计数在进入时即 +1：本帧局部搜索已失败，下一帧 lost_search()
        # 必须立刻拿到扩大后的搜索区，而不是再等一帧。
        self.state = STATE_LOST
        self.lost_frames = 1
        self.loss_started = idx
        return FrameRecord(
            frame=idx, pts_time=pts, status=LOST,
            bbox=(self.last_x, self.last_y, self.tw, self.th),
            score=score, confirmed=False, search=used,
        )

    def _update_lost(self, frame, idx, pts):
        from .model import FrameRecord

        # 计数在进入 LOST 时已递增；这里只按当前计数取扩大搜索区，
        # 每多丢一帧，下一轮调用时计数 +1
        self.lost_frames += 1
        search = self.lost_search()
        # 重搜一律用 anchor，防止漂移后的工作模板找不到原目标
        x, y, score, used = self._match(frame, self.anchor, search)
        if score_accepted(self.method, score, self.reacquire_threshold):
            self.state = STATE_REACQUIRE
            self._reacquire_count = 1
            self._anchor_pos = (x, y)
            if self.reacquire_frames == 1:
                return self._confirm_recovery(idx, pts, score, used)
        # 即便有候选也不移动框：未确认前 bbox 始终是最后已知位置
        return FrameRecord(
            frame=idx, pts_time=pts, status=LOST,
            bbox=(self.last_x, self.last_y, self.tw, self.th),
            score=score, confirmed=False, search=used,
        )

    def _update_reacquire(self, frame, idx, pts):
        from .model import FrameRecord

        # 重确认在「锚点局部窗」内匹配，要求位置连贯
        ax, ay = self._anchor_pos
        search = (
            ax - _REACQUIRE_JUMP, ay - _REACQUIRE_JUMP,
            ax + _REACQUIRE_JUMP, ay + _REACQUIRE_JUMP,
        )
        x, y, score, used = self._match(frame, self.anchor, search)
        coherent = (
            abs(x - ax) <= _REACQUIRE_JUMP
            and abs(y - ay) <= _REACQUIRE_JUMP
        )
        if (score_accepted(self.method, score, self.reacquire_threshold)
                and coherent):
            self._reacquire_count += 1
            self._anchor_pos = (x, y)
            if self._reacquire_count >= self.reacquire_frames:
                return self._confirm_recovery(idx, pts, score, used)
        else:
            # 任一帧不达标：确认失败，退回 LOST 继续扩大搜索
            self.state = STATE_LOST
            self._anchor_pos = None
        return FrameRecord(
            frame=idx, pts_time=pts, status=LOST,
            bbox=(self.last_x, self.last_y, self.tw, self.th),
            score=score, confirmed=False, search=used,
        )

    def _confirm_recovery(self, idx, pts, score, used):
        """重确认达标：记录恢复帧号并回到 TRACKED（事件由 pipeline 闭合）。"""
        from .model import FrameRecord

        x, y = self._anchor_pos
        self.recovery_frame = idx
        self.state = STATE_TRACKED
        self.lost_frames = 0
        self.loss_started = None
        self._anchor_pos = None
        self.last_x, self.last_y = x, y
        # 恢复后工作模板重置回 anchor，杜绝漂移累积
        self.work = bytes(self.anchor)
        return FrameRecord(
            frame=idx, pts_time=pts, status=TRACKED,
            bbox=(x, y, self.tw, self.th), score=score,
            confirmed=True, search=used,
        )

    # ------------------------------------------------------------ 辅助
    def _match(self, frame, template, search):
        """按方法分派，返回 (x, y, score, 实际搜索区)。"""
        if self.method == "zncc":
            x, y, score, used = best_match_zncc_coarse_fine(
                frame, self.width, self.height,
                template, self.tw, self.th, search=search,
            )
            return x, y, score, used
        x, y, score = best_match(
            frame, self.width, self.height,
            template, self.tw, self.th,
            method=self.method, search=search,
        )
        return x, y, score, _clip(search, self.width, self.height,
                                  self.tw, self.th)

    def _maybe_update_template(self, frame, x, y, score):
        """TRACKED 且分数足够好时，对工作模板做凸组合自适应。"""
        if not self.update_template:
            return
        if not score_accepted(self.method, score, self.update_score):
            return
        patch = extract_patch(frame, self.width, x, y, self.tw, self.th)
        a = self.update_alpha
        self.work = bytes(
            int(round(a * p + (1.0 - a) * w))
            for p, w in zip(patch, self.work)
        )


def _ideal_score(method):
    """首帧自匹配的理想分数：SAD/SSD 为 0，ZNCC 为 1。"""
    return 1.0 if higher_is_better(method) else 0.0


def _clip(search, width, height, tw, th):
    """搜索区夹边界（与 :func:`vtrack.template._clip_search` 同语义）。"""
    max_x = width - tw
    max_y = height - th
    x0, y0, x1, y1 = search
    return (
        max(0, min(x0, max_x)),
        max(0, min(y0, max_y)),
        max(0, min(x1, max_x)),
        max(0, min(y1, max_y)),
    )
