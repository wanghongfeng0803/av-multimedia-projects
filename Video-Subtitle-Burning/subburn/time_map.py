"""字幕时间轴变换：线性平移与非线性时间重映射（纯标准库实现）。

本模块在 :class:`~subburn.srt.Cue` 的毫秒整数时间模型上工作，提供两类变换：

* **线性平移** :func:`shift_cues` —— 对每条字幕的起止时间加统一偏移
  （可正可负，支持小数毫秒），并完成越界裁剪、截断与区间重排；
* **非线性重映射** :func:`remap_cues` —— 按速度曲线（关键帧 + 缓动）
  或显式锚点构造一个连续、**单调非减**的时间映射函数 ``T(t)``，把每条
  字幕的两个端点分别映射到新时间轴。

设计上钉死的语义：

1. 时间在模块内部以**浮点毫秒**计算，落盘前统一 ``floor`` 回整数毫秒，
   与 SRT 的整数毫秒模型一致；负时间戳全链路保留（映射函数本身支持
   负定义域），仅在写出 SRT 时由写出方拒绝；
2. 字幕仍是**半开区间** ``[start_ms, end_ms)``。变换后 ``end <= start``
   的区间（被裁空、被零速平台压扁、或整数化后宽度归零）直接丢弃并计数；
3. 速度允许为零（停顿/定格），因此映射函数是单调**非减**而非严格递增。
   零速平台会把不同的旧时刻压到同一新时刻，由此产生的重叠区间由
   :func:`_resolve_overlaps` 做确定性的「后区间让位」修复，保证最终
   结果在开启 ``resolve_overlaps`` 时严格不重叠；
4. 全程不假设输入有序、不假设区间互不重叠，空输入产出空结果且不报错，
   所有被丢弃/截断/修复的事件都计入统计并给出人类可读的告警。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

from .srt import Cue

__all__ = [
    "TimeMapResult",
    "SpeedKeyframe",
    "TimeWarper",
    "Easing",
    "EASINGS",
    "shift_cues",
    "remap_cues",
    "build_speed_warper",
]

#: 进度回调：参数为 0.0~1.0 的完成比例（与 pipeline.ProgressCallback 同形）。
ProgressCallback = Callable[[float], None]


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeMapResult:
    """一次时间轴变换的产出与统计（统计口径便于测试与 CLI 展示）。"""

    cues: list[Cue]
    input_count: int
    output_count: int
    dropped: int          # 完全落在界外或变换后宽度归零而丢弃的条数
    truncated: int        # 起点/终点被边界或映射定义域裁掉一部分的条数
    overlaps_fixed: int   # 重映射后发生重叠、被让位修复的条数
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 缓动函数：u -> [0,1]，要求单调非减、e(0)=0、e(1)=1
# ---------------------------------------------------------------------------

class Easing:
    """缓动函数包装器：同时给出任意点取值与其在 ``[0,1]`` 上的定积分。

    内置缓动的积分有解析值，零误差且无需数值求积；自定义缓动可只给
    ``func``，积分走自适应 Simpson 兜底（见 :meth:`integral`）。

    构造时会校验端点与单调性（按有限采样点），不合法直接抛
    :class:`ValueError` —— 非单调缓动会破坏时间映射的单调性，属于
    调用方错误，不容忍。
    """

    def __init__(
        self,
        name: str,
        func: Callable[[float], float],
        integral: float | None = None,
    ):
        self.name = name
        self._func = func
        self._integral = integral
        # 端点与采样单调性校验：缓动必须把 [0,1] 非减地映到 [0,1]
        endpoints = (func(0.0), func(1.0))
        if not (math.isclose(endpoints[0], 0.0, abs_tol=1e-12)
                and math.isclose(endpoints[1], 1.0, abs_tol=1e-12)):
            raise ValueError(
                f"缓动 {name!r} 端点必须为 f(0)=0、f(1)=1，实际 {endpoints}"
            )
        prev = -1.0
        for i in range(1, 41):
            v = func(i / 40.0)
            if v < prev - 1e-9:
                raise ValueError(f"缓动 {name!r} 不是单调非减函数")
            prev = v

    def __call__(self, u: float) -> float:
        return self._func(u)

    def integral(self) -> float:
        """``∫_0^1 e(u) du``；自定义缓动未给解析值时自适应求积。"""
        if self._integral is not None:
            return self._integral
        return _simpson_adaptive(self._func, 0.0, 1.0, 1e-10)

    def __repr__(self) -> str:
        return f"Easing({self.name!r})"


def _simpson_adaptive(
    func: Callable[[float], float], a: float, b: float, eps: float,
    depth: int = 18,
) -> float:
    """自适应复合 Simpson 求积，仅用于自定义缓动的积分兜底。"""
    m = (a + b) / 2.0
    whole = (b - a) / 6.0 * (func(a) + 4.0 * func(m) + func(b))
    left = (m - a) / 6.0 * (func(a) + 4.0 * func((a + m) / 2.0) + func(m))
    right = (b - m) / 6.0 * (func(m) + 4.0 * func((m + b) / 2.0) + func(b))
    if depth <= 0 or abs(left + right - whole) <= 15.0 * eps:
        return left + right + (left + right - whole) / 15.0
    return (_simpson_adaptive(func, a, m, eps / 2.0, depth - 1)
            + _simpson_adaptive(func, m, b, eps / 2.0, depth - 1))


EASINGS: dict[str, Easing] = {
    # 线性：积分 1/2
    "linear": Easing("linear", lambda u: u, 0.5),
    # 平滑启停（smoothstep 3u²-2u³）：积分 1/2
    "smoothstep": Easing(
        "smoothstep", lambda u: u * u * (3.0 - 2.0 * u), 0.5
    ),
    # 缓入：先慢后快，u² 积分 1/3
    "ease_in": Easing("ease_in", lambda u: u * u, 1.0 / 3.0),
    # 缓出：先快后慢，1-(1-u)² 积分 2/3
    "ease_out": Easing(
        "ease_out", lambda u: 1.0 - (1.0 - u) ** 2, 2.0 / 3.0
    ),
    # 缓入缓出（五次 smoothstep）：积分同样 1/2
    "ease_in_out": Easing(
        "ease_in_out",
        lambda u: u * u * u * (u * (6.0 * u - 15.0) + 10.0),
        0.5,
    ),
}


# ---------------------------------------------------------------------------
# 速度关键帧与分段结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeedKeyframe:
    """速度曲线上的关键帧。

    :param t_ms: 关键帧所在的旧时间轴时刻（毫秒，可为负）
    :param speed: 该时刻的**时间轴流速**，即映射斜率 ``dT/dt``（新毫秒 /
        旧毫秒，``>=0``）：``1`` 为原速、``2`` 为时间轴拉伸一倍（源每走
        1ms 新轴走 2ms）、``0.5`` 为压缩一半、``0`` 为定格（新轴停走）。
        取斜率而非播放倍率 ``dt/dT``，使「定格」对应自然的零值且全段
        不会出现除零。
    :param easing: 与**下一帧**之间的速度过渡方式，末帧该值被忽略
    """

    t_ms: float
    speed: float
    easing: str = "linear"

    def __post_init__(self) -> None:
        if self.speed < 0:
            raise ValueError(f"速度不能为负: {self.speed}")
        if self.easing not in EASINGS:
            raise ValueError(
                f"未知缓动 {self.easing!r}，可选: {', '.join(sorted(EASINGS))}"
            )


@dataclass(frozen=True)
class _Segment:
    """映射函数的一个内部分段（旧轴 ``[x0,x1]`` -> 新轴 ``[y0,y1]``）。

    两种形态共用同一结构：

    * **速度段**（``is_speed=True``）：段内瞬时速度按
      ``v(u) = v0 + (v1-v0)*e(u)`` 缓动变化，``y0`` 之外的累计量在
      映射时用积分现算；
    * **锚点段**（``is_speed=False``）：段内严格线性，``slope`` 即
      ``(y1-y0)/(x1-x1)``。
    """

    x0: float
    x1: float
    y0: float
    y1: float
    v0: float
    v1: float
    easing: Easing
    is_speed: bool

    @property
    def span(self) -> float:
        return self.x1 - self.x0

    @property
    def slope(self) -> float:
        """锚点段的线性斜率；速度段不应访问。"""
        return (self.y1 - self.y0) / self.span


# ---------------------------------------------------------------------------
# 时间映射函数
# ---------------------------------------------------------------------------

class TimeWarper:
    """连续、单调非减的分段时间映射 ``T: 旧毫秒 -> 新毫秒``。

    不直接 ``new = TimeWarper(...)`` 构造，使用两个工厂：

    * :meth:`from_speed_curve` —— 速度关键帧 + 缓动（非线性重映射）；
    * :meth:`from_anchors` —— 显式 ``(旧时刻, 新时刻)`` 锚点的分段线性
      映射（线性平移是它 ``T(t)=t+offset`` 的特例，也便于测试）。

    定义域之外按首尾速度/斜率做**一次函数外推**，因此任意有限时刻
    （含负时刻）都能映射，不会出现空洞。
    """

    def __init__(self, segments: Sequence[_Segment],
                 head_slope: float, tail_slope: float):
        if not segments:
            raise ValueError("TimeWarper 至少需要一个分段/两个关键帧")
        self._segments = list(segments)
        self._x_starts = [s.x0 for s in self._segments]
        self._head_slope = head_slope
        self._tail_slope = tail_slope

    # -- 工厂 -------------------------------------------------------------

    @classmethod
    def from_speed_curve(cls, keyframes: Sequence[SpeedKeyframe]) -> "TimeWarper":
        """把速度关键帧序列积分成时间映射。

        相邻关键帧 ``k_i -> k_{i+1}`` 构成一段：旧轴跨度
        ``dx = t_{i+1}-t_i`` 必须为正；新轴跨度由平均速度乘以 ``dx``
        得到，而平均速度依赖缓动积分 ``∫e``::

            v(u) = v_i + (v_{i+1}-v_i)·e(u),  u∈[0,1]
            dy   = dx · (v_i + (v_{i+1}-v_i)·∫_0^1 e(u)du)

        所有速度与积分值均非负，故 ``dy >= 0``，映射天然单调非减。
        """
        frames = list(keyframes)
        if len(frames) < 2:
            raise ValueError("速度曲线至少需要两个关键帧")
        segments: list[_Segment] = []
        y = 0.0
        for i in range(len(frames) - 1):
            k0, k1 = frames[i], frames[i + 1]
            dx = k1.t_ms - k0.t_ms
            if dx <= 0:
                raise ValueError(
                    f"第 {i} 段关键帧时刻必须严格递增: "
                    f"{k0.t_ms} -> {k1.t_ms}"
                )
            easing = EASINGS[k0.easing]
            mean_speed = k0.speed + (k1.speed - k0.speed) * easing.integral()
            dy = dx * mean_speed
            segments.append(_Segment(
                x0=k0.t_ms, x1=k1.t_ms, y0=y, y1=y + dy,
                v0=k0.speed, v1=k1.speed, easing=easing, is_speed=True,
            ))
            y += dy
        # 外推斜率取端点速度，保证分段交界处导数连续（线性/ease 段至少连续）
        return cls(segments, head_slope=frames[0].speed,
                   tail_slope=frames[-1].speed)

    @classmethod
    def from_anchors(cls, points: Sequence[tuple[float, float]]) -> "TimeWarper":
        """由 ``[(旧时刻 t, 新时刻 T(t)), ...]`` 构造分段线性映射。

        锚点旧时刻严格递增、新时刻单调非减（允许相等，表示定格）；
        首尾段斜率用于定义域外的线性外推。
        """
        pts = list(points)
        if len(pts) < 2:
            raise ValueError("锚点映射至少需要两个点")
        segments: list[_Segment] = []
        prev_y: float | None = None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x1 <= x0:
                raise ValueError(f"锚点旧时刻必须严格递增: {x0} -> {x1}")
            if prev_y is not None and y0 < prev_y - 1e-9:
                raise ValueError("锚点新时刻必须单调非减（不允许时间倒流）")
            if y1 < y0 - 1e-9:
                raise ValueError("锚点新时刻必须单调非减（不允许时间倒流）")
            linear = EASINGS["linear"]
            segments.append(_Segment(
                x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1),
                v0=0.0, v1=0.0, easing=linear, is_speed=False,
            ))
            prev_y = y0
        head = segments[0].slope
        tail = segments[-1].slope
        return cls(segments, head_slope=head, tail_slope=tail)

    # -- 映射 -------------------------------------------------------------

    @property
    def domain_start(self) -> float:
        return self._segments[0].x0

    @property
    def domain_end(self) -> float:
        return self._segments[-1].x1

    def map_ms(self, t_ms: float) -> float:
        """把旧时间轴时刻 ``t_ms`` 映射为新时间轴时刻（浮点毫秒）。"""
        segs = self._segments
        # 头部外推：一次函数，斜率取首段速度/斜率
        first = segs[0]
        if t_ms < first.x0:
            base = first.v0 if first.is_speed else first.slope
            return first.y0 + base * (t_ms - first.x0)
        # 尾部外推
        last = segs[-1]
        if t_ms >= last.x1:
            base = last.v1 if last.is_speed else last.slope
            return last.y1 + base * (t_ms - last.x1)
        # 定位分段：bisect 右插点减一，复杂度 O(log n)
        i = bisect.bisect_right(self._x_starts, t_ms) - 1
        i = max(0, min(i, len(segs) - 1))
        seg = segs[i]
        u = (t_ms - seg.x0) / seg.span
        if not seg.is_speed:
            return seg.y0 + seg.slope * (t_ms - seg.x0)
        # 速度段：T = y0 + dx·(v0·u + (v1-v0)·∫_0^u e)
        return (
            seg.y0
            + seg.span * (
                seg.v0 * u
                + (seg.v1 - seg.v0) * _partial_integral(seg.easing, u)
            )
        )


def _partial_integral(easing: Easing, u: float) -> float:
    """``∫_0^u e(s) ds``。内置缓动用解析原语，自定义缓动用数值求积。"""
    f = easing.name
    if f == "linear":
        return 0.5 * u * u
    if f == "smoothstep":  # e=3s²-2s³ -> 原语 u³ - 0.5u⁴
        return u ** 3 - 0.5 * u ** 4
    if f == "ease_in":  # s² -> u³/3
        return u ** 3 / 3.0
    if f == "ease_out":  # 1-(1-s)² = 2s-s² -> u² - u³/3
        return u * u - u ** 3 / 3.0
    if f == "ease_in_out":  # 6s⁵-15s⁴+10s³ -> u⁶-3u⁵+2.5u⁴
        return u ** 6 - 3.0 * u ** 5 + 2.5 * u ** 4
    return _simpson_adaptive(easing, 0.0, u, 1e-10)


def build_speed_warper(
    keyframes: Sequence[tuple[float, float] | SpeedKeyframe],
    easing: str = "linear",
) -> TimeWarper:
    """便捷构造：``(t_ms, speed)`` 元组或 :class:`SpeedKeyframe` 均可。

    元组形式统一套用 ``easing`` 指定的缓动；需要逐段指定不同缓动时
    直接传 :class:`SpeedKeyframe`。
    """
    frames = [
        k if isinstance(k, SpeedKeyframe)
        else SpeedKeyframe(t_ms=float(k[0]), speed=float(k[1]), easing=easing)
        for k in keyframes
    ]
    return TimeWarper.from_speed_curve(frames)


# ---------------------------------------------------------------------------
# 公共变换入口
# ---------------------------------------------------------------------------

def shift_cues(
    cues: Sequence[Cue],
    offset_ms: float,
    *,
    lower_bound_ms: float | None = 0.0,
    upper_bound_ms: float | None = None,
    resolve_overlaps: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TimeMapResult:
    """对全部字幕做统一时间平移，并完成越界裁剪、截断与区间重排。

    :param offset_ms: 统一偏移毫秒数，可正可负，允许小数（落盘前 floor
        为整数毫秒）。
    :param lower_bound_ms: 时间轴下界，默认 ``0``；传 ``None`` 表示不设
        下界（负时间戳予以保留）。
    :param upper_bound_ms: 时间轴上界（如视频时长），默认不限制。
    :param resolve_overlaps: 平移本身不会制造重叠，但输入可能自带重叠；
        为 True 时额外做一次不重叠修复。默认 False，保留输入的区间关系。
    :param progress_callback: 每处理一条回调一次完成比例。

    处理流程：平移（floor 取整）→ 边界裁剪/截断 → 丢弃零宽 → 排序重排
    （按新起点稳定排序并从 1 重新编号）→ 可选重叠修复。
    """
    return _transform(
        cues,
        map_func=lambda t: t + offset_ms,
        lower_bound_ms=lower_bound_ms,
        upper_bound_ms=upper_bound_ms,
        resolve_overlaps=resolve_overlaps,
        progress_callback=progress_callback,
        action=f"平移 {offset_ms:g}ms",
    )


def remap_cues(
    cues: Sequence[Cue],
    warper: TimeWarper,
    *,
    lower_bound_ms: float | None = 0.0,
    upper_bound_ms: float | None = None,
    resolve_overlaps: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> TimeMapResult:
    """按 :class:`TimeWarper` 对字幕时间轴做非线性重映射。

    每条字幕的起止端点分别经 ``warper.map_ms`` 映射；零速平台会把若干
    旧时刻压到同一新时刻，因此默认开启 ``resolve_overlaps``，对映射后
    相互重叠的区间做确定性让位，保证产出严格单调、互不重叠。

    边界参数作用于**新时间轴**：映射结果低于下界/高于上界的部分裁掉。
    """
    return _transform(
        cues,
        map_func=warper.map_ms,
        lower_bound_ms=lower_bound_ms,
        upper_bound_ms=upper_bound_ms,
        resolve_overlaps=resolve_overlaps,
        progress_callback=progress_callback,
        action="非线性重映射",
    )


# ---------------------------------------------------------------------------
# 共用变换管线
# ---------------------------------------------------------------------------

def _transform(
    cues: Sequence[Cue],
    map_func: Callable[[float], float],
    *,
    lower_bound_ms: float | None,
    upper_bound_ms: float | None,
    resolve_overlaps: bool,
    progress_callback: ProgressCallback | None,
    action: str,
) -> TimeMapResult:
    """平移与重映射共用的端点变换 + 裁剪 + 重排管线。"""
    warnings: list[str] = []
    total = len(cues)
    dropped = 0
    truncated = 0
    overlaps_fixed = 0
    out: list[Cue] = []

    # 边界参数容错：上界小于下界属于调用错误，直接抛出而非静默产出空表
    if (lower_bound_ms is not None and upper_bound_ms is not None
            and upper_bound_ms < lower_bound_ms):
        raise ValueError(
            f"上界 {upper_bound_ms}ms 不能小于下界 {lower_bound_ms}ms"
        )

    for pos, cue in enumerate(cues):
        # 内部一律浮点计算，最后统一 floor：负小数向负无穷取整，
        # 与 int() 的向零截断不同，负时间戳也能保持区间方向
        start = math.floor(map_func(float(cue.start_ms)))
        end = math.floor(map_func(float(cue.end_ms)))

        before_clip = (start, end)
        if lower_bound_ms is not None:
            lo = math.ceil(lower_bound_ms)
            start = max(start, lo)
            end = max(end, lo)
        if upper_bound_ms is not None:
            hi = math.floor(upper_bound_ms)
            start = min(start, hi)
            end = min(end, hi)
        if (start, end) != before_clip:
            truncated += 1

        # 半开区间宽度归零（含被裁空、零速压扁、floor 后等宽）：丢弃
        if end <= start:
            dropped += 1
            warnings.append(
                f"字幕 #{cue.index}（{cue.start_ms}->{cue.end_ms}ms）"
                f"{action}后区间为空，已丢弃"
            )
        else:
            out.append(replace(cue, start_ms=start, end_ms=end))

        if progress_callback is not None and total:
            progress_callback((pos + 1) / total)

    # 区间重排：映射单调非减并不保证输入有序，统一按 (起点, 终点) 稳定
    # 排序后从 1 重新编号，序号即新时间轴上的展示顺序
    out.sort(key=lambda c: (c.start_ms, c.end_ms))
    if resolve_overlaps:
        out, overlaps_fixed, fix_warnings = _resolve_overlaps(out)
        warnings.extend(fix_warnings)

    resequenced = [
        replace(c, index=i) for i, c in enumerate(out, start=1)
    ]
    if progress_callback is not None:
        progress_callback(1.0)
    return TimeMapResult(
        cues=resequenced,
        input_count=total,
        output_count=len(resequenced),
        dropped=dropped,
        truncated=truncated,
        overlaps_fixed=overlaps_fixed,
        warnings=warnings,
    )


def _resolve_overlaps(
    cues: Sequence[Cue],
) -> tuple[list[Cue], int, list[str]]:
    """对已按起点排序的区间做确定性不重叠修复。

    规则（贪心扫描线，游标只前进）：维护「上一条保留区间的终点
    ``cursor````」，当前区间起点若早于 ``cursor``，则把起点抬到
    ``cursor`` —— 即后出现的字幕为先来者让位；抬升后宽度归零则丢弃。
    起点相同时靠排序键中的终点次序打破平局，整个过程无随机行为。
    """
    fixed: list[Cue] = []
    fixes = 0
    warnings: list[str] = []
    cursor: int | None = None
    for cue in cues:
        start, end = cue.start_ms, cue.end_ms
        if cursor is not None and start < cursor:
            start = cursor
            fixes += 1
            warnings.append(
                f"字幕 #{cue.index} 与前序区间重叠，起点让位至 {start}ms"
            )
        if end <= start:
            warnings.append(
                f"字幕 #{cue.index} 让位后区间为空，已丢弃"
            )
            continue
        fixed.append(replace(cue, start_ms=start, end_ms=end))
        cursor = end
    return fixed, fixes, warnings
