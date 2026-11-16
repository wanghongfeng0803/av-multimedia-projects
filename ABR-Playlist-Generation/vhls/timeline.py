"""时间轴规划：把源帧序列切成等目标时长的分片计划（纯函数，无 ffmpeg）。

本模块是「列表写入与切片切割共用同一套时间轴」的唯一事实来源：

- 切割器（``cut.py``）按计划里的帧范围/起止毫秒调 ffmpeg；
- 播放列表（``playlist.py``）按同一份计划逐片写 EXTINF 与序号。

**为什么用整数毫秒而不是浮点秒**：每片时长若各自 ``round()`` 一次浮点，
N 片之后舍入误差会沿时间轴累积（典型表现：末片起点漂移几十毫秒、
EXTINF 之和与源时长对不上）。这里在入口把每个 PTS/帧时长一次性换算成
整数毫秒（半值向上取整），之后所有边界、比较、求和全部是整数运算，
不存在浮点表示误差，也不存在跨片累积。

**切点只允许落在真实帧边缘**：按目标时长贪心累积帧时长，装不下下一帧
就关片。这样每片都是整数个完整帧，切割不会把一帧劈成两半；代价是单片
实际长度相对目标值最多偏差一帧，属 HLS 规范允许范围（RFC 8216 只要求
不超过 TARGETDURATION）。

**毫秒量化的误差边界（已知且有界）**：入口把每个帧时长半值向上取整到
整数毫秒，单帧量化误差在 [-0.5, +0.5]ms，N 帧累计误差有界于 ±N/2 ms。
常规帧率下帧长本身就是毫秒整数（25fps=40ms、24fps≈42ms），误差为 0；
30fps（33.33ms）这类非整数帧长最坏约 0.5ms/帧——3 秒 91 帧的极端例子
累计 16ms，仍远小于一个帧间隔，且 EXTINF 标准本身就是为毫秒级（浮点
小数）精度设计的。换取的是全链路整数运算、零浮点漂移与完全可复现。
"""

from __future__ import annotations

from dataclasses import dataclass

from .probe import SourceInfo

__all__ = [
    "TimelineError",
    "SegmentSpec",
    "TimelinePlan",
    "MS_PER_SEC",
    "ms_to_seconds",
    "format_extinf",
    "build_timeline",
]

#: 内部时间单位：1 秒 = 1000 毫秒。全链路时间戳的唯一换算常量。
MS_PER_SEC = 1000


class TimelineError(ValueError):
    """时间轴规划参数非法（目标时长非正、帧清单为空等）。"""


@dataclass(frozen=True)
class SegmentSpec:
    """单个分片在统一时间轴上的规格（切割与写列表共用）。

    :ivar sequence: 媒体序列号（0 起算，全列表连续）。
    :ivar filename: 分片文件名（不含目录，由调用方统一命名）。
    :ivar start_frame: 起始帧在源帧清单中的下标（含）。
    :ivar end_frame: 结束帧在源帧清单中的下标（不含）。
    :ivar start_ms: 起点时间戳（毫秒，取自起始帧 PTS）。
    :ivar end_ms: 终点时间戳（毫秒，取自结束帧 PTS；末片取源总时长）。
    :ivar planned_duration_ms: 计划时长（毫秒，= end_ms - start_ms）。
    :ivar is_last: 是否为最后一片（末片通常不足目标时长，需单独标出）。
    :ivar is_short: 是否为不足目标时长的短尾片
        （计划时长 < 目标时长的 90%）。
    """

    sequence: int
    filename: str
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    planned_duration_ms: int
    is_last: bool
    is_short: bool

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class TimelinePlan:
    """整源的分片计划：切割与播放列表都只认这一份不可变对象。

    :ivar target_ms: 目标分片时长（毫秒）。
    :ivar source_duration_ms: 源总时长（毫秒）。
    :ivar segments: 逐片规格（序列号已连续编号）。
    :ivar segment_prefix: 分片文件名前缀。
    """

    target_ms: int
    source_duration_ms: int
    segments: tuple
    segment_prefix: str

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def target_duration(self) -> int:
        """EXT-X-TARGETDURATION：各片计划时长上限向上取整（整数秒）。"""
        worst = max(s.planned_duration_ms for s in self.segments)
        return ceil_div(worst, MS_PER_SEC)


def ceil_div(a: int, b: int) -> int:
    """正整数向上取整除法（避免 ``math.ceil`` 的浮点中间值）。"""
    return -((-a) // b)


def ms_to_seconds(ms: int) -> float:
    """毫秒转秒，仅在对外展示/写 EXTINF 的最后一步才允许出现浮点。"""
    return ms / MS_PER_SEC


def format_extinf(ms: int) -> str:
    """把毫秒时长格式化为 EXTINF 浮点字符串（3 位小数，半值向上取整）。

    毫秒本身已是整数，除以 1000 后用固定 3 位小数输出，避免 Python
    ``str(float)`` 吐出长串二进制尾数（如 ``3.003000000000000``）。
    3 位小数对应 1ms 分辨率，与内部时间单位一致，写出值与实际严格对应。
    """
    return f"{ms / MS_PER_SEC:.3f}"


def _to_ms(seconds: float) -> int:
    """秒（浮点）→ 毫秒整数，半值向上取整；负数视为非法输入。"""
    if seconds < 0:
        raise TimelineError(f"时间戳不能为负：{seconds}")
    return int(seconds * MS_PER_SEC + 0.5)


def build_timeline(source: SourceInfo, target_seconds: float,
                   segment_prefix: str = "seg",
                   segment_suffix: str = ".ts") -> TimelinePlan:
    """按目标分片时长对源帧做帧边界贪心分组，生成统一时间轴计划。

    分组规则：逐帧累加毫秒时长，当前片再装下一帧就会超过目标时长时
    关片、另开新片；最后一片吃掉所有剩余帧。因为输入是 VFR 也成立的
    逐帧 PTS，所以 CFR/VFR 走同一条代码路径。

    :param source: :func:`vhls.probe.probe_source` 的探测结果。
    :param target_seconds: 目标分片时长（秒），必须 > 0；换算为毫秒后
        至少要能容纳一帧，否则无法规划。
    :param segment_prefix: 分片文件名前缀，最终名为
        ``<prefix>_<5 位序号><suffix>``。
    :param segment_suffix: 分片文件名后缀（默认 ``.ts``）。
    :raises TimelineError: 参数非法或目标时长装不下单帧。
    """
    if target_seconds <= 0:
        raise TimelineError(f"目标分片时长必须为正：{target_seconds}")
    target_ms = _to_ms(target_seconds)
    if target_ms <= 0:
        raise TimelineError(
            f"目标分片时长 {target_seconds}s 不足 1ms，无法规划"
        )

    frames: tuple = source.frames
    if not frames:
        raise TimelineError("源帧清单为空，无法规划时间轴")

    # 入口一次性把浮点 PTS/帧时长换成整数毫秒，之后全程整数运算。
    pts_ms = [_to_ms(f.pts_time) for f in frames]
    dur_ms = [_to_ms(f.duration_time) for f in frames]
    # 探测补算的帧时长理论上为正；显式拦截 0/负值，防止死循环或负时长。
    for i, d in enumerate(dur_ms):
        if d <= 0:
            raise TimelineError(
                f"第 {i} 帧时长非正（{frames[i].duration_time}s），"
                "无法规划时间轴"
            )

    source_duration_ms = pts_ms[-1] + dur_ms[-1]
    frame_interval_ms = max(dur_ms)
    if target_ms < frame_interval_ms:
        raise TimelineError(
            f"目标分片时长 {target_ms}ms 小于最长单帧 "
            f"{frame_interval_ms}ms，至少需容纳一个完整帧"
        )

    # 贪心分组：[start, cursor) 为当前片已收纳的帧。
    bounds: list = []
    start = 0
    cursor = 0
    n = len(frames)
    while cursor < n:
        seg_start_ms = pts_ms[start]
        # 推进光标直到再吃下一帧就超目标（最后一片不受目标约束）。
        while cursor < n:
            frame_end_ms = pts_ms[cursor] + dur_ms[cursor]
            if cursor > start and frame_end_ms - seg_start_ms > target_ms:
                break
            cursor += 1
        bounds.append((start, cursor))
        start = cursor

    short_threshold_ms = (target_ms * 9 + 9) // 10  # target 的 90% 向上取整
    specs: list = []
    total = len(bounds)
    for seq, (f_start, f_end) in enumerate(bounds):
        start_ms = pts_ms[f_start]
        if f_end < n:
            end_ms = pts_ms[f_end]
        else:
            end_ms = source_duration_ms
        planned = end_ms - start_ms
        is_last = seq == total - 1
        specs.append(SegmentSpec(
            sequence=seq,
            filename=f"{segment_prefix}_{seq:05d}{segment_suffix}",
            start_frame=f_start,
            end_frame=f_end,
            start_ms=start_ms,
            end_ms=end_ms,
            planned_duration_ms=planned,
            is_last=is_last,
            # 短尾判定只看实际计划长度；正常片因帧对齐略短于目标不算短尾。
            is_short=is_last and planned < short_threshold_ms,
        ))

    return TimelinePlan(
        target_ms=target_ms,
        source_duration_ms=source_duration_ms,
        segments=tuple(specs),
        segment_prefix=segment_prefix,
    )
