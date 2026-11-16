"""GIF 时序规划：帧延迟精确对应源帧率，杜绝快慢漂移。

GIF 规范中每帧延迟写在图形控制扩展（GCE）里，单位是 **厘秒**
（centisecond，1/100 秒）且为无符号整数。常见实现直接用
``round(100 / fps)`` 反推延迟，会同时引入两类误差：

1. **帧率换算误差**：目标帧率与源帧率不是整数倍关系时，简单取帧会
   丢首尾帧或产生不均匀步长；
2. **厘秒量化累积误差**：例如 12fps 时 ``100/12≈8.33`` 取 8，每帧少
   0.33cs，长片段累积成可感知的播放加速。

本模块的口径：**延迟只由源时长与输出帧数决定，与调色板、缩放、量化
完全无关**。先在连续时间轴上做端点锚定的取帧（与 frconv 项目同一套
Bresenham 整数误差法），再把「源总时长 → 厘秒整数」的余数用
Bresenham 均匀撒到各帧，每帧延迟只可能相差 1 厘秒，总时长误差
严格小于 1 厘秒，且不随帧数累积。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MIN_GIF_DELAY_CS",
    "TimingError",
    "FramePick",
    "frame_pick_indices",
    "DelayPlan",
    "plan_gif_delays",
]

#: GIF 延迟字段可编码 0..65535 厘秒，但历史浏览器/查看器普遍把
#: 0 与 1 厘秒钳制为 10 厘秒（约 10fps 的旧 Gif87a 行为）。
#: 2 厘秒（=50fps）是事实上的安全下限，低于此值输出会被查看器
#: 改写时序，因此本包提前拒绝而非静默漂移。
MIN_GIF_DELAY_CS = 2


class TimingError(ValueError):
    """时序规划无法满足（时长非正、帧数为零或目标帧率超过安全下限）。"""


@dataclass(frozen=True)
class FramePick:
    """一个输出帧对应的源帧索引与该输出帧覆盖的源时间区间。

    :ivar index: 取用的源帧下标（0 起算）。
    :ivar start_ts: 该输出帧在源时间轴上的起始时间（秒）。
    :ivar end_ts: 该输出帧在源时间轴上的结束时间（秒，不含）。
    """

    index: int
    start_ts: float
    end_ts: float

    @property
    def duration(self) -> float:
        """该输出帧应持续的源时间长度（秒）。"""
        return self.end_ts - self.start_ts


def frame_pick_indices(n_src: int, n_out: int):
    """端点锚定 Bresenham：把 ``n_out`` 个输出帧映射到源帧下标。

    与 ``Video-Frame-Rate-Conversion`` 的取帧口径一致，用
    ``divmod(n_src-1, n_out-1)`` 把额外步长均匀撒出，保证：

    - 首项恒为 0、末项恒为 ``n_src-1``（强压缩也不丢首尾帧）；
    - 相邻下标间距只可能是 ``q`` 或 ``q+1``，节奏均匀；
    - 纯整数运算，结果确定性、可复现。

    :param n_src: 源帧总数（须 ≥1）。
    :param n_out: 输出帧总数（须 ≥1 且 ≤ ``n_src``）。
    :return: 长度 ``n_out`` 的源帧下标列表。
    :raises TimingError: 参数为零或输出帧数多于源帧数。
    """
    if n_src <= 0 or n_out <= 0:
        raise TimingError(f"帧数必须为正整数：n_src={n_src}, n_out={n_out}")
    if n_out > n_src:
        raise TimingError(
            f"输出帧数 {n_out} 多于源帧数 {n_src}，本包只做丢帧/重复式"
            "采样，不凭空插值造帧"
        )
    if n_out == 1:
        return [0]
    q, r = divmod(n_src - 1, n_out - 1)
    indices = []
    pos = 0
    for j in range(n_out):
        indices.append(pos)
        # 前 r 个间隔多走 1 步，把余数均匀铺开。
        pos += q + (1 if j < r else 0)
    # 浮点之外的整数保证：末项严格落在 n_src-1。
    indices[-1] = n_src - 1
    return indices


@dataclass(frozen=True)
class DelayPlan:
    """GIF 帧延迟规划结果。

    :ivar delays_cs: 逐帧延迟列表，单位厘秒，每帧只可能相差 1。
    :ivar total_delay_cs: 各帧延迟之和（厘秒）。
    :ivar source_duration: 源片段时长（秒，含末帧持续时间口径）。
    :ivar target_fps: 规划所用的目标帧率（每秒帧数）。
    """

    delays_cs: tuple
    total_delay_cs: int
    source_duration: float
    target_fps: float

    @property
    def gif_duration(self) -> float:
        """写出的 GIF 实际播放时长（秒）。"""
        return self.total_delay_cs / 100.0

    @property
    def duration_error(self) -> float:
        """GIF 时长相对源时长的偏差（秒），绝对值恒 < 0.01。"""
        return self.gif_duration - self.source_duration


def plan_gif_delays(source_duration: float, n_out: int,
                    target_fps: float | None = None) -> DelayPlan:
    """把源时长切成 ``n_out`` 个厘秒整数延迟，余数 Bresenham 均摊。

    延迟与调色板 / 量化 / 缩放无关，只取决于源时长与输出帧数，因此
    后续复用抽帧产物改任何画面参数都不会改变时序。

    厘秒化采用半值向上取整（避免 Python 银行家舍入的偶值偏向），
    再把总厘秒数按帧均分：基值 ``q``、余数 ``r``，前 ``r`` 帧取
    ``q+1`` 其余取 ``q``，各帧延迟差至多 1 且总和精确等于总厘秒数。

    :param source_duration: 源时长（秒），须为正，口径为「最后一帧
        PTS + 末帧持续时间」（CFR 即 N/fps），与 frconv 一致。
    :param n_out: 输出帧总数（须 ≥1）。
    :param target_fps: 仅记入结果便于追溯；为 None 时按
        ``n_out/source_duration`` 反推。
    :raises TimingError: 时长非正、帧数非正，或均摊后最小延迟低于
        :data:`MIN_GIF_DELAY_CS`（目标帧率超过查看器安全下限）。
    """
    if source_duration <= 0:
        raise TimingError(f"源时长必须为正：{source_duration!r}")
    if n_out <= 0:
        raise TimingError(f"输出帧数必须为正：{n_out!r}")

    # 半值向上取整：floor(x + 0.5)，对正负输入均成立；此处输入为正。
    total_cs = int(source_duration * 100.0 + 0.5)
    if total_cs <= 0:
        raise TimingError(
            f"源时长 {source_duration}s 不足 1 厘秒，无法编码 GIF 延迟"
        )

    q, r = divmod(total_cs, n_out)
    if q < MIN_GIF_DELAY_CS:
        raise TimingError(
            f"目标帧率过高：{n_out} 帧 / {source_duration:.3f}s 均摊延迟 "
            f"{q} 厘秒 < 安全下限 {MIN_GIF_DELAY_CS} 厘秒（50fps）。"
            "历史查看器会把 0/1 厘秒钳制成 10 厘秒造成时序漂移，"
            "请降低 --fps 或减少输出帧数"
        )
    delays = tuple(q + (1 if j < r else 0) for j in range(n_out))
    fps = target_fps if target_fps is not None else n_out / source_duration
    return DelayPlan(
        delays_cs=delays,
        total_delay_cs=sum(delays),
        source_duration=source_duration,
        target_fps=fps,
    )
