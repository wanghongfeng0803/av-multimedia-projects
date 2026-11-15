"""转场时间轴规划：把两段素材的帧坐标映射到成片坐标。

这是整个工具的「账本」模块，纯函数、无 I/O，输出时长可逐段核对。

时长恒等式（帧域）
==================

设两段素材帧数分别为 ``Na``、``Nb``，重叠区帧数为 ``No``，
成片由三段**互不交叠、首尾相接**的区间拼成::

    head     : A 的第 [0, Na-No) 帧        —— 纯 A，共 Na-No 帧
    overlap  : A 的第 [Na-No, Na) 帧       —— 交叉淡化，共 No 帧
               B 的第 [0, No) 帧
    tail     : B 的第 [No, Nb) 帧          —— 纯 B，共 Nb-No 帧

    成片总帧数 Nout = (Na-No) + No + (Nb-No) = Na + Nb - No

即成片比「两段直接拼接」恰好短一个重叠区，重叠区之外 A、B 的每一帧
都在成片里出现一次，一帧不丢。音频采样域复用同一张映射表（见
:mod:`video_transition.audio`），保证音视频淡化边界严格对齐。

淡化权重
========

重叠区第 k 帧（k 从 0 起）::

    w_b = No == 1 时取 1/2；否则 k / (No - 1)，w_a = 1 - w_b

首尾锚点是精确的：k=0 时 w_b=0（与 head 末帧的纯 A 画面逐字节连续），
k=No-1 时 w_b=1（与 tail 首帧的纯 B 画面逐字节连续），因此重叠区
边界不会出现跳变帧。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PlanError
from .probe import ClipMeta

__all__ = [
    "Segment",
    "FrameJob",
    "TransitionPlan",
    "build_plan",
    "fade_weight_b",
    "frame_jobs",
    "KIND_HEAD",
    "KIND_OVERLAP",
    "KIND_TAIL",
]

#: 成片三段的类型标签
KIND_HEAD = "head"
KIND_OVERLAP = "overlap"
KIND_TAIL = "tail"


@dataclass(frozen=True)
class Segment:
    """成片上的一个连续区间（全部坐标均为帧号，左闭右开）。

    :param kind: head/overlap/tail 之一；
    :param out_start/out_end: 成片帧号区间 ``[out_start, out_end)``；
    :param a_start: 该区间对应 A 素材的起始帧号，纯 B 段为 -1；
    :param b_start: 该区间对应 B 素材的起始帧号，纯 A 段为 -1。
    """

    kind: str
    out_start: int
    out_end: int
    a_start: int
    b_start: int

    @property
    def length(self) -> int:
        """区间帧数。"""
        return self.out_end - self.out_start


@dataclass(frozen=True)
class FrameJob:
    """成片单帧的取数与混合指令（帧调度器的输出，可纯逻辑单测）。

    :param out_index: 成片帧号（0 起，单调递增）；
    :param kind: 所在区间类型；
    :param index_a/index_b: 分别要从 A/B 读取器取的帧号，
        纯 A 段 ``index_b`` 为 -1，纯 B 段 ``index_a`` 为 -1；
    :param weight_b_num/weight_b_den: B 权重的有理数分子/分母，
        纯 A 段为 (0,1)，纯 B 段为 (1,1)，混合段为 (k, No-1)。
    """

    out_index: int
    kind: str
    index_a: int
    index_b: int
    weight_b_num: int
    weight_b_den: int

    @property
    def is_overlap(self) -> bool:
        """该帧是否需要双路混合。"""
        return self.kind == KIND_OVERLAP


@dataclass(frozen=True)
class TransitionPlan:
    """两段素材交叉淡化的完整帧映射表。"""

    clip_a: ClipMeta
    clip_b: ClipMeta
    overlap_frames: int
    segments: tuple[Segment, ...]

    @property
    def fps_num(self) -> int:
        """成片帧率分子（两段素材已校验一致）。"""
        return self.clip_a.fps_num

    @property
    def fps_den(self) -> int:
        """成片帧率分母。"""
        return self.clip_a.fps_den

    @property
    def width(self) -> int:
        """成片宽度（两段素材已校验一致）。"""
        return self.clip_a.width

    @property
    def height(self) -> int:
        """成片高度。"""
        return self.clip_a.height

    @property
    def output_frames(self) -> int:
        """成片总帧数 = Na + Nb - No（时长恒等式）。"""
        return self.clip_a.frame_count + self.clip_b.frame_count - self.overlap_frames

    @property
    def output_duration(self) -> float:
        """成片时长（秒），按 CFR 帧数 / 帧率计算。"""
        return self.output_frames * self.fps_den / self.fps_num

    @property
    def head(self) -> Segment:
        """纯 A 段。"""
        return self.segments[0]

    @property
    def overlap(self) -> Segment:
        """交叉淡化段。"""
        return self.segments[1]

    @property
    def tail(self) -> Segment:
        """纯 B 段。"""
        return self.segments[2]

    def weight_b_at(self, out_index: int) -> float:
        """成片第 ``out_index`` 帧上 B 素材的淡化权重。

        非重叠段恒为 0（纯 A）或 1（纯 B）；重叠段按帧线性插值。
        传入越界帧号抛 :class:`PlanError`。
        """
        if not 0 <= out_index < self.output_frames:
            raise PlanError(f"成片帧号越界: {out_index} / {self.output_frames}")
        ov = self.overlap
        if out_index < ov.out_start:
            return 0.0
        if out_index >= ov.out_end:
            return 1.0
        k = out_index - ov.out_start
        return fade_weight_b(k, self.overlap_frames)


def fade_weight_b(k: int, overlap_frames: int) -> float:
    """重叠区内第 k 帧（0 起）的 B 权重，A 权重为 ``1 - 返回值``。

    ``overlap_frames == 1`` 时没有插值空间，单帧取 50/50；
    否则线性插值且首尾锚点精确为 0 和 1。
    """
    if overlap_frames <= 0:
        raise PlanError(f"重叠帧数必须为正，收到 {overlap_frames}")
    if not 0 <= k < overlap_frames:
        raise PlanError(f"重叠区帧号越界: {k} / {overlap_frames}")
    if overlap_frames == 1:
        return 0.5
    return k / (overlap_frames - 1)


def frame_jobs(plan: TransitionPlan) -> list[FrameJob]:
    """按规划表展开成片每一帧的 :class:`FrameJob` 序列。

    纯逻辑函数，不触碰 ffmpeg；管道侧与测试侧共用同一份调度结果，
    保证「测试里核对的取帧指令」就是「实际执行的取帧指令」。
    """
    jobs: list[FrameJob] = []
    head, overlap_seg, tail = plan.segments
    no = plan.overlap_frames

    # head：纯 A，成片帧 f 即 A 帧 f
    for f in range(head.out_start, head.out_end):
        jobs.append(FrameJob(f, KIND_HEAD, f, -1, 0, 1))

    # overlap：成片帧对应 A[na-no+k] 与 B[k]，权重为精确分数
    for k in range(no):
        f = overlap_seg.out_start + k
        if no == 1:
            wnum, wden = 1, 2
        else:
            wnum, wden = k, no - 1
        jobs.append(
            FrameJob(
                f, KIND_OVERLAP,
                overlap_seg.a_start + k,
                overlap_seg.b_start + k,
                wnum, wden,
            )
        )

    # tail：纯 B，成片帧 tail.out_start+local 即 B 帧 no+local
    for local in range(tail.length):
        f = tail.out_start + local
        jobs.append(FrameJob(f, KIND_TAIL, -1, tail.b_start + local, 1, 1))

    return jobs


def build_plan(
    clip_a: ClipMeta,
    clip_b: ClipMeta,
    *,
    overlap_seconds: float | None = None,
    overlap_frames: int | None = None,
) -> TransitionPlan:
    """根据两段素材的探针结果与重叠配置构造 :class:`TransitionPlan`。

    重叠长度两种配置互斥：

    * ``overlap_frames``：直接指定重叠帧数（整数，语义最精确）；
    * ``overlap_seconds``：指定重叠秒数，按帧率换算为帧——
      ``No = round(seconds * fps)``，换算结果必须为正；

    两个参数都不给时抛 :class:`PlanError`，避免「零重叠」悄悄退化成
    硬切而调用方毫无感知。
    """
    # 1) 两段素材必须同帧率、同尺寸：帧读取器吐出的是固定尺寸裸流，
    #    淡化是逐像素整数运算，不做隐式缩放/重采样以免账本失真。
    if (clip_a.fps_num, clip_a.fps_den) != (clip_b.fps_num, clip_b.fps_den):
        raise PlanError(
            f"两段素材帧率不一致: {clip_a.fps_literal} vs {clip_b.fps_literal}；"
            "请先统一帧率后再做转场"
        )
    if (clip_a.width, clip_a.height) != (clip_b.width, clip_b.height):
        raise PlanError(
            f"两段素材尺寸不一致: {clip_a.width}x{clip_a.height} vs "
            f"{clip_b.width}x{clip_b.height}；请先统一分辨率"
        )

    # 2) 解析重叠帧数（秒数配置换算到帧域，之后全部按整数帧说话）
    if overlap_frames is not None and overlap_seconds is not None:
        raise PlanError("overlap_seconds 与 overlap_frames 只能指定一个")
    if overlap_frames is not None:
        no = overlap_frames
    elif overlap_seconds is not None:
        if overlap_seconds <= 0:
            raise PlanError(f"重叠时长必须为正，收到 {overlap_seconds!r}")
        no = round(overlap_seconds * clip_a.fps_num / clip_a.fps_den)
    else:
        raise PlanError("必须指定 overlap_seconds 或 overlap_frames")

    na, nb = clip_a.frame_count, clip_b.frame_count
    if no <= 0:
        raise PlanError(f"重叠帧数换算结果非正: {no}")
    if no >= na or no >= nb:
        raise PlanError(
            f"重叠帧数 {no} 必须短于任一素材（A={na} 帧, B={nb} 帧）；"
            "重叠区需要在两段素材内部各取一个同长区间"
        )

    # 3) 三段区间，坐标全部显式写出，便于逐段核对
    head = Segment(
        kind=KIND_HEAD,
        out_start=0,
        out_end=na - no,
        a_start=0,
        b_start=-1,
    )
    overlap_seg = Segment(
        kind=KIND_OVERLAP,
        out_start=na - no,
        out_end=na,
        a_start=na - no,
        b_start=0,
    )
    tail = Segment(
        kind=KIND_TAIL,
        out_start=na,
        out_end=na + nb - no,
        a_start=-1,
        b_start=no,
    )
    return TransitionPlan(
        clip_a=clip_a,
        clip_b=clip_b,
        overlap_frames=no,
        segments=(head, overlap_seg, tail),
    )
