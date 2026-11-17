"""检测结果数据契约：逐帧记录与事件 JSON 的内存模型和序列化。

JSON 是检测、标记、事件三个模块之间（以及后续背景建模复用区域结构
时）的唯一交换格式，字段保持显式、扁平、可 jsonl 化：

- ``frames[]``：逐帧记录，帧号严格连续、与解码顺序对齐，掩码本身
  不写入 JSON（体积不可接受），只写区域几何与统计；需要逐像素掩码
  时由标记流水线重放检测，区域列表足以支撑事件核对与框选回放；
- ``events[]``：:class:`~vmotion.events.MotionEvent` 的区间汇总；
- ``meta`` / ``params`` / ``stats``：来源信息、全部检测参数与整段
  统计，保证同一份 JSON 可以完整复现检测配置。
"""

import json

__all__ = [
    "SCHEMA_VERSION",
    "FrameRecord",
    "DetectionResult",
    "build_result",
    "save_detection_json",
    "load_detection_json",
]

#: JSON schema 版本，字段不兼容演进时 +1
SCHEMA_VERSION = 1


class FrameRecord:
    """单帧检测记录（不可变语义，按帧顺序排列）。

    属性
    ----
    frame:
        帧序号，从 0 开始、与解码顺序严格对齐。
    pts_time:
        帧显示时间（秒），由 avg_frame_rate 推导，VFR 素材为名义值。
    motion:
        滞后状态机判定的运动态（去抖/迟滞后的对外状态）。
    regions:
        本帧存活区域 dict 列表（:class:`~vmotion.morphology.Region`
        的 ``to_dict`` 形态），非运动态帧恒为空。
    stats:
        检测器调试统计（门控前后命中块数、MAD、光照抑制标志等）。
    event_id:
        本帧所属事件 id（无事件为 ``None``），由事件记录器回填，
        保证标记视频与事件列表逐帧可对照。
    """

    __slots__ = ("frame", "pts_time", "motion", "regions", "stats",
                 "event_id")

    def __init__(self, frame, pts_time, motion, regions, stats,
                 event_id=None):
        self.frame = frame
        self.pts_time = pts_time
        self.motion = bool(motion)
        self.regions = list(regions)
        self.stats = dict(stats)
        self.event_id = event_id

    def to_dict(self):
        return {
            "frame": self.frame,
            "pts_time": round(self.pts_time, 6),
            "motion": self.motion,
            "event_id": self.event_id,
            "regions": [dict(r) for r in self.regions],
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            frame=d["frame"],
            pts_time=d.get("pts_time", 0.0),
            motion=bool(d.get("motion", False)),
            regions=d.get("regions", []),
            stats=d.get("stats", {}),
            event_id=d.get("event_id"),
        )


class DetectionResult:
    """整段视频的检测结果：元信息 + 逐帧记录 + 事件 + 统计。"""

    __slots__ = ("meta", "params", "frames", "events", "stats")

    def __init__(self, meta, params, frames, events, stats):
        self.meta = meta
        self.params = params
        self.frames = frames
        self.events = events
        self.stats = stats

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "params": self.params,
            "stats": self.stats,
            "frames": [r.to_dict() for r in self.frames],
            "events": [e.to_dict() for e in self.events],
        }


def build_result(frames, events, meta, params):
    """由逐帧记录与事件汇总整段统计，组装 :class:`DetectionResult`。"""
    total = len(frames)
    motion_frames = sum(1 for f in frames if f.motion)
    held_off = sum(1 for f in frames if f.stats.get("held_off"))
    # 背景建模模式专有：预热期帧数与发生过块吸收的帧数；帧差模式
    # 的 stats 不含这两个键，.get 兜底为 0，旧 JSON 读回行为不变。
    warmup = sum(1 for f in frames if f.stats.get("warmup"))
    absorbing = sum(1 for f in frames
                    if f.stats.get("absorbed_blocks", 0) > 0)
    gated_total = sum(f.stats.get("gated_pixels", 0) for f in frames)
    stats = {
        "frames": total,
        "motion_frames": motion_frames,
        "motion_ratio": round(motion_frames / total, 6) if total else 0.0,
        "held_off_frames": held_off,
        "warmup_frames": warmup,
        "absorbing_frames": absorbing,
        "events": len(events),
        "gated_blocks_total": gated_total,
        "regions_total": sum(len(f.regions) for f in frames),
    }
    return DetectionResult(dict(meta), dict(params), list(frames),
                           list(events), stats)


def save_detection_json(result, path):
    """把 :class:`DetectionResult` 写成 UTF-8 JSON（ensure_ascii=False）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def load_detection_json(path):
    """读回检测 JSON，还原为 :class:`DetectionResult`。

    校验 schema 版本与逐帧序号连续性；缺字段抛 :class:`ValueError`。
    事件还原为轻量 dict（渲染侧只需要区间与 bbox，不依赖事件类）。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"检测 JSON schema 版本 {version!r} 不受支持，"
            f"当前仅支持 {SCHEMA_VERSION}"
        )
    frames = [FrameRecord.from_dict(d) for d in data.get("frames", [])]
    for i, rec in enumerate(frames):
        if rec.frame != i:
            raise ValueError(
                f"帧序号不连续：第 {i} 条记录的 frame={rec.frame}"
            )
    return DetectionResult(
        meta=data.get("meta", {}),
        params=data.get("params", {}),
        frames=frames,
        events=data.get("events", []),
        stats=data.get("stats", {}),
    )
