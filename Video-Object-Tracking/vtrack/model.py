"""轨迹数据契约：追踪结果的内存模型与 JSON 序列化。

JSON 是追踪与渲染之间（以及后续光流追踪、多目标追踪之间）的唯一交换
格式，字段保持显式、扁平、可 jsonl 化。状态只有两种对外值：

- ``"tracked"``：本帧匹配分数达到阈值，bbox 有效；
- ``"lost"``：本帧未达到阈值（含重确认中的帧），bbox 给出最后已知
  位置但 ``confirmed=False``，渲染侧据此决定是否画「丢失」标记。

重确认（REACQUIRE）是追踪器内部状态，对外统一报 ``"lost"``，
避免消费方理解三态。
"""

import json

__all__ = [
    "TRACKED",
    "LOST",
    "SCHEMA_VERSION",
    "FrameRecord",
    "LossEvent",
    "TrackResult",
    "build_track_result",
    "save_track_json",
    "load_track_json",
]

#: 对外可见的两种帧状态
TRACKED = "tracked"
LOST = "lost"

#: JSON schema 版本，字段不兼容演进时 +1
SCHEMA_VERSION = 1


class FrameRecord:
    """单帧追踪记录（不可变语义，按帧顺序排列）。

    属性
    ----
    frame:
        帧序号，从 0 开始、与解码顺序严格对齐。
    pts_time:
        帧显示时间（秒），由 avg_frame_rate 推导，VFR 素材为名义值。
    status:
        :data:`TRACKED` 或 :data:`LOST`。
    bbox:
        ``(x, y, w, h)``，左上角坐标 + 宽高。tracked 帧为本次匹配位置；
        lost 帧为最后已知位置（无任何已知位置时为初始框）。
    score:
        本帧最佳匹配的原始分数（SAD/SSD 越小越好，ZNCC 越大越好），
        lost 帧为搜索区内实际算出的最佳分（可能仍很差），不做编造。
    confirmed:
        该 bbox 是否为确认位置。重确认期与丢失帧恒为 ``False``。
    search:
        本帧实际搜索区域 ``(x0, y0, x1, y1)``（含端点的候选左上界），
        供调试与后续算法复用。
    """

    __slots__ = ("frame", "pts_time", "status", "bbox", "score",
                 "confirmed", "search")

    def __init__(self, frame, pts_time, status, bbox, score,
                 confirmed, search):
        self.frame = frame
        self.pts_time = pts_time
        self.status = status
        self.bbox = tuple(bbox)
        self.score = score
        self.confirmed = confirmed
        self.search = tuple(search)

    def to_dict(self):
        """转为可 JSON 序列化的 dict。"""
        x, y, w, h = self.bbox
        x0, y0, x1, y1 = self.search
        return {
            "frame": self.frame,
            "pts_time": round(self.pts_time, 6),
            "status": self.status,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "score": round(self.score, 6),
            "confirmed": self.confirmed,
            "search": [x0, y0, x1, y1],
        }

    @classmethod
    def from_dict(cls, d):
        """从 JSON dict 还原（容忍 search 字段缺失，缺省退化为 bbox 界）。"""
        bbox = (d["x"], d["y"], d["w"], d["h"])
        if "search" in d:
            search = tuple(d["search"])
        else:
            search = (d["x"], d["y"], d["x"], d["y"])
        return cls(
            frame=d["frame"],
            pts_time=d.get("pts_time", 0.0),
            status=d["status"],
            bbox=bbox,
            score=d.get("score", 0.0),
            confirmed=d.get("confirmed", d["status"] == TRACKED),
            search=search,
        )


class LossEvent:
    """一段连续丢失区间（进入丢失 → 恢复确认的前一帧）。"""

    __slots__ = ("lost_frame", "recovered_frame", "lost_pts",
                 "recovered_pts", "frames_lost", "end_of_stream")

    def __init__(self, lost_frame, recovered_frame, lost_pts,
                 recovered_pts, frames_lost, end_of_stream=False):
        self.lost_frame = lost_frame
        self.recovered_frame = recovered_frame
        self.lost_pts = lost_pts
        self.recovered_pts = recovered_pts
        self.frames_lost = frames_lost
        self.end_of_stream = end_of_stream

    def to_dict(self):
        d = {
            "lost_frame": self.lost_frame,
            "lost_pts": round(self.lost_pts, 6),
            "frames_lost": self.frames_lost,
            "end_of_stream": self.end_of_stream,
        }
        if self.recovered_frame is not None:
            d["recovered_frame"] = self.recovered_frame
            d["recovered_pts"] = round(self.recovered_pts, 6)
        return d


class TrackResult:
    """整段视频的追踪结果：元信息 + 逐帧记录 + 丢失事件 + 统计。"""

    __slots__ = ("meta", "params", "frames", "loss_events", "stats")

    def __init__(self, meta, params, frames, loss_events, stats):
        self.meta = meta
        self.params = params
        self.frames = frames
        self.loss_events = loss_events
        self.stats = stats

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "params": self.params,
            "frames": [r.to_dict() for r in self.frames],
            "loss_events": [e.to_dict() for e in self.loss_events],
            "stats": self.stats,
        }


def build_track_result(frames, loss_events, meta, params, fps):
    """由逐帧记录汇总统计并组装 :class:`TrackResult`。

    ``meta``/``params`` 原样写入；统计项包括总帧数、确认/丢失帧数、
    丢失占比、丢失事件数、末尾是否仍丢失、确认帧分数极值与均值。
    """
    total = len(frames)
    confirmed_recs = [f for f in frames if f.status == TRACKED]
    lost_count = total - len(confirmed_recs)
    scores = [f.score for f in confirmed_recs]
    last_lost = bool(total) and frames[-1].status == LOST
    stats = {
        "frames": total,
        "tracked_frames": len(confirmed_recs),
        "lost_frames": lost_count,
        "lost_ratio": round(lost_count / total, 6) if total else 0.0,
        "loss_events": len(loss_events),
        "still_lost_at_end": last_lost,
        "score_min": round(min(scores), 6) if scores else None,
        "score_max": round(max(scores), 6) if scores else None,
        "score_mean": round(sum(scores) / len(scores), 6) if scores else None,
    }
    return TrackResult(dict(meta), dict(params), list(frames),
                       list(loss_events), stats)


def save_track_json(result, path):
    """把 :class:`TrackResult` 写成 UTF-8 JSON（ensure_ascii=False）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def load_track_json(path):
    """读回轨迹 JSON，还原为 :class:`TrackResult`。

    校验 schema 版本与逐帧序号连续性；缺字段抛 :class:`ValueError`。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"轨迹 JSON schema 版本 {version!r} 不受支持，"
            f"当前仅支持 {SCHEMA_VERSION}"
        )
    frames = [FrameRecord.from_dict(d) for d in data.get("frames", [])]
    for i, rec in enumerate(frames):
        if rec.frame != i:
            raise ValueError(
                f"轨迹帧序号不连续：第 {i} 条记录的 frame={rec.frame}"
            )
        if rec.status not in (TRACKED, LOST):
            raise ValueError(f"帧 {i} 状态非法：{rec.status!r}")
    events = [
        LossEvent(
            lost_frame=e["lost_frame"],
            recovered_frame=e.get("recovered_frame"),
            lost_pts=e.get("lost_pts", 0.0),
            recovered_pts=e.get("recovered_pts"),
            frames_lost=e["frames_lost"],
            end_of_stream=e.get("end_of_stream", False),
        )
        for e in data.get("loss_events", [])
    ]
    return TrackResult(
        meta=data.get("meta", {}),
        params=data.get("params", {}),
        frames=frames,
        loss_events=events,
        stats=data.get("stats", {}),
    )
