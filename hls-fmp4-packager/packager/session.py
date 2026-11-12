# -*- coding: utf-8 -*-
"""
打包会话状态：持久化到输出目录的 session.json。

作用有两个：
1. 同一次打包中 ffmpeg 断流重拉时，内存里的状态机本身即可续点；
2. 进程退出后再次以 --resume 启动，从该文件恢复：已落盘分片不重打、
   媒体序列号与时间线继续递增，ffmpeg 用 -ss 从断点位置重新拉流。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .manifest import SegmentRecord, INIT_FILENAME

SESSION_FILENAME = "session.json"
INIT_FILENAME_DEFAULT = INIT_FILENAME


@dataclass
class PackagerState:
    input_path: str
    output_dir: str
    segment_duration: float
    next_sequence: int = 0
    # 时间线（输出侧，平移后的连续时间戳）
    timelines: dict = field(default_factory=dict)   # track_id -> TrackTimeline dict
    # 已完成分片
    segments: List[dict] = field(default_factory=list)
    # 输入侧已经消费到的位置（秒），重拉时作为 ffmpeg -ss 起点
    consumed_seconds: float = 0.0
    # 输入编码配置快照（stsd blob 用 hex 存），变化检测用
    stsd_snapshots: dict = field(default_factory=dict)
    # 下一个待落盘分片是否需要 DISCONTINUITY
    pending_discontinuity: bool = False
    # init 段版本管理：init.mp4 为第 0 份，编码变化后递增写 init1.mp4 ...
    next_init_index: int = 0
    current_init_file: str = "init.mp4"
    ended: bool = False

    @classmethod
    def new(cls, input_path: str, output_dir: str,
            segment_duration: float) -> "PackagerState":
        return cls(input_path=os.path.abspath(input_path),
                   output_dir=os.path.abspath(output_dir),
                   segment_duration=segment_duration)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "PackagerState":
        state = cls(
            input_path=data["input_path"],
            output_dir=data["output_dir"],
            segment_duration=data["segment_duration"],
        )
        state.next_sequence = data["next_sequence"]
        # JSON 的对象键只能是字符串：track_id 落盘后变成 "1"，这里统一转回
        # int，否则 resume 后用整数 id 查时间线会得到 None 而重建空时间线。
        state.timelines = {int(k): v for k, v in data["timelines"].items()}
        state.segments = data["segments"]
        state.consumed_seconds = data["consumed_seconds"]
        state.stsd_snapshots = {
            int(k): v for k, v in data["stsd_snapshots"].items()}
        state.pending_discontinuity = data["pending_discontinuity"]
        state.next_init_index = data.get("next_init_index", 0)
        state.current_init_file = data.get("current_init_file",
                                           INIT_FILENAME_DEFAULT)
        state.ended = data.get("ended", False)
        return state

    def save(self) -> None:
        path = os.path.join(self.output_dir, SESSION_FILENAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, output_dir: str) -> Optional["PackagerState"]:
        path = os.path.join(output_dir, SESSION_FILENAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def segment_records(self) -> List[SegmentRecord]:
        return [SegmentRecord(**item) for item in self.segments]
