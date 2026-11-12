# -*- coding: utf-8 -*-
"""
打包管线：ffmpeg 固定 GOP 转码管道 + 自有的 fMP4 切点状态机。

设计要点
--------
1. ffmpeg 只负责「转码 + 输出 fragmented mp4 字节流」，使用强制 2s IDR 的
   GOP 参数；它不知道任何 HLS 分片的存在，不使用 `-f hls`。
2. 切点完全由本模块决定：只在视频 IDR 样本上切，且累计时长达到目标时长
   （允许帧级误差），首片天然从 IDR 起播。
3. 断流：ffmpeg 进程退出（EOF / 错误）后按 -ss 已消费位置重拉，
   输出时间戳用 per-track 偏移平移，保证单调不回退；续接点的下一个分片
   标记 EXT-X-DISCONTINUITY。
4. 编码配置变化（新 moov 的 stsd 不同）：重写 init 段，同样打 DISCONTINUITY。
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from typing import Dict, List, Optional

from .fmp4writer import Fmp4Writer, TrackSamples
from .manifest import (
    SegmentRecord, INIT_FILENAME, MASTER_FILENAME, SEGMENTS_JSON,
    render_playlist, validate_playlist_text, validate_records,
)
from .mp4reader import Fragment, FragmentedMp4Reader, MoovEvent, Sample, TrackInfo
from .session import PackagerState

log = logging.getLogger("hls-packager")

READ_BLOCK = 64 * 1024
# 达到目标时长的 96% 即允许在最近的 IDR 切（2s 目标下容忍一帧误差）
CUT_RATIO = 0.96


class FfmpegError(RuntimeError):
    pass


class HlsPackager:
    def __init__(self, state: PackagerState, *,
                 max_reconnect: int = 10,
                 reconnect_backoff: float = 1.0,
                 simulate_disconnect_at: Optional[float] = None,
                 override_input: Optional[str] = None):
        self.state = state
        self.max_reconnect = max_reconnect
        self.reconnect_backoff = reconnect_backoff
        # 测试钩子：输入消费到该秒数时主动杀掉 ffmpeg，模拟断流
        self.simulate_disconnect_at = simulate_disconnect_at
        # 测试钩子：重连后换一个输入文件（用于制造编码参数变化）
        self.override_input = override_input
        self._simulate_done = False

        self.tracks: Dict[int, TrackInfo] = {}
        self.writer: Optional[Fmp4Writer] = None
        self.queues: Dict[int, List[Sample]] = {}
        self.proc: Optional[subprocess.Popen] = None
        self._ffmpeg_epoch: int = 0
        # 本次拉流各 track 的锚点：(首个样本输入 dts, 输入 timescale)。
        # 每次拉流重置；时间线本身以秒存在 state.timelines 里，跨进程/换
        # timescale 都连续。
        self._pull_anchors: Dict[int, tuple] = {}
        # 本次拉流的输入侧记账：-ss 起点（consumed_seconds 存在 state 里）
        self._pull_start_ss: float = 0.0

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def run(self) -> List[SegmentRecord]:
        os.makedirs(self.state.output_dir, exist_ok=True)
        # 跨进程 --resume：已有分片在前，本次新拉的第一个分片属于新编码会话，
        # 按 HLS 规范需要 DISCONTINUITY。
        if self.state.next_sequence > 0:
            self.state.pending_discontinuity = True
            self._pull_anchors = {}

        attempts = 0
        while True:
            pull_error = self._run_one_pull()
            if pull_error is None and not self._simulate_kill:
                break  # 输入自然结束（干净 EOF）
            attempts += 1
            if attempts > self.max_reconnect:
                raise FfmpegError(
                    f"断流重连超过 {self.max_reconnect} 次仍未成功: "
                    f"{pull_error or '模拟断开'}")
            # 先把断流前已解析的样本收口为独立分片，再准备续点重拉
            self._flush_open_segment(force=True)
            self.state.pending_discontinuity = True
            self._pull_anchors = {}
            log.warning("输入流中断（%s），%.1fs 后从输入侧 %.3fs 续点重拉",
                        pull_error or "模拟断流",
                        self.reconnect_backoff,
                        self.state.consumed_seconds)
            time.sleep(self.reconnect_backoff)

        self._flush_open_segment(force=True)
        self.state.ended = True
        self._write_manifest_and_meta()
        self.state.save()
        return self.state.segment_records()

    # ------------------------------------------------------------------
    # 一次 ffmpeg 拉流（可能是首次，也可能是重连）；返回 None 表示干净 EOF，
    # 返回字符串表示可重连的错误。
    # ------------------------------------------------------------------
    def _run_one_pull(self) -> Optional[str]:
        self._simulate_kill = False
        self._pull_start_ss = self.state.consumed_seconds
        cmd = self._build_ffmpeg_cmd()
        log.info("启动 ffmpeg: %s", " ".join(cmd))
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=creationflags)
        self._ffmpeg_epoch += 1

        # 后台排空 stderr：避免 ffmpeg 因 stderr 管道写满而阻塞，
        # 也避免「回调异常 → finally 里阻塞读 stderr」的互锁死等。
        import threading
        stderr_chunks: list = []

        def _drain_stderr():
            try:
                assert self.proc is not None and self.proc.stderr is not None
                while True:
                    data = self.proc.stderr.read(4096)
                    if not data:
                        break
                    stderr_chunks.append(data)
            except Exception:  # pragma: no cover
                pass

        drainer = threading.Thread(target=_drain_stderr, daemon=True)
        drainer.start()

        reader = FragmentedMp4Reader(
            on_moov=self._on_moov, on_fragment=self._on_fragment)
        assert self.proc.stdout is not None
        try:
            while True:
                chunk = self.proc.stdout.read(READ_BLOCK)
                if not chunk:
                    break
                reader.feed(chunk)
                self._maybe_simulate_disconnect()
                if self._simulate_kill:
                    break
            reader.flush()
        except BaseException:
            # 回调里的任何异常都先终止子进程，防止管道互锁后整进程挂死
            self.proc.kill()
            self.proc.wait()
            drainer.join(timeout=2)
            self.proc = None
            raise
        rc = self.proc.wait()
        drainer.join(timeout=2)
        self.proc = None
        stderr = b"".join(stderr_chunks)

        if self._simulate_kill:
            log.warning("已在输入侧 %.3fs 处主动断开（模拟断流）",
                        self.state.consumed_seconds)
            return "模拟断流"
        if rc != 0:
            tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
            return "ffmpeg rc=%s: %s" % (rc, " | ".join(tail))
        return None

    def _build_ffmpeg_cmd(self) -> List[str]:
        target, seek_seconds = self._current_target_and_seek()

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if seek_seconds > 0:
            # 续点重拉：-ss 放在 -i 前做输入定位，ffmpeg 从最近关键帧解码并
            # 丢弃定位点之前的帧，输出时间戳近似从 0 重新开始；
            # 少量重叠由 _shift_sample 的钳制逻辑兜底。
            cmd += ["-ss", f"{seek_seconds:.3f}"]
        cmd += ["-i", target]

        gop = max(1, int(round(self.state.segment_duration * self._fps())))
        # 视频：固定 GOP、关闭场景切帧、强制每 2s 一个 IDR
        cmd += [
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{self.state.segment_duration:.3f})",
        ]
        if self._has_audio(target):
            cmd += [
                "-map", "0:a:0?",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "48000",
                "-ac", "2",
            ]
        else:
            cmd += ["-an"]
        cmd += [
            "-f", "mp4",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
            "-video_track_timescale", str(self._video_timescale()),
            "pipe:1",
        ]
        return cmd

    def _upsert_timeline(self, tid: int, timescale: int) -> dict:
        """登记/更新 track 时间线。timescale 变化时保留已推进的秒位置。"""
        tl = self.state.timelines.get(tid)
        if tl is None:
            tl = {"track_id": tid, "timescale": timescale,
                  "next_seconds": 0.0}
            self.state.timelines[tid] = tl
        else:
            tl["timescale"] = timescale
        return tl

    # ------------------------------------------------------------------
    # moov 事件：初始化 / 编码变化检测
    # ------------------------------------------------------------------
    def _on_moov(self, event: MoovEvent) -> None:
        new_tracks = event.tracks
        if self.writer is None:
            self.tracks = dict(new_tracks)
            self.writer = Fmp4Writer(list(new_tracks.values()))
            self._write_init()   # 首份固定写 init.mp4
            for tid, info in new_tracks.items():
                self.queues.setdefault(tid, [])
                self._upsert_timeline(tid, info.timescale)
                self.state.stsd_snapshots[tid] = info.stsd_blob.hex()
            return

        # 后续 moov：检测 stsd 变化
        changed = False
        for tid, info in new_tracks.items():
            old_hex = self.state.stsd_snapshots.get(tid)
            if old_hex is not None and old_hex != info.stsd_blob.hex():
                changed = True
        if changed:
            log.warning("检测到编码配置变化，刷新 init 段并标记 DISCONTINUITY")
            self._flush_open_segment(force=True)
            self.state.pending_discontinuity = True
            self._pull_anchors = {}   # 新编码会话重新建立锚点
            self.tracks = dict(new_tracks)
            self.writer = Fmp4Writer(list(new_tracks.values()))
            # 不能覆盖旧 init：旧分片仍引用它。写一份新的 initN.mp4。
            self.state.next_init_index += 1
            self.state.current_init_file = (
                f"init{self.state.next_init_index}.mp4")
            self._write_init()
            for tid, info in new_tracks.items():
                self.state.stsd_snapshots[tid] = info.stsd_blob.hex()
                self.queues.setdefault(tid, [])
                # 保留 next_seconds，仅切换 timescale，保证时间线不回退
                self._upsert_timeline(tid, info.timescale)
        else:
            self.tracks = dict(new_tracks)

    # ------------------------------------------------------------------
    # fragment 事件：时间戳平移 + 切点状态机
    # ------------------------------------------------------------------
    def _on_fragment(self, frag: Fragment) -> None:
        for sample in frag.samples:
            self._track_input_position(sample)
            self._shift_sample(sample)
            self.queues.setdefault(sample.track_id, []).append(sample)

        video_id = self.writer.video_id if self.writer else None
        if video_id is None:
            return
        vq = self.queues[video_id]
        # 状态机：只在 IDR 上切。第一个进入队列的 IDR 开片（首片独立起播），
        # 之后到达的 IDR 若当前打开分片时长达标则收口。
        for idx in range(1, len(vq)):
            sample = vq[idx]
            if sample.sync and self._should_cut(sample):
                self._cut_at(sample)
                break

    def _track_input_position(self, sample: Sample) -> None:
        """记录本次拉流在「输入文件侧」消费到的位置，供下一次 -ss 续点。

        必须在 _shift_sample 之前调用，此时 pts 仍是 ffmpeg 本拉流的局部值。
        """
        info = self.tracks.get(sample.track_id)
        if info is None or info.media_type != "video":
            return
        self.state.consumed_seconds = (
            self._pull_start_ss
            + sample.pts / info.timescale
        )

    def _should_cut(self, arriving_idr: Sample) -> bool:
        video_id = self.writer.video_id
        vq = self.queues[video_id]
        if len(vq) < 2:
            return False
        first = vq[0]
        if not first.sync:
            # 防御：队列首样本不是 IDR 时，在到达的 IDR 处强制对齐开片
            return True
        info = self.tracks[video_id]
        span = (arriving_idr.pts - first.pts) / info.timescale
        return span >= self.state.segment_duration * CUT_RATIO

    def _shift_sample(self, sample: Sample) -> None:
        """
        把 ffmpeg 本次拉流的局部时间戳平移到全局连续时间线。

        全局时间线以「秒」记账（state.timelines[*].next_seconds），与具体
        timescale 解耦；每次拉流以该 track 第一个样本建立锚点：
            全局秒 = 锚点秒 + (本样本 dts - 锚点输入 dts) / 输入 timescale
        再按「当前」timescale 换算回整数 dts。这样重拉、换帧率/分辨率都能
        保持时间戳单调连续。cto（B 帧合成偏移）保持本拉流单位不变。
        """
        tid = sample.track_id
        info = self.tracks.get(tid)
        in_ts = info.timescale if info else 90000
        tl = self.state.timelines.get(tid)
        if tl is None:
            tl = self._upsert_timeline(tid, in_ts)
        out_ts = tl["timescale"]

        if tid not in self._pull_anchors:
            # 锚点：本拉流首个样本的输入 dts 对齐到全局时间线末尾
            self._pull_anchors[tid] = (sample.dts, in_ts,
                                       tl["next_seconds"])
        anchor_dts, anchor_ts, anchor_seconds = self._pull_anchors[tid]

        rel_seconds = (sample.dts - anchor_dts) / anchor_ts
        global_seconds = anchor_seconds + rel_seconds
        # 防御：重拉重叠导致回退时，钳制到时间线末尾
        if global_seconds < tl["next_seconds"]:
            global_seconds = tl["next_seconds"]

        new_dts = round(global_seconds * out_ts)
        # cto 用当前 timescale 换算，pts 由 dts + cto 得到
        cto_out = round(sample.cto * (out_ts / in_ts))
        sample.dts = new_dts
        sample.cto = cto_out
        sample.pts = new_dts + cto_out

    def _cut_at(self, idr_sample: Sample) -> None:
        """到达新 IDR：把它之前的样本收口为一个分片，IDR 留给下一片。"""
        video_id = self.writer.video_id
        vq = self.queues[video_id]
        cut_idx = vq.index(idr_sample)
        groups: List[TrackSamples] = []
        for tid, q in self.queues.items():
            if tid == video_id:
                picked, remain = q[:cut_idx], q[cut_idx:]
            else:
                # 音频按时间不超过视频切点取走
                bound_pts = idr_sample.pts
                picked, remain = [], []
                for s in q:
                    (picked if s.pts < bound_pts else remain).append(s)
            if picked:
                groups.append(TrackSamples(info=self.tracks[tid],
                                           samples=picked))
            self.queues[tid] = remain
        self._emit_segment(groups)

    def _flush_open_segment(self, force: bool) -> None:
        if not self.queues or not any(self.queues.values()):
            return
        groups = []
        for tid, q in self.queues.items():
            if q:
                groups.append(TrackSamples(info=self.tracks[tid], samples=q))
        if groups:
            self._emit_segment(groups)
        self.queues = {tid: [] for tid in self.queues}

    # ------------------------------------------------------------------
    # 分片落盘
    # ------------------------------------------------------------------
    def _emit_segment(self, groups: List[TrackSamples]) -> None:
        if not groups:
            return
        # 保证视频组在前面（sidx 时间线选择依赖顺序）
        groups.sort(key=lambda g: 0 if g.info.media_type == "video" else 1)
        seq = self.state.next_sequence
        filename = f"seg{seq}.m4s"

        # 推进各 track 时间线（按秒记账，下一片必须严格接在本段末尾之后）
        for g in groups:
            if not g.samples:
                continue
            last = g.samples[-1]
            tl = self.state.timelines[g.info.track_id]
            end_seconds = (last.dts + last.duration) / tl["timescale"]
            if end_seconds > tl["next_seconds"]:
                tl["next_seconds"] = end_seconds

        data = self.writer.build_segment(seq, groups)
        path = os.path.join(self.state.output_dir, filename)
        with open(path, "wb") as f:
            f.write(data)

        v_info = next((g.info for g in groups
                       if g.info.media_type == "video"), groups[0].info)
        v_samples = next((g.samples for g in groups
                          if g.info.media_type == "video"), groups[0].samples)
        # start_time 以 DTS（= tfdt/baseMediaDecodeTime）为准，这是分片在媒体
        # 时间线上的规范位置；首帧 PTS 可能因 B 帧合成偏移略大于 0，属正常。
        start_time = v_samples[0].dts / v_info.timescale
        duration = sum(s.duration for s in v_samples) / v_info.timescale
        first_idr = bool(v_samples[0].sync)

        rec = SegmentRecord(
            sequence_number=seq,
            filename=filename,
            start_time=round(start_time, 6),
            duration=round(duration, 6),
            discontinuity=self.state.pending_discontinuity,
            first_frame_idr=first_idr,
            size_bytes=len(data),
            init_file=self.state.current_init_file,
        )
        self.state.segments.append(rec.to_json_dict())
        self.state.next_sequence += 1
        # 该分片消费掉待处理的 DISCONTINUITY 标记；新的标记只由
        # 断流重拉 / 编码变化事件显式设置（见 _maybe_simulate_disconnect、
        # _on_moov 与 run() 的重连分支）。
        self.state.pending_discontinuity = False

        self._write_manifest_and_meta()
        self.state.save()
        log.info("落盘 %s start=%.3fs dur=%.3fs idr=%s discont=%s",
                 filename, rec.start_time, rec.duration,
                 first_idr, rec.discontinuity)

    # ------------------------------------------------------------------
    def _write_init(self) -> None:
        path = os.path.join(self.state.output_dir,
                            self.state.current_init_file)
        with open(path, "wb") as f:
            f.write(self.writer.build_init())

    def _write_manifest_and_meta(self) -> None:
        records = self.state.segment_records()
        target = max(1, round(self.state.segment_duration))
        text = render_playlist(records, target, ended=self.state.ended)
        validate_playlist_text(text)
        if records:
            validate_records(records, self.state.segment_duration)
        m3u8_path = os.path.join(self.state.output_dir, MASTER_FILENAME)
        with open(m3u8_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        meta_path = os.path.join(self.state.output_dir, SEGMENTS_JSON)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([r.to_json_dict() for r in records], f,
                      ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def _maybe_simulate_disconnect(self) -> None:
        if (self.simulate_disconnect_at is None or self._simulate_done
                or self.proc is None):
            return
        if self.state.consumed_seconds >= self.simulate_disconnect_at:
            self._simulate_done = True
            self._simulate_kill = True
            # 注意：这里不置 pending_discontinuity / 不动 _pull_anchors。
            # 断流前残片仍属于旧编码会话，DISCONTINUITY 只能挂在「新会话
            # 第一片」上；标记与锚点重置统一由 run() 在残片收口之后处理。
            self.proc.kill()

    # ------------------------------------------------------------------
    # 输入探测（ffprobe），按路径缓存
    # ------------------------------------------------------------------
    def _fps(self) -> float:
        target = self._current_target()
        cache = self.__dict__.setdefault("_fps_cache", {})
        if target in cache:
            return cache[target]
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                "-of", "json", target,
            ], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            st = json.loads(out)["streams"][0]
            for key in ("r_frame_rate", "avg_frame_rate"):
                num, den = st[key].split("/")
                if float(den) != 0:
                    val = float(num) / float(den)
                    if val > 0:
                        cache[target] = val
                        return val
        except Exception as e:  # pragma: no cover
            log.warning("ffprobe 帧率探测失败，回退 25fps: %s", e)
        cache[target] = 25.0
        return 25.0

    def _current_target(self) -> str:
        if (self.override_input is not None
                and self.state.consumed_seconds > 0):
            return self.override_input
        return self.state.input_path

    def _current_target_and_seek(self):
        """返回 (本次 ffmpeg 输入路径, -ss 秒数)。换源后续拉新文件从头开始。"""
        if (self.override_input is not None
                and self.state.consumed_seconds > 0):
            return self.override_input, 0.0
        return self.state.input_path, self.state.consumed_seconds

    def _video_timescale(self) -> int:
        # 用帧率的整数倍作为 timescale，25/30fps 下时间戳无小数
        fps = self._fps()
        return int(round(fps * 1000)) if fps <= 30 else 90000

    def _has_audio(self, path: str) -> bool:
        cache = self.__dict__.setdefault("_has_audio_cache", {})
        if path in cache:
            return cache[path]
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", path,
            ], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            cache[path] = b"audio" in out
        except Exception:
            cache[path] = False
        return cache[path]
