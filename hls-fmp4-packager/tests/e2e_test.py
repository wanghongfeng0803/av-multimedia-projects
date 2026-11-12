#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端验收脚本（标准库 + ffmpeg/ffprobe/ffplay）。

覆盖场景：
  1. 正常打包：2s 分片、IDR 对齐、首片独立起播
  2. 进程内断流：--simulate-disconnect 杀 ffmpeg 后自动续点重拉，
     DISCONTINUITY 位置正确、时间戳不回退
  3. 跨进程 --resume：杀掉进程后用 session.json 从断点继续，不从头
  4. 编码参数变化：续传换不同分辨率源，init 刷新 + DISCONTINUITY
最后：validate.py 全量校验 + ffplay 冷启动计时（验收线 3s）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "tests", "work")
SRC_DIR = os.path.join(WORK, "sources")

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          creationflags=CREATE_NO_WINDOW, **kw)


def ff(args, check=True):
    r = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args)
    if check and r.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {r.stderr.strip()}")
    return r


def make_sources():
    os.makedirs(SRC_DIR, exist_ok=True)
    # 主源：12s、25fps、不规则 GOP（源 GOP=17 帧）、带正弦音轨。
    # testsrc2 画面带时间戳数字，肉眼/日志都能识别续传位置。
    main = os.path.join(SRC_DIR, "main.mp4")
    if not os.path.exists(main):
        ff([
            "-f", "lavfi", "-i",
            "testsrc2=size=1280x720:rate=25:duration=12",
            "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=12",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-g", "17", "-keyint_min", "17", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "96k",
            main,
        ])

    # 续传源 B：8s，分辨率/帧率不同，用于编码参数变化场景
    src_b = os.path.join(SRC_DIR, "src_b_640x480.mp4")
    if not os.path.exists(src_b):
        ff([
            "-f", "lavfi", "-i",
            "testsrc2=size=640x480:rate=30:duration=8",
            "-f", "lavfi", "-i",
            "sine=frequency=880:sample_rate=48000:duration=8",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-g", "30", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "96k",
            src_b,
        ])

    # 纯视频源（无音频），覆盖无音轨分支
    video_only = os.path.join(SRC_DIR, "video_only.mp4")
    if not os.path.exists(video_only):
        ff([
            "-f", "lavfi", "-i",
            "testsrc2=size=854x480:rate=25:duration=6",
            "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-g", "25", "-sc_threshold", "0",
            video_only,
        ])
    return main, src_b, video_only


def packager(*args):
    cmd = [sys.executable, os.path.join(ROOT, "hls_packager.py")] + \
          [str(a) for a in args]
    return run(cmd)


def validate(output_dir):
    r = run([sys.executable, os.path.join(ROOT, "validate.py"), output_dir])
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise AssertionError(f"校验未通过: {output_dir}")
    return r.stdout


def load_segments(output_dir):
    with open(os.path.join(output_dir, "segments.json"), encoding="utf-8") as f:
        return json.load(f)


def assert_timeline(segments, expect_min_dur=1.0, label=""):
    prev_end = 0.0
    for i, s in enumerate(segments):
        assert s["sequence_number"] == i, f"{label}: 序号不连续"
        assert s["first_frame_idr"], f"{label}: 分片{i} 首帧非 IDR"
        assert s["duration"] >= expect_min_dur or i == len(segments) - 1, \
            f"{label}: 分片{i} 时长异常 {s['duration']}"
        assert abs(s["start_time"] - prev_end) < 0.01, \
            f"{label}: 分片{i} 起点 {s['start_time']} 与上一片结束 {prev_end} 不衔接"
        prev_end = s["start_time"] + s["duration"]
    return prev_end


def scenario_normal(main):
    print("\n[1/5] 正常打包")
    out = os.path.join(WORK, "case1_normal")
    shutil.rmtree(out, ignore_errors=True)
    r = packager(main, out)
    assert r.returncode == 0, r.stderr
    segs = load_segments(out)
    total = assert_timeline(segs, label="normal")
    assert len(segs) >= 5, f"12s 输入应至少 5 片，实际 {len(segs)}"
    assert all(not s["discontinuity"] for s in segs), "正常场景不应有 DISCONT"
    assert 11.0 < total <= 12.5, f"总时长 {total} 异常"
    print(f"  OK: {len(segs)} 片，总时长 {total:.3f}s，无 DISCONTINUITY")
    return out


def scenario_disconnect(main):
    print("\n[2/5] 进程内断流续传")
    out = os.path.join(WORK, "case2_disconnect")
    shutil.rmtree(out, ignore_errors=True)
    r = packager(main, out, "--simulate-disconnect", "4.5",
                 "--reconnect-backoff", "0.2")
    assert r.returncode == 0, r.stderr
    segs = load_segments(out)
    total = assert_timeline(segs, label="disconnect")
    disc = [s["sequence_number"] for s in segs if s["discontinuity"]]
    assert len(disc) == 1, f"应恰有 1 个 DISCONTINUITY，实际 {disc}"
    assert disc[0] >= 2, f"断流点后的分片序号异常: {disc}"
    print(f"  OK: {len(segs)} 片，DISCONTINUITY 位于分片 {disc[0]} 前，"
          f"总时长 {total:.3f}s")
    return out


def scenario_resume(main):
    print("\n[3/5] 跨进程 --resume 续传")
    out = os.path.join(WORK, "case3_resume")
    shutil.rmtree(out, ignore_errors=True)
    r1 = packager(main, out, "--simulate-disconnect", "3.0",
                  "--reconnect-backoff", "0.2", "--max-reconnect", "0")
    # max-reconnect=0：第一次断开即退出，模拟进程崩溃
    assert r1.returncode != 0, "max-reconnect=0 时应在断流后退出"
    first = load_segments(out)
    n_first = len(first)
    assert n_first >= 1, "崩溃前至少应落盘 1 片"
    print(f"  第一次进程退出，已落盘 {n_first} 片")

    r2 = packager(main, out, "--resume", "--reconnect-backoff", "0.2")
    assert r2.returncode == 0, r2.stderr
    segs = load_segments(out)
    assert len(segs) > n_first, "resume 后分片数未增长"
    total = assert_timeline(segs, label="resume")
    disc = [s["sequence_number"] for s in segs if s["discontinuity"]]
    assert disc == [n_first], f"resume 续接点 DISCONT 应在 {n_first}，实际 {disc}"
    print(f"  OK: 续传后共 {len(segs)} 片，DISCONTINUITY 位于分片 {n_first} 前，"
          f"总时长 {total:.3f}s")
    return out


def scenario_codec_change(main, src_b):
    print("\n[4/5] 编码参数变化（换分辨率源）")
    out = os.path.join(WORK, "case4_codec_change")
    shutil.rmtree(out, ignore_errors=True)
    r = packager(main, out, "--simulate-disconnect", "5.0",
                 "--reconnect-input", src_b,
                 "--reconnect-backoff", "0.2")
    assert r.returncode == 0, r.stderr
    segs = load_segments(out)
    total = assert_timeline(segs, label="codec-change")
    disc = [s["sequence_number"] for s in segs if s["discontinuity"]]
    assert len(disc) >= 1, f"换源后应有 DISCONTINUITY，实际 {disc}"
    # 必须保留两份 init，且变化点之后的分片引用新 init
    assert os.path.isfile(os.path.join(out, "init.mp4"))
    assert os.path.isfile(os.path.join(out, "init1.mp4")), "缺少版本化 init1.mp4"
    after = [s for s in segs if s["sequence_number"] >= disc[0]]
    assert all(s.get("init_file") == "init1.mp4" for s in after), \
        "编码变化后的分片必须引用 init1.mp4"
    before = [s for s in segs if s["sequence_number"] < disc[0]]
    assert all(s.get("init_file", "init.mp4") == "init.mp4" for s in before)
    with open(os.path.join(out, "master.m3u8"), encoding="utf-8") as f:
        playlist = f.read()
    assert playlist.count("#EXT-X-MAP") == 2, "m3u8 应有两个 EXT-X-MAP"
    print(f"  OK: {len(segs)} 片，DISCONTINUITY 位于 {disc}，"
          f"init.mp4→init1.mp4，总时长 {total:.3f}s")
    return out


def scenario_video_only(video_only):
    print("\n[5/5] 无音频源")
    out = os.path.join(WORK, "case5_video_only")
    shutil.rmtree(out, ignore_errors=True)
    r = packager(video_only, out)
    assert r.returncode == 0, r.stderr
    segs = load_segments(out)
    total = assert_timeline(segs, label="video-only")
    print(f"  OK: {len(segs)} 片，总时长 {total:.3f}s")
    return out


def ffprobe_keyframes(output_dir):
    """ffprobe 直连 m3u8，验证关键帧节奏 ≈ 2s。"""
    r = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=key_frame,best_effort_timestamp_time",
        "-of", "json", os.path.join(output_dir, "master.m3u8"),
    ])
    assert r.returncode == 0, r.stderr
    frames = json.loads(r.stdout)["frames"]
    kt = [float(f["best_effort_timestamp_time"]) for f in frames
          if f["key_frame"] == 1]
    gaps = [b - a for a, b in zip(kt, kt[1:])]
    bad = [g for g in gaps if abs(g - 2.0) > 0.12]
    assert not bad, f"关键帧间隔偏离 2s: {bad[:5]}"
    print(f"  ffprobe: {len(kt)} 个关键帧，间隔均为 2.0s±0.12s")


def decode_integrity(output_dir, label="", records=None):
    """
    整段解码到 null，要求 0 错误/无 corrupt，证明无花屏、可完整播放。

    普通场景（单一 init）：直接让 ffmpeg 解整个 m3u8。
    含 EXT-X-MAP 切换时：ffmpeg 的 HLS demuxer 对单 playlist 内 fMP4 的
    MAP 重载支持不完整（已知限制），此时按 init 分组合并后逐组解码——
    这是验证「每份 init 与其分片严格匹配」的正确做法，真实 HLS 客户端
    （Safari/hls.js/ExoPlayer）原生支持该 MAP 切换。
    """
    def assert_clean(r, tag):
        assert r.returncode == 0, f"{tag} 解码失败: {r.stderr[-800:]}"
        bad = [ln for ln in (r.stderr or "").splitlines()
               if "corrupt" in ln.lower() or "error while decoding" in ln.lower()]
        assert not bad, f"{tag} 解码出现错误帧: {bad[:5]}"

    map_groups = {}
    if records:
        for rec in records:
            map_groups.setdefault(rec.get("init_file", "init.mp4"),
                                  []).append(rec["filename"])
    if len(map_groups) <= 1:
        r = run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", os.path.join(output_dir, "master.m3u8"),
            "-f", "null", "-",
        ])
        assert_clean(r, label or os.path.basename(output_dir))
        print(f"  整段解码无错误（{label or os.path.basename(output_dir)}）")
        return

    # 多 init：逐组合并解码
    for init_file, names in map_groups.items():
        merged = os.path.join(output_dir, f"_chk_{init_file}")
        with open(merged, "wb") as o:
            with open(os.path.join(output_dir, init_file), "rb") as f:
                o.write(f.read())
            for n in names:
                with open(os.path.join(output_dir, n), "rb") as f:
                    o.write(f.read())
        r = run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-i", merged, "-f", "null", "-"])
        assert_clean(r, f"{label}/{init_file}")
        os.remove(merged)
        print(f"  init 组 {init_file}（{len(names)} 片）解码无错误（{label}）")


def ffplay_cold_start(output_dir, timeout_s=15.0):
    """
    冷启动计时：启动 ffplay 进程到它打开窗口/解码首帧的墙钟时间。
    用 -autoexit -t 2 让 ffplay 播放 2s 后自行退出；统计进程存活时长，
    再扣除 2s 播放时长，得到「启动到出画面」的近似值。
    """
    m3u8 = os.path.join(output_dir, "master.m3u8")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        ["ffplay", "-hide_banner", "-loglevel", "warning",
         "-autoexit", "-t", "2", "-x", "640", "-y", "360",
         "-window_title", "hls-coldstart", m3u8],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW)
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("ffplay 超时未退出")
    wall = time.monotonic() - t0
    startup = max(0.0, wall - 2.0)
    log = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
    assert rc == 0, f"ffplay 退出码 {rc}: {log[-500:]}"
    print(f"  ffplay 冷启动（墙钟 {wall:.2f}s - 播放 2s）≈ {startup:.2f}s")
    assert startup <= 3.0, f"冷启动 {startup:.2f}s 超过 3s 验收线"
    return startup


def main():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("需要 ffmpeg/ffprobe 在 PATH 中", file=sys.stderr)
        return 2
    os.makedirs(WORK, exist_ok=True)
    main_src, src_b, video_only = make_sources()

    outs = []
    outs.append(("normal", scenario_normal(main_src)))
    outs.append(("disconnect", scenario_disconnect(main_src)))
    outs.append(("resume", scenario_resume(main_src)))
    outs.append(("codec-change", scenario_codec_change(main_src, src_b)))
    outs.append(("video-only", scenario_video_only(video_only)))

    print("\n=== 独立校验脚本 ===")
    for name, out in outs:
        print(f"-- {name}")
        validate(out)

    print("\n=== 整段解码完整性（无花屏/无解码错误）===")
    for name, out in outs:
        decode_integrity(out, label=name, records=load_segments(out))

    print("\n=== ffprobe 关键帧节奏（normal 场景）===")
    ffprobe_keyframes(outs[0][1])

    if shutil.which("ffplay"):
        print("\n=== ffplay 冷启动计时（验收线 ≤ 3s）===")
        ffplay_cold_start(outs[0][1])
    else:
        print("\n[跳过] 未找到 ffplay")

    print("\n全部端到端验收通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
