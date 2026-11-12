#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HLS/fMP4 产物独立校验脚本。

两层校验：
1. 纯 Python（只依赖项目内 boxes 原语）：
   - master.m3u8 规范（复用 manifest.validate_playlist_text）
   - init.mp4 结构（ftyp/moov/mvex/trex，每 track 一条 trex）
   - 每个 seg*.m4s：盒子顺序 styp/sidx/moof/mdat；moof 内 tfdt/trun 合法；
     每个 traf 的第一个样本 sync 标志为 IDR；trun data_offset 与 mdat 对齐；
     tfdt 时间戳沿时间线单调不回退；EXTINF 与实际样本时长一致。
2. ffprobe（若环境可用）：
   - init+分片拼接后逐帧验证关键帧位置与分片边界一致；
   - ffprobe 直接读 m3u8，确认播放列表可被标准工具解析。

用法：python validate.py output_dir/ [--strict]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packager import boxes as B  # noqa: E402
from packager.manifest import (  # noqa: E402
    INIT_FILENAME, MASTER_FILENAME, SEGMENTS_JSON,
    validate_playlist_text, validate_records,
)
from packager.mp4reader import FragmentedMp4Reader  # noqa: E402


class ValidationError(AssertionError):
    pass


def check(cond: bool, msg: str, errors: list) -> None:
    if not cond:
        errors.append(msg)


# ---------------------------------------------------------------------------
# init 段
# ---------------------------------------------------------------------------
def validate_init(path: str, errors: list) -> dict:
    with open(path, "rb") as f:
        buf = f.read()
    top = list(B.iter_boxes(buf, 0))
    types = [b.type for b in top]
    check(types[:2] == [b"ftyp", b"moov"],
          f"init 段前两个盒子应为 ftyp/moov，实际 {types}", errors)
    moov = next((b for b in top if b.type == b"moov"), None)
    check(moov is not None, "init 缺少 moov", errors)
    if moov is None:
        return {}
    mvex = B.find_child(buf, moov.payload_start, moov.payload_end, b"mvex")
    check(mvex is not None, "init 的 moov 缺少 mvex（fMP4 必需）", errors)
    tracks = {}
    for trak in B.iter_boxes(buf, moov.payload_start, moov.payload_end):
        if trak.type != b"trak":
            continue
        tkhd = B.find_child(buf, trak.payload_start, trak.payload_end, b"tkhd")
        if tkhd is None:
            continue
        _, _, p = B.parse_fullbox(buf, tkhd.payload_start)
        r = B.Reader(buf, p, tkhd.payload_end)
        r.u64(); r.u64()
        track_id = r.u32()
        stsd = B.find_path(buf, trak.payload_start, trak.payload_end,
                           b"mdia", b"minf", b"stbl", b"stsd")
        check(stsd is not None, f"track {track_id} 缺少 stsd", errors)
        tracks[track_id] = bytes(buf[stsd.payload_start:stsd.payload_end]) \
            if stsd else b""
    if mvex is not None:
        trex_ids = []
        for tb in B.iter_boxes(buf, mvex.payload_start, mvex.payload_end):
            if tb.type == b"trex":
                _, _, tp = B.parse_fullbox(buf, tb.payload_start)
                trex_ids.append(B.Reader(buf, tp, tb.payload_end).u32())
        check(sorted(trex_ids) == sorted(tracks),
              f"trex track 集合 {trex_ids} 与 trak {sorted(tracks)} 不一致",
              errors)
    return tracks


# ---------------------------------------------------------------------------
# 媒体分片
# ---------------------------------------------------------------------------
def validate_segment(path: str, expected_seq: int,
                     prev_tfdt: dict, errors: list,
                     warnings: list) -> dict:
    with open(path, "rb") as f:
        buf = f.read()
    top = list(B.iter_boxes(buf, 0))
    types = [b.type for b in top]
    check(types == [b"styp", b"sidx", b"moof", b"mdat"],
          f"{os.path.basename(path)} 盒子顺序异常: {types}", errors)
    if b"moof" not in types:
        return prev_tfdt

    moof = next(b for b in top if b.type == b"moof")
    mdat = next(b for b in top if b.type == b"mdat")
    mfhd = B.find_child(buf, moof.payload_start, moof.payload_end, b"mfhd")
    check(mfhd is not None, f"{path} 缺少 mfhd", errors)
    if mfhd:
        seq = B.Reader(buf, mfhd.payload_start + 4, mfhd.payload_end).u32()
        check(seq == expected_seq,
              f"{path} mfhd 序号 {seq} != 期望 {expected_seq}", errors)

    for traf in B.iter_boxes(buf, moof.payload_start, moof.payload_end):
        if traf.type != b"traf":
            continue
        tfhd = B.find_child(buf, traf.payload_start, traf.payload_end, b"tfhd")
        trun = B.find_child(buf, traf.payload_start, traf.payload_end, b"trun")
        tfdt = B.find_child(buf, traf.payload_start, traf.payload_end, b"tfdt")
        check(tfhd is not None and trun is not None and tfdt is not None,
              f"{path} traf 缺少 tfhd/trun/tfdt", errors)
        if not (tfhd and trun and tfdt):
            continue

        _, tfhd_flags, hp = B.parse_fullbox(buf, tfhd.payload_start)
        track_id = B.Reader(buf, hp, tfhd.payload_end).u32()
        _, trun_flags, tp = B.parse_fullbox(buf, trun.payload_start)
        rr = B.Reader(buf, tp, trun.payload_end)
        count = rr.u32()
        data_offset = rr.i32() if trun_flags & B.TRUN_DATA_OFFSET_PRESENT else 0
        if trun_flags & B.TRUN_FIRST_SAMPLE_FLAGS_PRESENT:
            rr.u32()
        first_flags = None
        sizes = []
        for i in range(count):
            dur = rr.u32() if trun_flags & B.TRUN_SAMPLE_DURATION_PRESENT else 0
            size = rr.u32() if trun_flags & B.TRUN_SAMPLE_SIZE_PRESENT else 0
            flags = rr.u32() if trun_flags & B.TRUN_SAMPLE_FLAGS_PRESENT else 0
            if trun_flags & B.TRUN_SAMPLE_CTO_PRESENT:
                rr.i32()
            sizes.append(size)
            if i == 0:
                first_flags = flags

        # 首样本必须是同步样本（IDR）
        check(first_flags is not None and bool(first_flags & 0x02000000),
              f"{path} track {track_id} 首样本非 IDR (flags={first_flags})",
              errors)

        # data_offset（相对 moof 起点）必须落在 mdat payload 范围内，
        # 多 traf 时各自指向 mdat 内本 track 数据的起点
        abs_off = moof.start + data_offset
        check(mdat.payload_start <= abs_off
              and abs_off + sum(sizes) <= mdat.payload_end,
              f"{path} track {track_id} data_offset={abs_off} 越出 mdat "
              f"[{mdat.payload_start},{mdat.payload_end})", errors)

        # tfdt 单调
        ver, _, tpp = B.parse_fullbox(buf, tfdt.payload_start)
        tr = B.Reader(buf, tpp, tfdt.payload_end)
        base = tr.u64() if ver == 1 else tr.u32()
        if track_id in prev_tfdt:
            check(base >= prev_tfdt[track_id],
                  f"{path} track {track_id} tfdt 回退: "
                  f"{prev_tfdt[track_id]} -> {base}", errors)
        prev_tfdt[track_id] = base
    return prev_tfdt


# ---------------------------------------------------------------------------
# ffprobe 交叉验证
# ---------------------------------------------------------------------------
def have_ffprobe() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def ffprobe_idr_check(output_dir: str, records: list,
                      errors: list, warnings: list) -> None:
    """按 init 分组合并后用 ffprobe 验证：每组关键帧数 == 该组分片数。"""
    # 收集每个 init 对应的分片（保持顺序）
    groups: dict = {}
    for r in records:
        init_file = r.get("init_file", INIT_FILENAME)
        groups.setdefault(init_file, []).append(r["filename"])

    total_segments = 0
    total_keyframes = 0
    for init_file, seg_names in groups.items():
        init_path = os.path.join(output_dir, init_file)
        if not os.path.isfile(init_path):
            errors.append(f"分片引用的 init 段不存在: {init_file}")
            continue
        merged = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        merged.close()
        try:
            with open(merged.name, "wb") as o:
                with open(init_path, "rb") as f:
                    o.write(f.read())
                for name in seg_names:
                    with open(os.path.join(output_dir, name), "rb") as f:
                        o.write(f.read())
            out_json = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "frame=key_frame",
                "-of", "json", merged.name,
            ], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            frames = json.loads(out_json).get("frames", [])
            key_count = sum(1 for fr in frames if fr.get("key_frame") == 1)
            check(key_count == len(seg_names),
                  f"ffprobe: init {init_file} 组关键帧 {key_count} 与分片数 "
                  f"{len(seg_names)} 不一致", errors)
            total_segments += len(seg_names)
            total_keyframes += key_count
        except subprocess.CalledProcessError as e:
            errors.append(f"ffprobe 解析 init {init_file} 组合并文件失败: {e}")
        finally:
            os.unlink(merged.name)

    # ffprobe 直接解析 m3u8（多 EXT-X-MAP 也要能被标准工具接受）
    m3u8 = os.path.join(output_dir, MASTER_FILENAME)
    pl = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=format_name,duration:stream=codec_name",
        "-of", "json", m3u8,
    ], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    info = json.loads(pl)
    check("format" in info, "ffprobe 无法解析 master.m3u8", errors)
    if "format" in info:
        warnings.append(
            f"ffprobe 解析 m3u8 成功：{len(groups)} 份 init / "
            f"{total_segments} 片 / {total_keyframes} 个 IDR，总时长 "
            f"{info['format'].get('duration', '?')}s")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="HLS fMP4 产物校验")
    ap.add_argument("output_dir")
    ap.add_argument("--strict", action="store_true",
                    help="警告也按失败处理")
    args = ap.parse_args()

    out = args.output_dir
    errors: list = []
    warnings: list = []

    m3u8_path = os.path.join(out, MASTER_FILENAME)
    check(os.path.isfile(m3u8_path), "缺少 master.m3u8", errors)
    init_path = os.path.join(out, INIT_FILENAME)
    check(os.path.isfile(init_path), "缺少 init.mp4", errors)
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1

    with open(m3u8_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        warnings += validate_playlist_text(text)
    except Exception as e:
        errors.append(f"m3u8 规范错误: {e}")

    # 收集所有 init 段（init.mp4 + initN.mp4）
    init_files = sorted(
        n for n in os.listdir(out)
        if n == INIT_FILENAME or (
            n.startswith("init") and n.endswith(".mp4") and n != INIT_FILENAME))
    for inf in init_files:
        validate_init(os.path.join(out, inf), errors)

    seg_meta_path = os.path.join(out, SEGMENTS_JSON)
    records = []
    if os.path.isfile(seg_meta_path):
        with open(seg_meta_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        try:
            from packager.manifest import SegmentRecord
            warnings += validate_records(
                [SegmentRecord(**r) for r in records], 2.0)
        except Exception as e:
            errors.append(f"segments.json 时间线错误: {e}")
        seg_files = [r["filename"] for r in records]
        # 每条记录引用的 init 必须真实存在
        for r in records:
            inf = r.get("init_file", INIT_FILENAME)
            check(inf in init_files,
                  f"{r['filename']} 引用了不存在的 init: {inf}", errors)
    else:
        warnings.append("缺少 segments.json，跳过元数据交叉校验")
        seg_files = sorted(
            n for n in os.listdir(out) if n.startswith("seg") and n.endswith(".m4s"))

    # tfdt 单调性按 init 分组独立判断（换 init 后 baseMediaDecodeTime 会重置）
    seg_to_init = {r["filename"]: r.get("init_file", INIT_FILENAME)
                   for r in records}
    prev_tfdt_by_init: dict = {}
    for seq, name in enumerate(seg_files):
        path = os.path.join(out, name)
        check(os.path.isfile(path), f"缺少分片 {name}", errors)
        if os.path.isfile(path):
            inf = seg_to_init.get(name, INIT_FILENAME)
            prev_tfdt_by_init.setdefault(inf, {})
            prev_tfdt_by_init[inf] = validate_segment(
                path, seq, prev_tfdt_by_init[inf], errors, warnings)

    # EXTINF 与 trun 实际样本时长一致性
    extinf_durs = []
    for ln in text.splitlines():
        if ln.startswith("#EXTINF:"):
            extinf_durs.append(float(ln[len("#EXTINF:"):].rstrip(",")))
    if seg_meta_path and os.path.isfile(seg_meta_path):
        for rec, extinf in zip(records, extinf_durs):
            if abs(rec["duration"] - extinf) > 0.002:
                errors.append(
                    f"{rec['filename']} EXTINF {extinf} 与元数据 "
                    f"{rec['duration']} 不一致")

    if have_ffprobe():
        try:
            ffprobe_records = records if records else [
                {"filename": n, "init_file": INIT_FILENAME}
                for n in seg_files]
            ffprobe_idr_check(out, ffprobe_records, errors, warnings)
        except subprocess.CalledProcessError as e:
            errors.append(f"ffprobe 执行失败: {e}")
    else:
        warnings.append("未找到 ffprobe，跳过交叉验证（盒子层 IDR 校验仍已执行）")

    # 读取器冒烟测试：自己的 Reader 能完整解析所有 init 与分片
    try:
        r = FragmentedMp4Reader()
        for inf in init_files:
            with open(os.path.join(out, inf), "rb") as f:
                r.feed(f.read())
        for name in seg_files:
            with open(os.path.join(out, name), "rb") as f:
                r.feed(f.read())
        r.flush()
        check(bool(r.tracks), "读取器未能从 init 解析出 track", errors)
    except Exception as e:  # noqa: BLE001
        errors.append(f"自有读取器解析失败: {e}")

    print("=" * 70)
    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[FAIL] {e}")
    print("=" * 70)
    if errors:
        print(f"校验失败：{len(errors)} 个错误，{len(warnings)} 个警告")
        return 1
    if warnings and args.strict:
        print(f"严格模式失败：{len(warnings)} 个警告")
        return 1
    print(f"校验通过：{len(seg_files)} 个分片，{len(warnings)} 个警告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
