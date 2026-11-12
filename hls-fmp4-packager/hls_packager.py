#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hls-packager 命令行入口。

用法：
    python hls_packager.py input.mp4 output_dir/ [--segment-duration 2]
                           [--resume] [--force]
                           [--simulate-disconnect SEC]
                           [--reconnect-input FILE] [--max-reconnect N]

不使用 ffmpeg 的 hls muxer：ffmpeg 只产出 fragmented mp4 字节流，
切点规划、moof/mdat 重组、init 段、m3u8 与分片元数据全部由本项目代码生成。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# 允许直接以脚本方式运行（python hls_packager.py ...）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packager.manifest import (  # noqa: E402
    MASTER_FILENAME, SEGMENTS_JSON, validate_playlist_file, validate_records,
)
from packager.pipeline import HlsPackager  # noqa: E402
from packager.session import PackagerState, SESSION_FILENAME  # noqa: E402

log = logging.getLogger("hls-packager")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hls-packager",
        description="fMP4 HLS 打包器（自管 IDR 切点、断流续传、手写 m3u8）")
    p.add_argument("input", help="输入媒体文件（mp4/mkv/mov 等 ffmpeg 可读格式）")
    p.add_argument("output_dir", help="HLS 输出目录")
    p.add_argument("--segment-duration", type=float, default=2.0,
                   help="目标分片时长（秒），默认 2.0")
    p.add_argument("--resume", action="store_true",
                   help="从输出目录里的 session.json 续点打包，不从头开始")
    p.add_argument("--force", action="store_true",
                   help="输出目录已有会话时允许覆盖（与 --resume 互斥）")
    p.add_argument("--simulate-disconnect", type=float, metavar="SEC",
                   default=None,
                   help="测试钩子：输入消费到该秒数时杀掉 ffmpeg 模拟断流")
    p.add_argument("--reconnect-input", metavar="FILE", default=None,
                   help="测试钩子：重连后改用该输入（制造编码参数变化）")
    p.add_argument("--max-reconnect", type=int, default=10,
                   help="断流最大重连次数，默认 10")
    p.add_argument("--reconnect-backoff", type=float, default=1.0,
                   help="重连退避秒数，默认 1.0")
    p.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    return p.parse_args(argv)


def load_or_create_state(args: argparse.Namespace) -> PackagerState:
    existing = PackagerState.load(args.output_dir)
    if existing is not None:
        if args.resume:
            if existing.ended:
                log.warning("会话已处于结束态，仍按 --resume 继续追加")
            expected = os.path.abspath(args.input)
            if args.reconnect_input:
                # 跨进程续传并换源：把会话输入切到新文件，-ss 仍由会话记账
                expected = os.path.abspath(args.reconnect_input)
                existing.input_path = expected
            if os.path.abspath(existing.input_path) != expected:
                raise SystemExit(
                    f"--resume 的输入与会话记录不一致: "
                    f"{existing.input_path} != {expected}")
            if abs(existing.segment_duration - args.segment_duration) > 1e-6:
                raise SystemExit("--resume 不允许改变分片时长")
            log.info("从会话恢复：已完成 %d 片，续点 %.3fs",
                     existing.next_sequence, existing.consumed_seconds)
            return existing
        if not args.force:
            raise SystemExit(
                f"输出目录已存在 {SESSION_FILENAME}；"
                f"续点请加 --resume，覆盖请加 --force")
        # force：清空旧产物
        for name in os.listdir(args.output_dir):
            os.remove(os.path.join(args.output_dir, name))
    return PackagerState.new(args.input, args.output_dir,
                             args.segment_duration)


def print_segment_table(records, output_dir: str) -> None:
    print()
    print(f"{'seq':>4}  {'start_time(s)':>14}  {'duration(s)':>11}  "
          f"{'IDR':>5}  {'DISCONT':>8}  filename")
    print("-" * 78)
    for r in records:
        print(f"{r.sequence_number:>4}  {r.start_time:>14.3f}  "
              f"{r.duration:>11.3f}  "
              f"{('Y' if r.first_frame_idr else 'N'):>5}  "
              f"{('Y' if r.discontinuity else ''):>8}  {r.filename}")
    print("-" * 78)
    print(f"共 {len(records)} 个分片，"
          f"播放列表: {os.path.join(output_dir, MASTER_FILENAME)}，"
          f"元数据: {SEGMENTS_JSON}")


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    if args.segment_duration <= 0:
        raise SystemExit("--segment-duration 必须为正数")
    if not os.path.isfile(args.input):
        raise SystemExit(f"输入文件不存在: {args.input}")
    os.makedirs(args.output_dir, exist_ok=True)

    state = load_or_create_state(args)
    packager = HlsPackager(
        state,
        max_reconnect=args.max_reconnect,
        reconnect_backoff=args.reconnect_backoff,
        simulate_disconnect_at=args.simulate_disconnect,
        override_input=args.reconnect_input,
    )
    records = packager.run()

    # 落盘后再跑一遍完整校验（文本规范 + 时间线 + IDR 标记）
    m3u8_path = os.path.join(args.output_dir, MASTER_FILENAME)
    warnings = validate_playlist_file(m3u8_path)
    warnings += validate_records(records, state.segment_duration)
    for w in warnings:
        log.warning("m3u8 校验警告: %s", w)

    print_segment_table(records, args.output_dir)
    print(f"\n内置 m3u8 校验通过（{len(warnings)} 条警告）。"
          f"可再运行: python validate.py {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
