# -*- coding: utf-8 -*-
"""命令行入口：python -m video_qc <video> [-o report.json]"""
import argparse
import json
import sys

from .analyzer import analyze
from .detectors import Thresholds


def build_parser():
    p = argparse.ArgumentParser(
        prog="video-qc",
        description="视频质检工具：检测纯黑帧/静帧卡死/花屏马赛克/音频静音削波，输出帧级 JSON 报告")
    p.add_argument("video", help="输入视频路径（支持 mp4/mov/mkv/ts 等 ffmpeg 可解封装格式）")
    p.add_argument("-o", "--output", help="JSON 报告输出路径，缺省打印到标准输出")
    p.add_argument("--analyze-width", type=int, default=480,
                   help="分析分辨率宽度，默认 480（4K 性能与检测灵敏度的平衡点）")
    # 关键阈值可调（一般用默认值即可）
    p.add_argument("--black-min-sec", type=float, default=Thresholds.black_min_sec)
    p.add_argument("--freeze-mad", type=float, default=Thresholds.freeze_mad)
    p.add_argument("--freeze-min-sec", type=float, default=Thresholds.freeze_min_sec)
    p.add_argument("--blocky-ratio", type=float, default=Thresholds.blocky_ratio)
    p.add_argument("--silence-rms-db", type=float, default=Thresholds.silence_rms_db)
    p.add_argument("--clip-peak", type=float, default=Thresholds.clip_peak)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    th = Thresholds(
        black_min_sec=args.black_min_sec,
        freeze_mad=args.freeze_mad,
        freeze_min_sec=args.freeze_min_sec,
        blocky_ratio=args.blocky_ratio,
        silence_rms_db=args.silence_rms_db,
        clip_peak=args.clip_peak,
    )
    import video_qc.pipeline as pipeline
    pipeline.ANALYZE_WIDTH = args.analyze_width

    report = analyze(args.video, th)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"报告已写入 {args.output}；共发现 {len(report['issues'])} 个问题，"
              f"处理速度 {report['stats']['speed_factor']}x", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
