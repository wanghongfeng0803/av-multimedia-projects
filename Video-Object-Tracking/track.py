#!/usr/bin/env python3
"""视频目标追踪命令行入口（追踪与渲染两个子命令）。

用法：
    # 1) 追踪：逐帧模板匹配，产出轨迹 JSON（不产生视频）
    python track.py track 输入.mp4 轨迹.json --box x y w h
                            [--method sad|ssd|zncc] [--threshold 分数]
                            [--search-margin 像素] [--lost-growth 倍率]
                            [--reacquire-frames 帧数] [--reacquire-threshold 分数]
                            [--update-template] [--template-image 图片]
                            [--template-frame 帧号]

    # 2) 渲染：读轨迹 JSON 画框出视频，可对同一轨迹反复重渲染
    python track.py render 输入.mp4 轨迹.json 输出.mp4
                             [--font-scale 倍数] [--last-known]

追踪与渲染严格分离：追踪只输出 JSON 数据契约，供本模块渲染，也供后续
光流追踪、多目标追踪复用。退出码：0 成功；1 运行错误；2 追踪完成但
视频结束时目标仍处于丢失状态（轨迹 JSON 仍有效）。
"""

import argparse
import sys

from vtrack.io_utils import PipelineError
from vtrack.pipeline import render_video, track_video
from vtrack.template import DEFAULT_THRESHOLDS, TEMPLATE_METHODS


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="track",
        description="逐帧模板匹配目标追踪：追踪产出 JSON，渲染独立消费",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("track", help="逐帧模板匹配，输出轨迹 JSON")
    t.add_argument("input", help="输入视频路径")
    t.add_argument("track_json", help="输出轨迹 JSON 路径")
    t.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   default=None,
                   help="初始模板框 x y w h（使用 --template-image 全图时可省）")
    t.add_argument("--method", choices=TEMPLATE_METHODS, default="sad",
                   help="匹配方法：sad 绝对差（默认）/ ssd 平方差 / zncc 归一化相关")
    t.add_argument("--threshold", type=float, default=None,
                   help="接受阈值，缺省按方法取 sad=30 / ssd=900 / zncc=0.7；"
                        "sad/ssd 为每像素均值（越小越好），zncc 为相关系数（越大越好）")
    t.add_argument("--search-margin", type=int, default=12,
                   help="追踪态局部搜索区外扩像素边距，默认 12")
    t.add_argument("--lost-growth", type=float, default=1.5,
                   help="丢失后搜索区每帧复利增长因子，默认 1.5，铺满全帧后保持全帧")
    t.add_argument("--reacquire-frames", type=int, default=3,
                   help="恢复重确认所需连续达标帧数，默认 3")
    t.add_argument("--reacquire-threshold", type=float, default=None,
                   help="重确认专用阈值，缺省沿用 --threshold；可设更严格防误恢复")
    t.add_argument("--update-template", action="store_true",
                   help="启用工作模板自适应更新（默认关闭，全程用初始模板防漂移）")
    t.add_argument("--update-alpha", type=float, default=0.2,
                   help="模板更新凸组合系数（0..1），默认 0.2，仅配合 --update-template")
    t.add_argument("--update-score", type=float, default=None,
                   help="只有分数优于该值才更新模板，缺省取主阈值")
    t.add_argument("--template-image", default=None, metavar="IMAGE",
                   help="外部模板图片（替代从视频帧裁剪）")
    t.add_argument("--template-frame", type=int, default=0,
                   help="从视频第几帧裁剪初始模板，默认第 0 帧")
    t.add_argument("--target-id", default="target_0",
                   help="目标标识，写入 JSON 供多目标扩展，默认 target_0")

    r = sub.add_parser("render", help="按轨迹 JSON 渲染画框视频")
    r.add_argument("input", help="输入视频路径（须与追踪时同源同帧数）")
    r.add_argument("track_json", help="追踪产出的轨迹 JSON")
    r.add_argument("output", help="输出视频路径")
    r.add_argument("--font-scale", type=int, default=1,
                   help="点阵文字整数放大倍数，默认 1")
    r.add_argument("--last-known", action="store_true",
                   help="丢失帧额外画黄色虚线框标出最后已知位置")
    return parser


def _progress(prefix):
    def cb(n):
        print(f"\r{prefix} {n} 帧", end="", file=sys.stderr)
    return cb


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "track":
        init_box = tuple(args.box) if args.box else None
        try:
            result = track_video(
                args.input, args.track_json, init_box,
                method=args.method, threshold=args.threshold,
                search_margin=args.search_margin,
                lost_growth=args.lost_growth,
                reacquire_frames=args.reacquire_frames,
                reacquire_threshold=args.reacquire_threshold,
                update_template=args.update_template,
                update_alpha=args.update_alpha,
                update_score=args.update_score,
                template_image=args.template_image,
                template_frame=args.template_frame,
                target_id=args.target_id,
                progress=_progress("已追踪"),
            )
        except (PipelineError, ValueError) as exc:
            print(f"\n失败：{exc}", file=sys.stderr)
            return 1
        stats = result.stats
        print(
            f"\n完成：{stats['frames']} 帧，确认 {stats['tracked_frames']} / "
            f"丢失 {stats['lost_frames']}（{stats['lost_ratio'] * 100:.1f}%），"
            f"丢失事件 {stats['loss_events']} 段，方法 {result.params['method']}"
            f"（阈值 {result.params['threshold']}） -> {args.track_json}"
        )
        for ev in result.loss_events:
            if ev.end_of_stream:
                print(f"  警告：第 {ev.lost_frame} 帧起丢失，至片末未恢复",
                      file=sys.stderr)
            else:
                print(f"  丢失区间：帧 {ev.lost_frame}..{ev.recovered_frame - 1}"
                      f"（连续 {ev.frames_lost} 帧）", file=sys.stderr)
        return 2 if stats["still_lost_at_end"] else 0

    # render
    try:
        info = render_video(
            args.input, args.output, args.track_json,
            font_scale=args.font_scale,
            show_last_known=args.last_known,
            progress=_progress("已渲染"),
        )
    except (PipelineError, ValueError) as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1
    print(f"\n完成：渲染 {info['frames']} 帧，丢失事件 {info['loss_events']} 段"
          f" -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
