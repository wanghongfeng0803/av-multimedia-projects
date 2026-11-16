#!/usr/bin/env python3
"""视频防抖命令行入口：估计 / 平滑 / 补偿 / 裁剪 四步分离，另有 run 串联。

用法：
    python stabilize.py estimate  输入 运动.json [--est-width 96]
    python stabilize.py smooth    运动.json 平滑.json [--radius 15]
                                             [--method uniform|gaussian]
    python stabilize.py compensate 输入 平滑.json 稳定带黑边.mp4
                                  [--work-dir 目录] [--crf 18]
    python stabilize.py crop      稳定带黑边.mp4 平滑.json 输出.mp4
                                  [--inset 像素]
    python stabilize.py run       输入 输出.mp4 [--keep-work]

设计约束（与项目其他模块一致）：

- 原文件只读不写，每一步产物都是独立文件，参数 JSON 格式版本化、
  字段稳定，可人工检查后再执行后续步骤；
- 平移与旋转在 JSON 中始终是独立字段，补偿阶段也走两个独立滤镜；
- 估计/平滑/补偿解耦：同一份平滑 JSON 可反复用于不同裁剪量，
  裁剪步骤不重新估计；
- 零三方依赖，ffmpeg/ffprobe 走子进程。

退出码：0 成功；1 通用失败；2 参数/契约错误；8 ffmpeg 流水线失败。
"""

import argparse
import json
import sys
import tempfile

from stabilize import (
    DEFAULT_CRF,
    DEFAULT_RADIUS,
    EST_WIDTH,
    MOTION_FORMAT,
    SMOOTH_FORMAT,
    PipelineError,
    build_sendcmd,
    compensate_video,
    crop_video,
    dump_json,
    estimate_motion,
    load_json,
    smooth_motion,
)

#: 退出码与工具包其他模块对齐
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ARGS = 2
EXIT_PIPELINE = 8


def _progress(prefix):
    def cb(n):
        print(f"\r{prefix} {n} 帧", end="", file=sys.stderr)
    return cb


def cmd_estimate(args):
    doc = estimate_motion(
        args.input, args.motion_json,
        target_width=args.est_width,
        progress=_progress("已估计"),
    )
    print(f"\n完成：{doc['frame_count']} 帧，估计分辨率 "
          f"{doc['est_width']}x{doc['est_height']}（缩放比 "
          f"{doc['downsample_scale']}） -> {args.motion_json}")
    invalid = sum(1 for f in doc["frames"] if not f["valid"])
    cuts = sum(1 for f in doc["frames"] if f.get("scene_cut"))
    print(f"无效帧 {invalid}（其中场景切换 {cuts}）")
    return EXIT_OK


def cmd_smooth(args):
    motion = load_json(args.motion_json)
    if motion.get("format") != MOTION_FORMAT:
        print(f"参数文件格式不符：需要 {MOTION_FORMAT}，"
              f"收到 {motion.get('format')!r}", file=sys.stderr)
        return EXIT_ARGS
    doc = smooth_motion(motion, radius=args.radius, method=args.method)
    dump_json(doc, args.smooth_json)
    crop = doc["crop"]
    warn = "（触发上限钳制）" if crop["clamped"] else ""
    print(f"完成：{doc['params']['frame_count']} 帧，平滑窗 "
          f"2×{args.radius}+1（{args.method}），插值无效帧 "
          f"{doc['params']['invalid_frames']} -> {args.smooth_json}")
    print(f"推荐裁剪 inset={crop['inset']}px{warn}："
          f"平移峰值 x={crop['max_shift_x']} y={crop['max_shift_y']}，"
          f"旋转峰值 {crop['max_angle']}°")
    return EXIT_OK


def cmd_compensate(args):
    doc = load_json(args.smooth_json)
    if doc.get("format") != SMOOTH_FORMAT:
        print(f"参数文件格式不符：需要 {SMOOTH_FORMAT}，"
              f"收到 {doc.get('format')!r}", file=sys.stderr)
        return EXIT_ARGS
    info = compensate_video(
        doc, args.input, args.output, work_dir=args.work_dir,
        crf=args.crf, progress=_progress("已补偿"),
    )
    print(f"\n完成：{info['frames']} 帧补偿（含待裁黑边），"
          f"命令文件 {info['sendcmd']} -> {args.output}")
    return EXIT_OK


def cmd_crop(args):
    doc = load_json(args.smooth_json)
    if doc.get("format") != SMOOTH_FORMAT:
        print(f"参数文件格式不符：需要 {SMOOTH_FORMAT}，"
              f"收到 {doc.get('format')!r}", file=sys.stderr)
        return EXIT_ARGS
    info = crop_video(doc, args.input, args.output, inset=args.inset,
                      crf=args.crf)
    print(f"完成：居中裁剪 inset={info['inset']}px，"
          f"有效区 {info['crop_width']}x{info['crop_height']} "
          f"缩回 {info['width']}x{info['height']} -> {args.output}")
    return EXIT_OK


def cmd_run(args):
    """串联四步：中间产物放临时目录，--keep-work 时落到指定目录。"""
    if args.keep_work:
        work = args.work_dir or "stabilize_work"
        import os
        os.makedirs(work, exist_ok=True)
        cleanup = False
    else:
        work = tempfile.mkdtemp(prefix="stabilize_")
        cleanup = True
    import os
    motion_path = os.path.join(work, "motion.json")
    smooth_path = os.path.join(work, "smooth.json")
    stabilized = os.path.join(work, "stabilized_padded.mp4")
    try:
        motion = estimate_motion(
            args.input, motion_path, target_width=args.est_width,
            progress=_progress("已估计"),
        )
        print()
        smoothed = smooth_motion(motion, radius=args.radius,
                                 method=args.method)
        dump_json(smoothed, smooth_path)
        compensate_video(smoothed, args.input, stabilized,
                         work_dir=work, crf=args.crf,
                         progress=_progress("已补偿"))
        print()
        crop_video(smoothed, stabilized, args.output, crf=args.crf)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
    print(f"完成：{args.input} -> {args.output}")
    if not cleanup:
        print(f"中间产物保留在 {work}（motion.json / smooth.json / "
              f"stabilized_padded.mp4）")
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stabilize",
        description="视频防抖：全局平移+旋转估计、轨迹平滑、ffmpeg 补偿裁剪",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_est = sub.add_parser("estimate", help="估计逐帧全局运动，写 motion JSON")
    p_est.add_argument("input")
    p_est.add_argument("motion_json")
    p_est.add_argument("--est-width", type=int, default=EST_WIDTH,
                       help="估计用小图目标宽度，默认 96")
    p_est.set_defaults(func=cmd_estimate)

    p_sm = sub.add_parser("smooth", help="平滑轨迹并推导裁剪量，写 smooth JSON")
    p_sm.add_argument("motion_json")
    p_sm.add_argument("smooth_json")
    p_sm.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                      help=f"滑动平均半径（帧），默认 {DEFAULT_RADIUS}")
    p_sm.add_argument("--method", choices=["uniform", "gaussian"],
                      default="uniform", help="uniform 均匀窗（默认）/ "
                                              "gaussian 高斯加权窗")
    p_sm.set_defaults(func=cmd_smooth)

    p_cp = sub.add_parser("compensate", help="按 smooth JSON 做旋转/平移补偿")
    p_cp.add_argument("input")
    p_cp.add_argument("smooth_json")
    p_cp.add_argument("output")
    p_cp.add_argument("--work-dir", default="stabilize_work",
                      help="sendcmd 临时目录，默认 ./stabilize_work")
    p_cp.add_argument("--crf", type=int, default=DEFAULT_CRF,
                      help=f"x264 质量，默认 {DEFAULT_CRF}")
    p_cp.set_defaults(func=cmd_compensate)

    p_cr = sub.add_parser("crop", help="复用 smooth JSON 裁掉黑边并缩回原尺寸")
    p_cr.add_argument("input")
    p_cr.add_argument("smooth_json")
    p_cr.add_argument("output")
    p_cr.add_argument("--inset", type=float, default=None,
                      help="覆盖 JSON 推荐裁剪量（像素）")
    p_cr.add_argument("--crf", type=int, default=DEFAULT_CRF,
                      help=f"x264 质量，默认 {DEFAULT_CRF}")
    p_cr.set_defaults(func=cmd_crop)

    p_run = sub.add_parser("run", help="四步串联，直接产出成品视频")
    p_run.add_argument("input")
    p_run.add_argument("output")
    p_run.add_argument("--est-width", type=int, default=EST_WIDTH)
    p_run.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    p_run.add_argument("--method", choices=["uniform", "gaussian"],
                       default="uniform")
    p_run.add_argument("--crf", type=int, default=DEFAULT_CRF)
    p_run.add_argument("--keep-work", action="store_true",
                       help="保留中间 JSON 与带黑边视频（默认临时目录自动清理）")
    p_run.add_argument("--work-dir", default=None,
                       help="--keep-work 时的中间产物目录，默认 ./stabilize_work")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as exc:
        print(f"流水线失败：{exc}", file=sys.stderr)
        return EXIT_PIPELINE
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
