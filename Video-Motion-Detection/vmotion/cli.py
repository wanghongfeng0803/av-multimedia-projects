"""命令行入口：detect（检测出 JSON）与 mark（按 JSON 渲染标记视频）。

检测/标记/事件三模块在 CLI 层体现为两个独立子命令：``detect`` 只跑
检测流水线产出 JSON，``mark`` 只消费 JSON 出视频，两者可分开执行、
对同一份 JSON 反复重渲染。

退出码：0 成功且检测到运动事件；1 运行错误（文件/ffmpeg/参数）；
2 流程成功但全程无运动事件（JSON 仍有效，便于脚本编排区分）。
"""

import sys

from .detector import DEFAULT_CONFIG, MotionConfig
from .io_utils import PipelineError
from .pipeline import detect_video
from .render import render_video

__all__ = ["main", "build_parser"]


def build_parser():
    """构造 argparse 解析器（单独成函数供测试直接驱动）。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="vmotion",
        description="视频运动检测：分块帧差 + 局部自适应阈值 + "
                    "光照突变抑制，检测/标记/事件三模块解耦",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="逐帧运动检测，输出检测 JSON")
    d.add_argument("input", help="输入视频路径")
    d.add_argument("detection_json", help="输出检测 JSON 路径")
    d.add_argument("--block", type=int, default=DEFAULT_CONFIG.block,
                   help=f"分块边长（像素），默认 {DEFAULT_CONFIG.block}")
    d.add_argument("--local-radius", type=int,
                   default=DEFAULT_CONFIG.local_radius,
                   help="局部 MAD 邻域半径（块），默认 "
                        f"{DEFAULT_CONFIG.local_radius}")
    d.add_argument("--pixel-threshold", type=float,
                   default=DEFAULT_CONFIG.pixel_threshold,
                   help="块均值差硬阈值（0..255），默认 "
                        f"{DEFAULT_CONFIG.pixel_threshold}")
    d.add_argument("--gain", type=float, default=DEFAULT_CONFIG.gain,
                   help=f"局部 MAD 增益系数，默认 {DEFAULT_CONFIG.gain}")
    d.add_argument("--min-area", type=int, default=DEFAULT_CONFIG.min_area,
                   help=f"连通域最小面积（像素），默认 "
                        f"{DEFAULT_CONFIG.min_area}")
    d.add_argument("--open-radius", type=int,
                   default=DEFAULT_CONFIG.open_radius,
                   help="形态学开运算半径（0 跳过），默认 "
                        f"{DEFAULT_CONFIG.open_radius}")
    d.add_argument("--close-radius", type=int,
                   default=DEFAULT_CONFIG.close_radius,
                   help="形态学闭运算半径（0 跳过），默认 "
                        f"{DEFAULT_CONFIG.close_radius}")
    d.add_argument("--trigger-frames", type=int,
                   default=DEFAULT_CONFIG.trigger_frames,
                   help="连续命中确认帧数，默认 "
                        f"{DEFAULT_CONFIG.trigger_frames}")
    d.add_argument("--release-frames", type=int,
                   default=DEFAULT_CONFIG.release_frames,
                   help="连续无命中释放帧数，默认 "
                        f"{DEFAULT_CONFIG.release_frames}")
    d.add_argument("--illum-change-threshold", type=float,
                   default=DEFAULT_CONFIG.illum_change_threshold,
                   help="光照突变的全局均值差阈值（0..255），默认 "
                        f"{DEFAULT_CONFIG.illum_change_threshold}")
    d.add_argument("--illum-hold-frames", type=int,
                   default=DEFAULT_CONFIG.illum_hold_frames,
                   help="突变后抑制帧数，默认 "
                        f"{DEFAULT_CONFIG.illum_hold_frames}")

    m = sub.add_parser("mark", help="按检测 JSON 渲染标记视频")
    m.add_argument("input", help="输入视频路径（须与检测时同源同帧数）")
    m.add_argument("detection_json", help="detect 产出的检测 JSON")
    m.add_argument("output", help="输出标记视频路径")
    m.add_argument("--font-scale", type=int, default=1,
                   help="点阵文字整数放大倍数，默认 1")
    return parser


def _config_from_args(args):
    """把 detect 子命令的命名空间映射为 :class:`MotionConfig`。"""
    return MotionConfig(
        block=args.block,
        local_radius=args.local_radius,
        pixel_threshold=args.pixel_threshold,
        gain=args.gain,
        min_area=args.min_area,
        open_radius=args.open_radius,
        close_radius=args.close_radius,
        trigger_frames=args.trigger_frames,
        release_frames=args.release_frames,
        illum_change_threshold=args.illum_change_threshold,
        illum_hold_frames=args.illum_hold_frames,
    )


def _progress(prefix):
    def cb(n):
        print(f"\r{prefix} {n} 帧", end="", file=sys.stderr)
    return cb


def main(argv=None):
    """CLI 主入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect":
        try:
            config = _config_from_args(args)
            result = detect_video(
                args.input, args.detection_json, config,
                progress=_progress("已检测"),
            )
        except (PipelineError, ValueError) as exc:
            print(f"\n失败：{exc}", file=sys.stderr)
            return 1
        stats = result.stats
        print(
            f"\n完成：{stats['frames']} 帧，运动 {stats['motion_frames']} 帧"
            f"（{stats['motion_ratio'] * 100:.1f}%），"
            f"光照抑制 {stats['held_off_frames']} 帧，"
            f"事件 {stats['events']} 段，区域 {stats['regions_total']} 个"
            f" -> {args.detection_json}"
        )
        for ev in result.events:
            if ev.end_of_stream:
                print(f"  事件 {ev.event_id}：帧 {ev.start_frame} 起"
                      "（至片末未闭合）", file=sys.stderr)
            else:
                print(f"  事件 {ev.event_id}：帧 {ev.start_frame}.."
                      f"{ev.end_frame - 1}（{ev.motion_frames} 帧，"
                      f"峰值面积 {ev.peak_area}）", file=sys.stderr)
        return 0 if result.events else 2

    # mark
    try:
        info = render_video(
            args.input, args.output, args.detection_json,
            font_scale=args.font_scale,
            progress=_progress("已标记"),
        )
    except (PipelineError, ValueError) as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1
    print(f"\n完成：标记 {info['frames']} 帧，事件 {info['events']} 段"
          f" -> {args.output}")
    return 0
