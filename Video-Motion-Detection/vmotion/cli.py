"""命令行入口：detect（检测出 JSON）、mark（按 JSON 渲染标记视频）、
compare（两种检测模式同输入量化对比）。

检测/标记/事件三模块在 CLI 层体现为独立子命令：``detect`` 只跑
检测流水线产出 JSON，``mark`` 只消费 JSON 出视频，两者可分开执行、
对同一份 JSON 反复重渲染；``compare`` 在一次解码内并排跑帧差与背景
建模两种模式，产出逐帧不一致清单。

``detect`` 默认是旧帧差模式（默认参数与历史版本逐值一致），加
``--background`` 切换到背景建模模式，新增的预热/学习率/吸收帧数三个
参数只在该模式下生效。

退出码：0 成功且检测到运动事件；1 运行错误（文件/ffmpeg/参数）；
2 流程成功但全程无运动事件（JSON 仍有效，便于脚本编排）。
``compare`` 只做对比、不以有无事件决定退出码，成功恒为 0。
"""

import sys

from .background import DEFAULT_BACKGROUND_CONFIG, BackgroundConfig
from .comparison import compare_video
from .detector import DEFAULT_CONFIG, MotionConfig
from .io_utils import PipelineError
from .pipeline import detect_background_video, detect_video
from .render import render_video

__all__ = ["main", "build_parser"]


def _add_gate_args(p, defaults):
    """注册两种模式共用的门控/形态学/滞后/光照参数。"""
    p.add_argument("--block", type=int, default=defaults.block,
                   help=f"分块边长（像素），默认 {defaults.block}")
    p.add_argument("--local-radius", type=int,
                   default=defaults.local_radius,
                   help="局部 MAD 邻域半径（块），默认 "
                        f"{defaults.local_radius}")
    p.add_argument("--pixel-threshold", type=float,
                   default=defaults.pixel_threshold,
                   help="块均值差硬阈值（0..255），默认 "
                        f"{defaults.pixel_threshold}")
    p.add_argument("--gain", type=float, default=defaults.gain,
                   help=f"局部 MAD 增益系数，默认 {defaults.gain}")
    p.add_argument("--min-area", type=int, default=defaults.min_area,
                   help=f"连通域最小面积（像素），默认 "
                        f"{defaults.min_area}")
    p.add_argument("--open-radius", type=int,
                   default=defaults.open_radius,
                   help="形态学开运算半径（0 跳过），默认 "
                        f"{defaults.open_radius}")
    p.add_argument("--close-radius", type=int,
                   default=defaults.close_radius,
                   help="形态学闭运算半径（0 跳过），默认 "
                        f"{defaults.close_radius}")
    p.add_argument("--trigger-frames", type=int,
                   default=defaults.trigger_frames,
                   help="连续命中确认帧数，默认 "
                        f"{defaults.trigger_frames}")
    p.add_argument("--release-frames", type=int,
                   default=defaults.release_frames,
                   help="连续无命中释放帧数，默认 "
                        f"{defaults.release_frames}")
    p.add_argument("--illum-change-threshold", type=float,
                   default=defaults.illum_change_threshold,
                   help="光照突变的全局均值差阈值（0..255），默认 "
                        f"{defaults.illum_change_threshold}")
    p.add_argument("--illum-hold-frames", type=int,
                   default=defaults.illum_hold_frames,
                   help="突变后抑制帧数，默认 "
                        f"{defaults.illum_hold_frames}")


def _add_background_args(p):
    """注册背景建模专有参数。"""
    p.add_argument("--warmup-frames", type=int,
                   default=DEFAULT_BACKGROUND_CONFIG.warmup_frames,
                   help="背景建模预热帧数（首帧后累计平均、不出掩码），"
                        f"默认 {DEFAULT_BACKGROUND_CONFIG.warmup_frames}")
    p.add_argument("--learning-rate", type=float,
                   default=DEFAULT_BACKGROUND_CONFIG.learning_rate,
                   help="背景 EMA 学习率 α（(0,1]，越小越稳），默认 "
                        f"{DEFAULT_BACKGROUND_CONFIG.learning_rate}")
    p.add_argument("--absorb-frames", type=int,
                   default=DEFAULT_BACKGROUND_CONFIG.absorb_frames,
                   help="前景块连续多少帧后吸收进背景，默认 "
                        f"{DEFAULT_BACKGROUND_CONFIG.absorb_frames}")


def build_parser():
    """构造 argparse 解析器（单独成函数供测试直接驱动）。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="vmotion",
        description="视频运动检测：帧差/背景建模两种模式，局部自适应"
                    "阈值 + 光照突变抑制，检测/标记/事件三模块解耦",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="逐帧运动检测，输出检测 JSON")
    d.add_argument("input", help="输入视频路径")
    d.add_argument("detection_json", help="输出检测 JSON 路径")
    d.add_argument("--background", action="store_true",
                   help="切换到背景建模模式（默认帧差模式）")
    _add_gate_args(d, DEFAULT_CONFIG)
    _add_background_args(d)

    c = sub.add_parser(
        "compare",
        help="同输入并排跑帧差与背景建模两种模式，输出对比 JSON")
    c.add_argument("input", help="输入视频路径")
    c.add_argument("compare_json", help="输出对比 JSON 路径")
    _add_gate_args(c, DEFAULT_CONFIG)
    _add_background_args(c)

    m = sub.add_parser("mark", help="按检测 JSON 渲染标记视频")
    m.add_argument("input", help="输入视频路径（须与检测时同源同帧数）")
    m.add_argument("detection_json", help="detect 产出的检测 JSON")
    m.add_argument("output", help="输出标记视频路径")
    m.add_argument("--font-scale", type=int, default=1,
                   help="点阵文字整数放大倍数，默认 1")
    return parser


def _gate_kwargs(args):
    """从命名空间取两种模式共用的门控参数 dict。"""
    return dict(
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


def _config_from_args(args):
    """把 detect 子命令的命名空间映射为对应模式的配置对象。"""
    if getattr(args, "background", False):
        return BackgroundConfig(
            **_gate_kwargs(args),
            warmup_frames=args.warmup_frames,
            learning_rate=args.learning_rate,
            absorb_frames=args.absorb_frames,
        )
    return MotionConfig(**_gate_kwargs(args))


def _progress(prefix):
    def cb(n):
        print(f"\r{prefix} {n} 帧", end="", file=sys.stderr)
    return cb


def main(argv=None):
    """CLI 主入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        try:
            fcfg = MotionConfig(**_gate_kwargs(args))
            bcfg = BackgroundConfig(
                **_gate_kwargs(args),
                warmup_frames=args.warmup_frames,
                learning_rate=args.learning_rate,
                absorb_frames=args.absorb_frames,
            )
            payload = compare_video(
                args.input, args.compare_json, fcfg, bcfg,
                progress=_progress("已对比"),
            )
        except (PipelineError, ValueError) as exc:
            print(f"\n失败：{exc}", file=sys.stderr)
            return 1
        counts = payload["counts"]
        print(
            f"\n完成：{payload['frames']} 帧，运动态一致率 "
            f"{payload['verdict_agreement'] * 100:.1f}%，不一致 "
            f"{counts['frame_diff_only'] + counts['background_only']} 帧"
            f"（仅帧差 {counts['frame_diff_only']}，"
            f"仅背景 {counts['background_only']}，"
            f"区域数差异 {counts['region_count_diff']}），"
            f"事件 帧差 {payload['event_counts']['frame_diff']} 段 / "
            f"背景 {payload['event_counts']['background']} 段"
            f" -> {args.compare_json}"
        )
        return 0

    if args.command == "detect":
        try:
            config = _config_from_args(args)
            if args.background:
                result = detect_background_video(
                    args.input, args.detection_json, config,
                    progress=_progress("已检测"),
                )
            else:
                result = detect_video(
                    args.input, args.detection_json, config,
                    progress=_progress("已检测"),
                )
        except (PipelineError, ValueError) as exc:
            print(f"\n失败：{exc}", file=sys.stderr)
            return 1
        stats = result.stats
        mode = "背景建模" if args.background else "帧差"
        print(
            f"\n完成（{mode}）：{stats['frames']} 帧，运动 "
            f"{stats['motion_frames']} 帧"
            f"（{stats['motion_ratio'] * 100:.1f}%），"
            f"光照抑制 {stats['held_off_frames']} 帧，"
            f"事件 {stats['events']} 段，区域 {stats['regions_total']} 个"
            f" -> {args.detection_json}"
        )
        if args.background:
            print(f"  预热 {stats['warmup_frames']} 帧，"
                  f"发生块吸收的帧 {stats['absorbing_frames']} 个",
                  file=sys.stderr)
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
