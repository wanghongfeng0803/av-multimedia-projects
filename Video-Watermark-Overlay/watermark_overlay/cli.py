"""命令行入口：python -m watermark_overlay frame.png watermark.png -o out.png

单文件模式与批量模式（--batch 清单.json）二选一；
批量模式下单个任务失败不影响其余任务，全部成功退出码 0，
部分/全部失败退出码 1，清单或参数非法退出码 2。
"""

from __future__ import annotations

import argparse
import json
import sys

from .animation import MotionConfig, motion_from_dict
from .batch import load_batch_manifest, run_batch
from .config import WatermarkConfig
from .engine import overlay_files
from .errors import InvalidParameterError, WatermarkError
from .geometry import ANCHOR_CENTER, ANCHORS
from .image import read_png
from .scatter import ScatterConfig

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_INVALID_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark-overlay",
        description=(
            "图片水印叠加（纯标准库实现：预乘 alpha src-over 逐像素混合、"
            "双线性缩放、九宫格定位、内容自适应蒙版）"
        ),
    )
    parser.add_argument("frame", nargs="?", help="底图 PNG 文件路径")
    parser.add_argument("watermark", nargs="?", help="水印 PNG 文件路径（建议带 alpha 通道）")
    parser.add_argument("-o", "--output", help="输出 PNG 文件路径")
    parser.add_argument(
        "--batch",
        default=None,
        metavar="清单.json",
        help=(
            "批量模式：从 JSON 清单读取共享水印/配置与任务列表，"
            "单个任务失败不影响其余任务（此时忽略位置参数与 -o）"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="批量模式并发线程数（默认 1，纯顺序执行）",
    )

    parser.add_argument(
        "--anchor",
        default=ANCHOR_CENTER,
        choices=sorted(ANCHORS),
        help="九宫格锚点（默认 center）",
    )
    parser.add_argument("--offset-x", type=int, default=0, help="锚点定位后水平偏移像素")
    parser.add_argument("--offset-y", type=int, default=0, help="锚点定位后垂直偏移像素")
    parser.add_argument("--width", type=int, default=None, help="水印目标宽度（等比推算高度）")
    parser.add_argument("--height", type=int, default=None, help="水印目标高度（等比推算宽度）")
    parser.add_argument(
        "--opacity",
        type=float,
        default=1.0,
        help="全局不透明度，0.0~1.0（默认 1.0）",
    )

    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="启用基于帧内容的自适应蒙版（边缘/纹理处自动降低水印透明度）",
    )
    parser.add_argument(
        "--mask-strength",
        type=float,
        default=0.6,
        help="自适应蒙版最大压暗比例 0.0~1.0（默认 0.6）",
    )
    parser.add_argument("--detail-low", type=int, default=24, help="细节强度起压阈值 0~255")
    parser.add_argument("--detail-high", type=int, default=96, help="细节强度满压阈值 0~255")
    parser.add_argument("--mask-radius", type=int, default=2, help="局部统计窗口半径（像素）")

    parser.add_argument(
        "--motion",
        default=None,
        metavar="JSON文件",
        help=(
            "运动轨迹 JSON 文件："
            '{"keyframes": [{"time": 0, "x": 0, "y": 0, "angle": 0}, ...], '
            '"easing": "linear", "anchor_x": 0.5, "anchor_y": 0.5}'
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="动画模式下的整数帧索引（默认 0）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="动画模式下的帧率，支持非整数（默认 25.0）",
    )

    parser.add_argument(
        "--scatter",
        action="store_true",
        help=(
            "分散布局模式：把水印按确定性网格+抖动平铺到整帧（抗裁剪），"
            "与 --motion 互斥；此时 --anchor/--offset-x/--offset-y 不参与定位"
        ),
    )
    parser.add_argument("--scatter-rows", type=int, default=2, help="分散布局行数（默认 2）")
    parser.add_argument("--scatter-cols", type=int, default=3, help="分散布局列数（默认 3）")
    parser.add_argument(
        "--scatter-margin", type=int, default=0, help="四边留白像素（默认 0）"
    )
    parser.add_argument(
        "--scatter-jitter",
        type=float,
        default=0.3,
        help="单元格内位置抖动幅度 0.0~1.0（默认 0.3）",
    )
    parser.add_argument(
        "--scatter-angle-jitter",
        type=float,
        default=0.0,
        help="每实例随机旋转角幅度（度）0~180（默认 0）",
    )
    parser.add_argument(
        "--scatter-seed",
        type=int,
        default=0,
        help="布局随机种子，相同种子布局完全一致（默认 0）",
    )
    return parser


def _load_motion(path: str) -> MotionConfig:
    """从 JSON 文件加载运动配置，格式错误统一转为 InvalidParameterError。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise InvalidParameterError(f"无法读取运动配置文件 {path!r}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidParameterError(f"运动配置文件 {path!r} 不是合法 JSON：{exc}") from exc
    return motion_from_dict(data)


def _build_config(args: argparse.Namespace) -> WatermarkConfig:
    return WatermarkConfig(
        anchor=args.anchor,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        target_width=args.width,
        target_height=args.height,
        opacity=args.opacity,
        adaptive=args.adaptive,
        mask_strength=args.mask_strength,
        detail_low=args.detail_low,
        detail_high=args.detail_high,
        mask_radius=args.mask_radius,
    )


def _build_scatter(args: argparse.Namespace) -> ScatterConfig:
    return ScatterConfig(
        rows=args.scatter_rows,
        cols=args.scatter_cols,
        margin=args.scatter_margin,
        jitter=args.scatter_jitter,
        angle_jitter=args.scatter_angle_jitter,
        seed=args.scatter_seed,
    )


def _run_batch(args: argparse.Namespace) -> int:
    """批量模式：加载清单 → 解码一次水印 → 调度全部任务 → 汇总报告。"""
    spec = load_batch_manifest(args.batch)
    if args.workers < 1:
        raise InvalidParameterError(f"--workers 必须为正整数，收到 {args.workers}")
    watermark = read_png(spec.watermark_path)

    def _progress(outcome, done, total):
        if outcome.ok:
            print(f"[{done}/{total}] 完成：{outcome.job.output_path}"
                  f"（混合 {outcome.result.blended_pixels} 像素，"
                  f"耗时 {outcome.elapsed:.2f}s）")
        else:
            print(f"[{done}/{total}] 失败：{outcome.job.input_path}："
                  f"{outcome.error}", file=sys.stderr)

    report = run_batch(
        spec.jobs,
        watermark,
        config=spec.config,
        motion=spec.motion,
        workers=args.workers,
        on_progress=_progress,
        scatter=spec.scatter,
    )
    print(
        f"批量完成：成功 {report.succeeded}/{report.total}，"
        f"失败 {report.failed}，总耗时 {report.elapsed:.2f}s"
    )
    return EXIT_OK if report.ok else EXIT_PARTIAL_FAILURE


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch is not None:
            return _run_batch(args)
        if args.frame is None or args.watermark is None or args.output is None:
            raise InvalidParameterError(
                "单文件模式需要 frame、watermark 与 -o/--output；"
                "批量处理请使用 --batch 清单.json"
            )
        config = _build_config(args)
        if args.scatter and args.motion:
            raise InvalidParameterError("分散布局（--scatter）与运动轨迹（--motion）不能同时使用")
        scatter = _build_scatter(args) if args.scatter else None
        motion = _load_motion(args.motion) if args.motion else None
        result = overlay_files(
            args.frame,
            args.watermark,
            args.output,
            config,
            motion=motion,
            frame_index=args.frame_index,
            fps=args.fps,
            scatter=scatter,
        )
    except WatermarkError as exc:
        # 可预期的输入/参数问题：明确中文提示 + 非零退出码
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    if result.overlapped:
        if scatter is not None:
            detail = (
                f"实例 {result.instance_count} 个"
                f"（{scatter.rows}x{scatter.cols} 网格，种子 {scatter.seed}），"
                f"混合 {result.blended_pixels} 像素"
            )
        elif motion is not None:
            detail = (
                f"原点 {result.origin}，混合 {result.blended_pixels} 像素，"
                f"角度 {result.angle:.2f}°，运动偏移 "
                f"({result.motion_offset[0]:.2f}, {result.motion_offset[1]:.2f})"
            )
        else:
            detail = f"原点 {result.origin}，混合 {result.blended_pixels} 像素"
        print(
            f"完成：{args.output}（水印尺寸 {result.watermark_size[0]}x"
            f"{result.watermark_size[1]}，{detail}）"
        )
    else:
        print(
            f"警告：水印与帧无可见交集，未修改任何像素；仍已写出 {args.output}",
            file=sys.stderr,
        )
    return EXIT_OK
