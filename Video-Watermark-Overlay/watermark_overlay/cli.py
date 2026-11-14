"""命令行入口：python -m watermark_overlay frame.png watermark.png -o out.png"""

from __future__ import annotations

import argparse
import json
import sys

from .animation import Keyframe, MotionConfig
from .config import WatermarkConfig
from .engine import overlay_files
from .errors import InvalidParameterError, WatermarkError
from .geometry import ANCHOR_CENTER, ANCHORS

EXIT_OK = 0
EXIT_INVALID_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark-overlay",
        description=(
            "图片水印叠加（纯标准库实现：预乘 alpha src-over 逐像素混合、"
            "双线性缩放、九宫格定位、内容自适应蒙版）"
        ),
    )
    parser.add_argument("frame", help="底图 PNG 文件路径")
    parser.add_argument("watermark", help="水印 PNG 文件路径（建议带 alpha 通道）")
    parser.add_argument("-o", "--output", required=True, help="输出 PNG 文件路径")

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

    if not isinstance(data, dict):
        raise InvalidParameterError("运动配置必须是 JSON 对象")
    raw_keyframes = data.get("keyframes")
    if not isinstance(raw_keyframes, list):
        raise InvalidParameterError("运动配置缺少 keyframes 数组")

    keyframes = []
    for i, item in enumerate(raw_keyframes):
        if not isinstance(item, dict) or "time" not in item:
            raise InvalidParameterError(f"第 {i} 个关键帧必须是含 time 字段的对象")
        try:
            keyframes.append(
                Keyframe(
                    time=item["time"],
                    x=item.get("x", 0.0),
                    y=item.get("y", 0.0),
                    angle=item.get("angle", 0.0),
                )
            )
        except TypeError as exc:
            raise InvalidParameterError(f"第 {i} 个关键帧字段非法：{exc}") from exc

    easing = data.get("easing", "linear")
    motion = MotionConfig(
        keyframes=keyframes,
        easing=easing,
        anchor_x=data.get("anchor_x", 0.5),
        anchor_y=data.get("anchor_y", 0.5),
    )
    motion.validate()
    return motion


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _build_config(args)
        motion = _load_motion(args.motion) if args.motion else None
        result = overlay_files(
            args.frame,
            args.watermark,
            args.output,
            config,
            motion=motion,
            frame_index=args.frame_index,
            fps=args.fps,
        )
    except WatermarkError as exc:
        # 可预期的输入/参数问题：明确中文提示 + 非零退出码
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    if result.overlapped:
        motion_info = (
            f"，角度 {result.angle:.2f}°，运动偏移 "
            f"({result.motion_offset[0]:.2f}, {result.motion_offset[1]:.2f})"
            if motion is not None
            else ""
        )
        print(
            f"完成：{args.output}（水印尺寸 {result.watermark_size[0]}x"
            f"{result.watermark_size[1]}，原点 {result.origin}，"
            f"混合 {result.blended_pixels} 像素{motion_info}）"
        )
    else:
        print(
            f"警告：水印与帧无可见交集，未修改任何像素；仍已写出 {args.output}",
            file=sys.stderr,
        )
    return EXIT_OK
