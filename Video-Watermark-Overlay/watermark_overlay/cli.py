"""命令行入口：python -m watermark_overlay frame.png watermark.png -o out.png"""

from __future__ import annotations

import argparse
import sys

from .config import WatermarkConfig
from .engine import overlay_files
from .errors import WatermarkError
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
    return parser


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
        result = overlay_files(args.frame, args.watermark, args.output, config)
    except WatermarkError as exc:
        # 可预期的输入/参数问题：明确中文提示 + 非零退出码
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    if result.overlapped:
        print(
            f"完成：{args.output}（水印尺寸 {result.watermark_size[0]}x"
            f"{result.watermark_size[1]}，原点 {result.origin}，"
            f"混合 {result.blended_pixels} 像素）"
        )
    else:
        print(
            f"警告：水印与帧无可见交集，未修改任何像素；仍已写出 {args.output}",
            file=sys.stderr,
        )
    return EXIT_OK
