"""命令行入口：python -m hls_packager input.mp4 output_dir/"""

from __future__ import annotations

import argparse
import sys

from .encryption import DEFAULT_KEY_FILENAME, EncryptionConfig, EncryptionError
from .containers import SUPPORTED_CONTAINERS
from .errors import PackagerError
from .naming import DEFAULT_NAME_TEMPLATE
from .packager import PLAYLIST_FILENAME, package
from .planner import ALIGN_MODES
from .watermark.config import VALID_ANCHORS, WatermarkConfig

EXIT_OK = 0
EXIT_PACKAGER_ERROR = 2
EXIT_UNEXPECTED = 3


def _force_utf8_console() -> None:
    """Windows 控制台默认代码页可能不是 UTF-8，显式切换避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hls-packager",
        description=(
            "把视频按关键帧边界封装为 MPEG-TS 分片并生成 M3U8 播放列表"
            "（不重编码，切点严格对齐 GOP）"
        ),
    )
    parser.add_argument("input", help="输入媒体文件路径（mp4/mkv/mov/ts 等）")
    parser.add_argument("output_dir", help="HLS 产物输出目录（不存在会自动创建）")
    parser.add_argument(
        "-d", "--segment-duration",
        type=float,
        default=6.0,
        help="目标分片时长（秒），实际切点对齐最近关键帧，默认 6",
    )
    parser.add_argument(
        "-w", "--window-size",
        type=int,
        default=None,
        help="直播滑动窗口保留的分片数；不指定则输出 VOD 全量列表并带 ENDLIST",
    )
    parser.add_argument(
        "-s", "--media-sequence",
        type=int,
        default=0,
        help="首个分片的 EXT-X-MEDIA-SEQUENCE 序号，默认 0",
    )
    parser.add_argument(
        "-p", "--filename-prefix",
        default="seg",
        help="分片文件名前缀（命名模板中 {name} 的值），默认 seg",
    )
    parser.add_argument(
        "--name-template",
        default=None,
        metavar="TEMPLATE",
        help=(
            "分片命名模板，占位符：{name} 前缀、{sequence} 绝对序号、"
            "{index} 运行内序号、{ext} 扩展名；序号支持零填充如 "
            "{sequence:05d}。默认 "
            + DEFAULT_NAME_TEMPLATE
            + "（产出 seg_0000.ts ...）"
        ),
    )
    parser.add_argument(
        "--segment-format",
        choices=SUPPORTED_CONTAINERS,
        default="ts",
        help=(
            "分片封装格式：ts=MPEG-TS（默认，HLS 标准）、"
            "fmp4=自包含碎片化 MP4（CMAF 风格）、mp4=普通 ISO-BMFF、"
            "mkv=Matroska；仅影响封装层，GOP 对齐逻辑不变"
        ),
    )
    parser.add_argument(
        "--align-mode",
        choices=ALIGN_MODES,
        default="nearest",
        help=(
            "切点对齐关键帧的策略：nearest=距理想切点最近（默认，同距"
            "取靠前）、before=不超过目标时长、after=不短于目标时长；"
            "纯音频输入下该参数无效"
        ),
    )

    # ---- 文字水印（自行栅格化后重编码视频轨叠加；不指定则保持流拷贝） ----
    wm = parser.add_argument_group("文字水印")
    wm.add_argument(
        "--watermark-text",
        default=None,
        help=(
            "水印文字（UTF-8，支持多字节与特殊字符）；不指定则不叠加水印，"
            "保持流拷贝行为不变"
        ),
    )
    wm.add_argument(
        "--watermark-font",
        action="append",
        default=None,
        metavar="FONT_PATH",
        help="显式字体文件，可重复传入按顺序回退；缺省时自动发现系统字体",
    )
    wm.add_argument(
        "--watermark-size",
        type=int,
        default=28,
        help="水印字号（像素），默认 28",
    )
    wm.add_argument(
        "--watermark-color",
        default="white",
        help="文字颜色：#RGB / #RRGGBB / 命名色，默认 white",
    )
    wm.add_argument(
        "--watermark-opacity",
        type=float,
        default=1.0,
        help="不透明度 0.0~1.0，默认 1.0",
    )
    wm.add_argument(
        "--watermark-anchor",
        default="bottom-right",
        choices=VALID_ANCHORS,
        help="九宫格锚点，默认 bottom-right",
    )
    wm.add_argument(
        "--watermark-margin-x",
        type=int,
        default=24,
        help="水印距画面左右边缘的像素留白，默认 24",
    )
    wm.add_argument(
        "--watermark-margin-y",
        type=int,
        default=24,
        help="水印距画面上下边缘的像素留白，默认 24",
    )

    # ---- AES-128 分片加密（RFC 8216；不指定则分片保持明文） ----
    enc = parser.add_argument_group("AES-128 分片加密")
    enc.add_argument(
        "--encrypt",
        action="store_true",
        help=(
            "启用 AES-128-CBC 分片加密；未同时提供 --key-file/--key 时"
            "自动生成随机密钥与随机 IV，并把 16 字节裸密钥写入输出目录"
        ),
    )
    enc.add_argument(
        "--key-file",
        default=None,
        metavar="PATH",
        help=(
            "密钥文件：16 字节原始二进制，或只含 32 位 hex / Base64 文本；"
            "提供后即隐含启用加密"
        ),
    )
    enc.add_argument(
        "--key",
        dest="key_text",
        default=None,
        metavar="KEY",
        help=(
            "直接传入密钥：32 位十六进制或 Base64 字符串（解码后须为 16 字节）；"
            "提供后即隐含启用加密"
        ),
    )
    enc.add_argument(
        "--key-uri",
        default=DEFAULT_KEY_FILENAME,
        metavar="URI",
        help=(
            "EXT-X-KEY 的 URI 属性，默认 encryption.key（相对播放列表的"
            "裸文件名）；可换成 https:// 等绝对地址"
        ),
    )
    enc.add_argument(
        "--iv",
        dest="iv_text",
        default=None,
        metavar="IV",
        help=(
            "显式 IV：32 位十六进制（可带 0x 前缀）或 Base64；"
            "不指定时，--encrypt/--key 自动生成随机 IV，"
            "使用 --iv-mode sequence 时改为按分片序号隐式派生"
        ),
    )
    enc.add_argument(
        "--iv-mode",
        choices=("random", "sequence"),
        default="random",
        help=(
            "未显式给 --iv 时的 IV 策略：random（默认，随机显式 IV 并写入"
            "播放列表）或 sequence（RFC §5.2 序号派生，播放列表不写 IV）"
        ),
    )

    parser.add_argument(
        "-y", "--overwrite",
        action="store_true",
        help="允许写入非空输出目录并覆盖同名产物",
    )
    return parser


def _build_encryption_config(args) -> EncryptionConfig | None:
    """根据 CLI 参数构造加密配置；未启用加密时返回 None（旧行为不变）。"""
    if not (args.encrypt or args.key_file or args.key_text):
        return None
    if args.key_file and args.key_text:
        raise EncryptionError("--key-file 与 --key 只能二选一，不能同时提供")
    # 显式 --iv 与序号派生模式语义互斥，明确报错而不是静默覆盖
    if args.iv_text is not None and args.iv_mode == "sequence":
        raise EncryptionError(
            "已显式传入 --iv 时不能再指定 --iv-mode sequence"
            "（序号派生模式要求不提供 IV）"
        )

    key_uri = args.key_uri
    if args.key_file:
        # 未传 --iv 且要求序号模式时，from_key_file 的 iv=None 正好是该语义
        return EncryptionConfig.from_key_file(
            args.key_file, iv=args.iv_text, key_uri=key_uri
        )
    if args.key_text:
        return EncryptionConfig.from_key(
            args.key_text, iv=args.iv_text, key_uri=key_uri
        )
    # 仅 --encrypt：现场生成密钥；显式 --iv 优先解析（长度非法会明确报错），
    # 否则按 --iv-mode 生成随机 IV 或退化为 RFC 序号派生 IV
    generated_key = EncryptionConfig.generate(
        key_uri=key_uri, random_iv=(args.iv_mode == "random")
    )
    if args.iv_text is not None:
        return EncryptionConfig.from_key(
            generated_key.key, iv=args.iv_text, key_uri=key_uri
        )
    return generated_key


def _print_summary(result) -> None:
    segments = result.segments
    max_dur = max((s.duration for s in segments), default=0.0)
    total = sum(s.duration for s in segments)
    print(f"完成：{result.media.path}")
    print(f"  输出目录   : {result.output_dir}")
    print(f"  播放列表   : {PLAYLIST_FILENAME}")
    print(
        f"  分片数量   : {len(segments)}（TARGETDURATION="
        f"{result.target_duration}s，最大实际片长 {max_dur:.3f}s，"
        f"总时长 {total:.3f}s）"
    )
    print(
        "  列表模式   : "
        + (
            f"滑动窗口 {result.window_size} 片"
            if result.window_size is not None
            else "VOD（含 ENDLIST）"
        )
    )
    if result.extra.get("watermark_png"):
        print(f"  文字水印   : {result.extra['watermark_png']}")
    encryption_info = result.extra.get("encryption")
    if encryption_info:
        print("  AES-128 加密: 已启用（AES-128-CBC + PKCS#7）")
        print(f"    密钥文件 : {encryption_info['key_file']}")
        print(f"    密钥 URI : {encryption_info['key_uri']}")
        print(f"    密钥(hex): {encryption_info['key_hex']}")
        if encryption_info["iv_mode"] == "explicit":
            print(f"    IV(hex) : {encryption_info['iv_hex']}")
        else:
            print("    IV 策略 : 按分片 MEDIA-SEQUENCE 序号大端派生（RFC §5.2）")
    print("  分片明细（前 5 片）:")
    for seg in segments[:5]:
        print(
            f"    #{seg.sequence_number:04d} {seg.filename}  "
            f"src@{seg.source_start:7.3f}s  轴@{seg.start_time:7.3f}s  "
            f"时长 {seg.duration:6.3f}s"
        )
    if len(segments) > 5:
        print(f"    ... 其余 {len(segments) - 5} 片见 segments.json")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        watermark = None
        if args.watermark_text is not None:
            watermark = WatermarkConfig(
                text=args.watermark_text,
                font_files=tuple(args.watermark_font or ()),
                font_size=args.watermark_size,
                color=args.watermark_color,
                opacity=args.watermark_opacity,
                anchor=args.watermark_anchor,
                margin_x=args.watermark_margin_x,
                margin_y=args.watermark_margin_y,
            )
        encryption = _build_encryption_config(args)
        result = package(
            input_path=args.input,
            output_dir=args.output_dir,
            target_duration=args.segment_duration,
            window_size=args.window_size,
            media_sequence=args.media_sequence,
            filename_prefix=args.filename_prefix,
            overwrite=args.overwrite,
            watermark=watermark,
            encryption=encryption,
            name_template=args.name_template,
            container=args.segment_format,
            align_mode=args.align_mode,
        )
    except PackagerError as exc:
        # 可预期的输入/环境/切片问题：明确中文提示 + 非零退出码
        detail = f"：{exc.stderr_tail}" if getattr(exc, "stderr_tail", "") else ""
        print(f"错误：{exc}{detail}", file=sys.stderr)
        return EXIT_PACKAGER_ERROR
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_PACKAGER_ERROR

    _print_summary(result)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
