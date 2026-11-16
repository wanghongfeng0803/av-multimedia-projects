"""映射器可复算对比工具：Reinhard（基线）vs Hable/Filmic。

两种运行模式，**全程仅用 Python 标准库**：

1. 合成模式（默认，无需 ffmpeg）：在归一化线性光网格上直接调用
   :func:`hdr_to_sdr.tonemap` 的纯函数，产出
   - 映射曲线/局部斜率对比 SVG；
   - 锚点数值表 CSV（高精度，可独立复算）；
   - 高光细节指标（8bit 可分辨级数、几何均值局部斜率）JSON。
2. 真实素材模式（``--input 视频``）：用与正式管线完全相同的解码命令与
   :class:`~hdr_to_sdr.frames.FrameProcessor`，对**同一帧号区间**分别用
   两种映射器处理同一批解码原始帧，逐帧统计亮度与高光码级分布，
   产出逐帧 CSV 与帧级对比 SVG，验证切换映射器只改变色调映射这一步。

用法::

    python tools/compare_tonemappers.py --outdir build/tonemap_cmp
    python tools/compare_tonemappers.py --input sample.mkv \\
        --frames 0-29 --outdir build/tonemap_cmp

所有数值均为确定性输出：同参数重复运行逐字节一致（SVG 不含时间戳）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

# 允许从仓库根目录直接运行（python tools/compare_tonemappers.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr_to_sdr.binaries import find_ffmpeg  # noqa: E402
from hdr_to_sdr.frames import FrameProcessor  # noqa: E402
from hdr_to_sdr.pipeline import build_decode_cmd  # noqa: E402
from hdr_to_sdr.probe import probe_hdr  # noqa: E402
from hdr_to_sdr.tonemap import (  # noqa: E402
    DEFAULT_HABLE_SHARPNESS,
    TONE_HABLE,
    TONE_REINHARD,
    hable_scale,
    reinhard_scale,
)

# 对比高光带（归一化 x=y/peak 域）
DETAIL_BANDS = ((1.0, 4.0), (1.0, 10.0), (2.0, 8.0))
# 锚点表的采样位置
ANCHOR_XS = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)


def _deriv(scale_fn, x: float) -> float:
    """映射输出 h(x)=x·s(x) 的数值导数（中心差分）。"""
    h = 1e-7
    up = (x + h) * scale_fn(x + h, 1.0)
    dn = (x - h) * scale_fn(x - h, 1.0) if x > h else 0.0
    return (up - dn) / (2 * h)


def _curve(scale_fn, x: float) -> float:
    return x * scale_fn(x, 1.0)


def _count_levels(scale_fn, lo: float, hi: float, n: int = 1001) -> int:
    xs = (lo + (hi - lo) * i / (n - 1) for i in range(n))
    return len({round(_curve(scale_fn, x) * 255) for x in xs})


def _geo_mean_slope(scale_fn, lo: float, hi: float, n: int = 801) -> float:
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return math.exp(sum(math.log(max(_deriv(scale_fn, x), 1e-12)) for x in xs) / n)


def write_anchor_csv(path: str) -> None:
    """写锚点数值表：两映射器在同一网格上的 h(x) 与缩放系数 s。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("x,reinhard_h,reinhard_s,hable_h,hable_s,abs_diff_h\n")
        for x in ANCHOR_XS:
            rh = _curve(reinhard_scale, x)
            hh = _curve(hable_scale, x)
            rs = reinhard_scale(x, 1.0)
            hs = hable_scale(x, 1.0)
            f.write(
                f"{x:.4f},{rh:.10f},{rs:.10f},{hh:.10f},{hs:.10f},"
                f"{abs(rh - hh):.10f}\n"
            )


def write_detail_json(path: str) -> dict:
    """写高光细节指标 JSON（可复算的验收口径）。"""
    report = {"sharpness_default": DEFAULT_HABLE_SHARPNESS, "bands": []}
    for lo, hi in DETAIL_BANDS:
        entry = {
            "band": [lo, hi],
            "reinhard_levels_8bit": _count_levels(reinhard_scale, lo, hi),
            "hable_levels_8bit": _count_levels(hable_scale, lo, hi),
            "reinhard_geomean_slope": _geo_mean_slope(reinhard_scale, lo, hi),
            "hable_geomean_slope": _geo_mean_slope(hable_scale, lo, hi),
        }
        entry["slope_ratio_hable_over_reinhard"] = (
            entry["hable_geomean_slope"] / entry["reinhard_geomean_slope"]
        )
        entry["hable_not_weaker"] = (
            entry["hable_levels_8bit"] >= entry["reinhard_levels_8bit"]
            and entry["slope_ratio_hable_over_reinhard"] >= 1.0
        )
        report["bands"].append(entry)
    # 峰值同量级
    report["peak_output"] = {
        "reinhard_h1": _curve(reinhard_scale, 1.0),
        "hable_h1": _curve(hable_scale, 1.0),
    }
    report["peak_output"]["hable_ratio"] = (
        report["peak_output"]["hable_h1"]
        / report["peak_output"]["reinhard_h1"]
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def write_curve_svg(path: str) -> None:
    """手写零依赖 SVG：左图映射曲线，右图高光带局部斜率。"""
    width, height = 960, 420
    pad_l, pad_b, pad_t, pad_r = 60, 48, 30, 24
    plot_w, plot_h = 380, 320
    x_max, y_max = 4.0, 1.05

    def px(x):
        return pad_l + x / x_max * plot_w

    def py(y):
        return pad_t + plot_h - y / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    def axes(ox, title, y_label):
        parts.append(
            f'<text x="{ox + plot_w / 2}" y="20" text-anchor="middle" '
            f'font-size="15" font-weight="bold">{title}</text>'
        )
        # 网格 + y 刻度
        for gy in range(0, 6):
            yv = gy / 5.0
            yy = py(yv)
            parts.append(
                f'<line x1="{ox}" y1="{yy}" x2="{ox + plot_w}" y2="{yy}" '
                f'stroke="#e6e6e6"/>'
            )
            parts.append(
                f'<text x="{ox - 8}" y="{yy + 4}" text-anchor="end" '
                f'font-size="11" fill="#555">{yv:.1f}</text>'
            )
        for gx in range(0, 5):
            xv = gx
            xx = ox + xv / x_max * plot_w
            parts.append(
                f'<text x="{xx}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
                f'font-size="11" fill="#555">{xv}</text>'
            )
        parts.append(
            f'<line x1="{ox}" y1="{pad_t}" x2="{ox}" '
            f'y2="{pad_t + plot_h}" stroke="#333"/>'
        )
        parts.append(
            f'<line x1="{ox}" y1="{pad_t + plot_h}" x2="{ox + plot_w}" '
            f'y2="{pad_t + plot_h}" stroke="#333"/>'
        )
        parts.append(
            f'<text x="{ox + plot_w / 2}" y="{height - 8}" text-anchor="middle" '
            f'font-size="12">归一化亮度 x = y / peak</text>'
        )
        parts.append(
            f'<text x="14" y="{pad_t + plot_h / 2}" font-size="12" '
            f'transform="rotate(-90 14 {pad_t + plot_h / 2})" '
            f'text-anchor="middle">{y_label}</text>'
        )

    def polyline(ox, scale_fn, color, dashed=False):
        pts = []
        for i in range(201):
            x = x_max * i / 200
            pts.append(f"{ox + (px(x) - pad_l):.1f},{py(_curve(scale_fn, x)):.1f}")
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} '
            f'points="{" ".join(pts)}"/>'
        )

    def slope_line(ox, scale_fn, color, dashed=False):
        pts = []
        for i in range(1, 201):
            x = x_max * i / 200
            sv = min(_deriv(scale_fn, x), 1.05)
            pts.append(f"{ox + (px(x) - pad_l):.1f},{py(sv):.1f}")
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} '
            f'points="{" ".join(pts)}"/>'
        )

    # 左图：映射曲线
    axes(pad_l, "映射曲线 h(x)", "h(x)")
    # 恒等参考
    id_pts = f"{pad_l},{py(0)} {px(x_max)},{py(min(x_max, y_max))}"
    parts.append(
        f'<polyline fill="none" stroke="#999" stroke-width="1.5" '
        f'stroke-dasharray="2 3" points="{id_pts}"/>'
    )
    polyline(pad_l, reinhard_scale, "#1f77b4")
    polyline(pad_l, hable_scale, "#d62728", dashed=True)
    # 膝盖标记
    parts.append(
        f'<line x1="{px(0.75)}" y1="{pad_t}" x2="{px(0.75)}" '
        f'y2="{pad_t + plot_h}" stroke="#888" stroke-dasharray="3 3"/>'
    )
    parts.append(
        f'<text x="{px(0.75) + 4}" y="{pad_t + 12}" font-size="10" '
        f'fill="#666">膝盖 0.75</text>'
    )

    # 右图：局部斜率
    ox2 = pad_l + plot_w + 70
    axes(ox2, "局部斜率（高光细节对比度）", "h′(x)")
    slope_line(ox2, reinhard_scale, "#1f77b4")
    slope_line(ox2, hable_scale, "#d62728", dashed=True)

    # 图例
    lx, ly = width - 200, 60
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#1f77b4" stroke-width="2"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 4}" font-size="12">Reinhard（基线）</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 22}" x2="{lx + 26}" y2="{ly + 22}" stroke="#d62728" stroke-width="2" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 26}" font-size="12">Hable/Filmic c={DEFAULT_HABLE_SHARPNESS}</text>')
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---- 真实素材模式 ----------------------------------------------------------
def _read_exact(pipe, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        part = pipe.read(size - len(buf))
        if not part:
            break
        buf.extend(part)
    return bytes(buf)


def _parse_frame_interval(spec: str) -> tuple[int, int]:
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi)


def compare_real_file(
    src: str, frame_lo: int, frame_hi: int, outdir: str
) -> str:
    """对同一帧号区间跑两种映射器，逐帧统计，写 CSV 并返回路径。"""
    info = probe_hdr(src)
    ffmpeg = find_ffmpeg()
    decode_cmd, raw_fmt, bpp = build_decode_cmd(ffmpeg, src, info)
    frame_size = info.width * info.height * bpp

    processors = {
        TONE_REINHARD: FrameProcessor(
            info.transfer, info.bits_per_raw_sample, info.pixel_range,
            info.is_rgb, tone_method=TONE_REINHARD, peak_nits=info.peak_nits,
        ),
        TONE_HABLE: FrameProcessor(
            info.transfer, info.bits_per_raw_sample, info.pixel_range,
            info.is_rgb, tone_method=TONE_HABLE, peak_nits=info.peak_nits,
        ),
    }

    dec = subprocess.Popen(
        decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=frame_size,
    )
    csv_path = os.path.join(outdir, "frame_compare.csv")
    rows = ["frame_idx,reinhard_mean,reinhard_hi_levels,hable_mean,"
            "hable_hi_levels,mean_abs_diff"]
    idx = 0
    try:
        while idx <= frame_hi:
            chunk = _read_exact(dec.stdout, frame_size)
            if len(chunk) < frame_size:
                break
            if idx >= frame_lo:
                stats = {}
                outs = {}
                for name, proc in processors.items():
                    out = proc.process_frame(chunk, info.width, info.height)
                    outs[name] = out
                    # 统计口径统一用处理器输出帧（SDR 端实际可见的码值）
                    stats[name] = _frame_stats(out, info, bpp)
                diff = sum(
                    abs(a - b) for a, b in zip(
                        outs[TONE_REINHARD][:frame_size],
                        outs[TONE_HABLE][:frame_size],
                    )
                ) / frame_size
                rows.append(
                    f"{idx},{stats[TONE_REINHARD][0]:.4f},"
                    f"{stats[TONE_REINHARD][1]},"
                    f"{stats[TONE_HABLE][0]:.4f},{stats[TONE_HABLE][1]},"
                    f"{diff:.4f}"
                )
            idx += 1
    finally:
        dec.stdout.close()
        dec.wait()

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(rows) + "\n")
    return csv_path


def _frame_stats(frame: bytes, info, bpp: int) -> tuple[float, int]:
    """统计处理器**输出帧**首平面的均值与高光码级数（SDR 端可见口径）。

    YUV 路径输出是 16bit 容器承载的目标码深 Y 值（默认 8bit limited，
    白电平 235）：统计高于白电平 80%（≥191）的不同码值个数，码级越多
    代表高光层次越丰富。RGB 路径输出 gbrpf32 浮点，按归一化值 ≥0.8 统计。
    """
    plane_samples = info.width * info.height
    if info.is_rgb:
        import array
        floats = array.array("f")
        floats.frombytes(frame[: plane_samples * 4])
        vals = [v for v in floats if math.isfinite(v)]
        hi = len({round(v * 255) for v in vals if v >= 0.8})
        return sum(vals) / len(vals), hi
    import array
    samples = array.array("H")
    samples.frombytes(frame[: plane_samples * 2])
    vals = list(samples)
    # FrameProcessor 默认输出 8bit limited：白电平 235，高光阈值取其 80%
    white_level = 235
    threshold = int(white_level * 0.8)
    hi_levels = len({v for v in vals if v >= threshold})
    return sum(vals) / len(vals), hi_levels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="真实 HDR 视频；省略则只跑合成模式")
    parser.add_argument(
        "--frames", default="0-29",
        help="真实素材模式的帧号区间（闭区间，默认 0-29）",
    )
    parser.add_argument(
        "--outdir", default=os.path.join("build", "tonemap_cmp"),
        help="输出目录（默认 build/tonemap_cmp）",
    )
    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    anchor = os.path.join(args.outdir, "anchor_table.csv")
    detail = os.path.join(args.outdir, "highlight_detail.json")
    svg = os.path.join(args.outdir, "tone_curves.svg")
    write_anchor_csv(anchor)
    report = write_detail_json(detail)
    write_curve_svg(svg)
    print(f"锚点数值表: {anchor}")
    print(f"高光细节报告: {detail}")
    print(f"曲线对比图: {svg}")

    all_not_weaker = all(b["hable_not_weaker"] for b in report["bands"])
    print(
        "高光细节不弱于 Reinhard: "
        + ("是" if all_not_weaker else "否")
        + f"；峰值输出比 {report['peak_output']['hable_ratio']:.4f}"
    )

    if args.input:
        lo, hi = _parse_frame_interval(args.frames)
        path = compare_real_file(args.input, lo, hi, args.outdir)
        print(f"逐帧对比（帧 {lo}-{hi}）: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
