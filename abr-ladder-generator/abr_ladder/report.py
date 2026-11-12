"""Report generation (report.json + report.md).

The JSON is the machine-readable contract (per-rung measured bitrate, quality
scores, keyframe alignment, switch tests).  The Markdown file is the
human-readable delivery: why each resolution/bitrate pair was chosen, plus
the quality delta between adjacent rungs and whether it is visually
noticeable.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
from typing import List, Optional

from . import __version__, ff
from .config import Settings, SourceInfo, bpp
from .encode import EncodeResult
from .keyframes import KeyframePlan
from .quality import QualityResult


VMAF_BANDS = [
    (95, "极佳", "肉眼几乎不可见压缩痕迹"),
    (90, "优秀", "常规观看距离下难以察觉"),
    (80, "良好", "主流网络条件可接受"),
    (70, "一般", "弱网兜底可用，细节有损失"),
    (60, "较差", "明显压缩感，仅最后兜底"),
    (-1, "差", "不建议作为正式档位"),
]

SSIM_BANDS = [
    (0.98, "极佳", "肉眼几乎不可见压缩痕迹"),
    (0.95, "优秀", "常规观看距离下难以察觉"),
    (0.90, "良好", "主流网络条件可接受"),
    (0.85, "一般", "弱网兜底可用，细节有损失"),
    (0.75, "较差", "明显压缩感，仅最后兜底"),
    (-1, "差", "不建议作为正式档位"),
]

BPP_BANDS = [
    (0.15, "充足", "码率密度高，压缩压力小"),
    (0.075, "均衡", "码率密度与常规 CRF 水平接近"),
    (0.05, "偏紧", "需要编码器精细分配码率"),
    (0.03, "紧张", "复杂画面可能出现明显失真"),
    (-1, "很低", "仅适合极低复杂度画面"),
]


def _band(score: float, bands) -> tuple:
    for threshold, name, comment in bands:
        if score >= threshold:
            return name, comment
    return bands[-1][1], bands[-1][2]


def _quality_band(q: QualityResult) -> tuple:
    if q.metric == "vmaf":
        return _band(q.mean, VMAF_BANDS)
    return _band(q.mean, SSIM_BANDS)


def _bpp_band(value: float) -> tuple:
    return _band(value, BPP_BANDS)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def visible_judgement(delta_vmaf: Optional[float],
                     delta_ssim: Optional[float]) -> tuple:
    """Return (verdict, comment) describing an adjacent quality gap.

    Thresholds are empirical rules of thumb for VMAF deltas on SDR content;
    they are content- and display-dependent, so the report labels them as
    guidance rather than a hard guarantee.
    """
    if delta_vmaf is not None and delta_vmaf == delta_vmaf:
        d = abs(delta_vmaf)
        if d < 3:
            return "几乎无法分辨", "两档质量非常接近，带宽敏感场景可以更激进地降档"
        if d < 6:
            return "仔细对比可分辨", "并排对比才能看出差异，正常观看基本无感"
        if d < 12:
            return "可察觉", "切换时能感到清晰度变化，但仍在可接受范围"
        return "明显可察觉", "阶梯存在明显断层，弱网用户会感知到画质跳变"
    if delta_ssim is not None and delta_ssim == delta_ssim:
        d = abs(delta_ssim)
        if d < 0.005:
            return "几乎无法分辨", "两档质量非常接近"
        if d < 0.01:
            return "仔细对比可分辨", "并排对比才能看出差异"
        if d < 0.02:
            return "可察觉", "切换时能感到清晰度变化"
        return "明显可察觉", "阶梯存在明显断层"
    return "无法判定", "缺少质量指标"


def _mode_text(mode: str) -> str:
    if mode == "cbr-filler":
        return "CBR + filler（内容过易，VBR 无法吃满目标码率，用填充 NAL 精确达标）"
    return "2-pass capped VBR（pass-1 统计复用 + 闭环码率校正）"


def build_rung_rationale(source: SourceInfo, encode: EncodeResult,
                         quality: QualityResult) -> dict:
    rung = encode.rung
    density = bpp(encode.measured_kbps, rung.width, rung.height,
                  source.fps_float)
    bpp_name, bpp_comment = _bpp_band(density)
    q_name, q_comment = _quality_band(quality)
    metric = "VMAF" if quality.metric == "vmaf" else "SSIM"

    display = quality.display or {}
    display_score = display.get("mean")
    display_band = None
    display_comment = None
    if display_score is not None and metric == "VMAF":
        display_band, display_comment = _band(display_score, VMAF_BANDS)
    elif display_score is not None:
        display_band, display_comment = _band(display_score, SSIM_BANDS)

    worst = (display.get("worst_windows") or quality.worst_windows or [None])[0]

    parts = [
        f"{rung.width}x{rung.height} @ {rung.nominal_kbps} kb/s："
        f"实测 {encode.measured_kbps:.1f} kb/s（{encode.deviation_pct:+.2f}%），"
        f"像素码率 {density:.3f} bit/pixel/frame（{bpp_name}，{bpp_comment}）。",
        f"原生 {metric} 均值 {quality.mean:.2f}（{q_name}，{q_comment}）"
        + (f"，1% 分位 {quality.p1:.1f}，最低帧 {quality.min:.1f}"
           f" @ {quality.min_time:.1f}s。" if quality.metric == "vmaf"
           else f"，最低帧 {quality.min:.4f} @ {quality.min_time:.1f}s。"),
    ]
    if display_score is not None and display.get("target_width"):
        parts.append(
            f"同屏 {metric}（{display.get('label', '')}）均值 "
            f"{display_score:.2f}（{display_band}，{display_comment}），"
            f"最低 {display.get('min', 0):.2f} @ "
            f"{display.get('min_frame_time', 0):.1f}s。"
        )
    if worst:
        parts.append(
            f"最差 2 秒窗口 {worst['start']:.1f}–{worst['end']:.1f}s，"
            f"{metric} {worst['mean']:.2f}。"
        )
    if quality.ssim_mean is not None and quality.metric == "vmaf":
        parts.append(f"SSIM 均值 {quality.ssim_mean:.4f}。")

    decision_name = display_band or q_name
    reason = "该分辨率下维持这一码率密度，压缩压力与画质处于" + bpp_name + "区间。"
    if display_band is None and quality.metric == "vmaf":
        reason += f"原生 {metric} 达到{q_name}水平。"
    if decision_name in ("极佳", "优秀"):
        reason += f"同屏{metric}为{decision_name}，可作为该分辨率的正式档位。"
    elif decision_name == "良好":
        reason += f"同屏{metric}良好，适合作为主流网络下的主档位。"
    elif decision_name == "一般":
        reason += f"同屏{metric}一般，建议只作为弱网兜底档。"
    else:
        reason += f"同屏{metric}偏低，若业务重视画质需提高该档码率或移除该档。"

    return {
        "summary": " ".join(parts),
        "bpp": round(density, 4),
        "bpp_band": bpp_name,
        "quality_band": q_name,
        "display_quality_band": display_band,
        "display_score": round(display_score, 3)
        if display_score is not None else None,
        "metric": metric,
        "score": round(quality.mean, 3),
        "rate_mode": encode.mode,
        "rate_mode_text": _mode_text(encode.mode),
        "worst_window": worst,
    }


def build_adjacent_gaps(encodes: List[EncodeResult],
                        qualities: List[QualityResult]) -> List[dict]:
    gaps: List[dict] = []
    by_name = {q.rung_name: q for q in qualities}
    for upper, lower in zip(encodes, encodes[1:]):
        q_hi = by_name.get(upper.rung.name)
        q_lo = by_name.get(lower.rung.name)
        native_vmaf_delta = None
        ssim_delta = None
        display_vmaf_delta = None
        display_ssim_delta = None
        if q_hi and q_lo:
            if q_hi.metric == "vmaf" and q_lo.metric == "vmaf":
                native_vmaf_delta = q_hi.mean - q_lo.mean
            if q_hi.ssim_mean is not None and q_lo.ssim_mean is not None:
                ssim_delta = q_hi.ssim_mean - q_lo.ssim_mean
            d_hi = (q_hi.display or {}).get("mean")
            d_lo = (q_lo.display or {}).get("mean")
            if d_hi is not None and d_lo is not None:
                if (q_hi.display.get("metric") == "vmaf"
                        and q_lo.display.get("metric") == "vmaf"):
                    display_vmaf_delta = d_hi - d_lo
                else:
                    display_ssim_delta = d_hi - d_lo
            ds_hi = (q_hi.display or {}).get("ssim_mean")
            ds_lo = (q_lo.display or {}).get("ssim_mean")
            if ds_hi is not None and ds_lo is not None:
                display_ssim_delta = ds_hi - ds_lo
        # The same-screen metric is what a viewer on a top-resolution display
        # actually sees; it is the basis for the "visible?" verdict.
        verdict, comment = visible_judgement(
            display_vmaf_delta if display_vmaf_delta is not None
            else native_vmaf_delta,
            display_ssim_delta if display_ssim_delta is not None else ssim_delta,
        )
        saving = ((upper.measured_kbps - lower.measured_kbps)
                  / upper.measured_kbps * 100.0) if upper.measured_kbps else 0.0
        gaps.append({
            "from": upper.rung.name,
            "to": lower.rung.name,
            "metric": (qualities[0].metric if qualities else "vmaf"),
            "from_kbps": round(upper.measured_kbps, 2),
            "to_kbps": round(lower.measured_kbps, 2),
            "bandwidth_saving_pct": round(saving, 2),
            "native_delta": round(native_vmaf_delta, 3)
            if native_vmaf_delta is not None
            else (round(ssim_delta, 5) if ssim_delta is not None else None),
            "display_delta": round(display_vmaf_delta, 3)
            if display_vmaf_delta is not None
            else (round(display_ssim_delta, 5)
                  if display_ssim_delta is not None else None),
            "ssim_delta": round(ssim_delta, 5) if ssim_delta is not None else None,
            "display_ssim_delta": round(display_ssim_delta, 5)
            if display_ssim_delta is not None else None,
            "visible_judgement": verdict,
            "judgement_basis": "同屏指标" if display_vmaf_delta is not None
            or display_ssim_delta is not None else "原生指标",
            "comment": comment,
        })
    return gaps


def _ladder_summary(encodes: List[EncodeResult],
                    qualities: List[QualityResult],
                    gaps: List[dict]) -> dict:
    by_name = {q.rung_name: q for q in qualities}
    ordered = [by_name[e.rung.name] for e in encodes if e.rung.name in by_name]
    metric = ordered[0].metric if ordered else "vmaf"
    native_means = [q.mean for q in ordered]
    display_means = [(q.display or {}).get("mean") for q in ordered]
    display_means = [m for m in display_means if m is not None]
    ssim_means = [q.ssim_mean for q in ordered if q.ssim_mean is not None]
    notes = []
    small = 1.5 if metric == "vmaf" else 0.005
    big = 12.0 if metric == "vmaf" else 0.02
    for gap in gaps:
        d = gap.get("display_delta")
        if d is None:
            d = gap.get("native_delta")
        if d is not None and abs(d) < small:
            notes.append(
                f"{gap['from']}→{gap['to']} 同屏质量几乎相同但码率下降 "
                f"{gap['bandwidth_saving_pct']:.0f}%，该阶梯档存在冗余嫌疑。"
            )
        if d is not None and abs(d) > big:
            notes.append(
                f"{gap['from']}→{gap['to']} 同屏 {metric.upper()} 差 "
                f"{abs(d):.3f}，阶梯存在明显断层。"
            )
    return {
        "metric": metric,
        "total_nominal_kbps": sum(e.rung.nominal_kbps for e in encodes),
        "total_measured_kbps": round(sum(e.measured_kbps for e in encodes), 2),
        "native_min": round(min(native_means), 3) if native_means else None,
        "native_max": round(max(native_means), 3) if native_means else None,
        "native_span": round(max(native_means) - min(native_means), 3)
        if native_means else None,
        "display_min": round(min(display_means), 3) if display_means else None,
        "display_max": round(max(display_means), 3) if display_means else None,
        "display_span": round(max(display_means) - min(display_means), 3)
        if display_means else None,
        "ssim_min": round(min(ssim_means), 5) if ssim_means else None,
        "ssim_max": round(max(ssim_means), 5) if ssim_means else None,
        "lowest_rung": encodes[-1].rung.name if encodes else None,
        "notes": notes,
    }


def build_report(source: SourceInfo, settings: Settings, plan: KeyframePlan,
                 encodes: List[EncodeResult], qualities: List[QualityResult],
                 verification: dict, timings: dict, output_dir: str) -> dict:
    quality_by_name = {q.rung_name: q for q in qualities}
    rungs = []
    for enc in encodes:
        q = quality_by_name.get(enc.rung.name)
        rung = enc.to_dict()
        if q is not None:
            rung["quality"] = q.to_dict()
            rung["rationale"] = build_rung_rationale(source, enc, q)
        else:
            rung["quality"] = None
            rung["rationale"] = None
        if q is not None and q.warning:
            rung.setdefault("warnings", []).append(q.warning)
        rungs.append(rung)

    gaps = build_adjacent_gaps(encodes, qualities)
    summary = _ladder_summary(encodes, qualities, gaps)

    version = ff.version_info()
    total_wall = timings.get("total_wall_seconds", 0.0)
    within_budget = total_wall <= settings.time_budget_seconds if total_wall else None

    return {
        "schema_version": "1.0",
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
        "tool": {"name": "abr-ladder", "version": __version__},
        "source": source.to_dict(),
        "settings": settings.to_dict(),
        "keyframes": plan.to_dict(),
        "rungs": rungs,
        "adjacent_gaps": gaps,
        "ladder_summary": summary,
        "verification": verification,
        "performance": {
            "timings": timings,
            "time_budget_seconds": settings.time_budget_seconds,
            "within_time_budget": within_budget,
            "thread_budget": settings.threads,
        },
        "environment": {
            "ffmpeg": version.get("ffmpeg"),
            "libx264": ff.has_encoder("libx264"),
            "libvmaf": ff.has_filter("libvmaf"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "outputs": {
            "directory": os.path.abspath(output_dir),
            "videos": [os.path.join(output_dir, e.rung.name + ".mp4")
                       for e in encodes],
        },
    }


def render_markdown(report: dict) -> str:
    source = report["source"]
    settings = report["settings"]
    rungs = report["rungs"]
    gaps = report["adjacent_gaps"]
    keyframes = report["keyframes"]
    verification = report["verification"]
    perf = report["performance"]
    summary = report["ladder_summary"]
    primary_metric = (rungs[0]["quality"]["metric"].upper()
                      if rungs and rungs[0].get("quality") else "VMAF")

    lines: List[str] = []
    add = lines.append
    add("# ABR 阶梯质量报告")
    add("")
    add(f"生成时间：{report['generated_at']}  ")
    add(f"源片：`{source['path']}`  ")
    add(f"分辨率 {source['width']}x{source['height']}，帧率 "
        f"{source['fps_float']:.3f} fps，时长 {source['duration_seconds']:.1f}s  ")
    add(f"编码器：libx264 preset `{settings['preset']}`，线程预算 "
        f"{settings['threads']}，GOP {settings['gop_seconds']}s  ")
    add(f"质量指标：{primary_metric}"
        + (f"，模型：{rungs[0]['quality']['model']}"
           if rungs and primary_metric == "VMAF" else ""))
    add("")

    add("## 结论摘要")
    all_rate_ok = all(r.get("quality") is not None
                      and abs(r["deviation_pct"]) <= settings["rate_tolerance_pct"]
                      for r in rungs)
    align = verification.get("keyframe_alignment", {})
    switch = verification.get("switch_tests", [])
    switch_ok = all(t.get("ok") for t in switch) if switch else None
    add(f"- 四档实测码率：{'全部在 ±3% 内' if all_rate_ok else '存在超差档位'}"
        f"（总码率 {summary['total_measured_kbps'] / 1000:.2f} Mbps）。")
    add(f"- 关键帧对齐：{'全部对齐' if align.get('aligned') else '存在未对齐'}"
        f"，共 {keyframes['keyframe_count']} 个关键帧，"
        f"最大 GOP {keyframes['max_gap_frames']} 帧。")
    if switch_ok is not None:
        add(f"- 切档验证：{'通过' if switch_ok else '失败'}"
            f"（{len(switch)} 个切档点，最小边界 SSIM "
            f"{verification.get('switch_ssim_min')}）。")
    qdigits = 2 if primary_metric == "VMAF" else 4
    if summary.get("display_min") is not None:
        add(f"- 同屏 {primary_metric} 区间（放大到最高档）："
            f"{summary['display_min']:.{qdigits}f}–"
            f"{summary['display_max']:.{qdigits}f}"
            f"（差值 {summary['display_span']:.{qdigits}f}）。")
    if summary.get("native_min") is not None:
        add(f"- 原生 {primary_metric} 区间（各档分辨率下）："
            f"{summary['native_min']:.{qdigits}f}–"
            f"{summary['native_max']:.{qdigits}f}"
            f"（差值 {summary['native_span']:.{qdigits}f}）。")
    if perf.get("within_time_budget") is not None:
        add(f"- 总耗时 {perf['timings'].get('total_wall_seconds', 0):.0f}s，"
            f"{'满足' if perf['within_time_budget'] else '未达到'} "
            f"{perf['time_budget_seconds'] / 60:.0f} 分钟预算"
            f"（{perf['thread_budget']} 线程）。")
    if summary.get("notes"):
        for note in summary["notes"]:
            add(f"- {note}")
    add("")

    add("## 档位总表")
    add("")
    digits = 2 if primary_metric == "VMAF" else 4
    columns = ["档位", "分辨率", "标称码率", "实测码率", "偏差", "码控模式",
               f"平均 {primary_metric}（原生）",
               f"平均 {primary_metric}（同屏）", "同屏最低", "最差同屏 2s 窗口"]
    if primary_metric != "SSIM":
        columns.append("SSIM 均值")
    add("| " + " | ".join(columns) + " |")
    add("|" + "---|" * len(columns))
    for r in rungs:
        q = r["quality"] or {}
        display = q.get("display") or {}
        worst = (display.get("worst_windows") or [{}])[0] if display else {}
        worst_txt = (f"{_fmt(worst.get('mean'), digits)} "
                     f"@{worst.get('start', '—')}–{worst.get('end', '—')}s"
                     if worst else "—")
        cells = [
            r["name"], f"{r['width']}x{r['height']}",
            f"{r['nominal_kbps']} kb/s",
            f"{r['measured_kbps']:.1f} kb/s",
            f"{r['deviation_pct']:+.2f}%", r["mode"],
            _fmt(q.get("mean"), digits),
            _fmt(display.get("mean"), digits),
            _fmt(display.get("min"), digits),
            worst_txt,
        ]
        if primary_metric != "SSIM":
            cells.append(_fmt(q.get("ssim_mean"), 5))
        add("| " + " | ".join(cells) + " |")
    add("")

    add("## 各档选档依据")
    add("")
    for r in rungs:
        rationale = r.get("rationale")
        if not rationale:
            continue
        add(f"### {r['name']}（{r['width']}x{r['height']} @ {r['nominal_kbps']} kb/s）")
        add("")
        add(rationale["summary"])
        add("")
        add(f"- 码率控制：{rationale['rate_mode_text']}")
        add(f"- 闭环尝试次数：{len(r['attempts'])}")
        add(f"- 编码耗时：{r['encode_seconds']:.1f}s")
        add("")

    add("## 相邻档质量差（肉眼能否分辨）")
    add("")
    columns = ["相邻档", "码率", "节省带宽",
               f"原生 {primary_metric} 差", f"同屏 {primary_metric} 差"]
    if primary_metric != "SSIM":
        columns.append("SSIM 差（同屏）")
    columns += ["判断（同屏）", "说明"]
    add("| " + " | ".join(columns) + " |")
    add("|" + "---|" * len(columns))
    for g in gaps:
        cells = [
            f"{g['from']} → {g['to']}",
            f"{g['from_kbps']:.0f} → {g['to_kbps']:.0f} kb/s",
            f"{g['bandwidth_saving_pct']:.1f}%",
            _fmt(g.get("native_delta"), digits),
            _fmt(g.get("display_delta"), digits),
        ]
        if primary_metric != "SSIM":
            ssim_delta = (g.get("display_ssim_delta")
                          if g.get("display_ssim_delta") is not None
                          else g.get("ssim_delta"))
            cells.append(_fmt(ssim_delta, 5))
        cells += [g["visible_judgement"], g["comment"]]
        add("| " + " | ".join(cells) + " |")
    add("")
    add("“同屏 VMAF 差”把低档放大到最高档分辨率后与源片比较，包含分辨率损失，"
        "是“肉眼能否分辨”的判断依据；“原生 VMAF 差”只反映各档在自己分辨率下的"
        "编码质量，用于排查编码问题。")
    add("")
    add("判断阈值是对 SDR 内容的经验规则（同屏 VMAF 差 <3 几乎不可见，3–6 仔细可辨，"
        "6–12 可察觉，>12 明显），实际感知受内容和显示设备影响。")
    add("")

    add("## 关键帧对齐与切档验证")
    add("")
    add(f"- 关键帧计划：{keyframes['keyframe_count']} 个，"
        f"固定网格 {settings['gop_seconds']}s + "
        f"{len(keyframes['scene_cuts'])} 个场景切点；"
        f"最大间隔 {keyframes['max_gap_frames']} 帧"
        f"（{keyframes['max_gap_seconds']}s）。")
    add(f"- 跨档对齐：{'通过' if alignment_ok(align) else '失败'}"
        f"（帧号级完全一致）。")
    if verification.get("audio", {}).get("present"):
        add(f"- 音频一致性：{'通过' if verification['audio']['identical'] else '失败'}"
            f"（四档复用同一音频轨道）。")
    if switch:
        add("")
        add("| 切档 | 切点 | 帧号 | 解码+边界 SSIM | 结果 |")
        add("|---|---|---|---|---|")
        for t in switch:
            add(f"| {t['from']} → {t['to']} | {t['at_seconds']:.2f}s | "
                f"{t['frame_index']} | {_fmt(t['ssim_mean'], 4)} | "
                f"{'OK' if t['ok'] else 'FAIL'} |")
    add("")
    add("切档验证的做法：从目标档的关键帧处开始解码 8 帧（`-xerror` 下不允许任何"
        "解码错误），并与源片同时间点的缩放参考计算 SSIM。它覆盖了“切过去第一帧"
        "是否为完整 IDR、画面是否正常”这一关键路径。")
    add("")

    add("## 性能")
    add("")
    timings = perf.get("timings", {})
    add("| 阶段 | 耗时 |")
    add("|---|---|")
    for key, label in (
        ("probe", "源片探测"), ("keyframe_plan", "场景检测+关键帧计划"),
        ("prepare_audio", "音频准备"), ("verify", "码率/对齐/切档校验")
    ):
        if key in timings:
            add(f"| {label} | {timings[key]:.1f}s |")
    if timings.get("encodes"):
        add(f"| 四档编码（并行调度，累计耗时） | "
            f"{sum(timings['encodes'].values()):.1f}s |")
    if timings.get("quality"):
        add(f"| 质量评估（并行调度，累计耗时） | "
            f"{sum(timings['quality'].values()):.1f}s |")
    add(f"| **总墙钟** | **{timings.get('total_wall_seconds', 0):.1f}s** |")
    add("")
    add("编码和质量评估共享同一线程预算调度器；两-pass 的 pass-1 统计在码率校正"
        "时复用，因此每次校正只多一轮 pass-2。")
    add("")

    add("## 复现")
    add("")
    add("```bash")
    add(f"abr-ladder {json_quote(source['path'])} <output_dir> \\")
    add(f"  --preset {settings['preset']} --threads {settings['threads']} \\")
    add(f"  --gop-seconds {settings['gop_seconds']}")
    add(f"abr-ladder-verify <output_dir>")
    add("```")
    add("")

    env = report.get("environment", {})
    add("## 环境")
    add("")
    add(f"- {env.get('ffmpeg')}")
    add(f"- libx264: {env.get('libx264')}；libvmaf: {env.get('libvmaf')}")
    add(f"- {env.get('platform')}；Python {env.get('python')}；"
        f"逻辑核心 {env.get('logical_cpus')}；线程预算 {perf.get('thread_budget')}")
    add("")
    add("## 方法与边界")
    add("")
    add("- 实测码率按视频流 packet 字节数 / 视频时长计算，不含音频与容器开销；"
        "report.json 中同时给出文件总大小。")
    add("- 质量评估给出两个视角：原生指标（该档分辨率 vs 同分辨率参考，衡量编码"
        "质量）与同屏指标（该档放大到最高档分辨率 vs 源片，衡量包含分辨率损失"
        "的实际观感）。相邻档差值和肉眼判断使用同屏指标。")
    add("- VMAF 参考为源片按 bicubic 缩放到目标分辨率；VMAF 模型为 libvmaf "
        "vmaf_v0.6.1（可用 --vmaf-model 替换）。")
    add("- 本工具输出 H.264 8-bit SDR 阶梯；10-bit/HDR 源会被转换为 8-bit，"
        "不保留 HDR 元数据。")
    add("- 未做 HLS 切片/打包；输出为四个对齐的 MP4，可直接用于后续打包。")
    add("")
    return "\n".join(lines)


def alignment_ok(align: dict) -> bool:
    return bool(align.get("aligned"))


def json_quote(text: str) -> str:
    if any(ch in text for ch in " \t\"'&|<>"):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def write_report(report: dict, output_dir: str) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "report.json")
    md_path = os.path.join(output_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
        f.write("\n")
    return json_path, md_path
