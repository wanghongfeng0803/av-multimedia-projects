# -*- coding: utf-8 -*-
"""对照 samples/ground_truth.json 的验收测试。

判定规则：
- 无缺陷素材（normal/night/fade/slowmo）：必须零误报；
- 缺陷素材：必须检出对应类型，且报告区间与标注区间在容差内
  （起始帧偏差 <= 2 帧，持续帧数偏差 <= 4 帧，覆盖 IoU >= 0.8）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video_qc.analyzer import analyze  # noqa: E402

TOL_START = 2
TOL_DURATION = 4


def interval_iou(a_start, a_len, b_start, b_len):
    lo = max(a_start, b_start)
    hi = min(a_start + a_len, b_start + b_len)
    inter = max(0, hi - lo)
    union = a_len + b_len - inter
    return inter / union if union else 1.0


def main():
    gt = json.loads((ROOT / "samples" / "ground_truth.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in gt.items():
        path = ROOT / "samples" / name
        report = analyze(path)
        got = report["issues"]
        print(f"\n=== {name} ===")
        for issue in got:
            print(f"  {issue['type']:16s} start={issue['start_frame']:4d} "
                  f"dur={issue['duration_frames']:4d} conf={issue['confidence']:.2f}")

        if not expected:
            if got:
                failures.append(f"{name}: 期望零误报，实际 {len(got)} 个: {got}")
            continue

        for exp in expected:
            match = next((g for g in got if g["type"] == exp["type"]), None)
            if match is None:
                failures.append(f"{name}: 漏检 {exp['type']}")
                continue
            iou = interval_iou(match["start_frame"], match["duration_frames"],
                               exp["start_frame"], exp["duration_frames"])
            if (abs(match["start_frame"] - exp["start_frame"]) > TOL_START
                    or abs(match["duration_frames"] - exp["duration_frames"]) > TOL_DURATION
                    or iou < 0.8):
                failures.append(
                    f"{name}: {exp['type']} 定位偏差过大 "
                    f"got=({match['start_frame']},{match['duration_frames']}) "
                    f"expected=({exp['start_frame']},{exp['duration_frames']}) iou={iou:.2f}")
            # 缺陷素材不允许冒出标注之外的其他类型问题
            extra = [g["type"] for g in got if g["type"] != exp["type"]]
            if extra:
                failures.append(f"{name}: 出现标注外的误报 {extra}")

    print("\n" + "=" * 60)
    if failures:
        print("验收失败：")
        for f in failures:
            print(" -", f)
        return 1
    print(f"全部 {len(gt)} 个样本验收通过：检出与帧级定位准确，无漏检无误报。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
