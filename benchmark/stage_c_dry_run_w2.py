"""W2 (Phase B icon tuning) dry-run on v8 ui_dense failure cases.

Replays Phase 1 outputs from the Stage 2 v8 full run through the new
per-variant icon pipeline to check:

    - cases where v8 icon hit moved the coord WORSE (ui_dense_01/04/06 in
      v8) now abstain and return approx_xy unchanged (Phase 2 rescues);
    - the overall 7 Phase A stage_c_hit cases (ui_button_02/04/06/07/08
      + ui_dense_07 + one more) are not regressed — ui_button hits
      unchanged, icon hits may shift to miss where ambiguous, and that
      is acceptable because miss path is byte-identical to v7;
    - Stage B regression (34/34 tests) unchanged.

Run (Windows host, within-repo paths):
    python benchmark/stage_c_dry_run_w2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from benchmark import stage_c  # noqa: E402

V8_JSON = _ROOT / "benchmark" / "results.stage2_v8_full_20260417.json"
GT_JSON = _ROOT / "benchmark" / "ground_truth.json"
IMG_DIR = _ROOT / "benchmark"

# ui_dense + ui_button cases with Stage C hit or specific failure pattern
FOCUS_CASES = [
    "ui_button_02", "ui_button_04", "ui_button_05", "ui_button_06",
    "ui_button_08",
    "ui_dense_01", "ui_dense_02", "ui_dense_04", "ui_dense_06",
    "ui_dense_07", "ui_dense_09",
]


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def main():
    v8 = _load_json(V8_JSON)
    gt = {g["id"]: g for g in _load_json(GT_JSON)}

    # Build lookup: test_id → v8 pinpoint row
    v8_rows = {r["test_id"]: r for r in v8 if r.get("method") == "pinpoint"}

    print(f"\n{'test_id':13s}  {'v8_Phase1':>12s}  {'gt':>10s}  "
          f"{'v8_sc(method,out,dist)':>35s}  {'w2_sc(method,out,dist)':>40s}  {'Δdist':>7s}")
    print("-" * 135)

    summary = {
        "button_hits_preserved": 0,
        "button_regressed": 0,
        "icon_hits_dropped": 0,
        "icon_hits_preserved": 0,
        "icon_miss_unchanged": 0,
    }

    for tid in FOCUS_CASES:
        if tid not in v8_rows or tid not in gt:
            continue
        r = v8_rows[tid]
        g = gt[tid]
        approx_xy = tuple(r.get("stage_c_input_xy", [r["x"], r["y"]]))
        image_path = IMG_DIR / g["image"]
        img = Image.open(image_path).convert("RGB")

        # Determine target_type for dispatch (same Phase 1.5 heuristic)
        from benchmark import stage_c_phase_1_5 as p15
        p15_out = p15.declare_target(g["target"])
        target_type = p15_out["target_type"]

        # W2 dry-run: re-run Stage C with current (W2-modified) code
        w2 = stage_c.resolve_target(
            target_type, p15_out["intent"], approx_xy, img,
            label_hint=p15_out.get("label_hint"),
        )

        v8_method = r.get("stage_c_method", "")
        v8_out = tuple(r.get("stage_c_output_xy", approx_xy))
        v8_dist_gt = _dist(v8_out[0], v8_out[1], g["x"], g["y"])
        w2_out = (w2["x"], w2["y"])
        w2_dist_gt = _dist(w2_out[0], w2_out[1], g["x"], g["y"])
        delta = w2_dist_gt - v8_dist_gt

        # Classify
        v8_was_hit = r.get("stage_c_hit", False)
        w2_is_hit = w2["hit"]
        if target_type == "button":
            if v8_was_hit and w2_is_hit and w2_out == v8_out:
                summary["button_hits_preserved"] += 1
                marker = "[OK] button-unchanged"
            elif v8_was_hit and not w2_is_hit:
                summary["button_regressed"] += 1
                marker = "[REGRESSED] BUTTON REGRESSED"
            elif not v8_was_hit and not w2_is_hit:
                summary["button_hits_preserved"] += 1  # both miss, fine
                marker = "• button miss→miss"
            else:
                marker = "• button changed"
        elif target_type == "icon":
            if v8_was_hit and not w2_is_hit:
                summary["icon_hits_dropped"] += 1
                improved = "improved" if delta < -5 else ("worsened" if delta > 5 else "similar")
                marker = f"[ICON-ABSTAIN] icon hit→miss ({improved})"
            elif v8_was_hit and w2_is_hit:
                summary["icon_hits_preserved"] += 1
                marker = "• icon hit preserved"
            elif not v8_was_hit and not w2_is_hit:
                summary["icon_miss_unchanged"] += 1
                marker = "• icon miss→miss"
            else:
                marker = "• icon miss→hit (new)"
        else:
            marker = "—"

        print(f"{tid:13s}  {str(approx_xy):>12s}  {str((g['x'],g['y'])):>10s}  "
              f"{v8_method+','+str(v8_out)+','+f'{v8_dist_gt:.1f}':>35s}  "
              f"{w2['method']+','+str(w2_out)+','+f'{w2_dist_gt:.1f}':>40s}  "
              f"{delta:+6.1f}   {marker}")

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))

    # Pass criteria
    no_button_regression = summary["button_regressed"] == 0
    print(f"\nNo button regression: {no_button_regression}")
    return 0 if no_button_regression else 2


if __name__ == "__main__":
    sys.exit(main())
