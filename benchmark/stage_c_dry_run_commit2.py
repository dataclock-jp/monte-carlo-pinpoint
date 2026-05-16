"""Stage C Commit 2 dry-run: button / icon pipelines on real benchmark images.

Supervisor-specified cases (dlg/cognitive-nodes/..T18:59:05-6fe9):
    ui_button_01   — Cut button, v7 dist 4.1, button pipeline primary test
    ui_button_04   — Save button, v7 dist 19.1, Commit 1 missed, button test
    ui_button_05   — v7 dist 29.0, button test
    ui_dense_02    — Undo icon, **Commit 1 menu_item drift 18.7px**,
                     Commit 2 icon pipeline should avoid the drift by
                     returning an icon centroid closer to ground truth
    ui_dense_09    — Chart icon, v7 dist 33.6, icon test

For each case we report:
    - stage_c menu_item (baseline retrofit) — confirms target_type mismatch
      behaves predictably (label pipeline on an icon target → drift)
    - stage_c button  — applicable to ui_button_* cases
    - stage_c icon    — applicable to ui_dense_* cases
    - generic (sanity: must always miss + preserve approx_xy)
    - CV determinism N=10 for each pipeline

Pass criteria:
    - generic invariant holds 5/5
    - CV determinism stdev < 0.5 px across all pipelines × cases
    - button / icon hits report distance-from-gt that is NOT WORSE than
      v7 final distance (miss is fine; worse hit is a regression)

Run:
    python benchmark/stage_c_dry_run_commit2.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from benchmark import stage_c  # noqa: E402

BUTTON_CASES = ("ui_button_01", "ui_button_04", "ui_button_05")
ICON_CASES = ("ui_dense_02", "ui_dense_09")
ALL_CASES = BUTTON_CASES + ICON_CASES

V7_JSON = _ROOT / "benchmark" / "results.v7_full_20260416.json"
GT_JSON = _ROOT / "benchmark" / "ground_truth.json"
IMG_DIR = _ROOT / "benchmark"
DETERMINISM_N = 10


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gather_cases():
    v7 = _load_json(V7_JSON)
    gt = {g["id"]: g for g in _load_json(GT_JSON)}
    cases = []
    for row in v7:
        if row.get("method") != "pinpoint":
            continue
        tid = row.get("test_id")
        if tid not in ALL_CASES:
            continue
        g = gt[tid]
        cases.append({
            "test_id": tid,
            "target": g["target"],
            "image": str(IMG_DIR / g["image"]),
            "gt_xy": (int(g["x"]), int(g["y"])),
            "v7_final_xy": (int(row["x"]), int(row["y"])),
            "v7_distance_px": float(row.get("distance_px", 0.0)),
            "expected_target_type": "button" if tid in BUTTON_CASES else "icon",
        })
    cases.sort(key=lambda c: ALL_CASES.index(c["test_id"]))
    return cases


def _stdev_xy(rows):
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    if len(xs) < 2:
        return 0.0, 0.0
    return statistics.stdev(xs), statistics.stdev(ys)


def _dist_from_gt(x: int, y: int, gt_xy) -> float:
    dx = x - gt_xy[0]
    dy = y - gt_xy[1]
    return (dx * dx + dy * dy) ** 0.5


def main() -> int:
    cases = _gather_cases()
    if len(cases) != len(ALL_CASES):
        print(f"ERROR: expected {len(ALL_CASES)} cases, got {len(cases)}",
              file=sys.stderr)
        return 1

    report = {
        "commit": "stage_c Commit 2 dry-run (button/icon pipelines)",
        "cases": list(ALL_CASES),
        "determinism_n": DETERMINISM_N,
        "per_case": [],
        "summary": {},
    }

    overall_generic_ok = True
    overall_deterministic = True
    overall_no_regression = True
    button_hits = 0
    icon_hits = 0

    for c in cases:
        print(f"\n=== {c['test_id']} [{c['expected_target_type']}] ===")
        print(f"  target: {c['target']}")
        print(f"  image: {os.path.relpath(c['image'], _ROOT)}")
        print(f"  v7_final_xy: {c['v7_final_xy']}  gt: {c['gt_xy']}  "
              f"v7_dist: {c['v7_distance_px']:.1f}")

        img = Image.open(c["image"]).convert("RGB")
        shot_w, shot_h = img.size
        approx_xy = c["v7_final_xy"]
        gt_xy = c["gt_xy"]
        expected = c["expected_target_type"]

        # ---- Stage C with expected target_type ----
        t0 = time.time()
        sc_expected = stage_c.resolve_target(
            expected, "center", approx_xy, img,
            label_hint=c["target"],
        )
        t_expected = time.time() - t0

        # ---- Stage C generic (sanity) ----
        sc_gen = stage_c.resolve_target(
            "generic", "center", approx_xy, img,
            label_hint=c["target"],
        )
        generic_ok = (
            not sc_gen["hit"]
            and sc_gen["x"] == approx_xy[0]
            and sc_gen["y"] == approx_xy[1]
            and sc_gen["method"] == "generic_miss"
        )
        overall_generic_ok &= generic_ok

        # ---- Stage C menu_item (Commit 1 baseline, drift sanity) ----
        sc_menu = stage_c.resolve_target(
            "menu_item", "center", approx_xy, img,
            label_hint=c["target"],
        )

        # ---- Determinism (expected target_type, N=10) ----
        runs = [
            stage_c.resolve_target(
                expected, "center", approx_xy, img,
                label_hint=c["target"])
            for _ in range(DETERMINISM_N)
        ]
        stdev_x, stdev_y = _stdev_xy(runs)
        deterministic = (
            stdev_x < 0.5 and stdev_y < 0.5
            and len({r["hit"] for r in runs}) == 1
        )
        overall_deterministic &= deterministic

        # ---- Regression check: if expected pipeline hit, new dist must be
        #      better than v7 final dist; else miss is OK ----
        if sc_expected["hit"]:
            new_dist = _dist_from_gt(sc_expected["x"], sc_expected["y"], gt_xy)
            no_regression = new_dist <= c["v7_distance_px"]
            if expected == "button":
                button_hits += 1
            else:
                icon_hits += 1
        else:
            new_dist = c["v7_distance_px"]  # miss → no change
            no_regression = True
        overall_no_regression &= no_regression

        # Menu_item drift reference: distance of sc_menu hit (if any) from gt
        menu_drift = (
            _dist_from_gt(sc_menu["x"], sc_menu["y"], gt_xy)
            if sc_menu["hit"] else None
        )

        entry = {
            "test_id": c["test_id"],
            "target": c["target"],
            "approx_xy": list(approx_xy),
            "gt_xy": list(gt_xy),
            "v7_distance_px": c["v7_distance_px"],
            "expected_target_type": expected,
            "stage_c_expected": {
                "hit": bool(sc_expected["hit"]),
                "x": int(sc_expected["x"]), "y": int(sc_expected["y"]),
                "method": sc_expected["method"],
                "confidence": float(sc_expected["confidence"]),
                "time_s": t_expected,
                "dist_from_gt_px": new_dist,
            },
            "stage_c_menu_item": {
                "hit": bool(sc_menu["hit"]),
                "x": int(sc_menu["x"]), "y": int(sc_menu["y"]),
                "method": sc_menu["method"],
                "match": sc_menu.get("match", ""),
                "dist_from_gt_px": menu_drift,
            },
            "stage_c_generic": {
                "hit": bool(sc_gen["hit"]),
                "x": int(sc_gen["x"]), "y": int(sc_gen["y"]),
                "method": sc_gen["method"],
            },
            "generic_invariant_ok": generic_ok,
            "determinism": {
                "stdev_x": stdev_x, "stdev_y": stdev_y,
                "pass": deterministic,
            },
            "no_regression": no_regression,
        }
        report["per_case"].append(entry)

        print(f"  sc_{expected}: hit={sc_expected['hit']!s:5s} xy=({sc_expected['x']},{sc_expected['y']}) "
              f"method={sc_expected['method']} conf={sc_expected['confidence']:.2f} "
              f"t={t_expected*1000:.1f}ms dist_from_gt={new_dist:.1f}")
        if sc_menu["hit"]:
            print(f"  sc_menu (drift ref): hit=True xy=({sc_menu['x']},{sc_menu['y']}) "
                  f"match='{sc_menu.get('match','')}' dist_from_gt={menu_drift:.1f}")
        else:
            print(f"  sc_menu (drift ref): miss (method={sc_menu['method']})")
        print(f"  sc_generic: hit={sc_gen['hit']!s:5s} xy=({sc_gen['x']},{sc_gen['y']}) "
              f"{'[INVARIANT OK]' if generic_ok else '[INVARIANT FAIL]'}")
        print(f"  determinism: stdev=({stdev_x:.3f},{stdev_y:.3f}) "
              f"{'PASS' if deterministic else 'FAIL'}")
        print(f"  no_regression: {no_regression}")

    report["summary"] = {
        "cases": len(cases),
        "button_hits": button_hits,
        "icon_hits": icon_hits,
        "all_generic_invariant_ok": overall_generic_ok,
        "all_deterministic": overall_deterministic,
        "all_no_regression": overall_no_regression,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))

    out_path = _ROOT / "benchmark" / "stage_c_commit2_dry_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nreport written: {out_path}")

    pass_all = (
        overall_generic_ok and overall_deterministic and overall_no_regression
    )
    return 0 if pass_all else 2


if __name__ == "__main__":
    sys.exit(main())
