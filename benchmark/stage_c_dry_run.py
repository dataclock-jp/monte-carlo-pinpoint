"""Stage C Commit 1 dry-run harness.

Loads P1 candidate cases (supervisor-specified: ui_button_04 +
ui_dense_02/03/06/09), replays v7 final predictions as approx_xy, and runs
stage_c.resolve_target through both ``menu_item`` (Stage B retrofit) and
``generic`` (miss-only) pipelines. Produces a per-case report with:

    - stage_c_hit / stage_c_method / stage_c_confidence
    - byte-identical check: stage_c(menu_item) vs stage_b.apply_stage_b_snap
    - CV determinism: N=10 stdev(x), stdev(y)

This exercises the real OCR backend (not mocked), so it must run on the
Windows host with WinRT OCR available.

Usage:
    python benchmark/stage_c_dry_run.py
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

from benchmark import stage_b, stage_c  # noqa: E402

P1_CASES = ["ui_button_04", "ui_dense_02", "ui_dense_03", "ui_dense_06", "ui_dense_09"]
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
        if tid not in P1_CASES:
            continue
        g = gt[tid]
        cases.append({
            "test_id": tid,
            "target": g["target"],
            "image": str(IMG_DIR / g["image"]),
            "gt_xy": (int(g["x"]), int(g["y"])),
            "v7_final_xy": (int(row["x"]), int(row["y"])),
            "v7_stage_b_hit": row.get("stage_b_hit"),
            "v7_distance_px": float(row.get("distance_px", 0.0)),
        })
    cases.sort(key=lambda c: P1_CASES.index(c["test_id"]))
    return cases


def _run_stage_c(target_type, approx_xy, image, label_hint=None):
    return stage_c.resolve_target(
        target_type, "center", approx_xy, image,
        label_hint=label_hint,
    )


def _run_stage_b_ref(approx_xy, image, target, shot_w, shot_h):
    return stage_b.apply_stage_b_snap(
        image, approx_xy[0], approx_xy[1],
        True, True, target,
        shot_w, shot_h,
    )


def _stdev_xy(rows):
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    if len(xs) < 2:
        return 0.0, 0.0
    return statistics.stdev(xs), statistics.stdev(ys)


def main() -> int:
    cases = _gather_cases()
    if len(cases) != len(P1_CASES):
        print(f"ERROR: expected {len(P1_CASES)} cases, got {len(cases)}",
              file=sys.stderr)
        return 1

    report = {
        "commit": "stage_c Commit 1 dry-run",
        "p1_cases": P1_CASES,
        "determinism_n": DETERMINISM_N,
        "per_case": [],
        "summary": {},
    }

    overall_byte_identical = True
    overall_deterministic = True
    stage_c_hits_menu = 0
    stage_c_hits_generic = 0

    for c in cases:
        print(f"\n=== {c['test_id']} ===")
        print(f"  target: {c['target']}")
        print(f"  image: {os.path.relpath(c['image'], _ROOT)}")
        print(f"  v7_final_xy: {c['v7_final_xy']}  gt: {c['gt_xy']}  "
              f"v7_dist: {c['v7_distance_px']:.1f}  "
              f"v7_sb_hit: {c['v7_stage_b_hit']}")

        img = Image.open(c["image"]).convert("RGB")
        shot_w, shot_h = img.size
        approx_xy = c["v7_final_xy"]

        # ---- Reference: raw Stage B ----
        t0 = time.time()
        sb = _run_stage_b_ref(approx_xy, img, c["target"], shot_w, shot_h)
        sb_time = time.time() - t0

        # ---- Stage C menu_item ----
        sc_menu = _run_stage_c("menu_item", approx_xy, img,
                               label_hint=c["target"])

        # ---- Stage C generic ----
        sc_gen = _run_stage_c("generic", approx_xy, img,
                              label_hint=c["target"])

        # Byte-identical: (hit, x, y) must match between Stage B and Stage C menu_item
        byte_identical = (
            bool(sb["hit"]) == bool(sc_menu["hit"])
            and int(sb["x"]) == int(sc_menu["x"])
            and int(sb["y"]) == int(sc_menu["y"])
        )
        overall_byte_identical &= byte_identical

        # Generic must always miss and preserve approx_xy
        generic_ok = (
            not sc_gen["hit"]
            and sc_gen["x"] == approx_xy[0]
            and sc_gen["y"] == approx_xy[1]
            and sc_gen["method"] == "generic_miss"
        )

        # ---- CV determinism (menu_item) ----
        runs_menu = [
            _run_stage_c("menu_item", approx_xy, img, label_hint=c["target"])
            for _ in range(DETERMINISM_N)
        ]
        stdev_mx, stdev_my = _stdev_xy(runs_menu)
        runs_menu_hits = {int(r["hit"]) for r in runs_menu}
        menu_deterministic = (
            stdev_mx < 0.5 and stdev_my < 0.5 and len(runs_menu_hits) == 1
        )

        # ---- CV determinism (generic) ----
        runs_gen = [
            _run_stage_c("generic", approx_xy, img, label_hint=c["target"])
            for _ in range(DETERMINISM_N)
        ]
        stdev_gx, stdev_gy = _stdev_xy(runs_gen)
        gen_deterministic = (
            stdev_gx < 0.5 and stdev_gy < 0.5
            and all(not r["hit"] for r in runs_gen)
        )

        overall_deterministic &= menu_deterministic and gen_deterministic

        if sc_menu["hit"]:
            stage_c_hits_menu += 1
        if sc_gen["hit"]:
            stage_c_hits_generic += 1

        entry = {
            "test_id": c["test_id"],
            "target": c["target"],
            "approx_xy": list(approx_xy),
            "gt_xy": list(c["gt_xy"]),
            "v7_distance_px": c["v7_distance_px"],
            "stage_b_ref": {
                "hit": bool(sb["hit"]), "x": int(sb["x"]), "y": int(sb["y"]),
                "match": sb.get("match", ""), "time_s": sb_time,
            },
            "stage_c_menu_item": {
                "hit": bool(sc_menu["hit"]),
                "x": int(sc_menu["x"]), "y": int(sc_menu["y"]),
                "method": sc_menu["method"],
                "confidence": float(sc_menu["confidence"]),
                "match": sc_menu.get("match", ""),
                "time_s": float(sc_menu["time"]),
            },
            "stage_c_generic": {
                "hit": bool(sc_gen["hit"]),
                "x": int(sc_gen["x"]), "y": int(sc_gen["y"]),
                "method": sc_gen["method"],
                "confidence": float(sc_gen["confidence"]),
            },
            "byte_identical_menu_vs_stage_b": byte_identical,
            "generic_invariant_ok": generic_ok,
            "determinism_menu_item": {
                "stdev_x": stdev_mx, "stdev_y": stdev_my,
                "pass": menu_deterministic,
            },
            "determinism_generic": {
                "stdev_x": stdev_gx, "stdev_y": stdev_gy,
                "pass": gen_deterministic,
            },
        }
        report["per_case"].append(entry)

        print(f"  stage_b_ref: hit={sb['hit']!s:5s} xy=({sb['x']},{sb['y']}) "
              f"match='{sb.get('match','')}' t={sb_time*1000:.1f}ms")
        print(f"  stage_c_menu: hit={sc_menu['hit']!s:5s} xy=({sc_menu['x']},{sc_menu['y']}) "
              f"method={sc_menu['method']} conf={sc_menu['confidence']:.2f} "
              f"t={sc_menu['time']*1000:.1f}ms")
        print(f"  stage_c_generic: hit={sc_gen['hit']!s:5s} xy=({sc_gen['x']},{sc_gen['y']}) "
              f"method={sc_gen['method']} conf={sc_gen['confidence']:.2f}")
        print(f"  byte_identical (menu vs stage_b): {byte_identical}")
        print(f"  generic_invariant (miss + byte-identical): {generic_ok}")
        print(f"  determinism menu: stdev=({stdev_mx:.3f},{stdev_my:.3f}) "
              f"{'PASS' if menu_deterministic else 'FAIL'}")
        print(f"  determinism generic: stdev=({stdev_gx:.3f},{stdev_gy:.3f}) "
              f"{'PASS' if gen_deterministic else 'FAIL'}")

    report["summary"] = {
        "cases": len(cases),
        "stage_c_menu_item_hits": stage_c_hits_menu,
        "stage_c_generic_hits": stage_c_hits_generic,
        "all_byte_identical_menu_vs_stage_b": overall_byte_identical,
        "all_deterministic": overall_deterministic,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))

    out_path = _ROOT / "benchmark" / "stage_c_commit1_dry_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nreport written: {out_path}")

    return 0 if (overall_byte_identical and overall_deterministic) else 2


if __name__ == "__main__":
    sys.exit(main())
