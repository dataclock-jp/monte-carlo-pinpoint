"""pinpoint benchmark runner

4 methods x ground_truth.json test cases.

Methods:
  1. direct   -- VLM direct coordinate estimation
  2. som      -- SoM (OpenCV contour + numbered markers + VLM)
  3. grid     -- Grid binary search only (pinpoint Phase 1)
  4. pinpoint -- Phase 1 + 1.5 + 2 (full algorithm)

Usage:
  python benchmark/run_benchmark.py [--methods direct,som,grid,pinpoint]
                                    [--categories ui_button,natural,...]
                                    [--ids natural_00,ambiguous_02,...]
                                    [--repeat 3]
                                    [--model claude-sonnet-4-20250514]

Output:
  benchmark/results.json
  benchmark/summary.txt
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

# Bundle 2.15: shared Phase 2 primitive lives at repo root so the MCP tool
# (mcp_server.py) and this benchmark harness cannot diverge again.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pinpoint_core import hex_stencil_dots, clip_dots_to_bounds  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BENCHMARK_DIR = Path(__file__).parent
PROJECT_DIR = BENCHMARK_DIR.parent
GT_FILE = BENCHMARK_DIR / "ground_truth.json"
RESULTS_FILE = BENCHMARK_DIR / "results.json"
PARTIAL_FILE = BENCHMARK_DIR / "results.json.partial"
SUMMARY_FILE = BENCHMARK_DIR / "summary.txt"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from grid_core import (
    get_api_key, vlm_call, to_base64,
    grid_search_axis,
    pinpoint_declare_anchor, pinpoint_ask_majority,
    PINPOINT_COLORS, _adaptive_dot_colors,
)
import grid_core as _grid_core

_USE_CLI = False
_USE_STAGE_C = False  # Commit 3: when True, replace Stage B early-return with
                      # Stage C dispatch via heuristic declare_target.
_PHASE1_SAMPLES = 1   # Phase B W1 (decision 056): N samples for Phase 1 majority voting.
                      # N=1 = v8 baseline (single sample). N>=3 triggers per-axis
                      # aggregation (median by default) to reduce VLM variance.
_PHASE1_AGGREGATION = "median"  # aggregation method for N>=2: "median" | "mean".

# Phase C Priority 4 / 4.1 (decisions 062/063 candidate): W3 3-arm smoke
# dispatching.
#   arm A = v9 baseline (heuristic declare_target, Phase 1 N=3 median)
#   arm B = CLI-based LLM-routed structured output, N=1 (Priority 4.1
#           swapped transport from Anthropic SDK to claude CLI subprocess;
#           the `vlm_tool_use` enum value is retained for schema stability
#           but the mechanism is prompt-engineered JSON with post-hoc
#           schema validation, not true Messages-API tool-use)
#   arm C = same transport as B, N=3 samples with plurality-vote aggregation
# --arm A is a syntactic sugar over --use-stage-c --phase1-samples 3
# --phase1-aggregation median (per supervisor Q4-1).
ARM_CHOICES = ("A", "B", "C")
_ARM = "A"
_ARM_C_SAMPLES = 3  # declare_target call repetitions under arm C.


# ============================================================================
# Helpers
# ============================================================================

def _aggregate_centroid(samples, method: str = "median") -> dict:
    """Per-axis aggregation of N centroid samples (decision 056 at Phase 1.5
    level, used by arm C to median-collapse the tool-use N=3 outputs).

    ``samples`` is any iterable of length-2 ``(x, y)`` tuples / lists /
    objects with integer or float coordinates. Returns a dict with keys
    ``x`` / ``y`` / ``stdev_x`` / ``stdev_y``. When ``len(samples) < 2``
    the reported stdevs are ``0.0`` (degenerate case).
    """
    import statistics
    pts = list(samples)
    if not pts:
        return {"x": 0, "y": 0, "stdev_x": 0.0, "stdev_y": 0.0}
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    if method == "median":
        x = int(statistics.median(xs))
        y = int(statistics.median(ys))
    elif method == "mean":
        x = int(round(statistics.mean(xs)))
        y = int(round(statistics.mean(ys)))
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")
    sx = float(statistics.stdev(xs)) if len(xs) >= 2 else 0.0
    sy = float(statistics.stdev(ys)) if len(ys) >= 2 else 0.0
    return {"x": x, "y": y, "stdev_x": sx, "stdev_y": sy}


def _plurality_vote(values: list, default=None):
    """Plurality vote with first-occurrence tiebreak (supervisor Q4-4).

    Hashable values only. Returns ``default`` for empty input.
    """
    if not values:
        return default
    counts: dict = {}
    order: list = []
    for v in values:
        if v not in counts:
            counts[v] = 0
            order.append(v)
        counts[v] += 1
    # Sort by (-count, original_order) to get plurality with first-occurrence
    # tiebreak.
    best = min(order, key=lambda v: (-counts[v], order.index(v)))
    return best


def _aggregate_declare_target_samples(decls: list) -> dict:
    """Aggregate N ``declare_target_tool_use`` samples for arm C.

    Plurality vote on string fields (``target_type`` / ``intent`` /
    ``label_hint``), majority on ``visible`` bool, per supervisor Q4-4.
    The aggregated reasoning is a JSON-encoded list of the 3 raw inputs so
    provenance is preserved for §8.5.5 paper reporting.

    If **any** input sample has ``source == "vlm_tool_use_failed"`` the
    aggregated output is also marked failed (W3 v2 §6.3 Must-fix B: per-case
    full-exclusion, no partial aggregation). The aggregated reasoning
    carries the failing sample's ``parse_fail_<kind>`` tag for triage.
    """
    import json as _json
    if not decls:
        return {
            "target_type": "generic",
            "intent": "center",
            "label_hint": "",
            "source": "vlm_tool_use_failed",
            "reasoning": "parse_fail_empty_aggregation",
            "visible": True,
        }

    failed = [d for d in decls if d.get("source") == "vlm_tool_use_failed"]
    if failed:
        first_failed = failed[0]
        return {
            "target_type": "generic",
            "intent": "center",
            "label_hint": "",
            "source": "vlm_tool_use_failed",
            "reasoning": first_failed.get("reasoning", "parse_fail_unknown"),
            "visible": True,
        }

    tt = _plurality_vote([d["target_type"] for d in decls], default="generic")
    intent = _plurality_vote([d["intent"] for d in decls], default="center")
    label = _plurality_vote([d["label_hint"] for d in decls], default="")
    visible_votes = [bool(d["visible"]) for d in decls]
    visible = sum(visible_votes) >= (len(visible_votes) - sum(visible_votes))
    reasoning = _json.dumps(
        [d.get("reasoning", "") for d in decls],
        ensure_ascii=False, sort_keys=False)
    return {
        "target_type": tt,
        "intent": intent,
        "label_hint": label,
        "source": "vlm_tool_use",
        "reasoning": reasoning,
        "visible": visible,
    }


def _rollup_by_arm(rows, arm: str) -> dict:
    """Per-arm success-rate rollup with W3 v2 §6.3 full-exclusion.

    Filters ``rows`` down to the requested arm, then drops any row whose
    ``stage_c_source`` is ``"vlm_tool_use_failed"`` (or whose legacy
    ``arm_failed`` flag is True) before computing summary statistics.
    The dropped rows are still reported separately as ``n_failed`` so
    ``tool_use_parse_failure_rate`` can be derived at §8.5.5 reporting
    time without re-scanning the raw results.
    """
    arm_rows = [r for r in rows if r.get("arm") == arm]
    failed = [r for r in arm_rows
              if r.get("stage_c_source") == "vlm_tool_use_failed"
              or r.get("arm_failed") is True]
    valid = [r for r in arm_rows if r not in failed]
    hit_count = sum(1 for r in valid if r.get("hit_20px"))
    n_valid = len(valid)
    summary = {
        "arm": arm,
        "n_total": len(arm_rows),
        "n_valid": n_valid,
        "n_failed": len(failed),
        "hit_count_20px": hit_count,
        "hit_rate_20px": (hit_count / n_valid) if n_valid else 0.0,
        "parse_failure_rate": (
            len(failed) / len(arm_rows)) if arm_rows else 0.0,
    }
    return summary


def _declare_target_for_arm(target: str, arm: str,
                             api_key: str, model: str) -> dict:
    """Dispatch to the correct declare_target variant for the active arm.

    arm A → heuristic (no VLM call, deterministic, no subscription cost)
    arm B → tool-use N=1 (single Anthropic API call with retry)
    arm C → tool-use N=3 with plurality + majority aggregation

    Return shape is the existing 6-key declare_target dict so the caller in
    ``method_pinpoint`` does not need to branch on arm — schema is uniform.
    """
    if arm == "A":
        from stage_c_phase_1_5 import declare_target as _decl
        return _decl(target)
    if arm == "B":
        from stage_c_phase_1_5 import declare_target_tool_use as _decl_tu
        return _decl_tu(target, api_key=api_key, model=model)
    if arm == "C":
        from stage_c_phase_1_5 import declare_target_tool_use as _decl_tu
        decls = [_decl_tu(target, api_key=api_key, model=model)
                 for _ in range(_ARM_C_SAMPLES)]
        return _aggregate_declare_target_samples(decls)
    raise ValueError(f"unknown arm: {arm!r}")


def _aggregate_axis(samples: list, method: str = "median"):
    """Aggregate per-axis Phase 1 grid samples with N-sample majority voting.

    samples: list of int coordinates from grid_search_axis (raw, before scale
    map-back). Negative values are treated as failures and excluded from
    aggregation. Returns (agg_value, stdev) where stdev is over the valid
    samples only (0.0 if <2 valid).

    If all samples are invalid (all < 0), returns (-1, 0.0) to preserve the
    "grid failed" signal that the caller uses for direct-VLM fallback.
    """
    import statistics
    valid = [s for s in samples if s >= 0]
    if not valid:
        return (-1, 0.0)
    if method == "median":
        agg = int(statistics.median(valid))
    elif method == "mean":
        agg = int(round(statistics.mean(valid)))
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")
    stdev = statistics.stdev(valid) if len(valid) >= 2 else 0.0
    return (agg, stdev)


def _upscale_for_grid(pil_img: PILImage.Image, target_width: int = 2560) -> PILImage.Image:
    """Upscale image for better VLM grid line detection.
    Lines on a 1920px image are too close together for VLM to distinguish.
    Upscaling to 2560px increases spacing by ~33% without excessive token cost."""
    w, h = pil_img.size
    if w >= target_width:
        return pil_img
    scale = target_width / w
    return pil_img.resize((target_width, int(h * scale)), PILImage.Resampling.BICUBIC)


# ============================================================================
# Method 1: Direct Coordinate Estimation
# ============================================================================

def method_direct(pil_img: PILImage.Image, target: str,
                  api_key: str, model: str) -> dict:
    t0 = time.time()
    w, h = pil_img.size
    b64 = to_base64(pil_img)
    system = (
        "You are a precise visual coordinate estimator. "
        f"The image is {w}x{h} pixels. "
        "Given a target description, output the exact pixel coordinate. "
        "Reply with ONLY: x,y (two integers separated by comma). "
        "No explanation."
    )
    user = f"Target: {target}\nCoordinate (x,y):"
    text = vlm_call(b64, system, user, api_key, model, max_tokens=20)
    elapsed = time.time() - t0

    x, y = None, None
    if text:
        m = re.match(r"(\d+)\s*,\s*(\d+)", text)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
    return {"method": "direct", "x": x, "y": y, "queries": 1,
            "time_s": round(elapsed, 2)}


# ============================================================================
# Method 2: SoM (Set-of-Mark)
# ============================================================================

def method_som(pil_img: PILImage.Image, target: str,
               api_key: str, model: str) -> dict:
    t0 = time.time()
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if 15 <= bw <= 300 and 15 <= bh <= 300:
            elements.append({"x": bx + bw // 2, "y": by + bh // 2})

    if not elements:
        return {"method": "som", "x": None, "y": None, "queries": 0,
                "time_s": round(time.time() - t0, 2),
                "error": "no_elements_detected"}

    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    elements = elements[:50]
    for i, el in enumerate(elements):
        cx, cy = el["x"], el["y"]
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8],
                     fill=(255, 50, 50), outline=(255, 255, 255), width=1)
        draw.text((cx + 10, cy - 7), str(i + 1), fill=(255, 50, 50), font=font,
                  stroke_fill=(255, 255, 255), stroke_width=2)

    b64 = to_base64(annotated)
    system = (
        f"The image has {len(elements)} numbered red markers. "
        "Given a target description, identify which marker number is closest. "
        "Reply with ONLY the marker number (integer). If none match, reply NONE."
    )
    text = vlm_call(b64, system, f"Target: {target}\nMarker number:",
                    api_key, model, max_tokens=10)
    elapsed = time.time() - t0

    x, y = None, None
    if text and "NONE" not in text.upper():
        m = re.match(r"(\d+)", text.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(elements):
                x, y = elements[idx]["x"], elements[idx]["y"]

    return {"method": "som", "x": x, "y": y, "queries": 1,
            "time_s": round(elapsed, 2),
            "elements_detected": len(elements)}


# ============================================================================
# Method 3: Grid Find (Phase 1 only)
# ============================================================================

def method_grid(pil_img: PILImage.Image, target: str,
                api_key: str, model: str,
                num_lines: int = 12, precision: int = 20) -> dict:
    t0 = time.time()
    qc = [0]
    # Upscale small images for better VLM line detection
    work_img = _upscale_for_grid(pil_img)
    scale = work_img.width / pil_img.width
    x, _ = grid_search_axis(work_img, target, "x", num_lines, int(precision * scale),
                             api_key, model, query_counter=qc)
    y, _ = grid_search_axis(work_img, target, "y", num_lines, int(precision * scale),
                             api_key, model, query_counter=qc)
    # Map back to original coordinates
    if x >= 0:
        x = int(x / scale)
    if y >= 0:
        y = int(y / scale)
    return {"method": "grid",
            "x": x if x >= 0 else None,
            "y": y if y >= 0 else None,
            "queries": qc[0],
            "time_s": round(time.time() - t0, 2)}


# ============================================================================
# Method 4: Monte Carlo Pinpoint
# ============================================================================

def method_pinpoint(pil_img: PILImage.Image, target: str,
                    api_key: str, model: str,
                    num_dots: int = 7, max_rounds: int = 8,
                    grid_precision: int = 20, votes: int = 3) -> dict:
    t0 = time.time()
    qc = [0]
    w, h = pil_img.size

    # Phase 1: Grid search on upscaled image.
    # Phase B W1 (decision 056): when _PHASE1_SAMPLES >= 2, call grid_search_axis
    # N times per axis and aggregate (median by default) to reduce VLM variance.
    # N=1 path is bit-identical to v8 baseline.
    work_img = _upscale_for_grid(pil_img)
    scale = work_img.width / w
    N = max(1, int(_PHASE1_SAMPLES))
    x_samples_raw: list[int] = []
    y_samples_raw: list[int] = []
    for _sample_i in range(N):
        xs_i, _ = grid_search_axis(work_img, target, "x", 12, int(grid_precision * scale),
                                    api_key, model, query_counter=qc)
        ys_i, _ = grid_search_axis(work_img, target, "y", 12, int(grid_precision * scale),
                                    api_key, model, query_counter=qc)
        x_samples_raw.append(int(xs_i))
        y_samples_raw.append(int(ys_i))
    if N == 1:
        x_shot_raw, y_shot_raw = x_samples_raw[0], y_samples_raw[0]
        x_stdev_raw, y_stdev_raw = 0.0, 0.0
    else:
        x_shot_raw, x_stdev_raw = _aggregate_axis(x_samples_raw, _PHASE1_AGGREGATION)
        y_shot_raw, y_stdev_raw = _aggregate_axis(y_samples_raw, _PHASE1_AGGREGATION)
    # Map back to original coordinates
    phase1_ok_x = x_shot_raw >= 0
    phase1_ok_y = y_shot_raw >= 0
    x_shot = int(x_shot_raw / scale) if phase1_ok_x else x_shot_raw
    y_shot = int(y_shot_raw / scale) if phase1_ok_y else y_shot_raw
    # Map raw per-sample coords to original image space for results provenance.
    phase_1_samples = [
        [int(xi / scale) if xi >= 0 else None,
         int(yi / scale) if yi >= 0 else None]
        for xi, yi in zip(x_samples_raw, y_samples_raw)
    ]
    # Stdev in original coords (scale-invariant for log: divide by scale).
    phase_1_sample_stdev_xy = [
        round(x_stdev_raw / scale, 3),
        round(y_stdev_raw / scale, 3),
    ]
    phase_1_aggregation = _PHASE1_AGGREGATION if N >= 2 else "none"

    # Fallback: if grid failed on either axis, use direct VLM estimation
    if not phase1_ok_x or not phase1_ok_y:
        direct_result = method_direct(pil_img, target, api_key, model)
        qc[0] += 1
        if not phase1_ok_x:
            x_shot = direct_result["x"] if direct_result["x"] is not None else w // 2
        if not phase1_ok_y:
            y_shot = direct_result["y"] if direct_result["y"] is not None else h // 2

    phase1_queries = qc[0]
    phase1_time = time.time() - t0

    # ------------------------------------------------------------------
    # Stage B (default) / Stage C (--use-stage-c) refinement layer
    # ------------------------------------------------------------------
    # When --use-stage-c is off, the v7 Stage B OCR-snap runs unchanged
    # (byte-identical to that baseline). When on, the equivalent hook
    # point dispatches through stage_c.resolve_target based on a
    # heuristic Phase 1.5 target declaration; menu_item requests still
    # end up in stage_b.apply_stage_b_snap via the Stage C retrofit, so
    # the A/B comparison is apples-to-apples on text-label targets and
    # adds the button/icon pipelines on top.
    stage_c_hit = False
    stage_c_method = ""
    stage_c_confidence = 0.0
    stage_c_time = 0.0
    stage_c_input_xy = [int(x_shot), int(y_shot)]
    stage_c_output_xy = [int(x_shot), int(y_shot)]
    stage_c_source = ""
    stage_c_reasoning = ""
    phase_1_5_structured: dict = {}

    if _USE_STAGE_C:
        from stage_c import resolve_target as _resolve_target
        # Decision 052 9-field canonical: source + reasoning live at the top
        # level alongside the other stage_c_* fields. phase_1_5_structured
        # keeps only the pure declaration payload (target_type / intent /
        # label_hint / visible) so future tool-use variants can emit the same
        # dict shape verbatim.
        #
        # Phase C Priority 4: _declare_target_for_arm dispatches to the
        # heuristic (arm A) or tool-use (arm B / C) variant based on the
        # module-level _ARM flag. Arm C additionally runs declare_target N=3
        # times with plurality-vote aggregation on string fields.
        _decl = _declare_target_for_arm(target, _ARM, api_key, model)
        stage_c_source = _decl["source"]
        stage_c_reasoning = _decl["reasoning"]
        phase_1_5_structured = {
            k: _decl[k] for k in ("target_type", "intent", "label_hint", "visible")
        }
        _sc = _resolve_target(
            _decl["target_type"],
            _decl["intent"],
            (int(x_shot), int(y_shot)),
            pil_img,
            label_hint=_decl.get("label_hint") or target,
            shot_w=w, shot_h=h,
            phase1_x_ok=phase1_ok_x, phase1_y_ok=phase1_ok_y,
        )
        stage_c_hit = bool(_sc["hit"])
        stage_c_method = _sc["method"]
        stage_c_confidence = float(_sc["confidence"])
        stage_c_time = float(_sc["time"])
        stage_c_output_xy = [int(_sc["x"]), int(_sc["y"])]
        # Stage B fields are still reported but derived from the Stage C
        # menu_item sub-case (if that pipeline fired) so downstream
        # consumers see a consistent schema.
        stage_b_hit = stage_c_hit and stage_c_method == "menu_item"
        stage_b_match = _sc.get("match", "")
        stage_b_dist = 0.0
        stage_b_time = stage_c_time
    else:
        from stage_b import apply_stage_b_snap as _apply_stage_b_snap
        _sb = _apply_stage_b_snap(
            pil_img, x_shot, y_shot, phase1_ok_x, phase1_ok_y,
            target, w, h,
        )
        stage_b_hit = bool(_sb["hit"])
        stage_b_match = _sb["match"]
        stage_b_dist = float(_sb["dist"])
        stage_b_time = float(_sb["time"])

    # Hit → early return, skip Phase 1.5 / Phase 2 entirely (VLM-free path).
    early_hit = stage_c_hit if _USE_STAGE_C else stage_b_hit
    if early_hit:
        hit_x = stage_c_output_xy[0] if _USE_STAGE_C else int(_sb["x"])
        hit_y = stage_c_output_xy[1] if _USE_STAGE_C else int(_sb["y"])
        return {
            "method": "pinpoint",
            "x": hit_x, "y": hit_y,
            "queries": qc[0],
            "time_s": round(time.time() - t0, 2),
            "phase1_queries": phase1_queries,
            "phase2_rounds": 0,
            "phase2_hit": False,
            "absent_fired": 0,
            "anchor": "",
            "phase1_time": round(phase1_time, 3),
            "phase_1_5_time": 0.0,
            "phase2_time": 0.0,
            "stage_b_hit": stage_b_hit,
            "stage_b_time": round(stage_b_time, 3),
            "stage_b_match": stage_b_match,
            "stage_b_dist": round(stage_b_dist, 2),
            "stage_c_hit": stage_c_hit,
            "stage_c_method": stage_c_method,
            "stage_c_confidence": round(stage_c_confidence, 3),
            "stage_c_time": round(stage_c_time, 3),
            "stage_c_input_xy": stage_c_input_xy,
            "stage_c_output_xy": stage_c_output_xy,
            "stage_c_source": stage_c_source,
            "stage_c_reasoning": stage_c_reasoning,
            "phase_1_5_structured": phase_1_5_structured,
            "phase_1_samples": phase_1_samples,
            "phase_1_sample_stdev_xy": phase_1_sample_stdev_xy,
            "phase_1_aggregation": phase_1_aggregation,
        }
    # Miss → fall through to Phase 1.5 + Phase 2, byte-identical to v6.

    # Phase 1.5: Anchor declaration
    phase_1_5_t0 = time.time()
    # sr=100px covers 68% of Phase 1 errors (median 61px), while still
    # converging to sub-pixel in 8 rounds (100/2^8 = 0.39px)
    sr = 100
    anchor_radius = sr * 2 if (phase1_ok_x and phase1_ok_y) else sr * 3
    # P1: Claude Sonnet silently downsamples 1:1 images above ~1092x1092.
    # 1024 is the no-loss cap — previous 1200 was losing ~17% effective resolution.
    ZOOM = 1024
    cx1 = max(0, x_shot - anchor_radius); cy1 = max(0, y_shot - anchor_radius)
    cx2 = min(w, x_shot + anchor_radius); cy2 = min(h, y_shot + anchor_radius)
    cw, ch_ = cx2 - cx1, cy2 - cy1

    anchor_crop = pil_img.crop((cx1, cy1, cx2, cy2))
    sc = min(ZOOM / max(cw, 1), ZOOM / max(ch_, 1))
    anchor_zoomed = anchor_crop.resize(
        (max(1, int(cw * sc)), max(1, int(ch_ * sc))), PILImage.Resampling.BICUBIC)
    anchor_b64 = to_base64(anchor_zoomed)
    qc[0] += 1
    anchor, visible = pinpoint_declare_anchor(
        anchor_b64, target, api_key, model)

    # P2 (GR-0): If the VLM reports ABSENT, Phase-1 probably landed in the
    # wrong neighborhood. Retry anchor declaration at progressively wider
    # scopes to catch Mode B before it corrupts every subsequent round.
    absent_fired = 0
    if not visible:
        absent_fired = 1  # initial crop was ABSENT
        # Retry 1: ~1/3 screen crop
        wide_r = min(w, h) // 3
        cx1w = max(0, x_shot - wide_r); cy1w = max(0, y_shot - wide_r)
        cx2w = min(w, x_shot + wide_r); cy2w = min(h, y_shot + wide_r)
        cww, chw = cx2w - cx1w, cy2w - cy1w
        wide_crop = pil_img.crop((cx1w, cy1w, cx2w, cy2w))
        sc_w = min(ZOOM / max(cww, 1), ZOOM / max(chw, 1))
        wide_zoomed = wide_crop.resize(
            (max(1, int(cww * sc_w)), max(1, int(chw * sc_w))),
            PILImage.Resampling.BICUBIC)
        qc[0] += 1
        anchor2, visible2 = pinpoint_declare_anchor(
            to_base64(wide_zoomed), target, api_key, model)
        if visible2:
            anchor = anchor2
        else:
            absent_fired = 2  # second scope also ABSENT
            # Retry 2: full image (last-chance Mode B recovery)
            full_sc = min(ZOOM / max(w, 1), ZOOM / max(h, 1))
            full_zoomed = pil_img.resize(
                (max(1, int(w * full_sc)), max(1, int(h * full_sc))),
                PILImage.Resampling.BICUBIC)
            qc[0] += 1
            anchor3, visible3 = pinpoint_declare_anchor(
                to_base64(full_zoomed), target, api_key, model)
            if visible3:
                anchor = anchor3
            else:
                absent_fired = 3  # ABSENT at all scales -> Phase-1 likely wrong
                anchor = anchor2 or anchor

    phase_1_5_time = time.time() - phase_1_5_t0

    # Phase 2: hex stencil + closest-dot convergence (Bundle 2.15 / C-lite)
    # Q-MISS-A2 LOCK (cycle 1 v3): HIT recenters + halves; MISS retries
    # with same (center, sr), capped at max-3 consecutive retries before
    # break to Phase 1 fallback. Halving is HIT-only. The outer loop
    # iterates max_rounds * 4 batches and breaks when hit_round_count
    # reaches max_rounds, mirroring mcp_server.py so the two paths track
    # identically on (center, sr, miss_retry_count, hit_round_count).
    phase2_t0 = time.time()
    center_x, center_y = x_shot, y_shot
    nd = min(num_dots, 7)
    phase2_hit = False
    rounds_done = 0  # query batches issued (incl. MISS retries)
    hit_round_count = 0  # accepted HIT rounds (Theorem 1 round counter)
    miss_retry_count = 0

    # Widen search radius when Phase 1 had lower confidence
    if not phase1_ok_x or not phase1_ok_y:
        sr = sr * 2  # wider search for fallback cases (200px)

    # Adaptive dot colors based on background
    bg_crop = pil_img.crop((cx1, cy1, cx2, cy2))
    dot_colors = _adaptive_dot_colors(bg_crop, nd)

    for round_idx in range(max_rounds * 4):
        if hit_round_count >= max_rounds:
            break  # HIT budget exhausted (matches mcp_server.py)
        # Current search bounds
        dx1 = max(0, center_x - sr); dy1 = max(0, center_y - sr)
        dx2 = min(w, center_x + sr); dy2 = min(h, center_y + sr)
        if dx2 - dx1 < 3 or dy2 - dy1 < 3:
            break  # Converged to a point

        # Context for zoom (3x search radius)
        cx1 = max(0, center_x - sr * 3); cy1 = max(0, center_y - sr * 3)
        cx2 = min(w, center_x + sr * 3); cy2 = min(h, center_y + sr * 3)
        cw, ch_ = cx2 - cx1, cy2 - cy1

        ctx_crop = pil_img.crop((cx1, cy1, cx2, cy2))
        sc = min(ZOOM / max(cw, 1), ZOOM / max(ch_, 1))
        zw = max(1, int(cw * sc)); zh = max(1, int(ch_ * sc))
        zoomed = ctx_crop.resize((zw, zh), PILImage.Resampling.BICUBIC)

        # Place hex stencil dots in search area (deterministic 7-point
        # half-net, Bundle 2.15 / Mode T-2 C-lite). The stencil forms an
        # s/2-cover of the disk of radius sr around (center_x, center_y);
        # under perfect closest-dot selection, the picked dot is within
        # sr/2 of the target, so halving sr each round preserves
        # containment inductively (Theorem 1).
        all_dots = clip_dots_to_bounds(
            hex_stencil_dots(center_x, center_y, sr),
            dx1, dy1, dx2, dy2,
        )
        dots = all_dots[:nd]
        if not dots:
            break

        # Draw dots on zoomed image
        ann = zoomed.copy()
        draw = ImageDraw.Draw(ann)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for i, (sx, sy) in enumerate(dots):
            zx = int((sx - cx1) * sc); zy = int((sy - cy1) * sc)
            color = dot_colors[i % len(dot_colors)]
            # Larger dots for better visibility (radius 4 in zoomed space)
            draw.ellipse([(zx - 4, zy - 4), (zx + 4, zy + 4)],
                         fill=color, outline=(255, 255, 255), width=1)
            draw.text((zx + 7, zy - 10), str(i + 1), fill=color, font=font,
                      stroke_fill=(0, 0, 0), stroke_width=2)

        b64 = to_base64(ann)
        hit, unanimous = pinpoint_ask_majority(
            b64, anchor, len(dots), api_key, model, votes, query_counter=qc)

        rounds_done = round_idx + 1

        if hit is not None and 1 <= hit <= len(dots):
            # Q-MISS-A2 LOCK (Bundle 2.15 cycle 1 v3): HIT recenters and
            # halves; advances the HIT round counter; resets MISS retry
            # counter so subsequent MISSes are measured against the new
            # HIT round.
            center_x, center_y = dots[hit - 1]
            phase2_hit = True
            sr = max(2, sr // 2)
            hit_round_count += 1
            miss_retry_count = 0
        else:
            # Q-MISS-A2 LOCK: MISS retries with same (center, sr);
            # halving is HIT-only. The retry counter caps consecutive
            # MISSes at 3 (Theorem 1 anti-loop bound) and then breaks
            # to Phase 1 fallback. Avoids the boundary-case containment
            # failure that unconditional MISS-halving introduced.
            miss_retry_count += 1
            if miss_retry_count > 3:
                break  # Phase 1 fallback

    phase2_time = time.time() - phase2_t0
    found_x, found_y = center_x, center_y

    return {"method": "pinpoint", "x": found_x, "y": found_y,
            "queries": qc[0], "time_s": round(time.time() - t0, 2),
            "phase1_queries": phase1_queries,
            "phase2_rounds": rounds_done,
            "phase2_hit_rounds": hit_round_count,
            "phase2_hit": phase2_hit,
            "absent_fired": absent_fired,
            "anchor": anchor[:80],
            "phase1_time": round(phase1_time, 3),
            "phase_1_5_time": round(phase_1_5_time, 3),
            "phase2_time": round(phase2_time, 3),
            "stage_b_hit": False,
            "stage_b_time": round(stage_b_time, 3),
            "stage_b_match": "",
            "stage_b_dist": 0.0,
            "stage_c_hit": stage_c_hit,
            "stage_c_method": stage_c_method,
            "stage_c_confidence": round(stage_c_confidence, 3),
            "stage_c_time": round(stage_c_time, 3),
            "stage_c_input_xy": stage_c_input_xy,
            "stage_c_output_xy": stage_c_output_xy,
            "stage_c_source": stage_c_source,
            "stage_c_reasoning": stage_c_reasoning,
            "phase_1_5_structured": phase_1_5_structured,
            "phase_1_samples": phase_1_samples,
            "phase_1_sample_stdev_xy": phase_1_sample_stdev_xy,
            "phase_1_aggregation": phase_1_aggregation}


# ============================================================================
# Runner
# ============================================================================

METHODS = {
    "direct": method_direct,
    "som": method_som,
    "grid": method_grid,
    "pinpoint": method_pinpoint,
}


def _load_partial(path: Path) -> tuple[list, set]:
    """Read JSONL partial results and return (rows, done_keys).

    done_keys is a set of (test_id, method, repeat) tuples already completed.
    Corrupt trailing lines (from an interrupt mid-write) are silently dropped.
    """
    rows = []
    done = set()
    if not path.exists():
        return rows, done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
            done.add((r.get("test_id"), r.get("method"), r.get("repeat", 0)))
    return rows, done


def _append_partial(path: Path, result: dict) -> None:
    """Append a single result as one JSONL line and flush to disk."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False))
        f.write("\n")
        f.flush()


def run_benchmark(methods, categories, ids, repeat, model, resume=False):
    if _USE_CLI:
        api_key = "CLI_MODE"
        print("Using claude CLI (subscription) for VLM calls")
    else:
        api_key = get_api_key()
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not found", file=sys.stderr)
            sys.exit(1)

    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    if ids:
        gt = [g for g in gt if g["id"] in ids]
    elif categories:
        gt = [g for g in gt if g["category"] in categories]

    if resume:
        all_results, done_keys = _load_partial(PARTIAL_FILE)
        print(f"Resume: loaded {len(all_results)} completed rows from {PARTIAL_FILE.name}")
    else:
        if PARTIAL_FILE.exists():
            PARTIAL_FILE.unlink()
        all_results = []
        done_keys = set()

    total = len(gt) * len(methods) * repeat
    remaining = total - len(done_keys)
    print(f"Benchmark: {len(gt)} cases x {len(methods)} methods x {repeat} repeats "
          f"= {total} runs ({remaining} to do)")
    print(f"Model: {model}")
    print("=" * 70)

    for g in gt:
        img_path = BENCHMARK_DIR / g["image"]
        pil_img = None  # lazy-load only when we actually need to run a method
        gt_x, gt_y = g["x"], g["y"]
        header_shown = False

        for mname in methods:
            fn = METHODS[mname]
            for rep in range(repeat):
                key = (g["id"], mname, rep)
                if key in done_keys:
                    continue
                if not header_shown:
                    print(f"\n[{g['id']}] target='{g['target'][:50]}' gt=({gt_x},{gt_y})")
                    header_shown = True
                if pil_img is None:
                    pil_img = PILImage.open(img_path).convert("RGB")

                try:
                    result = fn(pil_img, g["target"], api_key, model)
                except Exception as e:
                    result = {"method": mname, "x": None, "y": None,
                              "queries": 0, "time_s": 0, "error": str(e)}

                dist = (math.hypot(result["x"] - gt_x, result["y"] - gt_y)
                        if result["x"] is not None and result["y"] is not None
                        else None)
                result.update({
                    "test_id": g["id"], "category": g["category"],
                    "gt_x": gt_x, "gt_y": gt_y,
                    "distance_px": round(dist, 1) if dist is not None else None,
                    "repeat": rep,
                })
                all_results.append(result)
                done_keys.add(key)
                _append_partial(PARTIAL_FILE, result)

                ds = f"{dist:.1f}px" if dist is not None else "FAIL"
                print(f"  {mname:10s} rep{rep}: ({result['x']},{result['y']}) "
                      f"dist={ds} Q={result['queries']} t={result['time_s']}s")

    RESULTS_FILE.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\nResults: {RESULTS_FILE}")
    _summary(all_results, methods)
    return all_results


def _summary(results, methods):
    cats = sorted(set(r["category"] for r in results))
    lines = ["=" * 80, "BENCHMARK SUMMARY", "=" * 80, ""]

    hdr = f"{'Category':<15s}"
    for m in methods:
        hdr += f" | {m:>10s} dist  {m:>6s} Q"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for cat in cats:
        row = f"{cat:<15s}"
        for m in methods:
            cr = [r for r in results if r["category"] == cat
                  and r["method"] == m and r["distance_px"] is not None]
            if cr:
                ds = [r["distance_px"] for r in cr]
                qs = [r["queries"] for r in cr]
                row += f" | {sum(ds)/len(ds):>8.1f}px  {sum(qs)/len(qs):>6.1f}"
            else:
                row += f" |     FAIL          --"
        lines.append(row)

    lines.extend(["", "SUCCESS RATE", "-" * 50])
    for th in [5, 10, 20, 50]:
        row = f"  <={th:>3d}px:"
        for m in methods:
            v = [r for r in results if r["method"] == m and r["distance_px"] is not None]
            if v:
                s = sum(1 for r in v if r["distance_px"] <= th)
                row += f"  {m}={s/len(v)*100:>5.1f}%"
        lines.append(row)

    # Stage B hit rate (pinpoint only)
    pinpoint_rows = [r for r in results if r["method"] == "pinpoint"
                     and "stage_b_hit" in r]
    if pinpoint_rows:
        hits = sum(1 for r in pinpoint_rows if r.get("stage_b_hit"))
        total = len(pinpoint_rows)
        lines.extend([
            "",
            "STAGE B",
            "-" * 50,
            f"  hit rate: {hits}/{total} = {hits/total*100:.1f}%",
        ])
        if hits:
            dists = [r["stage_b_dist"] for r in pinpoint_rows if r.get("stage_b_hit")]
            lines.append(
                f"  snap dist (hit): "
                f"mean={sum(dists)/len(dists):.1f}px  "
                f"max={max(dists):.1f}px"
            )

    text = "\n".join(lines)
    SUMMARY_FILE.write_text(text, encoding="utf-8")
    print(f"\n{text}")
    print(f"\nSummary: {SUMMARY_FILE}")


def main():
    global _USE_CLI, _USE_STAGE_C, _PHASE1_SAMPLES, _PHASE1_AGGREGATION, _ARM
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="direct,som,grid,pinpoint")
    p.add_argument("--categories", default=None)
    p.add_argument("--ids", default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--model", default="claude-sonnet-4-20250514")
    p.add_argument("--use-cli", action="store_true",
                   help="Use claude CLI (subscription) instead of API credits")
    p.add_argument("--use-stage-c", action="store_true",
                   help="Route Phase 1 output through Stage C "
                        "(heuristic declare_target + stage_c dispatcher) "
                        "instead of Stage B; decision 050")
    p.add_argument("--phase1-samples", type=int, default=1,
                   help="Phase 1 majority voting sample count (decision 056). "
                        "N=1 = v8 baseline (default). N>=3 calls grid_search_axis "
                        "N times per axis and aggregates to reduce VLM variance.")
    p.add_argument("--phase1-aggregation", default="median",
                   choices=["median", "mean"],
                   help="Aggregation method for --phase1-samples N>=2 "
                        "(default: median, robust to outliers).")
    p.add_argument("--arm", default=None, choices=list(ARM_CHOICES),
                   help="Phase C W3 3-arm dispatch (decisions 062/063 "
                        "candidate). A = v9 baseline syntactic sugar (forces "
                        "--use-stage-c --phase1-samples 3 "
                        "--phase1-aggregation median, heuristic declare_target). "
                        "B = CLI-based LLM-routed structured output, N=1. C = "
                        "same transport, N=3 with plurality/median "
                        "aggregation. --arm B/C auto-set --use-stage-c; the "
                        "claude CLI subprocess inherits OAuth subscription "
                        "auth (decision 063 candidate supersedes decision 062 "
                        "$2 API cap). Omit the flag to preserve pre-W3 "
                        "behaviour (driven by --use-stage-c / --phase1-samples "
                        "individually; arm defaults to \"A\" at the dispatch "
                        "layer so declare_target calls remain heuristic).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from results.json.partial, skipping completed (test_id, method, repeat) tuples")
    a = p.parse_args()

    if a.use_cli:
        _USE_CLI = True
        _grid_core._USE_CLI = True
    if a.use_stage_c:
        _USE_STAGE_C = True
    _PHASE1_SAMPLES = max(1, int(a.phase1_samples))
    _PHASE1_AGGREGATION = a.phase1_aggregation

    # Phase C Priority 4 / 4.1 arm dispatching (decisions 062/063 candidate).
    # --arm explicit on the CLI → syntactic sugar applies:
    #   A forces --use-stage-c --phase1-samples 3 --phase1-aggregation median
    #     (the v9 baseline) so `python run_benchmark.py --arm A` is a
    #     one-flag reproduction of the v9 Stage 2 full configuration.
    #   B/C auto-set --use-stage-c. Priority 4.1 moved the arm B/C transport
    #     to the claude CLI subprocess (decision 063 candidate), so no API
    #     key is required; the existing OAuth subscription covers all three
    #     arms and `--use-cli` on the Phase 1 grid path is orthogonal.
    # --arm omitted → _ARM stays at its module default of "A" so the
    # declare_target dispatcher still picks the heuristic path, but no
    # other flags are forced (backward-compat with pre-W3 invocations).
    if a.arm is not None:
        _ARM = a.arm
        if _ARM in ("B", "C"):
            _USE_STAGE_C = True
        elif _ARM == "A":
            _USE_STAGE_C = True
            if _PHASE1_SAMPLES < 3:
                _PHASE1_SAMPLES = 3

    methods = [m.strip() for m in a.methods.split(",")]
    cats = [c.strip() for c in a.categories.split(",")] if a.categories else None
    ids = [i.strip() for i in a.ids.split(",")] if a.ids else None
    run_benchmark(methods, cats, ids, a.repeat, a.model, resume=a.resume)


if __name__ == "__main__":
    main()
