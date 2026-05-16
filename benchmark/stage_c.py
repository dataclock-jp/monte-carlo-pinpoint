"""
Stage C: Semantic-Geometric Refinement Layer.

Dispatcher that takes a structured target declaration (target_type + intent +
approx_xy + optional label_hint) emitted by the VLM in Phase 1.5 tool-use and
delegates to the appropriate geometric / OCR pipeline. Decision 050.

target_type pipelines:
    generic    — miss-only, byte-identical fallback. Exists so Stage C can be
                 measured in isolation from Phase 1 (paper ablation: any gain
                 over v7 baseline when all targets route through `generic` is
                 attributable to VLM sampling variance, not Stage C).
    menu_item  — retrofit of stage_b.apply_stage_b_snap. Stage B is the
                 sub-case of Stage C for text-label targets.
    button     — Canny + findContours + aspect-ratio/area filter tuned for
                 rectangular toolbar buttons (AR 1.5-6.0, area >= 300 px^2).
                 centroid of the contour nearest approx_xy wins.
    icon      — same contour pipeline as button, different filter window
                 (AR 0.7-1.4, area 100-1500 px^2) so square/compact icons
                 pass and wide rectangles don't.

Reserved for later commits (rect / face / canvas_intent) short-circuit to
the `generic` miss path with a distinct method label so callers can see
they were requested but not yet implemented.

Contract:
    resolve_target(target_type, intent, approx_xy, image, label_hint=None,
                   *, shot_w=None, shot_h=None,
                   phase1_x_ok=True, phase1_y_ok=True) -> dict

Return schema (all paths):
    hit:        bool   — True only when the pipeline positively refined coords
    x, y:       int    — refined coords on hit; otherwise == approx_xy
    method:     str    — one of:
                    "generic_miss"
                    "menu_item"            (hit)
                    "menu_item_miss"       (miss after OCR)
                    "menu_item_no_label"   (label_hint absent)
                    "button"               (hit via button filter)
                    "button_miss"          (no qualifying contour)
                    "icon"                 (hit via icon filter)
                    "icon_miss"            (no qualifying contour)
                    "<target_type>_not_implemented"
    confidence: float  — in [0.0, 1.0]. 0.0 on miss.
    time:       float  — wall-clock seconds.
    match:      str    — OCR-matched label (menu_item hit only); else "".

Confidence thresholds (decision 050, 論点 3):
    >= CONF_SNAP  — full snap (hit)
    >= CONF_BIAS  — Phase 2 bias (reserved for the integration layer in
                    Commit 3; today the contour pipelines still return
                    hit / miss, but the float value carries the gradation)
    <  CONF_BIAS  — miss (byte-identical fallback)
"""
from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional, Tuple

import cv2 as _cv2
import numpy as _np
from PIL import Image as _Image

from benchmark import stage_b as _stage_b

CONF_SNAP = 0.8
CONF_BIAS = 0.5

SUPPORTED_TARGET_TYPES = (
    "generic",
    "menu_item",
    "button",
    "icon",
    "rect",
    "face",
    "canvas_intent",
)

# Search window around approx_xy for contour extraction. Phase A used a
# single 200 px window for both button and icon; Phase B W2 splits by
# variant because dense-toolbar icons (ui_dense regression, v8 -30 pt at
# ≤20 px) were being routed to wrong-neighbor centroids when the
# window swept in several candidate icons at once. The button window is
# unchanged per the Phase A contribution-protection rule.
BUTTON_SEARCH_RADIUS = 200  # unchanged from Phase A (tab:results-v8-category ui_button +10 pt)
ICON_SEARCH_RADIUS = 100    # Phase B W2: halved to cut false-positive neighbours
DEFAULT_SEARCH_RADIUS = BUTTON_SEARCH_RADIUS  # retained for back-compat

# Canny thresholds. Fixed (not auto-tuned) so the pipeline is deterministic
# under the CV-only determinism test; a follow-up could switch to the
# Otsu-based auto threshold once we're sure the real-image benchmark
# doesn't benefit from per-case tuning.
CANNY_LOW = 50
CANNY_HIGH = 150

# Filter windows — tuned from visual inspection of benchmark/images/ui_*.
# button variant: wide, rectangular toolbar buttons.
BUTTON_AR_MIN, BUTTON_AR_MAX = 1.5, 6.0
BUTTON_AREA_MIN = 300
BUTTON_AREA_MAX = 50_000  # upper bound so we don't match whole toolbars.

# icon variant: compact, roughly square icons.
ICON_AR_MIN, ICON_AR_MAX = 0.6, 1.7
ICON_AREA_MIN = 100
ICON_AREA_MAX = 3_000

# Confidence shaping for the contour pipelines. Distance-to-centroid maps
# to the 3-stage confidence fallback in decision 050.
# button variant (Phase A, unchanged): linear decay 15 → 60 px.
BUTTON_CONF_DIST_HIGH_PX = 15.0
BUTTON_CONF_DIST_LOW_PX = 60.0
# icon variant (Phase B W2, tighter): linear decay 10 → 40 px. At v8
# ui_dense_01 / 04, Phase 1 landed on a wrong icon 36 px from the true
# target, and the icon pipeline happily snapped there (hit, 51 / 70 px
# worse). The tighter window forces those cases to miss and fall through
# to Phase 2, which recovers them at v7-baseline accuracy.
ICON_CONF_DIST_HIGH_PX = 10.0
ICON_CONF_DIST_LOW_PX = 40.0

# Ambiguity abstention (Phase B W2): when more than one candidate contour
# survives the per-variant filter and the best two centroid-to-approx
# distances differ by less than this gap, the pipeline abstains (returns
# miss + byte-identical). Dense toolbars are precisely the environment
# where two neighbouring icons look almost equally likely — abstaining is
# safer than guessing.
BUTTON_AMBIGUITY_GAP_PX = 0.0   # 0 = off; button subset showed no false-hit pattern
ICON_AMBIGUITY_GAP_PX = 15.0    # icon abstention threshold

# label_hint adaptive tightening (Phase B W2): when the Phase 1.5
# declaration supplies a concrete label for an icon target, we know the
# VLM *intended* a specific icon rather than "any icon-shaped thing".
# Tighten the confidence window by an extra factor so that ambiguous
# snaps on dense toolbars are penalised further. Applied only to the
# icon variant; button already scores well under its existing window.
ICON_LABELED_TIGHTEN = 0.8  # multiplies ICON_CONF_DIST_{HIGH,LOW}_PX

# Legacy alias — kept because older callers or test fixtures may import
# CONF_DIST_HIGH_PX / CONF_DIST_LOW_PX. These are now the button values
# and should not be reused for icon tuning.
CONF_DIST_HIGH_PX = BUTTON_CONF_DIST_HIGH_PX
CONF_DIST_LOW_PX = BUTTON_CONF_DIST_LOW_PX


def resolve_target(
    target_type: str,
    intent: str,
    approx_xy: Tuple[int, int],
    image: _Image.Image,
    label_hint: Optional[str] = None,
    *,
    shot_w: Optional[int] = None,
    shot_h: Optional[int] = None,
    phase1_x_ok: bool = True,
    phase1_y_ok: bool = True,
) -> Dict[str, Any]:
    """Dispatch to the Stage C pipeline for ``target_type``.

    ``intent`` is accepted for API compatibility with the Phase 1.5 tool-use
    schema (Commit 3) but not yet consumed by Commit 1 pipelines. ``label_hint``
    is required for ``menu_item`` (OCR fuzzy-match target); unused elsewhere.
    """
    t0 = _time.time()
    x0, y0 = int(approx_xy[0]), int(approx_xy[1])
    out: Dict[str, Any] = {
        "hit": False,
        "x": x0,
        "y": y0,
        "method": "",
        "confidence": 0.0,
        "time": 0.0,
        "match": "",
    }

    iw, ih = image.size
    if shot_w is None:
        shot_w = iw
    if shot_h is None:
        shot_h = ih

    if target_type == "menu_item":
        _resolve_menu_item(
            out, image, x0, y0, label_hint,
            int(shot_w), int(shot_h), phase1_x_ok, phase1_y_ok,
        )
    elif target_type in ("button", "icon"):
        _resolve_contour(
            out, image, x0, y0, target_type,
            int(shot_w), int(shot_h),
            label_hint=label_hint,
        )
    elif target_type == "generic":
        _resolve_generic(out, x0, y0)
    elif target_type in SUPPORTED_TARGET_TYPES:
        _resolve_generic(out, x0, y0)
        out["method"] = f"{target_type}_not_implemented"
    else:
        _resolve_generic(out, x0, y0)
        out["method"] = "unknown_target_type"

    out["time"] = _time.time() - t0
    return out


def _resolve_generic(out: Dict[str, Any], x0: int, y0: int) -> None:
    """Miss-only pipeline.

    Returns input coords unchanged so Stage C routed through ``generic`` is
    byte-identical to the v6/v7 baseline without Stage C. This keeps the paper
    ablation interpretable: a non-``generic`` pipeline must demonstrate its
    own improvement, it cannot free-ride on Phase 1 variance.
    """
    out["method"] = "generic_miss"
    out["confidence"] = 0.0
    out["hit"] = False
    out["x"] = x0
    out["y"] = y0


def _resolve_menu_item(
    out: Dict[str, Any],
    image: _Image.Image,
    x0: int,
    y0: int,
    label_hint: Optional[str],
    shot_w: int,
    shot_h: int,
    phase1_x_ok: bool,
    phase1_y_ok: bool,
) -> None:
    """Retrofit Stage B under decision 050 (Stage C ⊃ Stage B)."""
    if not label_hint:
        out["method"] = "menu_item_no_label"
        out["confidence"] = 0.0
        return

    sb = _stage_b.apply_stage_b_snap(
        image, x0, y0, phase1_x_ok, phase1_y_ok, label_hint,
        shot_w, shot_h,
    )
    if sb["hit"]:
        out["hit"] = True
        out["x"] = int(sb["x"])
        out["y"] = int(sb["y"])
        out["method"] = "menu_item"
        # Stage B hit = bbox-containment, treated as high confidence under the
        # 3-tier fallback. Tightening this to an analog score (e.g. match_dist
        # / bbox_diag) is deferred to Commit 3 once real-data distributions are
        # available from the A/B smoke.
        out["confidence"] = 1.0
        out["match"] = sb.get("match", "")
    else:
        out["method"] = "menu_item_miss"
        out["confidence"] = 0.0
        out["x"] = x0
        out["y"] = y0


def _filter_window(variant: str) -> Tuple[float, float, int, int]:
    """Return (ar_min, ar_max, area_min, area_max) for the contour filter."""
    if variant == "button":
        return BUTTON_AR_MIN, BUTTON_AR_MAX, BUTTON_AREA_MIN, BUTTON_AREA_MAX
    if variant == "icon":
        return ICON_AR_MIN, ICON_AR_MAX, ICON_AREA_MIN, ICON_AREA_MAX
    raise ValueError(f"unknown contour variant: {variant}")


def _variant_conf_params(variant: str,
                          label_hint: Optional[str]) -> Tuple[float, float]:
    """Return (high_px, low_px) confidence thresholds for the variant.

    Phase A used a single (15, 60) pair for both button and icon; Phase B
    W2 splits them because the v8 ui_dense regression showed that
    dense-toolbar icons need a tighter window than wide toolbar buttons.
    When a ``label_hint`` is supplied for an icon target, Phase 1.5
    expressed a specific click intent and we tighten further (the
    ambiguity cost of snapping to a wrong icon is higher when the VLM
    named the target).
    """
    if variant == "icon":
        hi, lo = ICON_CONF_DIST_HIGH_PX, ICON_CONF_DIST_LOW_PX
        if label_hint:
            hi *= ICON_LABELED_TIGHTEN
            lo *= ICON_LABELED_TIGHTEN
        return hi, lo
    # button and future contour variants default to the Phase A window.
    return BUTTON_CONF_DIST_HIGH_PX, BUTTON_CONF_DIST_LOW_PX


def _variant_search_radius(variant: str) -> int:
    if variant == "icon":
        return ICON_SEARCH_RADIUS
    return BUTTON_SEARCH_RADIUS


def _variant_ambiguity_gap(variant: str) -> float:
    if variant == "icon":
        return ICON_AMBIGUITY_GAP_PX
    return BUTTON_AMBIGUITY_GAP_PX


def _distance_conf(dist: float,
                    high_px: float = BUTTON_CONF_DIST_HIGH_PX,
                    low_px: float = BUTTON_CONF_DIST_LOW_PX) -> float:
    """Piecewise-linear distance → confidence factor in [0, 1].

    Below ``high_px``: 1.0. Above ``low_px``: 0.0. Linear in between.
    The linearity is intentional (not Gaussian / sigmoid): the thresholds
    map directly to the 3-tier decision 050 fallback, and we want the
    Phase 2 bias region (0.5-0.8) to correspond to a visually intuitive
    distance band.

    ``high_px`` / ``low_px`` are defaulted to the button values for
    back-compat with older callers (tests / dry-run harness); variant
    callers should pass the per-variant values from
    ``_variant_conf_params``.
    """
    if dist <= high_px:
        return 1.0
    if dist >= low_px:
        return 0.0
    return 1.0 - (dist - high_px) / (low_px - high_px)


def _crop_bounds(x0: int, y0: int, shot_w: int, shot_h: int,
                 radius: int) -> Tuple[int, int, int, int]:
    x1 = max(0, x0 - radius)
    y1 = max(0, y0 - radius)
    x2 = min(shot_w, x0 + radius)
    y2 = min(shot_h, y0 + radius)
    return x1, y1, x2, y2


def _find_contours(gray: _np.ndarray) -> List[_np.ndarray]:
    """Canny + findContours. Shared between button/icon variants."""
    edges = _cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    # Close tiny gaps in button outlines so findContours sees a single loop
    # instead of disconnected arcs. 3x3 structuring element is small enough to
    # not merge neighbouring buttons in a dense toolbar.
    kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
    closed = _cv2.morphologyEx(edges, _cv2.MORPH_CLOSE, kernel)
    contours, _ = _cv2.findContours(
        closed, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def _resolve_contour(
    out: Dict[str, Any],
    image: _Image.Image,
    x0: int, y0: int,
    variant: str,
    shot_w: int, shot_h: int,
    label_hint: Optional[str] = None,
) -> None:
    """Button / icon pipeline (Phase B W2: per-variant tuning).

    1. Crop a variant-specific window (button=±200, icon=±100) around
       approx_xy.
    2. Canny + morphological close + findContours (external only).
    3. Reject contours whose area / aspect ratio fall outside the variant
       window (button vs icon; see module-level constants).
    4. Collect all surviving candidates; sort by centroid-to-approx
       distance.
    5. Ambiguity abstention: if the best two candidates are within
       ``_variant_ambiguity_gap`` of each other (icon: 15 px; button:
       disabled), abstain and return miss. Dense toolbars routinely
       present two almost-equally-close icon contours, and a
       coin-flip snap there is worse than Phase 2 fallthrough.
    6. Otherwise apply per-variant distance-to-confidence with extra
       tightening for labeled icon targets; hit iff ``conf >= CONF_BIAS``,
       miss otherwise.
    """
    ar_min, ar_max, area_min, area_max = _filter_window(variant)
    conf_high, conf_low = _variant_conf_params(variant, label_hint)
    search_radius = _variant_search_radius(variant)
    ambig_gap = _variant_ambiguity_gap(variant)

    x1, y1, x2, y2 = _crop_bounds(x0, y0, shot_w, shot_h, search_radius)
    if x2 <= x1 or y2 <= y1:
        out["method"] = f"{variant}_miss"
        return

    rgb = _np.asarray(image.crop((x1, y1, x2, y2)))
    if rgb.ndim == 3:
        gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb

    contours = _find_contours(gray)
    if not contours:
        out["method"] = f"{variant}_miss"
        return

    # Collect all qualifying candidates (distance, cx, cy) rather than
    # keeping only the nearest: the ambiguity check in the next step needs
    # at least the best two.
    candidates: List[Tuple[float, float, float]] = []
    for cnt in contours:
        area = float(_cv2.contourArea(cnt))
        if area < area_min or area > area_max:
            continue
        # minAreaRect returns ((cx, cy), (w, h), angle). Aspect ratio is
        # computed from the oriented rectangle so rotated buttons aren't
        # accidentally excluded.
        (cx_rel, cy_rel), (rw, rh), _angle = _cv2.minAreaRect(cnt)
        if rw <= 0 or rh <= 0:
            continue
        longer, shorter = max(rw, rh), min(rw, rh)
        ar = longer / shorter
        if ar < ar_min or ar > ar_max:
            continue
        cx = x1 + float(cx_rel)
        cy = y1 + float(cy_rel)
        dx = cx - x0
        dy = cy - y0
        d = (dx * dx + dy * dy) ** 0.5
        candidates.append((d, cx, cy))

    if not candidates:
        out["method"] = f"{variant}_miss"
        return

    candidates.sort(key=lambda c: c[0])
    best_dist, best_cx, best_cy = candidates[0]

    # Ambiguity abstention. Only applies when the variant enables it
    # (icon = 15 px; button disabled via 0.0 gap for back-compat with
    # Phase A's validated button behaviour).
    if ambig_gap > 0.0 and len(candidates) >= 2:
        second_dist = candidates[1][0]
        if (second_dist - best_dist) < ambig_gap:
            out["method"] = f"{variant}_miss"
            out["confidence"] = 0.0
            return

    conf = _distance_conf(best_dist, conf_high, conf_low)
    if conf < CONF_BIAS:
        out["method"] = f"{variant}_miss"
        out["confidence"] = float(conf)
        return

    out["hit"] = True
    out["x"] = int(round(best_cx))
    out["y"] = int(round(best_cy))
    out["method"] = variant
    out["confidence"] = float(conf)
