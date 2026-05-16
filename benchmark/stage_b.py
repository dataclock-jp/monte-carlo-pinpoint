"""
Stage B: Post-Phase 1 OCR-snap for pinpoint refinement.

Shared helper between mcp_server.py (MCP tool path) and benchmark/run_benchmark.py
(benchmark pipeline). Single source of truth per decision 049.

Contract: when Phase 1 lands within ±radius_px of target text, run Windows OCR
on the crop, fuzzy-match the target against word bboxes, and snap to the closest
matching bbox center. On hit the caller may skip Phase 1.5/2/3. On miss the
returned (x, y) equals the input (x_shot, y_shot) so downstream phases run
unchanged (byte-identical miss path).
"""
import re as _re
import time as _time
from typing import Any, Dict

from PIL import Image as _Image

# Snap gate: symmetric ±20px. Task 1.5 (a) analysis shows 5/5 P1 cases
# (max |dx|=14, max |dy|=16) fit within this radius. v7 may widen to ±25px
# as Stage B′.
DEFAULT_STAGE_B_RADIUS = 20

# OCR input crop: ±60px = 120×120 window. Live test showed WinRT OCR returns
# garbage on a 40×40 crop of toolbar-size (20pt) text but detects cleanly at
# 80×80 and 120×120; 60 gives comfortable margin without excessive OCR cost.
# Kept separate from the snap gate so Task 1.5 (a)'s ±20px symmetric radius
# remains the authoritative match-distance constraint.
DEFAULT_OCR_RADIUS = 60

# Quoted-label patterns used to extract the button/icon label from a
# natural-language target prompt. Order matters: most specific first.
_LABEL_PATTERNS = (
    _re.compile(r"'([^']+)'"),            # 'Import'
    _re.compile(r'"([^"]+)"'),            # "Save"
    _re.compile(r"\u300c([^\u300d]+)\u300d"),  # 「設定」 (CJK corner brackets)
)


def extract_target_label(target: str) -> str:
    """Pull a quoted label out of a natural-language target prompt.

    benchmark/ground_truth.json targets are natural-language like
        "the center of the 'Import' button in the toolbar"
    but Stage B needs the bare label "Import" to fuzzy-match against OCR text.

    Precedence: single quotes, then double quotes, then CJK corner brackets.
    When multiple match, the first quoted group wins.

    Bare-word inputs (no surrounding prose, no whitespace such as "Save") are
    returned as-is — this supports direct MCP callers passing just the label.

    Natural-language prompts with no quoted label (e.g. "the tip of the red
    triangle") return an empty string, which signals "no extractable label"
    and makes apply_stage_b_snap short-circuit to miss. Without this, long
    prose targets would coincidentally suffix-match arbitrary OCR words
    (e.g. OCR "Triangle" suffix-matching "...red triangle").
    """
    if not target:
        return ""
    for pat in _LABEL_PATTERNS:
        m = pat.search(target)
        if m:
            return m.group(1).strip()
    # Bare single word (no whitespace) → treat as already-extracted label.
    stripped = target.strip()
    if stripped and " " not in stripped and "\t" not in stripped:
        return stripped
    return ""


def _levenshtein_le_one(a: str, b: str) -> bool:
    """True iff Levenshtein distance between a and b is ≤ 1.

    Handles substitution, insertion, and deletion; used to tolerate single-
    character OCR typos like 'lmport' ↔ 'Import' (I→l) or 'Hel' ↔ 'Help'
    (trailing drop — already covered by prefix match but fine to allow here).
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    shorter, longer = (a, b) if la < lb else (b, a)
    i = j = 0
    diff = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            if diff >= 1:
                return False
            diff += 1
            j += 1
        else:
            i += 1
            j += 1
    return True


def fuzzy_match(ocr_text: str, target_text: str) -> bool:
    """Match OCR text to a target label with tolerance for OCR noise.

    Rules (applied in order):
      1. Exact match after strip + lower (e.g. 'Save' == 'SAVE ')
      2. Prefix or suffix, either direction (handles OCR truncation like
         'Hel' ↔ 'Help' or trailing-space artifacts)
      3. Levenshtein distance ≤ 1, only when the target label is ≥ 3 chars
         (catches 1-char OCR typos like 'lmport' ↔ 'Import' where I→l).
         Short labels (<3 chars) require complete match to avoid false
         positives.
    """
    a = ocr_text.strip().lower()
    b = target_text.strip().lower()
    if not a or not b:
        return False
    if (a == b
            or a.startswith(b) or a.endswith(b)
            or b.startswith(a) or b.endswith(a)):
        return True
    if len(b) >= 3 and abs(len(a) - len(b)) <= 1:
        return _levenshtein_le_one(a, b)
    return False


def apply_stage_b_snap(
    pil_img: _Image.Image,
    x_shot: int, y_shot: int,
    phase1_x_ok: bool, phase1_y_ok: bool,
    target: str,
    shot_w: int, shot_h: int,
    radius_px: int = DEFAULT_STAGE_B_RADIUS,
    ocr_radius_px: int = DEFAULT_OCR_RADIUS,
) -> Dict[str, Any]:
    """Run Stage B OCR-snap on a Phase 1 pinpoint result.

    Args:
        pil_img: Full-resolution screenshot (caller's shot coordinate system).
        x_shot, y_shot: Phase 1 predicted target coordinates.
        phase1_x_ok, phase1_y_ok: Phase 1 per-axis success flags. Stage B is
            a no-op unless both succeeded — Phase 1 center is otherwise
            unreliable and a false snap is worse than falling through to
            Phase 1.5 fallback.
        target: Target description (fuzzy-matched against OCR word text).
        shot_w, shot_h: Screenshot dimensions (for crop clamping).
        radius_px: Retained for API compatibility / diagnostic reporting;
            no longer used as a snap distance gate. Stage B now uses a
            conservative **bbox-containment** rule (see H3 evidence): hit
            only when Phase 1 pred lies inside a matched OCR word's bbox.
        ocr_radius_px: Half-side of the square crop passed to the OCR engine
            (default 60 → 120×120 window). WinRT OCR needs a larger input
            to reliably detect toolbar-size text.

    Returns a dict:
        hit (bool), x (int), y (int), match (str), dist (float),
        time (float wall-clock s), radius (int), ocr_radius (int).
        On miss: x/y == input x_shot/y_shot, match "", dist 0.0.
    """
    t0 = _time.time()
    out: Dict[str, Any] = {
        "hit": False, "x": int(x_shot), "y": int(y_shot),
        "match": "", "dist": 0.0, "time": 0.0,
        "radius": int(radius_px),
        "ocr_radius": int(ocr_radius_px),
    }

    if not (phase1_x_ok and phase1_y_ok):
        out["time"] = _time.time() - t0
        return out

    # Strip surrounding natural-language so fuzzy_match sees the bare label.
    # Raw benchmark prompts like "the center of the 'Import' button ..." need
    # extraction; bare labels pass through; non-quoted prose returns "" which
    # short-circuits to miss (avoids spurious suffix-match on OCR words).
    effective_target = extract_target_label(target)
    if not effective_target:
        out["time"] = _time.time() - t0
        return out

    # OCR crop uses ocr_radius_px (wider); snap gate below uses radius_px.
    sb_x1 = max(0, int(x_shot) - ocr_radius_px)
    sb_y1 = max(0, int(y_shot) - ocr_radius_px)
    sb_x2 = min(int(shot_w), int(x_shot) + ocr_radius_px)
    sb_y2 = min(int(shot_h), int(y_shot) + ocr_radius_px)

    try:
        sb_crop = pil_img.crop((sb_x1, sb_y1, sb_x2, sb_y2))
        from agent.ocr_windows import ocr_image_with_bboxes
        sb_words = ocr_image_with_bboxes(sb_crop)
    except Exception:
        sb_words = []

    # Conservative snap (H3 修正 β): hit iff Phase 1 pred is INSIDE the
    # matched OCR bbox. Distance gate was removed after live-data evidence
    # showed OCR text bbox centers sit 22-29px from button click-centers on
    # real toolbar buttons, so a center-distance gate rejected every real
    # match. Containment test avoids snapping when Phase 1 clearly missed
    # the text (snap would worsen distance).
    best = None  # (x, y, dist_from_pred, text)
    for w in sb_words:
        if not fuzzy_match(w.get("text", ""), effective_target):
            continue
        bb = w.get("bbox", {})
        bx1 = sb_x1 + float(bb.get("x", 0))
        by1 = sb_y1 + float(bb.get("y", 0))
        bw = float(bb.get("w", 0))
        bh = float(bb.get("h", 0))
        bx2 = bx1 + bw
        by2 = by1 + bh
        if not (bx1 <= x_shot <= bx2 and by1 <= y_shot <= by2):
            continue  # Phase 1 pred outside bbox → skip (conservative)
        cx = bx1 + bw / 2.0
        cy = by1 + bh / 2.0
        dxv = cx - x_shot
        dyv = cy - y_shot
        dv = (dxv * dxv + dyv * dyv) ** 0.5
        # Multiple containing bboxes → nearest center wins as tie-breaker.
        if best is None or dv < best[2]:
            best = (int(round(cx)), int(round(cy)), dv, w.get("text", ""))

    if best is not None:
        out["hit"] = True
        out["x"] = best[0]
        out["y"] = best[1]
        out["dist"] = best[2]
        out["match"] = best[3]

    out["time"] = _time.time() - t0
    return out
