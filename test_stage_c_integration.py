"""
test_stage_c_integration.py
run_benchmark.method_pinpoint の Stage C integration test。

Commit 3: --use-stage-c flag で method_pinpoint の Phase 1 後段が
Stage C (heuristic declare_target + stage_c.resolve_target) にルーティング
されることを検証。VLM 呼出は mock (Phase 1 / Phase 2 の grid_core 系 call)
で置換、Stage C dispatch と result schema (decision 052 9-field canonical)
のみを contract 検査。

Phase C Priority 3 (decision 052 canonical): nested
`phase_1_5_structured.source` / `.confidence` は top-level
`stage_c_source` / `stage_c_reasoning` に promote 済。
`phase_1_5_structured` は `{target_type, intent, label_hint, visible}` のみ。

Stage 1 A/B smoke で arm A (Stage B) / arm B (Stage C) 両 path を benchmark-
agent が走らせるため、本 test はそれぞれの path が result schema を正しく
出力することを保証する gate。
"""
import os
import sys
import unittest
from unittest.mock import patch

from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmark"))

from benchmark import run_benchmark


_STAGE_C_FIELDS = (
    "stage_c_hit", "stage_c_method", "stage_c_confidence",
    "stage_c_time", "stage_c_input_xy", "stage_c_output_xy",
    "stage_c_source", "stage_c_reasoning",
    "phase_1_5_structured",
)

_STAGE_B_FIELDS = (
    "stage_b_hit", "stage_b_time", "stage_b_match", "stage_b_dist",
)


def _button_image(w: int = 800, h: int = 600) -> Image.Image:
    """Synthetic image with a filled button-shaped rectangle near (360, 220)."""
    img = Image.new("RGB", (w, h), "white")
    px = img.load()
    for yy in range(200, 240):
        for xx in range(300, 420):
            px[xx, yy] = (40, 40, 40)  # type: ignore[index]
    return img


def _blank_image(w: int = 800, h: int = 600) -> Image.Image:
    return Image.new("RGB", (w, h), "white")


def _patch_phase1(x_shot: int = 362, y_shot: int = 221):
    """Return a context manager stubbing Phase 1 grid search to (x_shot, y_shot).

    Also disables the upscale-for-grid helper so ``scale`` stays 1.0 and the
    mocked coordinates survive the map-back-to-original step unchanged.
    """
    def _stub(work_img, target, axis, max_steps, precision,
              api_key, model, query_counter=None):
        if query_counter is not None:
            query_counter[0] += 1
        return (x_shot, 0) if axis == "x" else (y_shot, 0)

    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch(
        "benchmark.run_benchmark._upscale_for_grid",
        side_effect=lambda img, target_width=2560: img))
    stack.enter_context(patch(
        "benchmark.run_benchmark.grid_search_axis", side_effect=_stub))
    return stack


def _patch_phase_1_5_and_phase_2(anchor: str = "toolbar area"):
    """Stub anchor declaration + Monte Carlo ask so Phase 2 exits after 1 round."""
    def _stub_anchor(b64, target, api_key, model):
        return (anchor, True)  # (anchor, visible)

    def _stub_majority(b64, anchor, n_dots, api_key, model, votes,
                       query_counter=None):
        if query_counter is not None:
            query_counter[0] += 1
        return (None, False)  # no consensus → loop will exit on shrink

    # Apply both patches by returning a stack.
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("benchmark.run_benchmark.pinpoint_declare_anchor",
                              side_effect=_stub_anchor))
    stack.enter_context(patch("benchmark.run_benchmark.pinpoint_ask_majority",
                              side_effect=_stub_majority))
    return stack


class TestStageCIntegrationContract(unittest.TestCase):
    """--use-stage-c 有無で result schema が必ず decision 052 9 fields を含む。"""

    def setUp(self):
        run_benchmark._USE_STAGE_C = False

    def tearDown(self):
        run_benchmark._USE_STAGE_C = False

    def _run(self, target, img):
        with _patch_phase1(362, 221), _patch_phase_1_5_and_phase_2():
            return run_benchmark.method_pinpoint(
                img, target, api_key="stub", model="stub",
                max_rounds=1,  # keep Phase 2 short
            )

    def test_stage_b_path_emits_stage_c_fields_with_defaults(self):
        """--use-stage-c OFF: Stage B 走行でも schema に stage_c_* が含まれる。"""
        r = self._run("the tip of the red triangle", _blank_image())
        for f in _STAGE_C_FIELDS:
            self.assertIn(f, r, f"missing field {f}")
        for f in _STAGE_B_FIELDS:
            self.assertIn(f, r)
        self.assertFalse(r["stage_c_hit"])
        self.assertEqual(r["stage_c_method"], "")
        self.assertEqual(r["stage_c_source"], "")
        self.assertEqual(r["stage_c_reasoning"], "")
        self.assertEqual(r["phase_1_5_structured"], {})

    def test_stage_c_path_emits_heuristic_declaration(self):
        run_benchmark._USE_STAGE_C = True
        r = self._run(
            "the center of the 'Save' button in the toolbar",
            _button_image())
        for f in _STAGE_C_FIELDS:
            self.assertIn(f, r)
        self.assertEqual(r["phase_1_5_structured"]["target_type"], "button")
        self.assertEqual(r["phase_1_5_structured"]["label_hint"], "Save")
        # source / reasoning live at the top level per decision 052 canonical,
        # not inside phase_1_5_structured.
        self.assertNotIn("source", r["phase_1_5_structured"])
        self.assertNotIn("confidence", r["phase_1_5_structured"])
        self.assertEqual(r["stage_c_source"], "heuristic")
        self.assertEqual(r["stage_c_reasoning"], "button_label_match")

    def test_stage_c_button_hit_short_circuits_phase_2(self):
        """Commit 2 dry-run 再現: button pipeline が hit → Phase 2 skip。"""
        run_benchmark._USE_STAGE_C = True
        r = self._run(
            "the center of the 'Save' button in the toolbar",
            _button_image())
        self.assertTrue(r["stage_c_hit"])
        self.assertEqual(r["stage_c_method"], "button")
        self.assertEqual(r["phase2_rounds"], 0)  # Phase 2 skipped
        # Refined coords near the synthetic button centroid (360, 220).
        self.assertAlmostEqual(r["stage_c_output_xy"][0], 360, delta=3)
        self.assertAlmostEqual(r["stage_c_output_xy"][1], 220, delta=3)

    def test_stage_c_generic_falls_through_to_phase_2(self):
        """generic routing は miss-only → Phase 2 が走る (byte-identical)。"""
        run_benchmark._USE_STAGE_C = True
        r = self._run("the tip of the red triangle", _blank_image())
        self.assertFalse(r["stage_c_hit"])
        self.assertEqual(r["phase_1_5_structured"]["target_type"], "generic")
        # Phase 2 ran (even if no consensus, rounds_done should be >= 1).
        self.assertGreaterEqual(r["phase2_rounds"], 1)

    def test_stage_c_input_xy_matches_phase1(self):
        run_benchmark._USE_STAGE_C = True
        r = self._run(
            "the center of the 'Save' button in the toolbar",
            _button_image())
        self.assertEqual(r["stage_c_input_xy"], [362, 221])

    def test_stage_b_off_when_stage_c_routes_via_button(self):
        """button pipeline hit の場合、stage_b_hit は False のまま。"""
        run_benchmark._USE_STAGE_C = True
        r = self._run(
            "the center of the 'Save' button in the toolbar",
            _button_image())
        self.assertTrue(r["stage_c_hit"])
        # Stage B field reflects whether the menu_item sub-case fired
        # specifically, not Stage C as a whole.
        self.assertFalse(r["stage_b_hit"])


class TestStageCPathIdempotence(unittest.TestCase):
    """--use-stage-c OFF 走行が v7 と byte-identical である invariant。

    Stage B single call path 以外 (Phase 1 / Phase 1.5 / Phase 2) に Stage C
    導入で regression が入っていないことを、OFF 走行の result が Stage B-only
    baseline と schema 互換であることで担保する。
    """

    def setUp(self):
        run_benchmark._USE_STAGE_C = False

    def test_stage_c_off_yields_stage_b_identical_xy(self):
        with _patch_phase1(50, 50), _patch_phase_1_5_and_phase_2():
            r = run_benchmark.method_pinpoint(
                _blank_image(),
                "the tip of the red triangle",
                api_key="stub", model="stub",
                max_rounds=1,
            )
        # No Stage C → stage_c_output_xy == stage_c_input_xy == phase1 coords.
        self.assertEqual(r["stage_c_output_xy"], [50, 50])
        self.assertEqual(r["stage_c_input_xy"], [50, 50])


if __name__ == "__main__":
    unittest.main()
