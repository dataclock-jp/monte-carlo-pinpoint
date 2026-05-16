"""
test_stage_b.py
benchmark.stage_b helper の単体テスト。

外部依存 (Windows OCR) を mock で置換し、fuzzy match / distance gate / Phase 1
軸成功条件の 3 点を検証する。OCR 呼出自体の実機テストは test_ocr_windows.py で
既に網羅済み。
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

from benchmark import stage_b


class TestExtractTargetLabel(unittest.TestCase):
    """Natural-language prompt → bare label extraction (H1 fix)."""

    def test_single_quoted_label(self):
        t = "the center of the 'Import' button in the toolbar"
        self.assertEqual(stage_b.extract_target_label(t), "Import")

    def test_double_quoted_label(self):
        t = 'the center of the "Save" icon'
        self.assertEqual(stage_b.extract_target_label(t), "Save")

    def test_cjk_corner_bracket_label(self):
        t = "the center of the \u300c設定\u300d button"
        self.assertEqual(stage_b.extract_target_label(t), "設定")

    def test_bare_label_passes_through(self):
        self.assertEqual(stage_b.extract_target_label("Save"), "Save")

    def test_no_quoted_label_returns_empty(self):
        """Non-text prose returns "" — signals apply_stage_b_snap to miss
        so OCR words cannot accidentally suffix-match arbitrary prose."""
        self.assertEqual(
            stage_b.extract_target_label("the tip of the red triangle"), "")

    def test_empty_string(self):
        self.assertEqual(stage_b.extract_target_label(""), "")

    def test_first_quoted_group_wins(self):
        """When multiple quoted tokens exist, first occurrence wins."""
        t = "click 'Save' not 'Cancel'"
        self.assertEqual(stage_b.extract_target_label(t), "Save")

    def test_end_to_end_benchmark_prompt_matches_ocr(self):
        """Regression test for the reported 0% hit bug: ensure the
        extraction + fuzzy_match chain accepts OCR 'Import' for the
        benchmark-style natural-language prompt."""
        prompt = "the center of the 'Import' button in the toolbar"
        label = stage_b.extract_target_label(prompt)
        self.assertTrue(stage_b.fuzzy_match("Import", label))


class TestLevenshteinLeOne(unittest.TestCase):
    def test_equal(self):
        self.assertTrue(stage_b._levenshtein_le_one("abc", "abc"))

    def test_one_substitution(self):
        self.assertTrue(stage_b._levenshtein_le_one("lmport", "import"))

    def test_one_insertion(self):
        self.assertTrue(stage_b._levenshtein_le_one("abc", "abcd"))

    def test_one_deletion(self):
        self.assertTrue(stage_b._levenshtein_le_one("abcd", "abc"))

    def test_two_edits_rejected(self):
        self.assertFalse(stage_b._levenshtein_le_one("abcd", "abef"))

    def test_length_gap_two_rejected(self):
        self.assertFalse(stage_b._levenshtein_le_one("abc", "abcde"))


class TestFuzzyMatch(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(stage_b.fuzzy_match("Save", "Save"))

    def test_case_insensitive(self):
        self.assertTrue(stage_b.fuzzy_match("SAVE", "save"))

    def test_prefix(self):
        self.assertTrue(stage_b.fuzzy_match("Save ", "Save"))
        self.assertTrue(stage_b.fuzzy_match("Sav", "Save"))

    def test_suffix(self):
        self.assertTrue(stage_b.fuzzy_match("ave", "Save"))

    def test_mismatch(self):
        self.assertFalse(stage_b.fuzzy_match("Cancel", "Save"))

    def test_levenshtein_one_char_typo(self):
        """H4: 1-char OCR typo like I→l rescued for label length ≥3."""
        self.assertTrue(stage_b.fuzzy_match("lmport", "Import"))

    def test_levenshtein_requires_minimum_label_length(self):
        """Short labels (<3 chars) require exact match to avoid false pos."""
        # 'Ab' vs 'Cb' is 1 edit but label len < 3 → reject
        self.assertFalse(stage_b.fuzzy_match("Ab", "Cb"))

    def test_empty(self):
        self.assertFalse(stage_b.fuzzy_match("", "Save"))
        self.assertFalse(stage_b.fuzzy_match("Save", ""))


class TestApplyStageBSnap(unittest.TestCase):
    def _img(self):
        return Image.new("RGB", (800, 600), "white")

    def test_phase1_axis_failure_is_noop(self):
        img = self._img()
        with patch.object(stage_b, "apply_stage_b_snap", wraps=stage_b.apply_stage_b_snap):
            r = stage_b.apply_stage_b_snap(
                img, 100, 100, False, True, "Save", 800, 600)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (100, 100))
        self.assertEqual(r["match"], "")

    def test_miss_path_returns_phase1_center(self):
        """OCR が空の場合 → miss、x/y は Phase 1 入力そのまま (byte-identical)。"""
        img = self._img()
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=[]):
            r = stage_b.apply_stage_b_snap(
                img, 100, 100, True, True, "Save", 800, 600)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (100, 100))

    def test_hit_when_phase1_inside_bbox(self):
        """修正 β (H3): hit iff Phase 1 pred is INSIDE the matched bbox."""
        img = self._img()
        # crop origin (140, 140); bbox crop-relative (55, 55, 20, 20)
        # → abs bbox (195, 195) to (215, 215). Phase1 (200, 200) inside.
        fake = [{"text": "Save", "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600)
        self.assertTrue(r["hit"])
        # bbox center → abs (205, 205)
        self.assertEqual((r["x"], r["y"]), (205, 205))
        self.assertEqual(r["match"], "Save")

    def test_miss_when_phase1_outside_bbox(self):
        """修正 β (H3): Phase 1 pred outside bbox → miss even if fuzzy match.

        Evidence-based: ui_button_01 'Cut' had gt (174, 22) vs OCR bbox abs
        center (152, 26). Snapping the button-center Phase 1 to the text
        bbox would worsen distance — so we never snap from outside.
        """
        img = self._img()
        # crop origin (140, 140); bbox crop-rel (0, 0, 20, 20)
        # → abs bbox (140, 140)-(160, 160). Phase 1 (200, 200) OUTSIDE.
        fake = [{"text": "Save", "bbox": {"x": 0, "y": 0, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (200, 200))

    def test_multiple_containing_bboxes_picks_nearest_center(self):
        """Tie-breaker when Phase 1 pred is inside multiple matched bboxes."""
        img = self._img()
        # crop origin (140, 140); two bboxes both containing Phase1 (200, 200):
        # bbox A: abs (190, 190)-(210, 210), center (200, 200), dist=0
        # bbox B: abs (185, 195)-(225, 205), center (205, 200), dist=5
        fake = [
            {"text": "Save",    "bbox": {"x": 45, "y": 55, "w": 40, "h": 10}},  # B
            {"text": "Save",    "bbox": {"x": 50, "y": 50, "w": 20, "h": 20}},  # A
        ]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600)
        self.assertTrue(r["hit"])
        self.assertEqual((r["x"], r["y"]), (200, 200))  # A wins (closer center)

    def test_ocr_scope_and_snap_gate_are_independent(self):
        """Regression test for H2: even with snap gate 20px, OCR scope must
        be wider (default 60px) so WinRT can actually detect toolbar text.

        We verify by passing a custom narrow ocr_radius_px that would miss a
        distant-but-still-snappable match — confirming ocr_radius governs crop
        size independently of the snap gate.
        """
        img = self._img()
        # Target bbox center at abs (200, 170) — dist 30 from Phase1 (200,200).
        # With snap gate 40, this is inside snap; but if we force OCR crop to
        # ±20, the bbox at abs y=170 is outside the ±20 crop (y=180..220),
        # OCR sees nothing, miss.
        fake_if_ocr_wide = [{"text": "Save", "bbox": {
            "x": 60 - 10, "y": 30 - 10, "w": 20, "h": 20
        }}]  # crop origin (140,140) assumed by test_hit
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=fake_if_ocr_wide):
            r_narrow = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600,
                radius_px=40, ocr_radius_px=20)
        # With narrow OCR, even though snap gate allows 40px, the crop is
        # 40×40 and WinRT returns nothing useful. (Our mock returns the fake
        # regardless of crop, so we instead probe that the function honored
        # ocr_radius_px=20 in its computed crop origin by checking the output
        # remains at (200, 200) when the bbox is outside the ±20 crop.)

        # Direct assertion: with ocr_radius_px=20 (crop 180-220) but bbox at
        # crop-relative y=20 → abs y=200. Centerline fine. Hard to prove
        # param separation via mock. Instead assert the crop call shape:
        from unittest.mock import MagicMock
        crop_calls = []

        def spy_ocr(img_arg):
            crop_calls.append(img_arg.size)
            return []

        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   side_effect=spy_ocr):
            stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600,
                radius_px=20, ocr_radius_px=60)
        self.assertEqual(crop_calls[-1], (120, 120))

        crop_calls.clear()
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   side_effect=spy_ocr):
            stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600,
                radius_px=20, ocr_radius_px=20)
        self.assertEqual(crop_calls[-1], (40, 40))

    def test_ocr_exception_safe_miss(self):
        """OCR subprocess 失敗時は miss として返し、caller に伝播しない。"""
        img = self._img()
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   side_effect=RuntimeError("boom")):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (200, 200))

    def test_non_matching_text_is_miss(self):
        img = self._img()
        fake = [{"text": "Cancel", "bbox": {"x": 10, "y": 10, "w": 30, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600)
        self.assertFalse(r["hit"])

    def test_natural_language_prompt_hits(self):
        """H1 regression: natural-language prompt → label → fuzzy_match → hit.

        Uses bbox wide enough to contain Phase 1 pred (H3 修正 β).
        """
        img = self._img()
        # crop origin (140, 140); bbox abs (195, 195)-(215, 215), Phase1 inside.
        fake = [{"text": "Import", "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        prompt = "the center of the 'Import' button in the toolbar"
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, prompt, 800, 600)
        self.assertTrue(r["hit"])
        self.assertEqual(r["match"], "Import")

    def test_ocr_typo_matched_via_levenshtein(self):
        """H4: WinRT misreads 'Import' as 'lmport' (I→l). Levenshtein ≤1
        rescues the match; the bbox must also contain Phase 1 (H3)."""
        img = self._img()
        fake = [{"text": "lmport", "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Import", 800, 600)
        self.assertTrue(r["hit"])
        self.assertEqual(r["match"], "lmport")

    def test_natural_language_non_text_target_misses(self):
        """"the tip of the red triangle" has no quoted label; Stage B should
        miss (full prompt won't match any OCR word), which is correct."""
        img = self._img()
        fake = [{"text": "Triangle", "bbox": {"x": 20, "y": 20, "w": 10, "h": 10}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True,
                "the tip of the red triangle", 800, 600)
        self.assertFalse(r["hit"])

    def test_radius_px_preserved_in_output_for_diagnostics(self):
        """radius_px no longer gates snap (replaced by bbox containment) but
        is still reported in the output for benchmark diagnostic logs."""
        img = self._img()
        fake = [{"text": "Save", "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes", return_value=fake):
            r = stage_b.apply_stage_b_snap(
                img, 200, 200, True, True, "Save", 800, 600,
                radius_px=42, ocr_radius_px=60)
        self.assertEqual(r["radius"], 42)
        self.assertEqual(r["ocr_radius"], 60)


if __name__ == "__main__":
    unittest.main()
