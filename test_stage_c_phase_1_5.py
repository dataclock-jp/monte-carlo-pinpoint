"""
test_stage_c_phase_1_5.py
Phase 1.5 heuristic declare_target helper の unit test。

Commit 3 / Option iii (decision 050 論点 1 の heuristic 実装パス)。
Real VLM tool-use 版は後続 commit で追加予定、契約互換を本 test で担保。
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmark"))

from benchmark import stage_c, stage_c_phase_1_5 as p1_5


class TestSchema(unittest.TestCase):
    # Decision 052 9-field canonical: `source` and `reasoning` are promoted to
    # top-level `stage_c_source` / `stage_c_reasoning` by the caller, but
    # declare_target's internal return contract still exposes them so the
    # caller can populate the top-level fields from a single call.
    REQUIRED = {"target_type", "intent", "label_hint", "source",
                "reasoning", "visible"}

    VALID_REASONINGS = {
        "button_label_match", "icon_label_match", "menu_label_match",
        "label_only_fallback", "no_rule_fired", "empty_prompt",
    }

    def _assert(self, r):
        self.assertTrue(self.REQUIRED.issubset(r.keys()))
        self.assertIn(r["target_type"], stage_c.SUPPORTED_TARGET_TYPES)
        self.assertIsInstance(r["intent"], str)
        self.assertIsInstance(r["label_hint"], str)
        self.assertEqual(r["source"], "heuristic")
        self.assertIsInstance(r["reasoning"], str)
        self.assertIn(r["reasoning"], self.VALID_REASONINGS)
        self.assertIsInstance(r["visible"], bool)

    def test_ui_button_prompt(self):
        r = p1_5.declare_target("the center of the 'Save' button in the toolbar")
        self._assert(r)
        self.assertEqual(r["target_type"], "button")
        self.assertEqual(r["intent"], "center")
        self.assertEqual(r["label_hint"], "Save")
        self.assertEqual(r["reasoning"], "button_label_match")

    def test_ui_dense_icon_prompt(self):
        r = p1_5.declare_target("the center of the 'Undo' icon in the palette")
        self._assert(r)
        self.assertEqual(r["target_type"], "icon")
        self.assertEqual(r["label_hint"], "Undo")
        self.assertEqual(r["reasoning"], "icon_label_match")

    def test_ui_dense_icon_with_menu_strip_still_icon(self):
        """prompt に icon + menu strip 両 keyword があっても icon 優先。"""
        r = p1_5.declare_target(
            "the center of the 'Chart' icon in the menu strip")
        self.assertEqual(r["target_type"], "icon")
        self.assertEqual(r["label_hint"], "Chart")
        self.assertEqual(r["reasoning"], "icon_label_match")

    def test_menu_item_without_icon_or_button_keyword(self):
        r = p1_5.declare_target(
            "click the 'Settings' menu item")
        self.assertEqual(r["target_type"], "menu_item")
        self.assertEqual(r["label_hint"], "Settings")
        self.assertEqual(r["reasoning"], "menu_label_match")

    def test_natural_no_quoted_label_generic(self):
        r = p1_5.declare_target("the tip of the red triangle")
        self._assert(r)
        self.assertEqual(r["target_type"], "generic")
        self.assertEqual(r["label_hint"], "")
        self.assertEqual(r["intent"], "tip")
        self.assertEqual(r["reasoning"], "no_rule_fired")

    def test_ambiguous_no_quoted_label_generic(self):
        r = p1_5.declare_target("the slightly darker square")
        self.assertEqual(r["target_type"], "generic")
        self.assertEqual(r["reasoning"], "no_rule_fired")

    def test_precision_no_quoted_label_generic(self):
        r = p1_5.declare_target("the single red pixel")
        self.assertEqual(r["target_type"], "generic")
        self.assertEqual(r["intent"], "center")

    def test_empty_prompt_safe(self):
        r = p1_5.declare_target("")
        self._assert(r)
        self.assertEqual(r["target_type"], "generic")
        self.assertEqual(r["label_hint"], "")
        self.assertEqual(r["reasoning"], "empty_prompt")

    def test_intent_intersection_detected(self):
        r = p1_5.declare_target(
            "the intersection of the two diagonal lines")
        self.assertEqual(r["intent"], "intersection")

    def test_quoted_label_without_type_keyword_routes_to_menu_item(self):
        """v7 Stage B の permissive behavior を Stage C 下でも維持。"""
        r = p1_5.declare_target("click 'Preferences' at the top right")
        self.assertEqual(r["target_type"], "menu_item")
        self.assertEqual(r["label_hint"], "Preferences")
        # Inferred from label presence alone — distinct reasoning id so the
        # provenance is recoverable at report time without reading the prompt.
        self.assertEqual(r["reasoning"], "label_only_fallback")

    def test_determinism(self):
        """declare_target は純粋関数。同一入力で同一出力 (N=20)。"""
        prompts = [
            "the center of the 'Save' button in the toolbar",
            "the tip of the red triangle",
            "",
        ]
        for p in prompts:
            ref = p1_5.declare_target(p)
            for _ in range(20):
                self.assertEqual(p1_5.declare_target(p), ref)


class TestGroundTruthRouting(unittest.TestCase):
    """50-case benchmark dataset の category 別 routing が期待値通り。

    ground_truth.json の target prompt を直接 feed、target_type の
    category 別分布が decision 050 の paper ablation 設計と一致することを確認。
    """

    @classmethod
    def setUpClass(cls):
        import json
        with open(os.path.join(_ROOT, "benchmark", "ground_truth.json"),
                  encoding="utf-8") as f:
            cls.gt = json.load(f)

    def _routing_by_category(self):
        by_cat = {}
        for g in self.gt:
            r = p1_5.declare_target(g["target"])
            by_cat.setdefault(g["category"], []).append(r["target_type"])
        return by_cat

    def test_ui_button_all_button(self):
        by = self._routing_by_category()
        self.assertEqual(set(by["ui_button"]), {"button"})

    def test_ui_dense_all_icon(self):
        by = self._routing_by_category()
        self.assertEqual(set(by["ui_dense"]), {"icon"})

    def test_natural_all_generic(self):
        by = self._routing_by_category()
        self.assertEqual(set(by["natural"]), {"generic"})

    def test_ambiguous_all_generic(self):
        by = self._routing_by_category()
        self.assertEqual(set(by["ambiguous"]), {"generic"})

    def test_precision_all_generic(self):
        by = self._routing_by_category()
        self.assertEqual(set(by["precision"]), {"generic"})


if __name__ == "__main__":
    unittest.main()
