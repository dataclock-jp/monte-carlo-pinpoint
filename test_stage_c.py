"""
test_stage_c.py
Stage C dispatcher 単体テスト (Commit 1 scope)。

3 本立て (監督 decision 050 / CV-only determinism 指示):
    1. API contract: return schema + confidence range
    2. Stage B 互換: menu_item pipeline が apply_stage_b_snap を正しく retrofit
    3. CV-only 決定性: 固定入力で N=10 呼出、stdev(x) < 0.5 ∧ stdev(y) < 0.5

外部依存 (Windows OCR) は mock で置換。OCR 呼出自体の実機テストは
test_ocr_windows.py、Stage B 内部挙動は test_stage_b.py で網羅済。
"""
import os
import statistics
import sys
import unittest
from unittest.mock import patch

from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmark"))

from benchmark import stage_c


def _img(w: int = 800, h: int = 600) -> Image.Image:
    return Image.new("RGB", (w, h), "white")


class TestResolveTargetContract(unittest.TestCase):
    """API contract: 全 return path で schema が一貫し、confidence が [0, 1]。"""

    _REQUIRED_KEYS = {"hit", "x", "y", "method", "confidence", "time", "match"}

    def _assert_schema(self, r: dict) -> None:
        self.assertTrue(self._REQUIRED_KEYS.issubset(r.keys()))
        self.assertIsInstance(r["hit"], bool)
        self.assertIsInstance(r["x"], int)
        self.assertIsInstance(r["y"], int)
        self.assertIsInstance(r["method"], str)
        self.assertIsInstance(r["confidence"], float)
        self.assertIsInstance(r["time"], float)
        self.assertIsInstance(r["match"], str)
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)
        self.assertGreaterEqual(r["time"], 0.0)

    def test_generic_returns_miss_schema(self):
        r = stage_c.resolve_target("generic", "center", (100, 150), _img())
        self._assert_schema(r)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (100, 150))
        self.assertEqual(r["method"], "generic_miss")
        self.assertEqual(r["confidence"], 0.0)
        self.assertEqual(r["match"], "")

    def test_menu_item_no_label_returns_miss(self):
        r = stage_c.resolve_target(
            "menu_item", "center", (200, 200), _img(), label_hint=None)
        self._assert_schema(r)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (200, 200))
        self.assertEqual(r["method"], "menu_item_no_label")

    def test_unsupported_target_type_is_safe(self):
        """未知 target_type は generic miss-only に fall back (crash しない)。"""
        r = stage_c.resolve_target(
            "bogus_type", "center", (100, 100), _img())
        self._assert_schema(r)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (100, 100))
        self.assertEqual(r["method"], "unknown_target_type")

    def test_reserved_pipeline_marks_not_implemented(self):
        """Commit 3+ で実装される pipeline は識別可能な method label を返す。

        button/icon は Commit 2 で実装済なので別 test。
        """
        for t in ("rect", "face", "canvas_intent"):
            r = stage_c.resolve_target(t, "center", (50, 50), _img())
            self._assert_schema(r)
            self.assertFalse(r["hit"])
            self.assertEqual((r["x"], r["y"]), (50, 50))
            self.assertEqual(r["method"], f"{t}_not_implemented")

    def test_button_empty_image_returns_miss_schema(self):
        """白地のみ = contour なし → button_miss、approx_xy 維持。"""
        r = stage_c.resolve_target("button", "center", (400, 300), _img())
        self._assert_schema(r)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (400, 300))
        self.assertEqual(r["method"], "button_miss")

    def test_icon_empty_image_returns_miss_schema(self):
        r = stage_c.resolve_target("icon", "center", (400, 300), _img())
        self._assert_schema(r)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (400, 300))
        self.assertEqual(r["method"], "icon_miss")

    def test_coords_int_even_when_approx_xy_are_floats(self):
        """Phase 1.5 tool-use 側が float を渡しても int に正規化。"""
        r = stage_c.resolve_target(
            "generic", "center", (100.7, 150.2), _img())  # type: ignore[arg-type]
        self.assertEqual((r["x"], r["y"]), (100, 150))


class TestGenericMissIdentity(unittest.TestCase):
    """generic pipeline は v6 byte-identical (approx_xy passthrough) を保証。"""

    def test_generic_always_miss_regardless_of_label_hint(self):
        """label_hint があっても generic は OCR を走らせない。"""
        for label in (None, "", "Save", "Import"):
            r = stage_c.resolve_target(
                "generic", "center", (300, 400), _img(), label_hint=label)
            self.assertFalse(r["hit"])
            self.assertEqual((r["x"], r["y"]), (300, 400))

    def test_generic_does_not_invoke_ocr(self):
        """Paper ablation invariant: generic routes must be Phase 1-only."""
        with patch("agent.ocr_windows.ocr_image_with_bboxes") as ocr_mock:
            stage_c.resolve_target(
                "generic", "center", (100, 100), _img(),
                label_hint="Save")
            ocr_mock.assert_not_called()


class TestMenuItemStageBCompat(unittest.TestCase):
    """decision 049 + 050: Stage C ⊃ Stage B。既存 Stage B 挙動を retrofit。"""

    def test_menu_item_hit_delegates_to_stage_b(self):
        """Stage B が hit する入力で Stage C も hit、同じ座標を返す。"""
        # test_stage_b.TestApplyStageBSnap.test_hit_when_phase1_inside_bbox
        # と同じ fixture: crop origin (140, 140), bbox abs (195-215, 195-215).
        fake = [{"text": "Save",
                 "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=fake):
            r = stage_c.resolve_target(
                "menu_item", "center", (200, 200), _img(),
                label_hint="Save")
        self.assertTrue(r["hit"])
        self.assertEqual((r["x"], r["y"]), (205, 205))
        self.assertEqual(r["method"], "menu_item")
        self.assertEqual(r["match"], "Save")
        self.assertGreaterEqual(r["confidence"], stage_c.CONF_SNAP)

    def test_menu_item_miss_preserves_approx_xy(self):
        """Stage B miss → Stage C も miss、(x, y) は入力そのまま。"""
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=[]):
            r = stage_c.resolve_target(
                "menu_item", "center", (200, 200), _img(),
                label_hint="Save")
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (200, 200))
        self.assertEqual(r["method"], "menu_item_miss")
        self.assertEqual(r["confidence"], 0.0)

    def test_menu_item_phase1_axis_failure_is_miss(self):
        """Phase 1 軸失敗時は Stage B 側で no-op、Stage C も miss。"""
        fake = [{"text": "Save",
                 "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=fake):
            r = stage_c.resolve_target(
                "menu_item", "center", (200, 200), _img(),
                label_hint="Save",
                phase1_x_ok=False, phase1_y_ok=True)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (200, 200))

    def test_menu_item_natural_language_label_extraction(self):
        """stage_b.extract_target_label を経由: quoted label 抽出も機能。"""
        fake = [{"text": "Import",
                 "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=fake):
            r = stage_c.resolve_target(
                "menu_item", "center", (200, 200), _img(),
                label_hint="the center of the 'Import' button")
        self.assertTrue(r["hit"])
        self.assertEqual(r["match"], "Import")


def _button_image(
    w: int = 800, h: int = 600, *,
    bx: int = 300, by: int = 200, bw: int = 120, bh: int = 40,
    fill: tuple = (40, 40, 40),
) -> Image.Image:
    """White canvas with a single filled button-shaped rectangle.

    Default aspect ratio 120/40 = 3.0 (inside button window 1.5-6.0) and
    area 4800 px^2 (inside button area range). Centroid = (bx + bw/2, by + bh/2)
    = (360, 220) by default.
    """
    img = Image.new("RGB", (w, h), "white")
    px = img.load()
    for yy in range(by, by + bh):
        for xx in range(bx, bx + bw):
            px[xx, yy] = fill  # type: ignore[index]
    return img


def _icon_image(
    w: int = 800, h: int = 600, *,
    cx: int = 400, cy: int = 300, side: int = 30,
    fill: tuple = (30, 30, 30),
) -> Image.Image:
    """Icon-shaped (near-square) filled rectangle.

    Default 30x30 = area 900, AR 1.0 (inside icon window 0.6-1.7, area 100-3000).
    Centroid = (cx, cy).
    """
    img = Image.new("RGB", (w, h), "white")
    px = img.load()
    for yy in range(cy - side // 2, cy + side // 2):
        for xx in range(cx - side // 2, cx + side // 2):
            px[xx, yy] = fill  # type: ignore[index]
    return img


class TestButtonPipeline(unittest.TestCase):
    """Commit 2: button pipeline (Canny + findContours + filter + centroid)."""

    def test_button_happy_path_snaps_to_centroid(self):
        img = _button_image()  # button centroid (360, 220)
        r = stage_c.resolve_target("button", "center", (362, 221), img)
        self.assertTrue(r["hit"], f"unexpected miss: {r}")
        self.assertEqual(r["method"], "button")
        self.assertAlmostEqual(r["x"], 360, delta=2)
        self.assertAlmostEqual(r["y"], 220, delta=2)
        self.assertGreaterEqual(r["confidence"], stage_c.CONF_SNAP)

    def test_button_miss_when_approx_xy_far(self):
        """approx_xy が search window (±200px) 外 → contour 見えず miss。"""
        img = _button_image()  # button at (300-420, 200-240)
        r = stage_c.resolve_target("button", "center", (50, 50), img)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (50, 50))
        self.assertTrue(r["method"].endswith("_miss"))

    def test_button_aspect_ratio_filter_rejects_icon_shape(self):
        """Square 30x30 の icon shape (AR 1.0) は button window 外 → miss。"""
        img = _icon_image()
        r = stage_c.resolve_target("button", "center", (400, 300), img)
        self.assertFalse(r["hit"])
        self.assertEqual(r["method"], "button_miss")

    def test_icon_aspect_ratio_filter_rejects_wide_button(self):
        """120x40 の button shape (AR 3.0) は icon window 外 → miss。"""
        img = _button_image()
        r = stage_c.resolve_target("icon", "center", (362, 221), img)
        self.assertFalse(r["hit"])
        self.assertEqual(r["method"], "icon_miss")

    def test_icon_happy_path_snaps_to_centroid(self):
        img = _icon_image()  # icon centroid (400, 300)
        r = stage_c.resolve_target("icon", "center", (402, 301), img)
        self.assertTrue(r["hit"], f"unexpected miss: {r}")
        self.assertEqual(r["method"], "icon")
        self.assertAlmostEqual(r["x"], 400, delta=2)
        self.assertAlmostEqual(r["y"], 300, delta=2)

    def test_button_nearest_centroid_wins(self):
        """2 つの button のうち approx_xy に近い方を選ぶ。"""
        img = _button_image()  # first button at (300-420, 200-240)
        # add a second button far away
        px = img.load()
        for yy in range(500, 540):
            for xx in range(600, 720):
                px[xx, yy] = (50, 50, 50)  # type: ignore[index]
        r = stage_c.resolve_target("button", "center", (362, 221), img)
        self.assertTrue(r["hit"])
        # near button centroid (360, 220), not far one (660, 520)
        self.assertAlmostEqual(r["x"], 360, delta=2)
        self.assertAlmostEqual(r["y"], 220, delta=2)

    def test_confidence_decays_with_distance(self):
        """centroid から離れた approx_xy ほど confidence が下がる。"""
        img = _button_image()  # centroid (360, 220)
        near = stage_c.resolve_target("button", "center", (362, 221), img)
        far = stage_c.resolve_target("button", "center", (400, 260), img)  # ~55 px
        self.assertTrue(near["hit"])
        # Far case may still hit (conf ~0.1 > CONF_BIAS fails) or miss; we only
        # assert that if both hit, near > far; if far misses, near should still
        # be at CONF_SNAP.
        self.assertGreaterEqual(near["confidence"], stage_c.CONF_SNAP)
        if far["hit"]:
            self.assertLess(far["confidence"], near["confidence"])

    def test_intent_reserved_but_not_consumed(self):
        """Commit 2 では intent は contract 上受け取るだけで挙動に影響しない。"""
        img = _button_image()
        r_center = stage_c.resolve_target("button", "center", (362, 221), img)
        r_corner = stage_c.resolve_target("button", "corner", (362, 221), img)
        r_edge = stage_c.resolve_target("button", "edge", (362, 221), img)
        for r in (r_center, r_corner, r_edge):
            self.assertTrue(r["hit"])
            self.assertAlmostEqual(r["x"], 360, delta=2)
            self.assertAlmostEqual(r["y"], 220, delta=2)


class TestDistanceConfHelper(unittest.TestCase):
    """Piecewise-linear helper used by the contour pipelines."""

    def test_very_close_is_one(self):
        self.assertEqual(stage_c._distance_conf(0.0), 1.0)
        self.assertEqual(stage_c._distance_conf(stage_c.CONF_DIST_HIGH_PX - 1), 1.0)

    def test_far_is_zero(self):
        self.assertEqual(stage_c._distance_conf(stage_c.CONF_DIST_LOW_PX), 0.0)
        self.assertEqual(stage_c._distance_conf(1000.0), 0.0)

    def test_midpoint_linear(self):
        mid = (stage_c.CONF_DIST_HIGH_PX + stage_c.CONF_DIST_LOW_PX) / 2
        self.assertAlmostEqual(stage_c._distance_conf(mid), 0.5, places=3)

    def test_explicit_thresholds_decouple_from_button_defaults(self):
        """W2: _distance_conf supports per-variant thresholds."""
        # icon range (10, 40): midpoint 25 → 0.5
        self.assertAlmostEqual(stage_c._distance_conf(25, 10, 40), 0.5, places=3)
        # below icon hi → 1.0
        self.assertEqual(stage_c._distance_conf(5, 10, 40), 1.0)
        # above icon lo → 0.0
        self.assertEqual(stage_c._distance_conf(50, 10, 40), 0.0)


class TestW2VariantParameters(unittest.TestCase):
    """Phase B W2: per-variant thresholds and label_hint adaptive tightening."""

    def test_button_params_unchanged_from_phase_a(self):
        hi, lo = stage_c._variant_conf_params("button", None)
        self.assertEqual(hi, stage_c.BUTTON_CONF_DIST_HIGH_PX)
        self.assertEqual(lo, stage_c.BUTTON_CONF_DIST_LOW_PX)
        # label_hint MUST NOT change button params (Phase A contribution
        # protected: ui_button +10pt at v8 must remain identical).
        hi_lbl, lo_lbl = stage_c._variant_conf_params("button", "Save")
        self.assertEqual((hi_lbl, lo_lbl), (hi, lo))

    def test_icon_params_tighter_than_button(self):
        hi_icon, lo_icon = stage_c._variant_conf_params("icon", None)
        hi_btn, lo_btn = stage_c._variant_conf_params("button", None)
        self.assertLess(hi_icon, hi_btn)
        self.assertLess(lo_icon, lo_btn)

    def test_icon_label_hint_tightens_further(self):
        hi0, lo0 = stage_c._variant_conf_params("icon", None)
        hi1, lo1 = stage_c._variant_conf_params("icon", "Save")
        self.assertLess(hi1, hi0)
        self.assertLess(lo1, lo0)
        # tightening factor == ICON_LABELED_TIGHTEN
        self.assertAlmostEqual(hi1 / hi0, stage_c.ICON_LABELED_TIGHTEN, places=5)
        self.assertAlmostEqual(lo1 / lo0, stage_c.ICON_LABELED_TIGHTEN, places=5)

    def test_search_radius_split(self):
        self.assertEqual(stage_c._variant_search_radius("button"),
                         stage_c.BUTTON_SEARCH_RADIUS)
        self.assertEqual(stage_c._variant_search_radius("icon"),
                         stage_c.ICON_SEARCH_RADIUS)
        self.assertLess(stage_c._variant_search_radius("icon"),
                        stage_c._variant_search_radius("button"))

    def test_ambiguity_gap_icon_enabled_button_disabled(self):
        self.assertEqual(stage_c._variant_ambiguity_gap("button"), 0.0)
        self.assertGreater(stage_c._variant_ambiguity_gap("icon"), 0.0)


class TestW2AmbiguityAbstention(unittest.TestCase):
    """Phase B W2: dense-toolbar defense via ambiguity abstention."""

    @staticmethod
    def _dense_toolbar(w: int = 800, h: int = 600,
                        icons: tuple = ((400, 300), (440, 300))) -> Image.Image:
        """Two nearly-adjacent icons (40 px apart) to simulate a dense toolbar."""
        img = Image.new("RGB", (w, h), "white")
        px = img.load()
        for cx, cy in icons:
            for yy in range(cy - 15, cy + 15):
                for xx in range(cx - 15, cx + 15):
                    px[xx, yy] = (30, 30, 30)  # type: ignore[index]
        return img

    def test_icon_abstains_on_ambiguous_neighbour(self):
        """Two icons close together + approx_xy between them → miss.

        Without abstention the pipeline would snap to whichever icon happens
        to be nominally closer, even if the second is only a few pixels
        farther — a coin-flip with catastrophic error potential on dense
        toolbars (v8 ui_dense regression root cause).
        """
        img = self._dense_toolbar(icons=((400, 300), (440, 300)))
        # approx_xy placed between the two icons, closer to (400, 300).
        r = stage_c.resolve_target("icon", "center", (418, 300), img)
        self.assertFalse(r["hit"])
        self.assertEqual(r["method"], "icon_miss")
        self.assertEqual((r["x"], r["y"]), (418, 300))

    def test_button_does_not_abstain_on_ambiguity(self):
        """Phase A contribution protected: button abstention gap is 0,
        so neighbouring buttons do not suppress a hit."""
        # Two wide button rectangles 40 px apart
        img = Image.new("RGB", (800, 600), "white")
        px = img.load()
        for yy in range(200, 240):
            for xx in range(300, 420):
                px[xx, yy] = (40, 40, 40)  # type: ignore[index]
        for yy in range(200, 240):
            for xx in range(450, 570):
                px[xx, yy] = (40, 40, 40)  # type: ignore[index]
        # approx_xy close to first button centroid (360, 220)
        r = stage_c.resolve_target("button", "center", (362, 221), img)
        self.assertTrue(r["hit"])  # button still hits regardless of neighbour

    def test_icon_hits_when_neighbour_is_far_away(self):
        """Ambiguity abstention must not kill unambiguous hits."""
        img = self._dense_toolbar(icons=((400, 300), (700, 300)))  # 300 px apart
        r = stage_c.resolve_target("icon", "center", (402, 301), img)
        self.assertTrue(r["hit"])
        self.assertAlmostEqual(r["x"], 400, delta=2)
        self.assertAlmostEqual(r["y"], 300, delta=2)


class TestW2ConfidenceTightening(unittest.TestCase):
    """Phase B W2: icon pipeline tighter distance-to-confidence window."""

    def test_icon_misses_at_distance_that_button_would_hit(self):
        """36 px movement (v8 ui_dense_01/_04 observed) must miss for icon.

        Under Phase A's (15, 60) window, a 36 px centroid→approx distance
        produced conf ≈ 0.53 and therefore hit. Under Phase B W2's icon
        window (10, 40), the same distance yields conf ≈ 0.13 → miss, and
        the downstream Phase 2 gets to rescue the case instead of being
        short-circuited to a wrong-icon snap.
        """
        # Synthetic single-icon image with centroid at (400, 300), area fits
        # the icon window, AR ≈ 1.
        img = Image.new("RGB", (800, 600), "white")
        px = img.load()
        for yy in range(285, 315):
            for xx in range(385, 415):
                px[xx, yy] = (30, 30, 30)  # type: ignore[index]
        # approx_xy 36 px away from centroid
        r = stage_c.resolve_target("icon", "center", (436, 300), img)
        self.assertFalse(r["hit"])
        self.assertEqual(r["method"], "icon_miss")
        self.assertEqual((r["x"], r["y"]), (436, 300))

    def test_icon_labeled_tightens_further(self):
        """Labeled icon call uses tighter thresholds than unlabeled.

        A distance that survives the unlabeled threshold should drop below
        CONF_BIAS once the label-tighten factor is applied.
        """
        # Icon at (385-415, 285-315) has its detected centroid near (399, 299)
        # due to pixel sampling. approx (420, 299) gives effective distance
        # ~21 px:
        #   Unlabeled (10, 40): conf ≈ (40-21)/30 ≈ 0.633 → HIT
        #   Labeled (8, 32):    conf ≈ (32-21)/24 ≈ 0.458 < 0.5 → MISS
        img = Image.new("RGB", (800, 600), "white")
        px = img.load()
        for yy in range(285, 315):
            for xx in range(385, 415):
                px[xx, yy] = (30, 30, 30)  # type: ignore[index]
        approx = (420, 299)
        r_unlabeled = stage_c.resolve_target("icon", "center", approx, img)
        r_labeled = stage_c.resolve_target(
            "icon", "center", approx, img, label_hint="Save")
        self.assertTrue(r_unlabeled["hit"], f"unlabeled should hit: {r_unlabeled}")
        self.assertFalse(r_labeled["hit"], f"labeled should miss: {r_labeled}")
        self.assertEqual(r_labeled["method"], "icon_miss")

    def test_button_unaffected_by_label_hint(self):
        """Phase A button behaviour must be bit-identical with and without
        label_hint."""
        img = Image.new("RGB", (800, 600), "white")
        px = img.load()
        for yy in range(200, 240):
            for xx in range(300, 420):
                px[xx, yy] = (40, 40, 40)  # type: ignore[index]
        r_none = stage_c.resolve_target("button", "center", (362, 221), img)
        r_lbl = stage_c.resolve_target(
            "button", "center", (362, 221), img, label_hint="Save")
        self.assertEqual(
            (r_none["hit"], r_none["x"], r_none["y"], r_none["method"]),
            (r_lbl["hit"], r_lbl["x"], r_lbl["y"], r_lbl["method"]))


class TestW2IconSearchRadius(unittest.TestCase):
    """Phase B W2: icon uses a tighter search window than button."""

    def test_icon_misses_when_outside_icon_window_but_inside_button_window(self):
        """An icon 150 px away is inside the button window (200) but outside
        the icon window (100) → must miss for icon, regardless of what
        button might do on the same geometry."""
        img = Image.new("RGB", (800, 600), "white")
        px = img.load()
        # Icon at (400, 300)
        for yy in range(285, 315):
            for xx in range(385, 415):
                px[xx, yy] = (30, 30, 30)  # type: ignore[index]
        # approx_xy 150 px away → outside icon's ±100 window
        r = stage_c.resolve_target("icon", "center", (550, 300), img)
        self.assertFalse(r["hit"])
        self.assertEqual((r["x"], r["y"]), (550, 300))


class TestCvOnlyDeterminism(unittest.TestCase):
    """論点 2 (監督 `dlg/research/..T11:33:22-8a9f`) CV-only 決定性 test。

    固定 input (approx_xy / image) で pipeline を N 回呼出し、出力 (x, y) の
    stdev が 0.5px 未満であることを assert。Commit 1 scope の ``generic`` /
    ``menu_item`` は OCR mock を固定すれば厳密に決定的、stdev = 0 を想定。
    非決定性 (CUDA backend / seeded-RNG operator) が混入すれば即検出。
    """

    N = 10
    STDEV_CAP = 0.5  # px

    def _assert_deterministic(self, coords):
        if not coords:
            return
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if len(xs) >= 2:
            self.assertLess(
                statistics.stdev(xs), self.STDEV_CAP,
                f"x non-deterministic: xs={xs}")
            self.assertLess(
                statistics.stdev(ys), self.STDEV_CAP,
                f"y non-deterministic: ys={ys}")

    def test_generic_determinism(self):
        coords = [
            (r["x"], r["y"]) for r in (
                stage_c.resolve_target("generic", "center", (123, 456), _img())
                for _ in range(self.N)
            )
        ]
        self._assert_deterministic(coords)
        self.assertEqual(len(set(coords)), 1)  # 完全同一

    def test_menu_item_determinism_on_hit(self):
        fake = [{"text": "Save",
                 "bbox": {"x": 55, "y": 55, "w": 20, "h": 20}}]
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=fake):
            coords = [
                (r["x"], r["y"]) for r in (
                    stage_c.resolve_target(
                        "menu_item", "center", (200, 200), _img(),
                        label_hint="Save")
                    for _ in range(self.N)
                )
            ]
        self._assert_deterministic(coords)
        self.assertEqual(len(set(coords)), 1)

    def test_menu_item_determinism_on_miss(self):
        with patch("agent.ocr_windows.ocr_image_with_bboxes",
                   return_value=[]):
            coords = [
                (r["x"], r["y"]) for r in (
                    stage_c.resolve_target(
                        "menu_item", "center", (200, 200), _img(),
                        label_hint="Save")
                    for _ in range(self.N)
                )
            ]
        self._assert_deterministic(coords)
        self.assertEqual(len(set(coords)), 1)

    def test_button_determinism_on_hit(self):
        img = _button_image()
        coords = [
            (r["x"], r["y"]) for r in (
                stage_c.resolve_target(
                    "button", "center", (362, 221), img)
                for _ in range(self.N)
            )
        ]
        self._assert_deterministic(coords)
        self.assertEqual(len(set(coords)), 1)

    def test_icon_determinism_on_hit(self):
        img = _icon_image()
        coords = [
            (r["x"], r["y"]) for r in (
                stage_c.resolve_target(
                    "icon", "center", (402, 301), img)
                for _ in range(self.N)
            )
        ]
        self._assert_deterministic(coords)
        self.assertEqual(len(set(coords)), 1)

    def test_button_determinism_on_miss(self):
        img = _img()  # blank canvas, no contours
        coords = [
            (r["x"], r["y"]) for r in (
                stage_c.resolve_target(
                    "button", "center", (400, 300), img)
                for _ in range(self.N)
            )
        ]
        self._assert_deterministic(coords)
        self.assertEqual(len(set(coords)), 1)


if __name__ == "__main__":
    unittest.main()
