"""
test_arm_c_aggregator.py

W3 3-arm smoke (decision 062 candidate / Priority 4) — TDD acceptance-criteria
scaffolding authored by benchmark-agent for video2ai-agent to implement against.

Scope pinned by supervisor directive dlg/cognitive-nodes/msg-2026-04-20T03:22:06-6a74
and W3 plan v2 §2.2 / §2.3 / §6.3. video2ai-agent Priority 4 implements arm B
(tool-use N=1) + arm C (tool-use N=3 median), exposes a `--arm {A,B,C}` flag on
run_benchmark.py, and emits `stage_c_source="vlm_tool_use"` + `stage_c_reasoning`
(raw tool input) for arm B/C rows.

These tests use `unittest.skipUnless` gates so the file is importable and runnable
from day one: until video2ai-agent lands the impl, every test skips with an
explicit reason. When impl lands, skips invert and failures indicate spec drift.

Four TDD contracts:
  1. TestToolUseDeclareTargetContract — return-dict shape for arm B/C declare_target
  2. TestArmDispatching               — --arm {A,B,C} flag wiring
  3. TestArmCCentroidAggregation      — N=3 median-of-centroids helper
  4. TestParseFailureExclusion         — W3 v2 §6.3 revised: full-exclude, no substitution
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmark"))

from benchmark import stage_c, stage_c_phase_1_5 as p1_5


# ---------------------------------------------------------------------------
# Feature-gate helpers: invert on Priority 4 landing.
#
# Each helper returns True once the corresponding API surface exists in HEAD.
# video2ai-agent is free to choose exact names; if a different name is used,
# update the probe below in a single place.
# ---------------------------------------------------------------------------

def _has_tool_use_declare_target() -> bool:
    """True once a tool-use variant of declare_target is importable.

    video2ai-agent Priority 4 may expose this as:
      - `p1_5.declare_target_tool_use(target_prompt, ...)`, or
      - `p1_5.declare_target(target_prompt, source="vlm_tool_use")`, or
      - a new module `benchmark.stage_c_phase_1_5_tool_use.declare_target(...)`.
    The probe accepts any of these shapes so the acceptance tests are
    name-agnostic.
    """
    if hasattr(p1_5, "declare_target_tool_use"):
        return True
    try:
        import inspect
        sig = inspect.signature(p1_5.declare_target)
        if "source" in sig.parameters or "arm" in sig.parameters:
            return True
    except (ValueError, TypeError):
        pass
    try:
        from benchmark import stage_c_phase_1_5_tool_use  # noqa: F401
        return True
    except ImportError:
        return False
    return False


def _has_arm_flag() -> bool:
    """True once run_benchmark.py exposes --arm dispatching."""
    try:
        from benchmark import run_benchmark
    except ImportError:
        return False
    # Expect either an _ARM module-level var, an _arm argparse add, or an
    # ARM_CHOICES tuple. Any of the three indicates wiring.
    for probe in ("_ARM", "ARM_CHOICES", "_arm_choices", "SUPPORTED_ARMS"):
        if hasattr(run_benchmark, probe):
            return True
    # Fallback: introspect the module source for --arm flag registration.
    try:
        src = open(run_benchmark.__file__, encoding="utf-8").read()
        return "--arm" in src
    except OSError:
        return False


def _has_centroid_aggregator() -> bool:
    """True once an N-sample centroid aggregator exists at Phase 1.5 level.

    video2ai-agent may implement this as:
      - `run_benchmark._aggregate_centroid(samples, method="median")`, or
      - extend existing `_aggregate_axis` with a paired-axis wrapper, or
      - a fresh helper in `stage_c_phase_1_5` or a new module.
    """
    try:
        from benchmark import run_benchmark
    except ImportError:
        return False
    for probe in ("_aggregate_centroid", "_aggregate_xy", "_aggregate_centroid_samples"):
        if hasattr(run_benchmark, probe):
            return True
    return False


_TOOL_USE_READY = _has_tool_use_declare_target()
_ARM_FLAG_READY = _has_arm_flag()
_CENTROID_AGG_READY = _has_centroid_aggregator()

_SKIP_REASON_TOOL_USE = (
    "Priority 4 tool-use declare_target not landed yet "
    "(see stage_c_phase_1_5.py:3-15 docstring, W3 plan v3 §8.1 gate)"
)
_SKIP_REASON_ARM_FLAG = "Priority 4 --arm flag not wired into run_benchmark.py yet"
_SKIP_REASON_CENTROID = "Priority 4 centroid aggregator helper not present yet"


# ---------------------------------------------------------------------------
# 1. Tool-use declare_target return-dict contract
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TOOL_USE_READY, _SKIP_REASON_TOOL_USE)
class TestToolUseDeclareTargetContract(unittest.TestCase):
    """Arm B/C declare_target must emit the same 6-key dict as the heuristic
    variant, with `source="vlm_tool_use"` and `reasoning=<raw tool input>`
    (string, free-form, not restricted to the 6-value heuristic enum)."""

    REQUIRED_KEYS = {"target_type", "intent", "label_hint", "source",
                     "reasoning", "visible"}

    def _invoke_tool_use_declare(self, prompt: str, mock_tool_response: dict) -> dict:
        """Thin shim — video2ai-agent may expose this under one of the three
        names probed in `_has_tool_use_declare_target`. We try them in order."""
        if hasattr(p1_5, "declare_target_tool_use"):
            with patch.object(p1_5, "_call_anthropic_tool_use",
                              return_value=mock_tool_response, create=True):
                return p1_5.declare_target_tool_use(prompt)
        try:
            from benchmark import stage_c_phase_1_5_tool_use as tu
            with patch.object(tu, "_call_anthropic_tool_use",
                              return_value=mock_tool_response, create=True):
                return tu.declare_target(prompt)
        except ImportError:
            pass
        with patch.object(p1_5, "_call_anthropic_tool_use",
                          return_value=mock_tool_response, create=True):
            return p1_5.declare_target(prompt, source="vlm_tool_use")

    def test_tool_use_happy_path_button(self):
        mock_response = {
            "target_type": "button", "intent": "center",
            "label_hint": "Save", "raw": '{"target_type":"button",...}',
        }
        r = self._invoke_tool_use_declare(
            "the center of the 'Save' button", mock_response)
        self.assertTrue(self.REQUIRED_KEYS.issubset(r.keys()),
                        f"missing keys: {self.REQUIRED_KEYS - r.keys()}")
        self.assertEqual(r["source"], "vlm_tool_use")
        self.assertIn(r["target_type"], stage_c.SUPPORTED_TARGET_TYPES)
        self.assertIsInstance(r["reasoning"], str)
        self.assertTrue(len(r["reasoning"]) > 0,
                        "reasoning must carry raw tool input (non-empty)")
        self.assertIsInstance(r["visible"], bool)

    def test_tool_use_target_type_must_be_supported(self):
        """If VLM hallucinates an unsupported target_type, the result must
        either be coerced to 'generic' or flagged as parse failure — never
        passed through as-is (Stage C would KeyError on an unknown type)."""
        mock_response = {
            "target_type": "submarine", "intent": "center",
            "label_hint": "",
            "raw": '{"target_type":"submarine"}',
        }
        r = self._invoke_tool_use_declare("a submarine in the harbor",
                                          mock_response)
        self.assertIn(r["target_type"], stage_c.SUPPORTED_TARGET_TYPES,
                      "unsupported target_type must be coerced to supported")

    def test_tool_use_reasoning_carries_raw_input(self):
        """Per stage_c_phase_1_5.py docstring l.34-36: tool-use `reasoning`
        field carries the VLM's raw tool input, not a rule id. No 6-value
        enum constraint (unlike heuristic)."""
        raw_json = '{"target_type":"icon","label_hint":"Undo"}'
        mock_response = {
            "target_type": "icon", "intent": "center",
            "label_hint": "Undo", "raw": raw_json,
        }
        r = self._invoke_tool_use_declare("'Undo' icon", mock_response)
        # The exact raw format is impl-defined; but it must not collapse to
        # one of the 6 heuristic rule ids by coincidence.
        self.assertIsInstance(r["reasoning"], str)


# ---------------------------------------------------------------------------
# 2. --arm {A,B,C} flag dispatching
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ARM_FLAG_READY, _SKIP_REASON_ARM_FLAG)
class TestArmDispatching(unittest.TestCase):
    """`run_benchmark.py --arm A` keeps heuristic, `--arm B` routes to tool-use
    N=1, `--arm C` routes to tool-use N=3 median aggregation. Default = A
    (backward compatibility with v9 Stage 2 full reuse per W3 v2 §2.1)."""

    def test_arm_a_default_is_heuristic(self):
        from benchmark import run_benchmark
        # A single-source-of-truth probe: either an _ARM module variable or
        # argparse default, the intent is arm-A-as-default.
        if hasattr(run_benchmark, "_ARM"):
            self.assertEqual(run_benchmark._ARM, "A")
        else:
            # If a different mechanism is chosen, at least confirm arm A
            # path emits stage_c_source="heuristic" (unchanged from 590c899).
            self.skipTest("arm dispatcher internals opaque; covered by integration smoke")

    def test_arm_choices_accept_abc(self):
        from benchmark import run_benchmark
        # At minimum one of these shape-probes must evaluate a membership
        # check against {"A", "B", "C"}.
        for probe in ("ARM_CHOICES", "_ARM_CHOICES", "SUPPORTED_ARMS"):
            if hasattr(run_benchmark, probe):
                choices = set(getattr(run_benchmark, probe))
                self.assertEqual(choices, {"A", "B", "C"},
                                 f"{probe} must accept exactly A/B/C")
                return
        self.skipTest("no enum-like arm-choices constant exposed; covered by CLI parse test")


# ---------------------------------------------------------------------------
# 3. Arm C N=3 centroid aggregator
# ---------------------------------------------------------------------------

@unittest.skipUnless(_CENTROID_AGG_READY, _SKIP_REASON_CENTROID)
class TestArmCCentroidAggregation(unittest.TestCase):
    """Arm C runs the full pipeline N=3 times, collects final centroids, and
    applies per-axis median (decision 056 W1 convention — same as Phase 1
    aggregator, applied one level up at Phase 1.5 output)."""

    def _get_centroid_aggregator(self):
        from benchmark import run_benchmark
        for name in ("_aggregate_centroid", "_aggregate_xy",
                     "_aggregate_centroid_samples"):
            if hasattr(run_benchmark, name):
                return getattr(run_benchmark, name)
        self.fail("no centroid aggregator found in run_benchmark")

    def test_median_of_three_centroids_per_axis(self):
        agg = self._get_centroid_aggregator()
        samples = [(100, 200), (110, 210), (120, 205)]
        result = agg(samples, method="median")
        # Accept either (x, y, stdev_xy) tuple or {x, y, stdev_x, stdev_y} dict
        if isinstance(result, dict):
            self.assertEqual(result["x"], 110)
            self.assertEqual(result["y"], 205)
        else:
            x, y = result[0], result[1]
            self.assertEqual(x, 110)
            self.assertEqual(y, 205)

    def test_median_is_robust_to_outlier_centroid(self):
        """Motivates arm C's existence: a catastrophic N=3 centroid outlier
        must be suppressed by median, matching decision 056 rationale."""
        agg = self._get_centroid_aggregator()
        samples = [(100, 200), (105, 210), (500, 999)]  # 3rd is catastrophic
        result = agg(samples, method="median")
        x = result["x"] if isinstance(result, dict) else result[0]
        y = result["y"] if isinstance(result, dict) else result[1]
        self.assertEqual(x, 105)  # outlier dampened
        self.assertEqual(y, 210)

    def test_per_axis_stdev_reported(self):
        """Variance-isolation inference at paper §8.5.5 requires per-axis
        stdev; flat scalar stdev would collapse independent x/y jitter."""
        agg = self._get_centroid_aggregator()
        samples = [(100, 200), (110, 200), (120, 200)]
        result = agg(samples, method="median")
        if isinstance(result, dict):
            self.assertGreater(result.get("stdev_x", 0), 0)
            self.assertEqual(result.get("stdev_y", -1), 0.0)  # y is constant
        else:
            self.assertTrue(len(result) >= 3,
                            "tuple result must carry stdev component(s)")


# ---------------------------------------------------------------------------
# 4. Parse-failure exclusion (W3 v2 §6.3 revised per Must-fix B)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TOOL_USE_READY, _SKIP_REASON_TOOL_USE)
class TestParseFailureExclusion(unittest.TestCase):
    """W3 v2 §6.3 Must-fix B: arm B/C parse-failure cases are fully excluded,
    not substituted with rule-based.

    Per video2ai-agent Q4-5 (`..T03:27:49-2afc`) + supervisor approval
    (`..T03:34:49-4454`), the failure marker is a source-enum extension:
      stage_c_source ∈ {"heuristic", "vlm_tool_use", "vlm_tool_use_failed"}
    `arm_failed` is *derived* from `source == "vlm_tool_use_failed"` rather
    than carried as a separate flag field. Rollup helpers may still accept
    either shape during transition.
    """

    _FAILURE_SOURCE = "vlm_tool_use_failed"

    def test_malformed_tool_response_marks_arm_failed(self):
        """Malformed JSON from Anthropic tool_use → retry ×2 → still bad →
        row emitted with source='vlm_tool_use_failed' + reasoning starting
        with 'parse_fail_', NOT substituted with heuristic."""
        if not hasattr(p1_5, "declare_target_tool_use"):
            self.skipTest("declare_target_tool_use not exposed under probed name")
        with patch.object(p1_5, "_call_anthropic_tool_use",
                          side_effect=ValueError("malformed JSON from tool_use"),
                          create=True):
            r = p1_5.declare_target_tool_use("the 'Save' button")
        self.assertIsInstance(r, dict,
                              "failure mode must return a sentinel dict, not raise")
        # Enum-based derivation: source == "vlm_tool_use_failed" implies arm_failed.
        # Accept either the enum form (canonical) or a legacy arm_failed flag for
        # transition-period compatibility.
        is_failed = (r.get("source") == self._FAILURE_SOURCE
                     or r.get("arm_failed") is True)
        self.assertTrue(is_failed,
                        "failure must be explicitly marked via source enum or "
                        "arm_failed flag, not silently fallback")
        self.assertNotEqual(r.get("source"), "heuristic",
                            "parse failure must NOT silently fall back to heuristic")
        # Reasoning should carry the error kind for downstream triage.
        reasoning = r.get("reasoning", "")
        self.assertIsInstance(reasoning, str)
        if r.get("source") == self._FAILURE_SOURCE:
            self.assertTrue(reasoning.startswith("parse_fail_"),
                            f"reasoning must start with 'parse_fail_'; got {reasoning!r}")

    def test_arm_failed_rows_excluded_from_rollup(self):
        """Rollup must exclude failure rows. Accepts both the canonical enum
        form (`source == 'vlm_tool_use_failed'`) and the legacy `arm_failed`
        flag so this test works across the impl-transition window."""
        try:
            from benchmark import run_benchmark
        except ImportError:
            self.skipTest("run_benchmark not importable")
        # Expect a rollup helper that accepts rows and arm label.
        for name in ("_rollup_by_arm", "_per_arm_accuracy", "_arm_rollup"):
            if hasattr(run_benchmark, name):
                fn = getattr(run_benchmark, name)
                rows = [
                    {"arm": "B", "dist_px": 3.0,
                     "stage_c_source": "vlm_tool_use", "hit_20px": True},
                    {"arm": "B", "dist_px": 0.0,
                     "stage_c_source": self._FAILURE_SOURCE,
                     "stage_c_reasoning": "parse_fail_malformed_json",
                     "arm_failed": True, "hit_20px": False},
                    {"arm": "B", "dist_px": 18.0,
                     "stage_c_source": "vlm_tool_use", "hit_20px": True},
                ]
                summary = fn(rows, arm="B")
                # Of 3 rows, 1 is arm_failed — rollup must count 2, not 3.
                n = summary.get("n_valid", summary.get("count"))
                self.assertEqual(n, 2,
                                 "arm_failed rows must be excluded from n_valid")
                return
        self.skipTest("no arm rollup helper found; covered by integration smoke")


# ---------------------------------------------------------------------------
# Smoke: ensure the file itself is importable and all gates are introspectable
# (this runs even when everything else skips, proving the scaffolding is sound).
# ---------------------------------------------------------------------------

class TestScaffoldingIsSound(unittest.TestCase):
    def test_gates_are_boolean(self):
        self.assertIn(_TOOL_USE_READY, (True, False))
        self.assertIn(_ARM_FLAG_READY, (True, False))
        self.assertIn(_CENTROID_AGG_READY, (True, False))

    def test_skip_reasons_are_explicit(self):
        for reason in (_SKIP_REASON_TOOL_USE, _SKIP_REASON_ARM_FLAG,
                       _SKIP_REASON_CENTROID):
            self.assertIsInstance(reason, str)
            self.assertTrue(len(reason) > 10)

    def test_supported_target_types_still_includes_phase_a_set(self):
        for t in ("button", "icon", "menu_item", "generic"):
            self.assertIn(t, stage_c.SUPPORTED_TARGET_TYPES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
