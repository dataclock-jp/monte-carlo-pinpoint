"""
test_phase1_majority_voting.py

Phase B Workstream 1 (decision 056) unit + integration tests:
- `_aggregate_axis` helper: per-axis median/mean with outlier rejection
- `--phase1-samples N` flag wiring + 3 new result fields

The aggregate function is the core mechanism for VLM variance reduction.
Integration tests verify the flag is propagated and that Phase 1 N-sample
loops produce the expected schema (phase_1_samples, phase_1_sample_stdev_xy,
phase_1_aggregation).
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


class TestAggregateAxis(unittest.TestCase):
    """Unit tests for `_aggregate_axis` helper (core mechanism)."""

    def test_median_odd_samples(self):
        agg, stdev = run_benchmark._aggregate_axis([10, 20, 30], "median")
        self.assertEqual(agg, 20)
        self.assertGreater(stdev, 0.0)

    def test_median_with_outlier(self):
        """Median must be robust to a catastrophic outlier (decision 056 rationale)."""
        agg, _ = run_benchmark._aggregate_axis([100, 105, 500], "median")
        self.assertEqual(agg, 105)  # outlier 500 dampened

    def test_median_even_samples(self):
        agg, _ = run_benchmark._aggregate_axis([10, 20, 30, 40], "median")
        self.assertEqual(agg, 25)  # statistics.median returns mean of middle two

    def test_mean(self):
        agg, stdev = run_benchmark._aggregate_axis([10, 20, 30], "mean")
        self.assertEqual(agg, 20)
        self.assertGreater(stdev, 0.0)

    def test_all_invalid_returns_minus_one(self):
        """Grid-failed samples (all < 0) must propagate the failure signal."""
        agg, stdev = run_benchmark._aggregate_axis([-1, -1, -1], "median")
        self.assertEqual(agg, -1)
        self.assertEqual(stdev, 0.0)

    def test_mixed_valid_invalid_uses_valid_only(self):
        """Partial failures: aggregate only the valid samples."""
        agg, _ = run_benchmark._aggregate_axis([-1, 100, 110, -1], "median")
        self.assertEqual(agg, 105)

    def test_single_valid_sample_stdev_zero(self):
        agg, stdev = run_benchmark._aggregate_axis([42, -1, -1], "median")
        self.assertEqual(agg, 42)
        self.assertEqual(stdev, 0.0)

    def test_bit_identical_samples(self):
        """Cross-run reproducibility: identical inputs yield identical outputs."""
        agg1, s1 = run_benchmark._aggregate_axis([100, 100, 100], "median")
        agg2, s2 = run_benchmark._aggregate_axis([100, 100, 100], "median")
        self.assertEqual(agg1, agg2)
        self.assertEqual(s1, s2)
        self.assertEqual(agg1, 100)
        self.assertEqual(s1, 0.0)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            run_benchmark._aggregate_axis([1, 2, 3], "unknown")


class TestPhase1SamplesIntegration(unittest.TestCase):
    """--phase1-samples N wiring + new schema fields."""

    def setUp(self):
        self._saved_samples = run_benchmark._PHASE1_SAMPLES
        self._saved_agg = run_benchmark._PHASE1_AGGREGATION
        self._saved_stage_c = run_benchmark._USE_STAGE_C
        run_benchmark._USE_STAGE_C = False

    def tearDown(self):
        run_benchmark._PHASE1_SAMPLES = self._saved_samples
        run_benchmark._PHASE1_AGGREGATION = self._saved_agg
        run_benchmark._USE_STAGE_C = self._saved_stage_c

    def _run_with_phase1_stub(self, samples_per_axis_x, samples_per_axis_y,
                              N=3, aggregation="median"):
        """Run method_pinpoint with deterministic Phase 1 samples."""
        run_benchmark._PHASE1_SAMPLES = N
        run_benchmark._PHASE1_AGGREGATION = aggregation

        call_count = [0]

        def _stub(work_img, target, axis, max_steps, precision,
                  api_key, model, query_counter=None):
            if query_counter is not None:
                query_counter[0] += 1
            idx = call_count[0] // 2  # pairs of (x, y)
            call_count[0] += 1
            if axis == "x":
                return (samples_per_axis_x[idx], 0)
            return (samples_per_axis_y[idx], 0)

        # Stub Phase 1.5 + Phase 2 to exit quickly
        def _stub_anchor(b64, target, api_key, model):
            return ("test anchor", True)

        def _stub_majority(b64, anchor, n_dots, api_key, model, votes,
                           query_counter=None):
            if query_counter is not None:
                query_counter[0] += 1
            return (None, False)

        from contextlib import ExitStack
        img = Image.new("RGB", (800, 600), "white")
        with ExitStack() as stack:
            stack.enter_context(patch(
                "benchmark.run_benchmark._upscale_for_grid",
                side_effect=lambda img, target_width=2560: img))
            stack.enter_context(patch(
                "benchmark.run_benchmark.grid_search_axis", side_effect=_stub))
            stack.enter_context(patch(
                "benchmark.run_benchmark.pinpoint_declare_anchor",
                side_effect=_stub_anchor))
            stack.enter_context(patch(
                "benchmark.run_benchmark.pinpoint_ask_majority",
                side_effect=_stub_majority))
            return run_benchmark.method_pinpoint(
                img, "the test target", api_key="stub", model="stub",
                max_rounds=1,
            )

    def test_n1_baseline_fields_present(self):
        """N=1 (default v8 baseline): 3 new fields present with degenerate values."""
        r = self._run_with_phase1_stub([362], [221], N=1)
        self.assertIn("phase_1_samples", r)
        self.assertIn("phase_1_sample_stdev_xy", r)
        self.assertIn("phase_1_aggregation", r)
        self.assertEqual(len(r["phase_1_samples"]), 1)
        self.assertEqual(r["phase_1_samples"][0], [362, 221])
        self.assertEqual(r["phase_1_sample_stdev_xy"], [0.0, 0.0])
        self.assertEqual(r["phase_1_aggregation"], "none")

    def test_n3_median_aggregation(self):
        """N=3 with median: final (x, y) is median of samples."""
        r = self._run_with_phase1_stub(
            [100, 110, 200], [50, 55, 500],  # x median=110, y median=55
            N=3, aggregation="median")
        self.assertEqual(len(r["phase_1_samples"]), 3)
        self.assertEqual(r["phase_1_aggregation"], "median")
        self.assertEqual(r["x"], 110)
        self.assertEqual(r["y"], 55)
        # stdev should be non-zero
        self.assertGreater(r["phase_1_sample_stdev_xy"][0], 0.0)
        self.assertGreater(r["phase_1_sample_stdev_xy"][1], 0.0)

    def test_n3_samples_preserved_in_results(self):
        """Raw samples are preserved for provenance + stdev analysis."""
        r = self._run_with_phase1_stub(
            [100, 110, 120], [50, 55, 60],
            N=3, aggregation="median")
        samples = r["phase_1_samples"]
        self.assertEqual(samples[0], [100, 50])
        self.assertEqual(samples[1], [110, 55])
        self.assertEqual(samples[2], [120, 60])

    def test_n3_with_grid_fail_handles_gracefully(self):
        """If one sample's x fails (-1), aggregation uses the remaining valid samples."""
        r = self._run_with_phase1_stub(
            [-1, 100, 110], [50, 55, 60],
            N=3, aggregation="median")
        # x median of [100, 110] = 105
        # y median of [50, 55, 60] = 55
        self.assertEqual(r["x"], 105)
        self.assertEqual(r["y"], 55)
        # Raw samples: first one has None for x (failure)
        self.assertEqual(r["phase_1_samples"][0], [None, 50])

    def test_bit_identical_aggregation(self):
        """Same inputs → same outputs → cross-run reproducibility preserved."""
        r1 = self._run_with_phase1_stub(
            [100, 105, 110], [50, 55, 60], N=3, aggregation="median")
        r2 = self._run_with_phase1_stub(
            [100, 105, 110], [50, 55, 60], N=3, aggregation="median")
        self.assertEqual(r1["x"], r2["x"])
        self.assertEqual(r1["y"], r2["y"])
        self.assertEqual(r1["phase_1_samples"], r2["phase_1_samples"])
        self.assertEqual(r1["phase_1_sample_stdev_xy"],
                         r2["phase_1_sample_stdev_xy"])


if __name__ == "__main__":
    unittest.main()
