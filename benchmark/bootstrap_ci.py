"""
bootstrap_ci.py

Bootstrap confidence-interval helpers for W3 Stage 1/2 variance-isolation
reporting (decision 062 candidate / W3 plan v2 §5.4 Must-fix C).

Non-Gaussian heavy-tail distance distributions (catastrophic misses at >100px
coexisting with ≤5px path-A hits) violate normal-theory CI coverage. W3 v2
mandates percentile bootstrap (1000 iter) for:

  * per-arm summary statistics (mean / median / success-rate at each threshold)
  * arm-to-arm delta statistics (routing_variance_delta, aggregation_variance_delta)
  * per-category rollup intervals for paper §8.5.5 table

Pure numpy, stdlib-clean, stateless. Consumer side (Stage 1/2 report) imports
the three public helpers and feeds numpy arrays of per-case distances.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


_DEFAULT_N_BOOT = 1000
_DEFAULT_CI_PCT = 95.0


def _resolve_ci_bounds(ci_pct: float) -> tuple[float, float]:
    """Return (lower_percentile, upper_percentile) for a two-sided ci_pct CI.

    ci_pct=95 → (2.5, 97.5); ci_pct=90 → (5.0, 95.0)."""
    if not (0.0 < ci_pct < 100.0):
        raise ValueError(f"ci_pct must be in (0, 100); got {ci_pct}")
    tail = (100.0 - ci_pct) / 2.0
    return (tail, 100.0 - tail)


def bootstrap_ci(
    data: Iterable[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = _DEFAULT_N_BOOT,
    ci_pct: float = _DEFAULT_CI_PCT,
    seed: int | None = None,
) -> dict:
    """Percentile bootstrap CI for a scalar statistic of a 1-D sample.

    Parameters
    ----------
    data : iterable of float
        Per-case measurements (e.g. distance in px for one arm).
    statistic_fn : callable ndarray -> float
        The point statistic (np.mean, np.median, or a custom lambda such as
        `lambda x: (x <= 20).mean()` for ≤20px success-rate).
    n_boot : int
        Resample count. Default 1000 per W3 plan v2 §5.4.
    ci_pct : float
        Two-sided coverage (e.g. 95 for 95% CI).
    seed : int | None
        Optional numpy seed for reproducibility. None → uses default Generator.

    Returns
    -------
    dict with keys:
        point_estimate : float
        ci_lower, ci_upper : float (percentile bounds)
        ci_pct, n_boot : int
        method : "percentile"
        n : sample size
    """
    arr = np.asarray(list(data), dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError("bootstrap_ci requires at least 1 sample")

    rng = np.random.default_rng(seed)
    # Vectorized resampling: draw n_boot × n indices in one shot.
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = arr[idx]  # shape (n_boot, n)

    # Apply statistic_fn along axis=1 per-resample.
    # For np.mean/np.median we can pass axis=1 directly; for custom fns we
    # iterate. Detect the fast path by trying once.
    try:
        boot_stats = statistic_fn(resamples) if statistic_fn in (np.mean, np.median) \
            else np.asarray([statistic_fn(r) for r in resamples], dtype=float)
        # np.mean/np.median with a 2D input collapse across all axes by default;
        # we need the axis=1 variant.
        if statistic_fn is np.mean:
            boot_stats = np.mean(resamples, axis=1)
        elif statistic_fn is np.median:
            boot_stats = np.median(resamples, axis=1)
    except TypeError:
        boot_stats = np.asarray([statistic_fn(r) for r in resamples], dtype=float)

    lo_pct, hi_pct = _resolve_ci_bounds(ci_pct)
    ci_lower = float(np.percentile(boot_stats, lo_pct))
    ci_upper = float(np.percentile(boot_stats, hi_pct))
    point = float(statistic_fn(arr))

    return {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_pct": ci_pct,
        "n_boot": n_boot,
        "method": "percentile",
        "n": int(n),
    }


def bootstrap_ci_paired_delta(
    data_a: Iterable[float],
    data_b: Iterable[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = _DEFAULT_N_BOOT,
    ci_pct: float = _DEFAULT_CI_PCT,
    seed: int | None = None,
) -> dict:
    """Paired-sample bootstrap CI for the delta statistic_fn(data_b) -
    statistic_fn(data_a).

    Paired resampling (same row indices for both arms) preserves per-case
    correlation, which is critical for the W3 cross-arm comparison: same
    50 cases × 3 arms, so arm-to-arm deltas should sample case indices jointly.

    Returns the same dict schema as bootstrap_ci with `point_estimate` being
    the delta (b - a), plus `point_a` and `point_b` for the individual arms.
    """
    arr_a = np.asarray(list(data_a), dtype=float)
    arr_b = np.asarray(list(data_b), dtype=float)
    if arr_a.size != arr_b.size:
        raise ValueError(
            f"paired bootstrap requires equal-length arrays; "
            f"got {arr_a.size} vs {arr_b.size}"
        )
    n = arr_a.size
    if n == 0:
        raise ValueError("paired bootstrap requires at least 1 sample")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples_a = arr_a[idx]
    resamples_b = arr_b[idx]

    if statistic_fn is np.mean:
        stats_a = np.mean(resamples_a, axis=1)
        stats_b = np.mean(resamples_b, axis=1)
    elif statistic_fn is np.median:
        stats_a = np.median(resamples_a, axis=1)
        stats_b = np.median(resamples_b, axis=1)
    else:
        stats_a = np.asarray([statistic_fn(r) for r in resamples_a], dtype=float)
        stats_b = np.asarray([statistic_fn(r) for r in resamples_b], dtype=float)

    delta_boot = stats_b - stats_a
    lo_pct, hi_pct = _resolve_ci_bounds(ci_pct)
    ci_lower = float(np.percentile(delta_boot, lo_pct))
    ci_upper = float(np.percentile(delta_boot, hi_pct))

    point_a = float(statistic_fn(arr_a))
    point_b = float(statistic_fn(arr_b))

    return {
        "point_estimate": point_b - point_a,
        "point_a": point_a,
        "point_b": point_b,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_pct": ci_pct,
        "n_boot": n_boot,
        "method": "percentile_paired",
        "n": int(n),
    }


def ci_overlap(result_a: dict, result_b: dict) -> bool:
    """Return True if two bootstrap_ci result dicts have overlapping intervals.

    Used for W3 v2 §5.4 STRONG GO criterion: routing_variance_delta and
    aggregation_variance_delta both have non-overlapping CI between arms.
    """
    return not (result_a["ci_upper"] < result_b["ci_lower"]
                or result_b["ci_upper"] < result_a["ci_lower"])


def success_rate_at(threshold_px: float) -> Callable[[np.ndarray], float]:
    """Factory: return a statistic_fn computing the fraction of distances
    ≤ threshold_px. Usage:

        bootstrap_ci(distances, statistic_fn=success_rate_at(20))
    """
    def _fn(arr: np.ndarray) -> float:
        return float(np.mean(arr <= threshold_px))
    _fn.__name__ = f"success_rate_le_{int(threshold_px)}px"
    return _fn
