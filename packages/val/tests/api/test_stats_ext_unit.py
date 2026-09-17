"""tests/api/test_stats_ext_unit.py — unit tests for stoptimes.val.api.stats_ext.

Pure-function tests: no HTTP, no engine, no storage — build ValidationResult
rows and RunSummary/LatencyStats objects directly.
"""

from __future__ import annotations

from stoptimes.val.api.stats_ext import (
    STATS_SEED,
    compute_calibration,
    compute_significance,
    compute_stats_extension,
)
from stoptimes.core.domain.models import ValidationResult
from stoptimes.val.engine.runtime import RunSummary
from stoptimes.val.provenance import LatencyStats


def _result(
    predictor_id: str,
    trip_key: str,
    stop_id: str,
    error_s: float,
    horizon_s: float = 90.0,
    stop_sequence: int = 1,
) -> ValidationResult:
    return ValidationResult(
        trip_key=trip_key,
        predictor_id=predictor_id,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        error_s=error_s,
        horizon_s=horizon_s,
        predicted_arrival_epoch=1_000.0 + error_s,
        actual_arrival_epoch=1_000.0,
        made_at_epoch=1_000.0 - horizon_s,
    )


# ── pairing / averaging on (trip, stop) ─────────────────────────────────────────


def test_significance_pairs_and_averages_by_trip_stop() -> None:
    # predictor "a" has two observations at the same (trip, stop) key —
    # they must be averaged to a single |error| of 20 before differencing.
    results = {
        "a": [
            _result("a", "t1", "s1", error_s=10.0),
            _result("a", "t1", "s1", error_s=30.0),
            _result("a", "t2", "s1", error_s=5.0),
        ],
        "b": [
            _result("b", "t1", "s1", error_s=0.0),
            _result("b", "t2", "s1", error_s=5.0),
        ],
    }
    sig = compute_significance(results)
    # avg|error| for a@(t1,s1) = (10+30)/2 = 20; b@(t1,s1) = 0 -> diff 20
    # a@(t2,s1) = 5; b@(t2,s1) = 5 -> diff 0
    # diffs = [20, 0] -> not degenerate (n=2, not all-equal)
    assert {row["test"] for row in sig} == {"wilcoxon", "diebold_mariano"}
    for row in sig:
        assert row["a"] == "a"
        assert row["b"] == "b"
        assert row["n"] == 2
        assert abs(row["diff_mean"] - 10.0) < 1e-9


def test_significance_only_uses_shared_trip_stop_keys() -> None:
    results = {
        "a": [_result("a", "t1", "s1", error_s=10.0), _result("a", "t3", "s1", error_s=1.0)],
        "b": [_result("b", "t1", "s1", error_s=0.0), _result("b", "t2", "s1", error_s=1.0)],
    }
    sig = compute_significance(results)
    # Only (t1, s1) is shared -> n=1 -> degenerate (n<2) -> skipped entirely.
    assert sig == []


# ── degenerate-skip ──────────────────────────────────────────────────────────────


def test_significance_skips_all_equal_diffs() -> None:
    results = {
        "a": [
            _result("a", "t1", "s1", error_s=10.0),
            _result("a", "t2", "s1", error_s=10.0),
        ],
        "b": [
            _result("b", "t1", "s1", error_s=0.0),
            _result("b", "t2", "s1", error_s=0.0),
        ],
    }
    sig = compute_significance(results)
    # diffs = [10, 10] -> all-equal -> degenerate -> skipped.
    assert sig == []


def test_significance_skips_n_less_than_2() -> None:
    results = {
        "a": [_result("a", "t1", "s1", error_s=10.0)],
        "b": [_result("b", "t1", "s1", error_s=0.0)],
    }
    sig = compute_significance(results)
    assert sig == []


def test_significance_single_predictor_yields_no_pairs() -> None:
    results = {"a": [_result("a", "t1", "s1", error_s=10.0)]}
    assert compute_significance(results) == []


# ── Holm-per-family ────────────────────────────────────────────────────────────


def test_significance_holm_correction_applied_within_each_family() -> None:
    # Three predictors -> three pairs per family; verify p_adjusted differs
    # from raw p_value in general and reject is a valid bool per family.
    results = {
        "a": [_result("a", f"t{i}", "s1", error_s=float(i)) for i in range(10)],
        "b": [_result("b", f"t{i}", "s1", error_s=1.5 * i + 5.0) for i in range(10)],
        "c": [_result("c", f"t{i}", "s1", error_s=2.0 * i - 1.0) for i in range(10)],
    }
    sig = compute_significance(results)
    by_family: dict[str, list[dict]] = {}
    for row in sig:
        by_family.setdefault(row["test"], []).append(row)

    for family_rows in by_family.values():
        pairs = {(r["a"], r["b"]) for r in family_rows}
        assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}
        for row in family_rows:
            assert row["p_adjusted"] >= row["p_value"] - 1e-9
            assert 0.0 <= row["p_adjusted"] <= 1.0


# ── calibration ──────────────────────────────────────────────────────────────────


def test_compute_calibration_empty_results() -> None:
    assert compute_calibration([]) == []


def test_compute_calibration_returns_bins_for_run() -> None:
    results = [_result("a", f"t{i}", "s1", error_s=5.0, horizon_s=90.0) for i in range(5)]
    bins = compute_calibration(results)
    assert len(bins) > 0
    for b in bins:
        assert b["n"] == 5
        assert b["band_label"] == "<2m"


# ── latency-from-summary parsing ─────────────────────────────────────────────────


def test_stats_extension_latency_present_for_known_predictor() -> None:
    results = [_result("a", "t1", "s1", error_s=5.0)]
    latency_stats = LatencyStats(count=3, p50_s=0.1, p95_s=0.2, p99_s=0.3)
    summary = RunSummary(inference_latency={"a": latency_stats})
    ext = compute_stats_extension(results, summary)
    assert ext["per_predictor_ext"]["a"]["latency"] == {
        "count": 3,
        "p50_s": 0.1,
        "p95_s": 0.2,
        "p99_s": 0.3,
    }


def test_stats_extension_latency_null_when_missing() -> None:
    results = [_result("a", "t1", "s1", error_s=5.0)]
    ext = compute_stats_extension(results, summary=None)
    assert ext["per_predictor_ext"]["a"]["latency"] is None

    ext2 = compute_stats_extension(results, summary=RunSummary())
    assert ext2["per_predictor_ext"]["a"]["latency"] is None


# ── mae_ci / seed ──────────────────────────────────────────────────────────────


def test_stats_extension_mae_ci_shape_and_seed() -> None:
    results = [_result("a", f"t{i}", "s1", error_s=float(i)) for i in range(5)]
    ext = compute_stats_extension(results, summary=None)
    assert ext["seed"] == STATS_SEED == 0
    ci = ext["per_predictor_ext"]["a"]["mae_ci"]
    assert set(ci) == {"lo", "hi", "ci", "n_boot"}
    assert ci["ci"] == 0.95
    assert ci["n_boot"] == 1000
    assert ci["lo"] <= ci["hi"]


# ── caching (computed once) — RunManager-level, verified via identity in test_stats_ext.py ──


def test_compute_stats_extension_deterministic_given_seed() -> None:
    results = [_result("a", f"t{i}", "s1", error_s=float(i) % 3) for i in range(20)]
    ext1 = compute_stats_extension(results, summary=None)
    ext2 = compute_stats_extension(results, summary=None)
    assert ext1["per_predictor_ext"]["a"]["mae_ci"] == ext2["per_predictor_ext"]["a"]["mae_ci"]
