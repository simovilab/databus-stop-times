"""Tests for models/evaluation/cost.py (roadmap 5.4).

Fast and synthetic throughout: no real model training, no real registry
lookups except where explicitly exercising the graceful-degradation path
against a genuinely missing model_key.
"""

import time

import numpy as np
import pytest

from stoptimes.train.evaluation.cost import (
    LATENCY_QUANTILE_KEYS,
    TRAINING_COST_KEYS,
    TrainingCost,
    cost_accuracy_table,
    latency_quantiles,
    measure_serving_latency,
    time_single_calls,
    training_cost,
)


# ---------------------------------------------------------------------------
# TrainingCost
# ---------------------------------------------------------------------------


def test_training_cost_measures_known_sleep():
    """A known sleep(0.2) should show up as ~0.2s wall time, generously
    tolerant since CI/dev machines are noisy."""
    with training_cost() as tc:
        time.sleep(0.2)

    assert 0.15 <= tc.result["wall_seconds"] <= 0.6


def test_training_cost_result_has_documented_keys():
    with training_cost() as tc:
        time.sleep(0.01)

    assert set(tc.result.keys()) == set(TRAINING_COST_KEYS)
    for key in ("wall_seconds", "cpu_seconds", "peak_rss_bytes", "peak_rss_mb"):
        assert isinstance(tc.result[key], (int, float))
        assert tc.result[key] >= 0


def test_training_cost_context_manager_propagates_exceptions():
    with pytest.raises(ValueError):
        with training_cost():
            raise ValueError("boom")


def test_training_cost_decorator_preserves_return_value_and_reports_cost():
    @TrainingCost.wrap
    def train():
        time.sleep(0.05)
        return {"mae": 12.3}

    out, cost = train()

    assert out == {"mae": 12.3}
    assert set(cost.keys()) == set(TRAINING_COST_KEYS)
    assert cost["wall_seconds"] >= 0.04


# ---------------------------------------------------------------------------
# latency_quantiles
# ---------------------------------------------------------------------------


def test_latency_quantiles_on_known_array():
    # 1..100 ms as seconds
    latencies_s = [i / 1000.0 for i in range(1, 101)]
    q = latency_quantiles(latencies_s)

    assert set(q.keys()) == set(LATENCY_QUANTILE_KEYS)
    assert q["n"] == 100
    assert q["min_ms"] == pytest.approx(1.0)
    assert q["max_ms"] == pytest.approx(100.0)
    assert q["mean_ms"] == pytest.approx(50.5)
    # Cross-check against numpy directly rather than hand-deriving the
    # interpolated percentile values.
    ms = np.asarray(latencies_s) * 1000.0
    assert q["p50_ms"] == pytest.approx(np.percentile(ms, 50))
    assert q["p95_ms"] == pytest.approx(np.percentile(ms, 95))
    assert q["p99_ms"] == pytest.approx(np.percentile(ms, 99))


def test_latency_quantiles_rejects_empty_input():
    with pytest.raises(ValueError):
        latency_quantiles([])


# ---------------------------------------------------------------------------
# time_single_calls
# ---------------------------------------------------------------------------


def test_time_single_calls_separates_cold_from_warm():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.05)  # simulate a slow first ("cold") call
        else:
            time.sleep(0.001)

    result = time_single_calls(fn, n_calls=10)

    assert calls["n"] == 10
    assert result["cold_ms"] >= 40
    assert set(result["warm"].keys()) == set(LATENCY_QUANTILE_KEYS)
    assert result["warm"]["n"] == 9
    # Warm calls were all fast; cold call was clearly slower.
    assert result["warm"]["mean_ms"] < result["cold_ms"]
    assert len(result["all_latencies_ms"]) == 10


def test_time_single_calls_rejects_zero_calls():
    with pytest.raises(ValueError):
        time_single_calls(lambda: None, n_calls=0)


# ---------------------------------------------------------------------------
# measure_serving_latency: graceful degradation
# ---------------------------------------------------------------------------


def test_measure_serving_latency_degrades_gracefully_for_missing_model():
    """A model_key that doesn't exist in the registry must not raise --
    the estimator's own model-not-found error path should be caught and
    surfaced as an 'error' key, so a multi-family benchmark loop can keep
    going for the families that *are* available."""
    result = measure_serving_latency(
        "definitely_not_a_real_model_key_12345", n_calls=3
    )

    assert result["model_key"] == "definitely_not_a_real_model_key_12345"
    assert "error" in result
    assert "cold_ms" not in result


# ---------------------------------------------------------------------------
# cost_accuracy_table
# ---------------------------------------------------------------------------


def test_cost_accuracy_table_sorts_by_mae_and_requires_keys():
    rows = [
        {"model": "b", "mae_seconds": 20.0, "train_seconds": 5.0, "p95_inference_ms": 3.0},
        {"model": "a", "mae_seconds": 10.0, "train_seconds": 50.0, "p95_inference_ms": 1.0},
    ]
    df = cost_accuracy_table(rows)

    assert list(df["model"]) == ["a", "b"]
    assert list(df.columns) == ["model", "mae_seconds", "train_seconds", "p95_inference_ms"]


def test_cost_accuracy_table_requires_documented_keys():
    with pytest.raises(ValueError):
        cost_accuracy_table([{"model": "a", "mae_seconds": 1.0}])
