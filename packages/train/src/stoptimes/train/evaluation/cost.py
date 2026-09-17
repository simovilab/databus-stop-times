"""Cost instrumentation (roadmap 5.4).

Two independent measurements, both meant to be stamped into model metadata
or dropped straight into a report table:

1. **Training cost** -- ``training_cost`` is a context manager (also usable
   as a decorator) that wraps an *arbitrary* training call from the outside
   and reports wall-clock seconds, CPU seconds, and peak RSS. It does not
   require any change to a trainer's own code.

2. **Inference latency measured through the serving path** --
   ``measure_serving_latency`` calls ``eta_service.estimator
   .estimate_stop_times`` the way production actually calls it: once per
   vehicle position, for a handful of upcoming stops -- not
   ``model.predict()`` vectorised over a whole test set. It reports the
   first call (cold: model load + unpickle + import warmup all included)
   separately from the repeated warm calls (p50/p95/p99/mean), because for
   a serving system that distinction is a real operational number, not
   noise to discard. Note the registry has no in-process model cache (see
   ``models/common/registry.py::load_model``), so even "warm" calls
   re-unpickle the model from disk on every call -- this module reports
   what's actually happening, it doesn't paper over it.

Also provided: ``cost_accuracy_table``, a small Pareto-view helper (MAE vs.
training seconds vs. p95 inference latency) for roadmap 6.3.

Nothing here is trainer-specific; it's all wrapping/measuring from outside.
"""

from __future__ import annotations

import functools
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Documented key sets, used both to build results and to assert their shape
# in tests -- keep these in sync with the dict literals below.
TRAINING_COST_KEYS = (
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_bytes",
    "peak_rss_mb",
    "started_at",
    "finished_at",
)

LATENCY_QUANTILE_KEYS = ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "min_ms", "max_ms", "n")


# ---------------------------------------------------------------------------
# Training cost
# ---------------------------------------------------------------------------


def _peak_rss_bytes(ru: "resource.struct_rusage") -> int:
    """Normalize ``ru_maxrss`` across platforms.

    macOS (darwin) reports ru_maxrss in bytes; Linux reports it in
    kilobytes. Both report the same across a process (it's the OS's peak
    value observed at the point ``getrusage`` was called, not a delta), so
    a "delta" here is really "peak seen by the end of the block" -- callers
    should not expect this to be zero for a trivial block, only that a
    training run visibly larger than the baseline shows a larger number.
    """
    if sys.platform == "darwin":
        return int(ru.ru_maxrss)
    return int(ru.ru_maxrss) * 1024


@dataclass
class TrainingCost:
    """Wall-clock / CPU / memory cost of one training run.

    Use as a context manager::

        with TrainingCost() as tc:
            train_xgboost(...)
        tc.result  # -> dict, ready to stamp into model metadata

    or as a decorator::

        @TrainingCost.wrap
        def train():
            return train_xgboost(...)

        output, cost = train()
    """

    result: Dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False)
    _ru0: Optional["resource.struct_rusage"] = field(default=None, repr=False)

    def __enter__(self) -> "TrainingCost":
        self._ru0 = resource.getrusage(resource.RUSAGE_SELF)
        self._t0 = time.perf_counter()
        self._started_at = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        t1 = time.perf_counter()
        finished_at = time.time()
        ru1 = resource.getrusage(resource.RUSAGE_SELF)

        wall_seconds = t1 - self._t0
        cpu0 = self._ru0.ru_utime + self._ru0.ru_stime
        cpu1 = ru1.ru_utime + ru1.ru_stime
        cpu_seconds = max(0.0, cpu1 - cpu0)
        peak_rss_bytes = _peak_rss_bytes(ru1)

        self.result = {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mb": peak_rss_bytes / (1024 * 1024),
            "started_at": self._started_at,
            "finished_at": finished_at,
        }
        # Never swallow the training exception; just fall through so it
        # propagates normally (returning falsy/None from __exit__ does that).

    @staticmethod
    def wrap(fn: Callable) -> Callable:
        """Decorator form: returns ``(fn(...), cost_dict)`` instead of just
        ``fn(...)``. Does not modify ``fn`` -- purely an external wrapper,
        so trainers need no changes to be measured this way."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            with TrainingCost() as tc:
                out = fn(*args, **kwargs)
            return out, tc.result

        return wrapper


def training_cost() -> TrainingCost:
    """Convenience constructor: ``with training_cost() as tc: ...``."""
    return TrainingCost()


# ---------------------------------------------------------------------------
# Latency quantiles (generic, reusable)
# ---------------------------------------------------------------------------


def latency_quantiles(latencies_seconds: Sequence[float]) -> Dict[str, float]:
    """p50/p95/p99/mean/min/max/n (all times in milliseconds) for a sequence
    of per-call latencies measured in seconds.

    Uses ``numpy.percentile`` default (linear) interpolation -- the same
    convention used elsewhere in this codebase (models/evaluation/stats.py).
    """
    arr = np.asarray(latencies_seconds, dtype=float)
    if arr.size == 0:
        raise ValueError("latency_quantiles() received an empty sequence")
    ms = arr * 1000.0
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "n": int(arr.size),
    }


def time_single_calls(
    fn: Callable[[], Any], n_calls: int, warmup: int = 0
) -> Dict[str, Any]:
    """Time ``n_calls`` invocations of a zero-arg callable, one at a time
    (never batched -- this is the point: it measures the single-call path).

    Returns a dict with:
      - ``cold``: latency (ms) of the very first call *after* any
        ``warmup`` calls have already run (so "cold" reflects whatever
        state the caller intended -- e.g. warmup=0 makes the first call
        genuinely cold w.r.t. this process/model).
      - ``warm``: quantile dict (see ``latency_quantiles``) over the
        remaining calls.
      - ``all_latencies_ms``: every measured latency, in call order.
    """
    if n_calls < 1:
        raise ValueError("n_calls must be >= 1")

    for _ in range(warmup):
        fn()

    latencies: List[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - t0)

    cold_ms = latencies[0] * 1000.0
    warm = latencies[1:] if len(latencies) > 1 else latencies
    return {
        "cold_ms": cold_ms,
        "warm": latency_quantiles(warm),
        "all_latencies_ms": [l * 1000.0 for l in latencies],
    }


# ---------------------------------------------------------------------------
# Inference latency through the serving path
# ---------------------------------------------------------------------------


def _default_vehicle_position() -> Dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "vehicle_id": "cost_probe_vehicle",
        "route": "unknown",
        "lat": 9.9281,
        "lon": -84.0907,
        "speed": 8.0,  # m/s
        "heading": 90,
        "current_status": "IN_TRANSIT_TO",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _default_upcoming_stops(n: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "stop_id": f"cost_probe_stop_{i}",
            "stop_sequence": i + 1,
            "total_stop_sequence": n + 2,
            "lat": 9.9281 + 0.0025 * (i + 1),
            "lon": -84.0907 + 0.0025 * (i + 1),
        }
        for i in range(n)
    ]


def measure_serving_latency(
    model_key: str,
    *,
    n_calls: int = 200,
    max_stops: int = 3,
    vehicle_position: Optional[Dict[str, Any]] = None,
    upcoming_stops: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Measure ``eta_service.estimator.estimate_stop_times`` latency for one
    pinned model, single call at a time -- this is the production call
    shape (one vehicle position, a handful of upcoming stops), not a
    vectorised ``model.predict(big_df)``.

    Degrades gracefully: any exception raised while importing or calling
    the estimator (e.g. the model family's predict module is unavailable,
    or the model_key isn't in the registry) is caught and returned as
    ``{"model_key": ..., "error": "..."}`` rather than propagating, so a
    caller benchmarking several model families doesn't lose the rest of
    the table because one family is broken/unavailable.
    """
    vp = vehicle_position or _default_vehicle_position()
    stops = upcoming_stops or _default_upcoming_stops(max_stops)

    try:
        from eta_service.estimator import estimate_stop_times
    except Exception as e:  # pragma: no cover - import-time failure path
        return {
            "model_key": model_key,
            "error": f"estimator import failed: {e}",
        }

    def _call() -> None:
        result = estimate_stop_times(
            vp,
            stops,
            model_key=model_key,
            max_stops=max_stops,
        )
        if result.get("error"):
            raise RuntimeError(result["error"])
        for pred in result["predictions"]:
            if pred.get("error"):
                raise RuntimeError(pred["error"])

    try:
        timed = time_single_calls(_call, n_calls=n_calls, warmup=0)
    except Exception as e:
        return {
            "model_key": model_key,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }

    return {
        "model_key": model_key,
        "n_calls": n_calls,
        "cold_ms": timed["cold_ms"],
        "warm": timed["warm"],
    }


def measure_serving_latency_by_family(
    model_keys: Dict[str, str],
    *,
    n_calls: int = 200,
    max_stops: int = 3,
) -> pd.DataFrame:
    """Run ``measure_serving_latency`` for several {family_name: model_key}
    pairs and return one row per family. Families that error out get a row
    with NaN latency columns and an ``error`` string rather than being
    dropped -- the caller can see which families were skipped and why.
    """
    rows = []
    for family, model_key in model_keys.items():
        r = measure_serving_latency(model_key, n_calls=n_calls, max_stops=max_stops)
        if "error" in r:
            rows.append({
                "family": family,
                "model_key": model_key,
                "cold_ms": np.nan,
                "p50_ms": np.nan,
                "p95_ms": np.nan,
                "p99_ms": np.nan,
                "mean_ms": np.nan,
                "n": 0,
                "error": r["error"],
            })
        else:
            warm = r["warm"]
            rows.append({
                "family": family,
                "model_key": model_key,
                "cold_ms": r["cold_ms"],
                "p50_ms": warm["p50_ms"],
                "p95_ms": warm["p95_ms"],
                "p99_ms": warm["p99_ms"],
                "mean_ms": warm["mean_ms"],
                "n": warm["n"],
                "error": None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost / accuracy Pareto table (roadmap 6.3)
# ---------------------------------------------------------------------------


def cost_accuracy_table(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Build the MAE vs. training-seconds vs. p95-inference-latency table.

    Each row of ``rows`` is a dict with at least ``model``, ``mae_seconds``,
    ``train_seconds``, ``p95_inference_ms`` -- extra keys pass through
    untouched. Sorted by MAE ascending (best model first), matching
    ``models/evaluation/stats.py::compare_all``'s convention.
    """
    required = {"model", "mae_seconds", "train_seconds", "p95_inference_ms"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"cost_accuracy_table row missing keys: {missing}")
    df = pd.DataFrame(rows)
    return df.sort_values("mae_seconds", kind="stable").reset_index(drop=True)
