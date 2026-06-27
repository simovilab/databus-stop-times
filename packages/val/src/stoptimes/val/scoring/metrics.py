"""Metrics (pandas). Phase P5.

compute_metrics(results) -> MetricsReport

MetricsReport contains:
- per_predictor: MAE, RMSE, MAPE (|error_s| / horizon_s × 100), bias, count
- horizon_buckets: per (predictor × bucket) MAE / bias / count
  Buckets (fixed edges in seconds): <2 min, 2–5 min, 5–10 min, >10 min
- convergence: per (trip_key, stop_id, predictor_id) — does |error_s| decrease
  monotonically as horizon shrinks (prediction series sorted horizon high→low)?
  Also reports the settling_horizon_s: first horizon_s where |error_s| < 60 s.
- head_to_head: for stops predicted by >1 predictor, win-rate (smallest |error_s|)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from etaval.domain.models import ValidationResult

# ── Horizon bucket edges (seconds) ────────────────────────────────────────────

_BUCKET_LABELS: list[str] = ["<2m", "2-5m", "5-10m", ">10m"]
_BUCKET_BREAKPOINTS: list[float] = [0.0, 120.0, 300.0, 600.0, float("inf")]


def _assign_bucket(horizon_s: float) -> str:
    """Map a horizon in seconds to a human-readable bucket label."""
    if horizon_s < 120.0:
        return "<2m"
    if horizon_s < 300.0:
        return "2-5m"
    if horizon_s < 600.0:
        return "5-10m"
    return ">10m"


# ── Result dataclasses ─────────────────────────────────────────────────────────


@dataclass
class PerPredictorMetrics:
    predictor_id: str
    mae: float            # mean |error_s|
    rmse: float           # root-mean-square error_s
    mape: float           # mean |error_s| / horizon_s × 100  (%)
    bias: float           # mean signed error_s  (+ = late, − = early)
    count: int


@dataclass
class HorizonBucket:
    label: str
    predictor_id: str
    mae: float
    bias: float
    count: int


@dataclass
class ConvergenceSeries:
    trip_key: str
    stop_id: str
    predictor_id: str
    is_converging: bool           # |error_s| monotonically non-increasing as horizon ↓
    settling_horizon_s: float | None  # first horizon_s where |error_s| < 60 s, or None


@dataclass
class WinRate:
    predictor_id: str
    wins: int
    total: int                    # contested stops (predicted by ≥2 predictors)
    win_fraction: float


@dataclass
class MetricsReport:
    """Structured analysis report returned by :func:`compute_metrics`."""

    per_predictor: list[PerPredictorMetrics] = field(default_factory=list)
    horizon_buckets: list[HorizonBucket] = field(default_factory=list)
    convergence: list[ConvergenceSeries] = field(default_factory=list)
    head_to_head: list[WinRate] = field(default_factory=list)

    def to_records(self) -> list[dict]:
        """Flat list of per-predictor summary dicts (for CLI rendering)."""
        return [
            {
                "predictor_id": m.predictor_id,
                "mae": m.mae,
                "rmse": m.rmse,
                "mape": m.mape,
                "bias": m.bias,
                "count": m.count,
            }
            for m in self.per_predictor
        ]

    def summary(self) -> dict:
        """Nested dict representation of the full report."""
        return {
            "per_predictor": self.to_records(),
            "horizon_buckets": [
                {
                    "label": b.label,
                    "predictor_id": b.predictor_id,
                    "mae": b.mae,
                    "bias": b.bias,
                    "count": b.count,
                }
                for b in self.horizon_buckets
            ],
            "convergence": [
                {
                    "trip_key": c.trip_key,
                    "stop_id": c.stop_id,
                    "predictor_id": c.predictor_id,
                    "is_converging": c.is_converging,
                    "settling_horizon_s": c.settling_horizon_s,
                }
                for c in self.convergence
            ],
            "head_to_head": [
                {
                    "predictor_id": w.predictor_id,
                    "wins": w.wins,
                    "total": w.total,
                    "win_fraction": w.win_fraction,
                }
                for w in self.head_to_head
            ],
        }


# ── Main entry point ──────────────────────────────────────────────────────────


def compute_metrics(results: list[ValidationResult]) -> MetricsReport:
    """Compute a full analysis report from a list of ValidationResults.

    Handles the empty-input case gracefully (returns an empty MetricsReport with
    no crash).

    Args:
        results: ValidationResults collected from the engine (or loaded from storage).

    Returns:
        :class:`MetricsReport` with all four analysis sections populated.
    """
    if not results:
        return MetricsReport()

    df = pd.DataFrame([r.model_dump() for r in results])
    df["abs_error"] = df["error_s"].abs()
    df["bucket"] = df["horizon_s"].apply(_assign_bucket)

    # ── Per-predictor metrics ─────────────────────────────────────────────────
    per_predictor: list[PerPredictorMetrics] = []
    for pid, grp in df.groupby("predictor_id"):
        errors = grp["error_s"]
        abs_errors = grp["abs_error"]
        # MAPE: |error_s| as a fraction of the prediction horizon.
        # Use max(horizon_s, 1) to avoid division by zero for degenerate cases.
        mape_vals = abs_errors / grp["horizon_s"].clip(lower=1.0) * 100.0
        per_predictor.append(
            PerPredictorMetrics(
                predictor_id=str(pid),
                mae=float(abs_errors.mean()),
                rmse=float(math.sqrt(float((errors**2).mean()))),
                mape=float(mape_vals.mean()),
                bias=float(errors.mean()),
                count=int(len(grp)),
            )
        )

    # ── Error-vs-horizon buckets ──────────────────────────────────────────────
    horizon_buckets: list[HorizonBucket] = []
    for pid, pred_grp in df.groupby("predictor_id"):
        for label in _BUCKET_LABELS:
            bgrp = pred_grp[pred_grp["bucket"] == label]
            if bgrp.empty:
                continue
            horizon_buckets.append(
                HorizonBucket(
                    label=label,
                    predictor_id=str(pid),
                    mae=float(bgrp["abs_error"].mean()),
                    bias=float(bgrp["error_s"].mean()),
                    count=int(len(bgrp)),
                )
            )

    # ── Convergence per (trip_key, stop_id, predictor_id) ────────────────────
    convergence: list[ConvergenceSeries] = []
    for (tk, sid, pid), grp in df.groupby(["trip_key", "stop_id", "predictor_id"]):
        if len(grp) < 2:
            continue
        # Sort by horizon_s descending: earliest predictions first (large horizon → small).
        sorted_grp = grp.sort_values("horizon_s", ascending=False)
        abs_err_seq = sorted_grp["abs_error"].tolist()

        # Monotonically non-increasing abs_error as horizon shrinks?
        is_converging = all(
            abs_err_seq[i] >= abs_err_seq[i + 1] for i in range(len(abs_err_seq) - 1)
        )

        # Settling horizon: first horizon_s where |error_s| drops below 60 s.
        settling: float | None = None
        for _, row in sorted_grp.iterrows():
            if row["abs_error"] < 60.0:
                settling = float(row["horizon_s"])
                break

        convergence.append(
            ConvergenceSeries(
                trip_key=str(tk),
                stop_id=str(sid),
                predictor_id=str(pid),
                is_converging=is_converging,
                settling_horizon_s=settling,
            )
        )

    # ── Head-to-head win-rate ─────────────────────────────────────────────────
    wins: dict[str, int] = {}
    totals: dict[str, int] = {}
    for (_tk, _sid), grp in df.groupby(["trip_key", "stop_id"]):
        preds = grp.groupby("predictor_id")["abs_error"].min()
        if len(preds) < 2:
            continue
        winner = str(preds.idxmin())
        for pid_raw in preds.index:
            pid_str = str(pid_raw)
            totals[pid_str] = totals.get(pid_str, 0) + 1
            if pid_str == winner:
                wins[pid_str] = wins.get(pid_str, 0) + 1

    head_to_head: list[WinRate] = [
        WinRate(
            predictor_id=pid,
            wins=wins.get(pid, 0),
            total=totals[pid],
            win_fraction=wins.get(pid, 0) / totals[pid],
        )
        for pid in totals
    ]

    return MetricsReport(
        per_predictor=per_predictor,
        horizon_buckets=horizon_buckets,
        convergence=convergence,
        head_to_head=head_to_head,
    )
