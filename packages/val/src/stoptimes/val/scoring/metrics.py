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
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from scipy import stats as scipy_stats

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


# ── Stops-ahead bucket edges ───────────────────────────────────────────────────

_STOPS_AHEAD_LABELS: list[str] = ["1", "2-3", "4-5", "6-10", ">10"]


def _assign_stops_ahead_bucket(stops_remaining: int) -> str:
    """Map a stops_remaining count to a human-readable band label."""
    if stops_remaining <= 1:
        return "1"
    if stops_remaining <= 3:
        return "2-3"
    if stops_remaining <= 5:
        return "4-5"
    if stops_remaining <= 10:
        return "6-10"
    return ">10"


# ── Time-of-day bucket edges (local-hour, 3-hour bands) ────────────────────────

_TIME_OF_DAY_LABELS: list[str] = [
    "00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24",
]


def _assign_time_of_day_bucket(actual_arrival_epoch: float, tz: str) -> str:
    """Map an epoch timestamp to a 3-hour local-time-of-day band label."""
    dt = datetime.fromtimestamp(actual_arrival_epoch, tz=UTC).astimezone(ZoneInfo(tz))
    band_idx = dt.hour // 3
    return _TIME_OF_DAY_LABELS[band_idx]


# ── Result dataclasses ─────────────────────────────────────────────────────────


@dataclass
class PerPredictorMetrics:
    predictor_id: str
    mae: float            # mean |error_s|
    rmse: float           # root-mean-square error_s
    mape: float           # mean |error_s| / horizon_s × 100  (%)
    bias: float           # mean signed error_s  (+ = late, − = early)
    count: int
    r2: float = math.nan             # coefficient of determination; NaN on zero variance
    medae: float = math.nan          # median |error_s|
    abs_error_p50: float = math.nan  # 50th percentile of |error_s|
    abs_error_p90: float = math.nan  # 90th percentile of |error_s|
    abs_error_p95: float = math.nan  # 95th percentile of |error_s|
    abs_error_p99: float = math.nan  # 99th percentile of |error_s|
    within_30s: float = math.nan     # fraction of |error_s| <= 30 s
    within_60s: float = math.nan     # fraction of |error_s| <= 60 s
    skew: float = math.nan           # skewness of error_s; NaN when n < 3
    excess_kurtosis: float = math.nan  # excess kurtosis of error_s; NaN when n < 3


@dataclass
class RouteMetrics:
    route_id: str
    predictor_id: str
    mae: float
    bias: float
    count: int


@dataclass
class TimeOfDayMetrics:
    label: str             # e.g. "06-09" (local-hour 3-hour band)
    predictor_id: str
    mae: float
    bias: float
    count: int


@dataclass
class StopsAheadMetrics:
    label: str             # "1", "2-3", "4-5", "6-10", ">10"
    predictor_id: str
    mae: float
    bias: float
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
    per_route: list[RouteMetrics] = field(default_factory=list)
    per_time_of_day: list[TimeOfDayMetrics] = field(default_factory=list)
    per_stops_ahead: list[StopsAheadMetrics] = field(default_factory=list)

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
                "r2": m.r2,
                "medae": m.medae,
                "abs_error_p50": m.abs_error_p50,
                "abs_error_p90": m.abs_error_p90,
                "abs_error_p95": m.abs_error_p95,
                "abs_error_p99": m.abs_error_p99,
                "within_30s": m.within_30s,
                "within_60s": m.within_60s,
                "skew": m.skew,
                "excess_kurtosis": m.excess_kurtosis,
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
            "per_route": [
                {
                    "route_id": r.route_id,
                    "predictor_id": r.predictor_id,
                    "mae": r.mae,
                    "bias": r.bias,
                    "count": r.count,
                }
                for r in self.per_route
            ],
            "per_time_of_day": [
                {
                    "label": t.label,
                    "predictor_id": t.predictor_id,
                    "mae": t.mae,
                    "bias": t.bias,
                    "count": t.count,
                }
                for t in self.per_time_of_day
            ],
            "per_stops_ahead": [
                {
                    "label": s.label,
                    "predictor_id": s.predictor_id,
                    "mae": s.mae,
                    "bias": s.bias,
                    "count": s.count,
                }
                for s in self.per_stops_ahead
            ],
        }


# ── Main entry point ──────────────────────────────────────────────────────────


def compute_metrics(results: list[ValidationResult], tz: str = "UTC") -> MetricsReport:
    """Compute a full analysis report from a list of ValidationResults.

    Handles the empty-input case gracefully (returns an empty MetricsReport with
    no crash).

    Args:
        results: ValidationResults collected from the engine (or loaded from storage).
        tz:      IANA timezone name used to bucket ``per_time_of_day`` by local
            hour. The core stays agency-agnostic — callers pass their own
            timezone; it is never hardcoded to a specific city.

    Returns:
        :class:`MetricsReport` with all analysis sections populated.
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
        n = int(len(grp))
        # MAPE: |error_s| as a fraction of the prediction horizon.
        # Use max(horizon_s, 1) to avoid division by zero for degenerate cases.
        mape_vals = abs_errors / grp["horizon_s"].clip(lower=1.0) * 100.0

        # R²: 1 - SS_res / SS_tot, comparing predicted vs actual arrival epochs.
        # NaN when the actual values have zero variance (SS_tot == 0).
        ss_res = float((errors**2).sum())
        actual = grp["actual_arrival_epoch"]
        ss_tot = float(((actual - actual.mean()) ** 2).sum())
        r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

        if n >= 3:
            skew = float(scipy_stats.skew(errors.to_numpy()))
            excess_kurtosis = float(scipy_stats.kurtosis(errors.to_numpy(), fisher=True))
        else:
            skew = math.nan
            excess_kurtosis = math.nan

        per_predictor.append(
            PerPredictorMetrics(
                predictor_id=str(pid),
                mae=float(abs_errors.mean()),
                rmse=float(math.sqrt(float((errors**2).mean()))),
                mape=float(mape_vals.mean()),
                bias=float(errors.mean()),
                count=n,
                r2=r2,
                medae=float(abs_errors.median()),
                abs_error_p50=float(abs_errors.quantile(0.50)),
                abs_error_p90=float(abs_errors.quantile(0.90)),
                abs_error_p95=float(abs_errors.quantile(0.95)),
                abs_error_p99=float(abs_errors.quantile(0.99)),
                within_30s=float((abs_errors <= 30.0).mean()),
                within_60s=float((abs_errors <= 60.0).mean()),
                skew=skew,
                excess_kurtosis=excess_kurtosis,
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

    # ── Per-route slice ────────────────────────────────────────────────────────
    per_route: list[RouteMetrics] = []
    route_df = df[df["route_id"].notna()]
    if not route_df.empty:
        for (route_id, pid), grp in route_df.groupby(["route_id", "predictor_id"]):
            per_route.append(
                RouteMetrics(
                    route_id=str(route_id),
                    predictor_id=str(pid),
                    mae=float(grp["abs_error"].mean()),
                    bias=float(grp["error_s"].mean()),
                    count=int(len(grp)),
                )
            )

    # ── Per-time-of-day slice (local-hour, 3-hour bands) ──────────────────────
    df["_time_of_day"] = df["actual_arrival_epoch"].apply(
        lambda e: _assign_time_of_day_bucket(float(e), tz)
    )
    per_time_of_day: list[TimeOfDayMetrics] = []
    for (label, pid), grp in df.groupby(["_time_of_day", "predictor_id"]):
        per_time_of_day.append(
            TimeOfDayMetrics(
                label=str(label),
                predictor_id=str(pid),
                mae=float(grp["abs_error"].mean()),
                bias=float(grp["error_s"].mean()),
                count=int(len(grp)),
            )
        )

    # ── Per-stops-ahead slice ──────────────────────────────────────────────────
    per_stops_ahead: list[StopsAheadMetrics] = []
    stops_df = df[df["stops_remaining"].notna()]
    if not stops_df.empty:
        stops_df = stops_df.copy()
        stops_df["_stops_ahead"] = stops_df["stops_remaining"].apply(
            lambda s: _assign_stops_ahead_bucket(int(s))
        )
        for (label, pid), grp in stops_df.groupby(["_stops_ahead", "predictor_id"]):
            per_stops_ahead.append(
                StopsAheadMetrics(
                    label=str(label),
                    predictor_id=str(pid),
                    mae=float(grp["abs_error"].mean()),
                    bias=float(grp["error_s"].mean()),
                    count=int(len(grp)),
                )
            )

    return MetricsReport(
        per_predictor=per_predictor,
        horizon_buckets=horizon_buckets,
        convergence=convergence,
        head_to_head=head_to_head,
        per_route=per_route,
        per_time_of_day=per_time_of_day,
        per_stops_ahead=per_stops_ahead,
    )
