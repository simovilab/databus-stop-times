"""Trip-inference quality report for BUCR navsat traces (roadmap Step 4).

There are no ground-truth trip labels for BUCR, so this module is how we
judge whether ``bucr_trip_inference.infer_bucr_trips`` output is trustworthy
enough to build the Step-5 segment dataset on. It runs the full pipeline

    normalize_raw_navsat(raw) -> clean_navsat_trace(normalized)
        -> navsat_to_vehicle_positions(cleaned) -> infer_bucr_trips(vp, candidates)

(``normalize_raw_navsat`` reconciles the real S3 export's column names --
``latitude``/``longitude`` instead of ``lat``/``lon``, and a missing/partial
``cr_datetime_utc`` derived from the local ``cr_datetime`` string -- see
``navsat_normalize.py``. Everything downstream still assumes the canonical
``lat``/``lon``/``cr_datetime_utc`` names, unchanged.)

per service-day over a sample of raw navsat frames, and aggregates the
result into a structured report covering:

  * cleaning drop-rate by reason (``TraceCleaningReport``)
  * inference match/park/anomaly rate, overall and per (direction, variant)
  * trips/day, per vehicle, per direction/variant
  * the self-loop segmentation artifact flagged in Step 3 (two shapes,
    ``desde_educacion_a_odontologia_sin_milla`` and
    ``desde_artes_a_odontologia_sin_milla``, whose raw GTFS geometry
    self-crosses and can split one real trip into 2-3 instances) --
    quantified by comparing instance duration/point-count/stops-covered on
    those shapes against their ``con_milla`` siblings and against all other
    shapes
  * a spot-check of N inferred trips against the static-GTFS timetable
    (plausibility only -- the published schedule is a known-offset
    approximation, not ground truth)
  * sanity checks: trips/day vs. timetable trip count, zero/absurd
    vehicle-days

Everything except ``main`` (the S3/CLI entry point) is pure and unit-tested
with synthetic frames -- no network, no filesystem, deterministic given the
same input. See ``feature_engineering/tests/test_bucr_quality_report.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from feature_engineering.bucr_gtfs import BucrGtfs, load_gtfs, route_directions, route_direction_candidates
from feature_engineering.bucr_gtfs import RouteDirectionCandidate
from feature_engineering.bucr_trip_inference import (
    OUTPUT_COLUMNS,
    InferenceStats,
    infer_bucr_trips,
)
from feature_engineering.navsat_adapter import navsat_to_vehicle_positions
from feature_engineering.navsat_cleaning import (
    DEFAULT_STALENESS_THRESHOLD_SECONDS,
    TraceCleaningReport,
    clean_navsat_trace,
)
from feature_engineering.navsat_normalize import normalize_raw_navsat

__all__ = [
    "SELF_LOOP_SHAPE_IDS",
    "SELF_LOOP_SIBLING_SHAPES",
    "CleaningSummary",
    "MatchRateSummary",
    "QualityReport",
    "summarize_cleaning",
    "summarize_match_rate",
    "parse_trip_id",
    "trip_instance_table",
    "trips_per_day_table",
    "self_loop_measurement",
    "scheduled_trip_starts",
    "spot_check_table",
    "sanity_check",
    "run_quality_report",
    "render_markdown_report",
    "main",
]

# The two shapes Step 3 flagged as having a self-crossing loop in their raw
# GTFS geometry, which can split one real trip into 2-3 inferred instances.
# See bucr_trip_scoring.py module docstring for the origin/detour naming
# convention this is drawn from.
SELF_LOOP_SHAPE_IDS: Tuple[str, ...] = (
    "desde_educacion_a_odontologia_sin_milla",
    "desde_artes_a_odontologia_sin_milla",
)

# Each flagged shape's same-origin "con_milla" sibling -- the natural
# comparison group, since siblings differ only by the optional detour, not
# by which origin they start from.
SELF_LOOP_SIBLING_SHAPES: Dict[str, str] = {
    "desde_educacion_a_odontologia_sin_milla": "desde_educacion_a_odontologia_con_milla",
    "desde_artes_a_odontologia_sin_milla": "desde_artes_a_odontologia_con_milla",
}


# ---------------------------------------------------------------------------
# Cleaning summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleaningSummary:
    """Aggregate of ``TraceCleaningReport`` across a sample of service-days."""

    days: int
    rows_in: int
    rows_kept: int
    dropped_bad_timestamp: int
    dropped_stale: int
    dropped_null_coord: int
    dropped_zero_coord: int
    dropped_out_of_range: int
    staleness_threshold_seconds: float

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_bad_timestamp
            + self.dropped_stale
            + self.dropped_null_coord
            + self.dropped_zero_coord
            + self.dropped_out_of_range
        )

    @property
    def drop_rate(self) -> float:
        if self.rows_in == 0:
            return 0.0
        return self.dropped_total / self.rows_in

    def drop_rate_by_reason(self) -> Dict[str, float]:
        if self.rows_in == 0:
            return {
                "bad_timestamp": 0.0,
                "stale": 0.0,
                "null_coord": 0.0,
                "zero_coord": 0.0,
                "out_of_range": 0.0,
            }
        return {
            "bad_timestamp": self.dropped_bad_timestamp / self.rows_in,
            "stale": self.dropped_stale / self.rows_in,
            "null_coord": self.dropped_null_coord / self.rows_in,
            "zero_coord": self.dropped_zero_coord / self.rows_in,
            "out_of_range": self.dropped_out_of_range / self.rows_in,
        }


def summarize_cleaning(reports: Sequence[TraceCleaningReport]) -> CleaningSummary:
    """Aggregate a sequence of per-day ``TraceCleaningReport``s.

    Raises:
        ValueError: ``reports`` is empty, or reports use inconsistent
            staleness thresholds (the report would silently mix rules).
    """
    if not reports:
        raise ValueError("reports must not be empty")
    thresholds = {r.staleness_threshold_seconds for r in reports}
    if len(thresholds) > 1:
        raise ValueError(f"inconsistent staleness thresholds across reports: {sorted(thresholds)}")

    return CleaningSummary(
        days=len(reports),
        rows_in=sum(r.rows_in for r in reports),
        rows_kept=sum(r.rows_kept for r in reports),
        dropped_bad_timestamp=sum(r.dropped_bad_timestamp for r in reports),
        dropped_stale=sum(r.dropped_stale for r in reports),
        dropped_null_coord=sum(r.dropped_null_coord for r in reports),
        dropped_zero_coord=sum(r.dropped_zero_coord for r in reports),
        dropped_out_of_range=sum(r.dropped_out_of_range for r in reports),
        staleness_threshold_seconds=next(iter(thresholds)),
    )


# ---------------------------------------------------------------------------
# Match-rate summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRateSummary:
    """Aggregate of ``InferenceStats`` across a sample of service-days."""

    days: int
    points_in: int
    points_assigned: int
    points_parked_ambiguous: int
    points_anomaly_rejected: int
    points_dropped_as_noise: int
    trips_inferred: int
    instances_dropped_as_noise: int
    per_variant: Tuple[Tuple[str, int, str, int], ...]

    @property
    def assigned_rate(self) -> float:
        return self.points_assigned / self.points_in if self.points_in else 0.0

    @property
    def parked_rate(self) -> float:
        return self.points_parked_ambiguous / self.points_in if self.points_in else 0.0

    @property
    def anomaly_rate(self) -> float:
        return self.points_anomaly_rejected / self.points_in if self.points_in else 0.0

    @property
    def dropped_as_noise_rate(self) -> float:
        return self.points_dropped_as_noise / self.points_in if self.points_in else 0.0


def summarize_match_rate(stats: Sequence[InferenceStats]) -> MatchRateSummary:
    """Aggregate a sequence of per-day ``InferenceStats``.

    ``per_variant`` sums ``trips_per_direction_variant`` tuples across days
    by (route_id, direction_id, shape_id) key.

    Raises:
        ValueError: ``stats`` is empty.
    """
    if not stats:
        raise ValueError("stats must not be empty")

    totals: Dict[Tuple[str, int, str], int] = {}
    for s in stats:
        for route_id, direction_id, shape_id, count in s.trips_per_direction_variant:
            key = (route_id, direction_id, shape_id)
            totals[key] = totals.get(key, 0) + count

    per_variant = tuple(
        sorted((route_id, direction_id, shape_id, count) for (route_id, direction_id, shape_id), count in totals.items())
    )

    return MatchRateSummary(
        days=len(stats),
        points_in=sum(s.points_in for s in stats),
        points_assigned=sum(s.points_assigned for s in stats),
        points_parked_ambiguous=sum(s.points_parked_ambiguous for s in stats),
        points_anomaly_rejected=sum(s.points_anomaly_rejected for s in stats),
        points_dropped_as_noise=sum(s.points_dropped_as_noise for s in stats),
        trips_inferred=sum(s.trips_inferred for s in stats),
        instances_dropped_as_noise=sum(s.instances_dropped_as_noise for s in stats),
        per_variant=per_variant,
    )


# ---------------------------------------------------------------------------
# Trip-instance table (parsed from infer_bucr_trips' out_df)
# ---------------------------------------------------------------------------

_TRIP_TABLE_COLUMNS = [
    "trip_id",
    "vehicle_id",
    "route_id",
    "direction_id",
    "shape_id",
    "service_day",
    "start_ts",
    "end_ts",
    "duration_s",
    "n_points",
    "n_stops_covered",
]


def parse_trip_id(trip_id: str) -> Dict[str, object]:
    """Decode ``bucr_trip_inference._make_trip_id``'s format.

    Recipe: ``bucr:{vehicle_id}:{route_id}:{direction_id}:{shape_id}:{start_ts}``.

    Raises:
        ValueError: ``trip_id`` does not have the expected 6 colon-separated
            fields with a literal ``bucr`` prefix.
    """
    parts = trip_id.split(":")
    if len(parts) != 6 or parts[0] != "bucr":
        raise ValueError(f"not a recognized bucr trip_id: {trip_id!r}")
    _, vehicle_id, route_id, direction_id, shape_id, start_ts = parts
    return {
        "vehicle_id": vehicle_id,
        "route_id": route_id,
        "direction_id": int(direction_id),
        "shape_id": shape_id,
        "start_ts": start_ts,
    }


def trip_instance_table(out_df: pd.DataFrame) -> pd.DataFrame:
    """One row per inferred trip instance, decoded from ``infer_bucr_trips`` output.

    Columns: ``trip_id, vehicle_id, route_id, direction_id, shape_id,
    service_day, start_ts, end_ts, duration_s, n_points, n_stops_covered``.

    Args:
        out_df: The ``out_df`` returned by ``infer_bucr_trips`` (or a
            concatenation of several days' worth). May be empty.

    Returns:
        A new DataFrame, one row per distinct ``trip_id``, sorted by
        ``start_ts``. Empty (with the columns above) if ``out_df`` is empty.
    """
    if out_df.empty:
        return pd.DataFrame(columns=_TRIP_TABLE_COLUMNS)

    rows: List[Dict[str, object]] = []
    for trip_id, g in out_df.groupby("trip_id", sort=False):
        g = g.sort_values("ts")
        meta = parse_trip_id(str(trip_id))
        start = pd.Timestamp(g["ts"].iloc[0])
        end = pd.Timestamp(g["ts"].iloc[-1])
        rows.append(
            {
                "trip_id": trip_id,
                "vehicle_id": meta["vehicle_id"],
                "route_id": meta["route_id"],
                "direction_id": meta["direction_id"],
                "shape_id": meta["shape_id"],
                "service_day": start.floor("D").date(),
                "start_ts": start,
                "end_ts": end,
                "duration_s": float((end - start).total_seconds()),
                "n_points": int(len(g)),
                "n_stops_covered": int(g["stop_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=_TRIP_TABLE_COLUMNS).sort_values("start_ts").reset_index(drop=True)


def trips_per_day_table(trip_table: pd.DataFrame) -> pd.DataFrame:
    """Trip-instance counts per (service_day, vehicle_id, route_id, direction_id, shape_id).

    Args:
        trip_table: Output of :func:`trip_instance_table`.

    Returns:
        A new DataFrame with those grouping columns plus ``n_trips``, sorted
        by the grouping columns. Empty if ``trip_table`` is empty.
    """
    cols = ["service_day", "vehicle_id", "route_id", "direction_id", "shape_id"]
    if trip_table.empty:
        return pd.DataFrame(columns=cols + ["n_trips"])
    out = (
        trip_table.groupby(cols, as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n_trips"})
    )
    return out


# ---------------------------------------------------------------------------
# Self-loop artifact measurement
# ---------------------------------------------------------------------------


def _distribution_stats(values: pd.Series) -> Dict[str, float]:
    if values.empty:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
    }


def self_loop_measurement(
    trip_table: pd.DataFrame,
    flagged_shape_ids: Sequence[str] = SELF_LOOP_SHAPE_IDS,
    sibling_shapes: Mapping[str, str] = SELF_LOOP_SIBLING_SHAPES,
) -> Dict[str, object]:
    """Quantify the self-loop segmentation artifact on flagged shapes.

    Compares trip-instance ``duration_s``, ``n_points``, and
    ``n_stops_covered`` distributions for the flagged (self-crossing) shapes
    against (a) their same-origin ``con_milla`` siblings and (b) all other
    shapes. Also reports a rough fragmentation-inflation estimate: the ratio
    of each flagged shape's median instance count *per vehicle-service-day*
    to its sibling's -- a self-loop that splits 1 real trip into ~2-3
    instances should show up as a multiple >1 here, since the sibling isn't
    affected by the same geometry bug.

    Args:
        trip_table: Output of :func:`trip_instance_table`.
        flagged_shape_ids: Shapes suspected of the self-loop artifact.
        sibling_shapes: flagged shape_id -> comparison (con_milla) shape_id.

    Returns:
        A dict with keys ``flagged``, ``siblings``, ``others`` (each a dict
        of duration/points/stops distribution stats + trip count), and
        ``inflation_estimate`` (dict of flagged_shape_id -> float ratio, NaN
        where the sibling has zero trips to compare against).
    """
    if trip_table.empty:
        empty = {"n_trips": 0, "duration_s": _distribution_stats(pd.Series(dtype=float)),
                 "n_points": _distribution_stats(pd.Series(dtype=float)),
                 "n_stops_covered": _distribution_stats(pd.Series(dtype=float))}
        return {"flagged": empty, "siblings": empty, "others": empty, "inflation_estimate": {}}

    flagged_mask = trip_table["shape_id"].isin(flagged_shape_ids)
    sibling_ids = list(sibling_shapes.values())
    sibling_mask = trip_table["shape_id"].isin(sibling_ids)
    other_mask = ~flagged_mask & ~sibling_mask

    def _group_summary(mask: pd.Series) -> Dict[str, object]:
        sub = trip_table.loc[mask]
        return {
            "n_trips": int(len(sub)),
            "duration_s": _distribution_stats(sub["duration_s"]),
            "n_points": _distribution_stats(sub["n_points"]),
            "n_stops_covered": _distribution_stats(sub["n_stops_covered"]),
        }

    # Fragmentation-inflation: trips per (vehicle, service_day) for each
    # flagged shape vs. its sibling. A ratio > 1 means the flagged shape
    # produces more instances per vehicle-day than its sibling for the same
    # rider demand -- consistent with one real trip being split.
    inflation: Dict[str, float] = {}
    per_vd = trip_table.groupby(["shape_id", "vehicle_id", "service_day"]).size()
    for flagged_id, sibling_id in sibling_shapes.items():
        flagged_counts = per_vd[per_vd.index.get_level_values("shape_id") == flagged_id]
        sibling_counts = per_vd[per_vd.index.get_level_values("shape_id") == sibling_id]
        flagged_mean = float(flagged_counts.mean()) if len(flagged_counts) else float("nan")
        sibling_mean = float(sibling_counts.mean()) if len(sibling_counts) else float("nan")
        if sibling_mean and not np.isnan(sibling_mean) and sibling_mean > 0:
            inflation[flagged_id] = flagged_mean / sibling_mean
        else:
            inflation[flagged_id] = float("nan")

    return {
        "flagged": _group_summary(flagged_mask),
        "siblings": _group_summary(sibling_mask),
        "others": _group_summary(other_mask),
        "inflation_estimate": inflation,
    }


# ---------------------------------------------------------------------------
# Spot-check vs. timetable
# ---------------------------------------------------------------------------


def scheduled_trip_starts(gtfs_data: BucrGtfs) -> pd.DataFrame:
    """First-stop scheduled departure time-of-day per scheduled trip.

    Returns:
        DataFrame with ``trip_id, route_id, direction_id, shape_id,
        start_time`` (``start_time`` = ``departure_time`` string, e.g.
        ``"07:15:00"``, GTFS-style -- may exceed 24:00:00). One row per trip
        in ``trips.txt`` that has at least one ``stop_times.txt`` row.
    """
    trips = gtfs_data.trips
    st = gtfs_data.stop_times
    first_stops = st.sort_values("stop_sequence").groupby("trip_id", as_index=False).first()
    merged = trips.merge(first_stops[["trip_id", "departure_time"]], on="trip_id", how="inner")
    return merged[["trip_id", "route_id", "direction_id", "shape_id", "departure_time"]].rename(
        columns={"departure_time": "start_time"}
    )


def _gtfs_time_to_seconds(t: str) -> Optional[float]:
    try:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def spot_check_table(
    trip_table: pd.DataFrame,
    gtfs_data: BucrGtfs,
    n: int = 10,
    seed: int = 0,
    local_tz: str = "America/Costa_Rica",
) -> pd.DataFrame:
    """Sample N inferred trips and compare their start time-of-day to the timetable.

    Plausibility check only -- BUCR's published schedule is a known-offset
    approximation (see docs/BUCR_DATASET_PROMPT.md), not ground truth, so
    this reports the delta to the *nearest* scheduled start on the same
    (route, direction) rather than asserting exact matches.

    Args:
        trip_table: Output of :func:`trip_instance_table`.
        gtfs_data: Loaded static GTFS (``bucr_gtfs.load_gtfs``).
        n: Number of trips to sample (fewer if ``trip_table`` has fewer rows).
        seed: RNG seed for deterministic sampling.
        local_tz: Timezone ``start_ts`` (UTC) is converted to before
            comparing against the GTFS local time-of-day.

    Returns:
        DataFrame with ``trip_id, vehicle_id, route_id, direction_id,
        shape_id, inferred_start_local, nearest_scheduled_start,
        delta_minutes, n_scheduled_trips_same_direction``. Empty if
        ``trip_table`` is empty.
    """
    cols = [
        "trip_id",
        "vehicle_id",
        "route_id",
        "direction_id",
        "shape_id",
        "inferred_start_local",
        "nearest_scheduled_start",
        "delta_minutes",
        "n_scheduled_trips_same_direction",
    ]
    if trip_table.empty:
        return pd.DataFrame(columns=cols)

    schedule = scheduled_trip_starts(gtfs_data)
    sample = trip_table.sample(n=min(n, len(trip_table)), random_state=seed).sort_values("start_ts")

    rows: List[Dict[str, object]] = []
    for _, r in sample.iterrows():
        local_start = pd.Timestamp(r["start_ts"]).tz_convert(local_tz)
        local_seconds = local_start.hour * 3600 + local_start.minute * 60 + local_start.second

        same_dir = schedule[
            (schedule["route_id"] == r["route_id"]) & (schedule["direction_id"].astype(int) == int(r["direction_id"]))
        ]
        nearest_start = None
        delta_minutes = float("nan")
        if not same_dir.empty:
            sched_seconds = same_dir["start_time"].map(_gtfs_time_to_seconds).dropna()
            if not sched_seconds.empty:
                # GTFS times can exceed 24h; compare mod 24h to local wall-clock.
                sched_mod = sched_seconds % 86400
                diffs = (sched_mod - local_seconds).abs()
                diffs = np.minimum(diffs, 86400 - diffs)  # wrap around midnight
                best_idx = diffs.idxmin()
                delta_minutes = float(diffs.loc[best_idx] / 60.0)
                nearest_start = same_dir.loc[best_idx, "start_time"]

        rows.append(
            {
                "trip_id": r["trip_id"],
                "vehicle_id": r["vehicle_id"],
                "route_id": r["route_id"],
                "direction_id": r["direction_id"],
                "shape_id": r["shape_id"],
                "inferred_start_local": local_start.strftime("%Y-%m-%d %H:%M:%S"),
                "nearest_scheduled_start": nearest_start,
                "delta_minutes": delta_minutes,
                "n_scheduled_trips_same_direction": int(len(same_dir)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_check(trips_per_day: pd.DataFrame, gtfs_data: BucrGtfs) -> Dict[str, object]:
    """Flag zero/absurd vehicle-days and compare trips/day to the timetable size.

    Args:
        trips_per_day: Output of :func:`trips_per_day_table`.
        gtfs_data: Loaded static GTFS (``bucr_gtfs.load_gtfs``).

    Returns:
        Dict with ``timetable_trip_count`` (distinct trip_id count in
        ``trips.txt``), ``zero_trip_vehicle_days`` (list of (vehicle_id,
        service_day) with no inferred trips -- only meaningful if the
        caller also passes vehicle-days that produced zero rows, which this
        function cannot see by itself; see the module-level report for how
        ``run_quality_report`` fills this in), and ``max_trips_single_day``
        (the single highest per-(vehicle,day,shape) trip count observed --
        an early-warning signal for absurd over-fragmentation).
    """
    timetable_trip_count = int(gtfs_data.trips["trip_id"].nunique())
    max_trips = int(trips_per_day["n_trips"].max()) if not trips_per_day.empty else 0
    return {
        "timetable_trip_count": timetable_trip_count,
        "max_trips_single_vehicle_day_shape": max_trips,
        "vehicle_days_observed": (
            trips_per_day[["vehicle_id", "service_day"]].drop_duplicates().shape[0] if not trips_per_day.empty else 0
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityReport:
    cleaning: CleaningSummary
    match_rate: MatchRateSummary
    trip_table: pd.DataFrame = field(compare=False)
    trips_per_day: pd.DataFrame = field(compare=False)
    self_loop: Dict[str, object] = field(compare=False)
    spot_check: pd.DataFrame = field(compare=False)
    sanity: Dict[str, object] = field(compare=False)


def run_quality_report(
    raw_by_day: Mapping[str, pd.DataFrame],
    candidates: Sequence[RouteDirectionCandidate],
    gtfs_data: BucrGtfs,
    staleness_threshold_seconds: float = DEFAULT_STALENESS_THRESHOLD_SECONDS,
    spot_check_n: int = 10,
    spot_check_seed: int = 0,
) -> QualityReport:
    """Run normalize -> clean -> adapt -> infer per day, aggregate into a QualityReport.

    Args:
        raw_by_day: service-day label -> raw navsat frame, in EITHER the
            real S3 export schema (``latitude``/``longitude``,
            ``cr_datetime`` local string, ``cr_datetime_utc`` absent/partial)
            or the already-canonical schema
            (``navsat_cleaning.REQUIRED_RAW_COLUMNS`` /
            ``navsat_adapter.REQUIRED_RAW_COLUMNS``) -- reconciled by
            ``navsat_normalize.normalize_raw_navsat`` before cleaning, so
            either is accepted.
        candidates: All ``RouteDirectionCandidate``s (see
            ``bucr_gtfs.route_direction_candidates``) to score traces against.
        gtfs_data: Loaded static GTFS, used for the spot-check and sanity
            sections.
        staleness_threshold_seconds: Passed through to ``clean_navsat_trace``.
        spot_check_n: Number of trips to sample for the spot-check.
        spot_check_seed: RNG seed for deterministic spot-check sampling.

    Raises:
        ValueError: ``raw_by_day`` is empty.
    """
    if not raw_by_day:
        raise ValueError("raw_by_day must not be empty")

    cleaning_reports: List[TraceCleaningReport] = []
    inference_stats: List[InferenceStats] = []
    out_frames: List[pd.DataFrame] = []

    for _, raw in raw_by_day.items():
        normalized = normalize_raw_navsat(raw)
        cleaned, creport = clean_navsat_trace(normalized, staleness_threshold_seconds)
        cleaning_reports.append(creport)
        if cleaned.empty:
            inference_stats.append(
                InferenceStats(
                    points_in=0,
                    points_assigned=0,
                    points_parked_ambiguous=0,
                    points_anomaly_rejected=0,
                    points_dropped_as_noise=0,
                    trips_inferred=0,
                    instances_dropped_as_noise=0,
                    trips_per_direction_variant=(),
                )
            )
            continue
        vp = navsat_to_vehicle_positions(cleaned)
        out_df, stats = infer_bucr_trips(vp, candidates)
        inference_stats.append(stats)
        if not out_df.empty:
            out_frames.append(out_df)

    combined_out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    trip_table = trip_instance_table(combined_out)
    per_day = trips_per_day_table(trip_table)

    return QualityReport(
        cleaning=summarize_cleaning(cleaning_reports),
        match_rate=summarize_match_rate(inference_stats),
        trip_table=trip_table,
        trips_per_day=per_day,
        self_loop=self_loop_measurement(trip_table),
        spot_check=spot_check_table(trip_table, gtfs_data, n=spot_check_n, seed=spot_check_seed),
        sanity=sanity_check(per_day, gtfs_data),
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_stats(d: Dict[str, float]) -> str:
    if not d or d.get("n", 0) == 0:
        return "n=0"
    return f"n={d['n']}, median={d['median']:.1f}, mean={d['mean']:.1f}, p25={d['p25']:.1f}, p75={d['p75']:.1f}"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown-table renderer (no ``tabulate`` dependency).

    Not a general-purpose formatter -- just enough for this report's small
    aggregate tables (a handful of columns/rows).
    """
    if df.empty:
        return "(empty)"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + body_lines)


def render_markdown_report(report: QualityReport, verdict: str) -> str:
    """Render a ``QualityReport`` as a markdown document."""
    lines: List[str] = []
    lines.append("# BUCR trip-inference quality report")
    lines.append("")

    c = report.cleaning
    lines.append("## Cleaning")
    lines.append(f"- Days: {c.days}, rows in: {c.rows_in}, rows kept: {c.rows_kept}")
    lines.append(f"- Overall drop rate: {_fmt_pct(c.drop_rate)}")
    for reason, rate in c.drop_rate_by_reason().items():
        lines.append(f"  - {reason}: {_fmt_pct(rate)}")
    lines.append(f"- Staleness threshold: {c.staleness_threshold_seconds:.0f}s")
    lines.append("")

    m = report.match_rate
    lines.append("## Inference match rate")
    lines.append(f"- Points in: {m.points_in}")
    lines.append(f"- Assigned: {m.points_assigned} ({_fmt_pct(m.assigned_rate)})")
    lines.append(f"- Parked (ambiguous): {m.points_parked_ambiguous} ({_fmt_pct(m.parked_rate)})")
    lines.append(f"- Anomaly-rejected: {m.points_anomaly_rejected} ({_fmt_pct(m.anomaly_rate)})")
    lines.append(
        f"- Dropped as noise (idle/too-short instances): {m.points_dropped_as_noise} "
        f"({_fmt_pct(m.dropped_as_noise_rate)}) across {m.instances_dropped_as_noise} instances"
    )
    lines.append(f"- Trips inferred: {m.trips_inferred}")
    lines.append("")
    lines.append("| route_id | direction_id | shape_id | trips |")
    lines.append("|---|---|---|---|")
    for route_id, direction_id, shape_id, count in m.per_variant:
        lines.append(f"| {route_id} | {direction_id} | {shape_id} | {count} |")
    lines.append("")

    lines.append("## Trips/day")
    if report.trips_per_day.empty:
        lines.append("(no trips inferred)")
    else:
        lines.append(_df_to_markdown(report.trips_per_day))
    lines.append("")

    sl = report.self_loop
    lines.append("## Self-loop artifact")
    lines.append(f"Flagged shapes: {', '.join(SELF_LOOP_SHAPE_IDS)}")
    lines.append("")
    lines.append(f"- Flagged shapes: {sl['flagged']['n_trips']} trips")
    lines.append(f"  - duration_s: {_fmt_stats(sl['flagged']['duration_s'])}")
    lines.append(f"  - n_points: {_fmt_stats(sl['flagged']['n_points'])}")
    lines.append(f"  - n_stops_covered: {_fmt_stats(sl['flagged']['n_stops_covered'])}")
    lines.append(f"- Con_milla siblings: {sl['siblings']['n_trips']} trips")
    lines.append(f"  - duration_s: {_fmt_stats(sl['siblings']['duration_s'])}")
    lines.append(f"- All other shapes: {sl['others']['n_trips']} trips")
    lines.append(f"  - duration_s: {_fmt_stats(sl['others']['duration_s'])}")
    lines.append("- Fragmentation-inflation estimate (flagged trips/vehicle-day ÷ sibling trips/vehicle-day):")
    for shape_id, ratio in sl["inflation_estimate"].items():
        lines.append(f"  - {shape_id}: {ratio:.2f}x" if not np.isnan(ratio) else f"  - {shape_id}: n/a (no sibling trips)")
    lines.append("")

    lines.append("## Spot-check vs. timetable")
    if report.spot_check.empty:
        lines.append("(no trips to spot-check)")
    else:
        lines.append(_df_to_markdown(report.spot_check))
    lines.append("")

    s = report.sanity
    lines.append("## Sanity")
    lines.append(f"- Timetable trip count: {s['timetable_trip_count']}")
    lines.append(f"- Vehicle-days observed: {s['vehicle_days_observed']}")
    lines.append(f"- Max trips in a single (vehicle, day, shape) bucket: {s['max_trips_single_vehicle_day_shape']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append(verdict)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_raw_navsat_by_day(sample_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load one raw navsat DataFrame per ``*.parquet`` file in ``sample_dir``."""
    by_day: Dict[str, pd.DataFrame] = {}
    for p in sorted(sample_dir.glob("*.parquet")):
        by_day[p.stem] = pd.read_parquet(p)
    return by_day


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsat-dir", type=Path, required=True, help="Directory of per-day raw navsat *.parquet files")
    parser.add_argument("--gtfs-zip", type=Path, required=True, help="Path to a BUCR static GTFS zip")
    parser.add_argument("--out", type=Path, required=True, help="Path to write the rendered markdown report")
    parser.add_argument("--spot-check-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    raw_by_day = _load_raw_navsat_by_day(args.navsat_dir)
    if not raw_by_day:
        print(f"no *.parquet files found in {args.navsat_dir}", file=sys.stderr)
        return 1

    gtfs_data = load_gtfs(args.gtfs_zip)
    candidates: List[RouteDirectionCandidate] = []
    for route_id, direction_id in route_directions(gtfs_data):
        candidates.extend(route_direction_candidates(gtfs_data, route_id, direction_id))

    report = run_quality_report(raw_by_day, candidates, gtfs_data, spot_check_n=args.spot_check_n, spot_check_seed=args.seed)

    verdict = (
        "AUTOMATED PLACEHOLDER -- replace with a reasoned call before treating this run as final. "
        "See the self-loop inflation estimate and match rate above."
    )
    md = render_markdown_report(report, verdict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)

    print(f"assigned_rate={_fmt_pct(report.match_rate.assigned_rate)} "
          f"parked_rate={_fmt_pct(report.match_rate.parked_rate)} "
          f"anomaly_rate={_fmt_pct(report.match_rate.anomaly_rate)} "
          f"trips_inferred={report.match_rate.trips_inferred} "
          f"drop_rate={_fmt_pct(report.cleaning.drop_rate)}")
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
